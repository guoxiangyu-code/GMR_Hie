"""
Proposal-to-Event Adapter (P0-selection) and supporting modules.

Implements the full Part 2 Section 5 specification:
  - RelationEncoder: per-candidate relation feature r_i
  - Greedy diversity seed selection (on stop_gradient B0 quantities)
  - Two-layer decoder (mode-SA → proposal cross-attn → FFN+LN)
  - P0-selection: event_span frozen from seed spans (no regression)
  - Hungarian matching on detached cost, focal event loss, SmoothL1 quality loss
  - P0 inference: threshold 0.5 on sigmoid(event_logit), no NMS

Design rules enforced here (tests in tests/test_event_matching.py):
  - event_mask is True=valid throughout
  - padding modes never enter Hungarian cost matrix
  - RelationEncoder receives gradients from L_event and L_quality
  - seed selection uses stop_gradient z_i / s_i only (no trainable dependency)
  - P0-selection loss has NO span regression term
  - null queries: all valid modes get event target=0
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from models.flash_vtg_gmr.event_interface import EventInterfaceV1, M, FEATURE_DIM

logger = logging.getLogger(__name__)

# ── Fixed hyper-parameters (locked by document §5) ────────────────────────────
K_CANDIDATES: int = 50          # top-K from FlashVTG
NUM_MODES: int = M               # = 10 event slots
LAMBDA_DIV: float = 0.5          # diversity weight, fixed
EVENT_THRESHOLD: float = 0.5     # P0 inference threshold, fixed


# ── Focal loss helper ──────────────────────────────────────────────────────────
def focal_loss_binary(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss. logits and targets both (N,)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ── RelationEncoder ────────────────────────────────────────────────────────────
class RelationEncoder(nn.Module):
    """
    Maps a candidate's (feat, span, scale, logit) → relation feature r_i ∈ R^D.

    Input dim: D + 2 + 1 + 1 = D+4  (span=(start,end), scale scalar, logit scalar)
    """

    def __init__(self, feat_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        in_dim = feat_dim + 4   # feat | start | end | scale | logit
        self.proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(
        self,
        candidate_feat: torch.Tensor,   # (B, K, D)
        candidate_span: torch.Tensor,   # (B, K, 2)
        candidate_scale: torch.Tensor,  # (B, K, 1) or (B, K)
        candidate_logit: torch.Tensor,  # (B, K, 1) or (B, K)
    ) -> torch.Tensor:                  # (B, K, D)
        if candidate_scale.dim() == 2:
            candidate_scale = candidate_scale.unsqueeze(-1)
        if candidate_logit.dim() == 2:
            candidate_logit = candidate_logit.unsqueeze(-1)
        x = torch.cat([candidate_feat, candidate_span, candidate_scale, candidate_logit], dim=-1)
        return self.proj(x)


# ── Greedy diversity seed selection ───────────────────────────────────────────
def select_seeds(
    z: torch.Tensor,          # (B, K, D) stop_grad L2-normed feats
    s: torch.Tensor,          # (B, K)   stop_grad normalised scores
    candidate_mask: torch.Tensor,  # (B, K) bool, True=valid
    num_modes: int = NUM_MODES,
    lambda_div: float = LAMBDA_DIV,
) -> torch.Tensor:            # (B, num_modes) int64 seed indices (< K), -1 for padding
    """
    Greedy diversity selection (document §5.1).
    All inputs are stop_gradient'd by the caller.
    """
    B, K = s.shape
    device = s.device

    # Replace invalid candidates with very negative score so they are never selected.
    valid_s = s.clone()
    valid_s[~candidate_mask] = -1e9

    seed_indices = torch.full((B, num_modes), -1, dtype=torch.long, device=device)

    for b in range(B):
        selected: list[int] = []
        # Count how many valid candidates exist for this sample
        n_valid_b = int(candidate_mask[b].sum().item())
        max_seeds = min(num_modes, n_valid_b)

        for m_idx in range(max_seeds):
            if m_idx == 0:
                # seed_1 = argmax s_i
                score = valid_s[b].clone()
            else:
                # diversity: score_i = s_i + lambda_div * min_j(1 - cos(z_i, z_seed_j))
                z_seeds = z[b, selected, :]  # (m_idx, D)
                # cosine similarity between all K and each selected seed
                cos_sim = F.cosine_similarity(
                    z[b].unsqueeze(1),    # (K, 1, D)
                    z_seeds.unsqueeze(0), # (1, m_idx, D)
                    dim=-1,
                )  # (K, m_idx)
                min_dist = (1 - cos_sim).min(dim=1).values  # (K,)
                score = valid_s[b].clone() + lambda_div * min_dist

            # Mask out already-selected and invalid candidates
            for prev in selected:
                score[prev] = -1e9

            best = int(score.argmax().item())
            if valid_s[b, best] < -1e8:
                break  # no more valid candidates
            selected.append(best)
            seed_indices[b, m_idx] = best

    return seed_indices


# ── Two-layer event decoder ────────────────────────────────────────────────────
class EventDecoder(nn.Module):
    """
    Two-layer decoder per §5.1:
      1. mode self-attention (mask invalid queries/keys)
      2. mode-to-all-proposals cross-attention (mask query w/ event_mask, key w/ candidate_mask)
      3. FFN + LayerNorm
    """

    def __init__(self, d_model: int = FEATURE_DIM, nhead: int = 4, dim_ff: int = 512) -> None:
        super().__init__()
        # Layer 1: mode self-attention
        self.sa = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_sa = nn.LayerNorm(d_model)

        # Layer 2: cross-attention (mode → proposals)
        self.ca = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln_ca = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        event_emb: torch.Tensor,         # (B, M, D)
        event_mask: torch.Tensor,         # (B, M) bool True=valid
        proposal_mem: torch.Tensor,       # (B, K, D)
        candidate_mask: torch.Tensor,     # (B, K) bool True=valid
    ) -> torch.Tensor:                    # (B, M, D)
        B, Mq, D = event_emb.shape

        # MHA key_padding_mask: True means IGNORE
        # event_mask is True=valid → invert for attn
        sa_key_pad = ~event_mask   # (B, M) True=ignore

        # self-attention
        h = event_emb
        sa_out, _ = self.sa(h, h, h, key_padding_mask=sa_key_pad)
        h = self.ln_sa(h + sa_out)

        # cross-attention
        ca_key_pad = ~candidate_mask   # (B, K) True=ignore
        ca_out, _ = self.ca(h, proposal_mem, proposal_mem, key_padding_mask=ca_key_pad)
        h = self.ln_ca(h + ca_out)

        # FFN
        h = self.ln_ffn(h + self.ffn(h))

        # Zero-out invalid slots
        h = h * event_mask.unsqueeze(-1).float()
        return h


# ── P0 Adapter ─────────────────────────────────────────────────────────────────
class ProposalToEventAdapter(nn.Module):
    """
    Proposal-to-Event Adapter (P0-selection).

    §5.2: event_span = stop_gradient(seed_span).
    No boundary regression in the training loss.
    """

    def __init__(
        self,
        feat_dim: int = FEATURE_DIM,
        num_modes: int = NUM_MODES,
        num_heads: int = 4,
        dim_ff: int = 512,
        variant: str = "P0",  # "P0" or "P0-R" (P0-R adds residual span regression)
        residual_rho: float = 0.5,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.num_modes = num_modes
        self.variant = variant
        self.residual_rho = residual_rho

        # RelationEncoder: candidate → relation feature r_i
        self.relation_encoder = RelationEncoder(feat_dim)

        # Seed initialisation projections
        self.W_seed = nn.Linear(feat_dim, feat_dim, bias=False)
        self.W_query = nn.Linear(feat_dim, feat_dim, bias=False)
        self.slot_embeddings = nn.Embedding(num_modes, feat_dim)

        # Decoder
        self.decoder = EventDecoder(feat_dim, num_heads, dim_ff)

        # Output heads
        self.event_head = nn.Linear(feat_dim, 1)    # → event_logit
        self.quality_head = nn.Linear(feat_dim, 1)  # → quality_logit

        # P0-R only: residual span head (zero-init last layer)
        if variant == "P0-R":
            self.span_residual_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.GELU(),
                nn.Linear(feat_dim, 2),
            )
            # Zero-init last layer weights/biases for P0-R (§5.3)
            nn.init.zeros_(self.span_residual_head[-1].weight)
            nn.init.zeros_(self.span_residual_head[-1].bias)

    def forward(
        self,
        candidate_feat: torch.Tensor,   # (B, K, D) – from backbone
        candidate_mask: torch.Tensor,   # (B, K) bool True=valid
        candidate_span: torch.Tensor,   # (B, K, 2) normalised [0,1]
        candidate_logit: torch.Tensor,  # (B, K) pre-sigmoid
        candidate_scale: torch.Tensor,  # (B, K)
        query_global: torch.Tensor,     # (B, D)
    ) -> dict:
        B, K, D = candidate_feat.shape
        device = candidate_feat.device
        M = self.num_modes

        # ── Step 1: relation features (trainable, enters gradients) ───────────
        r = self.relation_encoder(candidate_feat, candidate_span, candidate_scale, candidate_logit)
        # (B, K, D) – these enter the decoder memory and the seed init → gradient flows

        # ── Step 2: diversity seed selection on DETACHED quantities ──────────
        with torch.no_grad():
            # z_i = stop_gradient(L2Norm(candidate_feat_i))
            z = F.normalize(candidate_feat.detach(), dim=-1)
            # s_i = stop_gradient(normalised score)
            raw_score = torch.sigmoid(candidate_logit.detach())
            s_normalised = raw_score  # already in [0,1]; no additional normalisation needed

        seed_idx = select_seeds(z, s_normalised, candidate_mask, M, LAMBDA_DIV)
        # seed_idx: (B, M), -1 means padding slot

        # ── Step 3: event mask (True = real seed) ─────────────────────────────
        event_mask = (seed_idx >= 0)   # (B, M) bool

        # ── Step 4: gather seed relation features and spans ───────────────────
        # Clamp -1 indices to 0 temporarily (masked out later)
        safe_idx = seed_idx.clamp(min=0)   # (B, M)

        # Gather seed relation features: (B, M, D)
        idx_expanded = safe_idx.unsqueeze(-1).expand(-1, -1, D)
        seed_r = torch.gather(r, 1, idx_expanded)   # (B, M, D) – differentiable

        # Gather seed spans (stop_gradient by design for P0):
        seed_span_raw = torch.gather(
            candidate_span, 1, safe_idx.unsqueeze(-1).expand(-1, -1, 2)
        )  # (B, M, 2)
        seed_span = seed_span_raw.detach()   # frozen for P0

        # Gather seed scale (for P0-R duration)
        if self.variant == "P0-R":
            seed_scale = torch.gather(candidate_scale, 1, safe_idx)  # (B, M)
            seed_duration = (seed_span[:, :, 1] - seed_span[:, :, 0]).detach()  # (B, M)

        # ── Step 5: initialise event slot embeddings ──────────────────────────
        slot_emb = self.slot_embeddings.weight.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        query_proj = self.W_query(query_global).unsqueeze(1).expand(-1, M, -1)  # (B, M, D)
        seed_proj = self.W_seed(seed_r)                                          # (B, M, D)
        event_emb = seed_proj + slot_emb + query_proj                            # (B, M, D)
        # Zero-out invalid slots
        event_emb = event_emb * event_mask.unsqueeze(-1).float()

        # ── Step 6: two-layer decoder ─────────────────────────────────────────
        event_feat = self.decoder(event_emb, event_mask, r, candidate_mask)
        # event_feat: (B, M, D), invalid slots already zeroed

        # ── Step 7: output heads ──────────────────────────────────────────────
        event_logit = self.event_head(event_feat).squeeze(-1)     # (B, M)
        quality_logit = self.quality_head(event_feat).squeeze(-1) # (B, M)

        # Mask invalid slots' logits to a large negative value (no contribution to loss)
        INF = 1e4
        event_logit = event_logit.masked_fill(~event_mask, -INF)
        quality_logit = quality_logit.masked_fill(~event_mask, -INF)

        # ── Step 8: event span ────────────────────────────────────────────────
        if self.variant == "P0-R":
            delta = self.span_residual_head(event_feat)           # (B, M, 2)
            span_residual = torch.tanh(delta) * self.residual_rho * seed_duration.unsqueeze(-1)
            event_span = seed_span + span_residual
            event_span = event_span.clamp(0.0, 1.0)
        else:
            # P0-selection: span is exactly the frozen seed span
            event_span = seed_span  # already detached

        # Zero invalid slot spans
        event_span = event_span * event_mask.unsqueeze(-1).float()

        return dict(
            event_feat=event_feat,          # (B, M, D) – for CountHead / HMSA
            event_logit=event_logit,        # (B, M)
            quality_logit=quality_logit,    # (B, M)
            event_span=event_span,          # (B, M, 2)
            event_mask=event_mask,          # (B, M) bool
            seed_idx=seed_idx,              # (B, M)
            seed_span=seed_span,            # (B, M, 2) detached
        )


# ── Hungarian matching ─────────────────────────────────────────────────────────
def hungarian_matching(
    event_span: torch.Tensor,    # (M_valid, 2) – only valid modes
    event_logit: torch.Tensor,   # (M_valid,)
    gt_spans: torch.Tensor,      # (G, 2)
) -> Tuple[list, list]:
    """
    One-to-one Hungarian matching per §5.4.
    Returns (mode_indices, gt_indices) in matched pairs.
    All inputs are CPU numpy-compatible.
    """
    M_v = event_span.shape[0]
    G = gt_spans.shape[0]

    if M_v == 0 or G == 0:
        return [], []

    with torch.no_grad():
        # Cost matrix C(m, j) = 2*L1 + 2*(1-tIoU) - sigmoid(logit_m)
        span_m = event_span.detach().unsqueeze(1).expand(-1, G, -1)   # (M_v, G, 2)
        span_g = gt_spans.unsqueeze(0).expand(M_v, -1, -1)             # (M_v, G, 2)

        # L1 cost
        l1_cost = (span_m - span_g).abs().sum(-1)  # (M_v, G)

        # tIoU cost
        inter_start = torch.max(span_m[:, :, 0], span_g[:, :, 0])
        inter_end   = torch.min(span_m[:, :, 1], span_g[:, :, 1])
        inter = (inter_end - inter_start).clamp(min=0)
        union = (
            (span_m[:, :, 1] - span_m[:, :, 0])
            + (span_g[:, :, 1] - span_g[:, :, 0])
            - inter
        ).clamp(min=1e-6)
        tiou = inter / union   # (M_v, G)

        sig = torch.sigmoid(event_logit.detach()).unsqueeze(1).expand(-1, G)  # (M_v, G)

        cost = 2 * l1_cost + 2 * (1 - tiou) - sig
        cost_np = cost.float().cpu().numpy()

    row_inds, col_inds = linear_sum_assignment(cost_np)
    return row_inds.tolist(), col_inds.tolist()


# ── Adapter losses ────────────────────────────────────────────────────────────
def compute_adapter_losses(
    event_logit: torch.Tensor,    # (B, M)
    quality_logit: torch.Tensor,  # (B, M)
    event_span: torch.Tensor,     # (B, M, 2)  – may be detached
    event_mask: torch.Tensor,     # (B, M) bool
    targets: list,                # list of per-sample dicts with 'relevant_windows' (normalised) and 'is_null'
    variant: str = "P0",
) -> dict:
    """
    Compute L_event (focal) + L_quality (SmoothL1) for all samples.
    For P0-selection: NO span regression loss.
    For P0-R: also computes span L1 + tIoU for matched modes.
    These losses must be computed BEFORE positive-query filtering.
    """
    B, Mn = event_logit.shape
    device = event_logit.device

    total_event_loss = torch.zeros(1, device=device)
    total_quality_loss = torch.zeros(1, device=device)
    total_span_loss = torch.zeros(1, device=device)
    count = 0

    for b in range(B):
        mask_b = event_mask[b]           # (M,) bool
        valid_idx = torch.where(mask_b)[0]  # indices of valid modes
        Mv = valid_idx.numel()

        if Mv == 0:
            continue

        logit_b = event_logit[b, valid_idx]    # (Mv,)
        qual_b = quality_logit[b, valid_idx]   # (Mv,)
        span_b = event_span[b, valid_idx]       # (Mv, 2)

        is_null = targets[b].get("is_null", False)
        gt_windows = targets[b].get("relevant_windows_norm", None)

        if is_null or gt_windows is None or len(gt_windows) == 0:
            # Null query: all valid modes get event target=0
            event_target = torch.zeros(Mv, device=device)
            quality_target = torch.zeros(Mv, device=device)
            # focal loss
            loss_e = focal_loss_binary(logit_b, event_target, reduction="mean")
            loss_q = F.smooth_l1_loss(
                torch.sigmoid(qual_b), quality_target, reduction="mean"
            )
            total_event_loss = total_event_loss + loss_e
            total_quality_loss = total_quality_loss + loss_q
            count += 1
            continue

        # Convert GT to tensor
        gt_spans = torch.tensor(gt_windows, dtype=torch.float32, device=device)  # (G, 2)
        # Remove padding rows (inf entries from dataset)
        valid_gt = (gt_spans[:, 0] < 1e8) & (gt_spans[:, 1] < 1e8)
        gt_spans = gt_spans[valid_gt]
        G = gt_spans.shape[0]
        if G == 0:
            # Treat as null
            event_target = torch.zeros(Mv, device=device)
            quality_target = torch.zeros(Mv, device=device)
            total_event_loss = total_event_loss + focal_loss_binary(logit_b, event_target, reduction="mean")
            total_quality_loss = total_quality_loss + F.smooth_l1_loss(
                torch.sigmoid(qual_b), quality_target, reduction="mean"
            )
            count += 1
            continue

        # Hungarian matching
        mode_inds, gt_inds = hungarian_matching(span_b, logit_b, gt_spans)

        # Build event targets
        event_target = torch.zeros(Mv, device=device)
        for mi in mode_inds:
            event_target[mi] = 1.0

        # focal event loss
        loss_e = focal_loss_binary(logit_b, event_target, reduction="mean")
        total_event_loss = total_event_loss + loss_e

        # quality target = max_j tIoU(event_span_m, gt_j) – detached
        with torch.no_grad():
            span_m_exp = span_b.unsqueeze(1).expand(-1, G, -1)
            span_g_exp = gt_spans.unsqueeze(0).expand(Mv, -1, -1)
            inter_s = torch.max(span_m_exp[:, :, 0], span_g_exp[:, :, 0])
            inter_e = torch.min(span_m_exp[:, :, 1], span_g_exp[:, :, 1])
            inter = (inter_e - inter_s).clamp(min=0)
            union = (
                (span_m_exp[:, :, 1] - span_m_exp[:, :, 0])
                + (span_g_exp[:, :, 1] - span_g_exp[:, :, 0])
                - inter
            ).clamp(min=1e-6)
            tiou_mat = (inter / union)   # (Mv, G)
            quality_target = tiou_mat.max(dim=1).values   # (Mv,)

        loss_q = F.smooth_l1_loss(torch.sigmoid(qual_b), quality_target, reduction="mean")
        total_quality_loss = total_quality_loss + loss_q

        # P0-R: span regression for matched modes only
        if variant == "P0-R" and len(mode_inds) > 0:
            m_t = torch.tensor(mode_inds, dtype=torch.long, device=device)
            g_t = torch.tensor(gt_inds, dtype=torch.long, device=device)
            pred_spans_matched = span_b[m_t]
            gt_spans_matched = gt_spans[g_t]
            # span L1 + tIoU
            span_l1 = F.l1_loss(pred_spans_matched, gt_spans_matched, reduction="mean")
            # tIoU term
            ps, pe = pred_spans_matched[:, 0], pred_spans_matched[:, 1]
            gs, ge = gt_spans_matched[:, 0], gt_spans_matched[:, 1]
            inter_sp = torch.max(ps, gs)
            inter_ep = torch.min(pe, ge)
            inter_r = (inter_ep - inter_sp).clamp(min=0)
            union_r = (pe - ps) + (ge - gs) - inter_r
            tiou_r = inter_r / union_r.clamp(min=1e-6)
            span_tiou_loss = (1 - tiou_r).mean()
            total_span_loss = total_span_loss + 5.0 * span_l1 + 2.0 * span_tiou_loss

        count += 1

    if count > 0:
        total_event_loss = total_event_loss / count
        total_quality_loss = total_quality_loss / count
        total_span_loss = total_span_loss / count

    losses = {
        "loss_event": total_event_loss,
        "loss_quality": total_quality_loss,
    }
    if variant == "P0-R":
        losses["loss_span"] = total_span_loss
    return losses


# ── P0 Inference: build EventInterfaceV1 ──────────────────────────────────────
@torch.no_grad()
def p0_inference(
    adapter_out: dict,
    query_global: torch.Tensor,
    b0_sha: Optional[str] = None,
    p0_sha: Optional[str] = None,
    fm_sha: Optional[str] = None,
) -> EventInterfaceV1:
    """
    Convert adapter forward() output to EventInterfaceV1.
    Inference threshold 0.5 on sigmoid(event_logit); no NMS.
    """
    event_feat = adapter_out["event_feat"]
    event_logit = adapter_out["event_logit"]
    quality_logit = adapter_out["quality_logit"]
    event_span = adapter_out["event_span"]
    event_mask = adapter_out["event_mask"]

    return EventInterfaceV1(
        event_feat=event_feat,
        event_span=event_span,
        adapter_event_logit=event_logit,
        adapter_quality_logit=quality_logit,
        event_mask=event_mask,
        query_global=query_global,
        baseline_checkpoint_sha256=b0_sha,
        public_p0_checkpoint_sha256=p0_sha,
        feature_manifest_sha256=fm_sha,
    )
