"""
CountHeadV1 and Adaptive Event Cardinality (AEC) module.

Implements Part 2 §8 (AGC-Direct) and §9 (AEC-CE / AEC + contrastive).

Key rules:
  - G0 and C1 use CountHeadV1 with **identical initialisation** (isolated RNG fork).
  - C1/C2 use AEC-CE (raw proposals → event modes as input).
  - C2 adds count contrastive (weight 0.1).
  - C1-Enhanced is non-blocking; C1/C2 NEVER include max/expected-count/consistency.
  - Unique empty-set decision: argmax P_count == 0. No second gate.
  - count 4+ uses tau_mode threshold for P0 event modes, and tau_raw for raw proposals.
  - No NMS in either G0 or C1/C2 output selection.

Effective-number class weighting formula (§8):
  w[c] = 1 / (1 - beta^n_c)   beta = (N-1)/N  clipped to [0.5, 2.0]
  where N = total training samples, n_c = count of class c.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NUM_COUNT_CLASSES: int = 5   # {0, 1, 2, 3, 4+}
COUNT_W_MIN: float = 0.5
COUNT_W_MAX: float = 2.0


# ── Masked pooling utilities ───────────────────────────────────────────────────
def masked_mean(x: torch.Tensor, mask: torch.Tensor, audit_counter: Optional[list] = None) -> torch.Tensor:
    """
    Mean-pool x (B, L, D) over valid positions given mask (B, L) bool True=valid.
    Returns (B, D). If entire row is masked → zero vector + audit log.
    """
    mask_f = mask.float().unsqueeze(-1)   # (B, L, 1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)   # (B, 1)
    out = (x * mask_f).sum(dim=1) / denom      # (B, D)

    # Audit: detect zero-mask rows
    zero_rows = (mask.sum(dim=1) == 0)
    if zero_rows.any():
        n_zero = int(zero_rows.sum().item())
        if audit_counter is not None:
            audit_counter[0] += n_zero
        else:
            logger.debug("masked_mean: %d row(s) have no valid entries; returning zero vector.", n_zero)
    return out


# ── CountHeadV1 ───────────────────────────────────────────────────────────────
class CountHeadV1(nn.Module):
    """
    Shared count head for G0 (raw proposals) and C1/C2 (event modes).

    Architecture (§8):
        t = LayerNorm(GELU(Linear(512,256)(text_mean)))
        s = LayerNorm(GELU(Linear(256,256)(set_mean)))
        g = LayerNorm(GELU(Linear(512,256)(concat(t,s))))
        count_logits = Linear(256,5)(g)

    text_mean: (B, 512) – from query token MaskedMean (512-dim because CLIP text dim).
    set_mean:  (B, 256) – from candidate/event MaskedMean.

    Dropout is FIXED at 0 (§8): ensures G0/C1 are parameter-for-parameter comparable.
    Does NOT use query_global, video-memory max, set max, proposal/event logits, or
    expected count in the main path.
    """

    def __init__(
        self,
        text_dim: int = 512,   # CLIP text token dim (may differ from hidden_dim)
        set_dim: int = 256,    # candidate/event feature dim
        hidden_dim: int = 256,
        num_classes: int = NUM_COUNT_CLASSES,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.set_proj = nn.Sequential(
            nn.Linear(set_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def encode(self, text_mean: torch.Tensor, set_mean: torch.Tensor) -> torch.Tensor:
        """Returns g (B, hidden_dim)."""
        t = self.text_proj(text_mean)
        s = self.set_proj(set_mean)
        g = self.gate(torch.cat([t, s], dim=-1))
        return g

    def forward(self, text_mean: torch.Tensor, set_mean: torch.Tensor) -> torch.Tensor:
        """Returns count_logits (B, 5)."""
        g = self.encode(text_mean, set_mean)
        return self.classifier(g)


def init_count_head_isolated(
    module: CountHeadV1,
    seed: int,
    rng_key: str = "CountHeadV1",
) -> None:
    """
    Initialise CountHeadV1 using an isolated RNG fork so that creation order of
    other modules does NOT affect CountHeadV1 parameters.

    Both G0 and C1 must call this with the same (seed, rng_key) to guarantee
    element-wise identical initialisation (§9.1).
    """
    # Save current RNG state
    saved_state = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # Derive a deterministic seed from (seed, rng_key)
    import hashlib
    digest = hashlib.sha256(f"{seed}:{rng_key}".encode()).digest()
    fork_seed = int.from_bytes(digest[:4], "little") & 0x7FFFFFFF

    try:
        torch.manual_seed(fork_seed)
        module.apply(_reset_linear_weight)
    finally:
        # Restore RNG state
        torch.set_rng_state(saved_state)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _reset_linear_weight(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ── Count contrastive (class-balanced queue) ───────────────────────────────────
class CountContrastiveHead(nn.Module):
    """
    Count contrastive loss with class-balanced memory queue (§9.4).
    g_con = L2Norm(Projection(g)) ; Projection is a simple linear layer.

    Temperature tau is a hyperparameter (not trained).
    Queue stores detached embeddings per count class {0,1,2,3,4+}.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        proj_dim: int = 128,
        num_classes: int = NUM_COUNT_CLASSES,
        queue_size_per_class: int = 64,
        tau: float = 0.07,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim, proj_dim)
        self.num_classes = num_classes
        self.qsize = queue_size_per_class
        self.tau = tau
        # Queues: one FIFO per class (not parameters, just buffers)
        self.register_buffer(
            "queue", torch.zeros(num_classes, queue_size_per_class, proj_dim)
        )
        self.register_buffer(
            "queue_labels", -torch.ones(num_classes, queue_size_per_class, dtype=torch.long)
        )
        self.register_buffer(
            "queue_ptr", torch.zeros(num_classes, dtype=torch.long)
        )

    @torch.no_grad()
    def _enqueue(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Enqueue (detached) embeddings per class."""
        for cls in range(self.num_classes):
            mask = (labels == cls)
            if not mask.any():
                continue
            embs = embeddings[mask]   # (n, proj_dim)
            ptr = int(self.queue_ptr[cls].item())
            for emb in embs:
                self.queue[cls, ptr % self.qsize] = emb.detach()
                self.queue_labels[cls, ptr % self.qsize] = cls
                ptr += 1
            self.queue_ptr[cls] = ptr

    def forward(
        self,
        g: torch.Tensor,           # (B, feat_dim)
        count_class: torch.Tensor, # (B,) int64 in {0,1,2,3,4}
    ) -> torch.Tensor:
        """Returns scalar contrastive loss."""
        proj = F.normalize(self.proj(g), dim=-1)   # (B, proj_dim)

        # Build all negative/positive keys from queue
        # queue: (num_classes, qsize, proj_dim) → flatten valid entries
        all_keys = self.queue.view(-1, self.queue.shape[-1])      # (C*Q, proj_dim)
        all_labels = self.queue_labels.view(-1)                    # (C*Q,)
        valid = (all_labels >= 0)
        if valid.sum() < 2:
            # Not enough queue entries yet – return 0
            self._enqueue(proj, count_class)
            return g.sum() * 0.0

        keys = all_keys[valid]         # (N_k, proj_dim)
        key_labels = all_labels[valid] # (N_k,)

        loss = torch.tensor(0.0, device=g.device)
        n_terms = 0
        for i in range(proj.shape[0]):
            ci = count_class[i].item()
            sim = F.cosine_similarity(proj[i].unsqueeze(0), keys, dim=-1) / self.tau  # (N_k,)
            # Positive mask: same class in queue
            pos_mask = (key_labels == ci)
            if not pos_mask.any():
                continue
            # Exclude self if present (unlikely in queue; no exact match by construction)
            # Softmax denominator: all entries with index != self (a != i)
            # Here all keys are from the queue (previous batch), so no exact i overlap
            log_denom = torch.logsumexp(sim, dim=0)
            log_pos = torch.logsumexp(sim[pos_mask], dim=0) - math.log(pos_mask.sum().item())
            loss = loss - log_pos + log_denom
            n_terms += 1

        if n_terms > 0:
            loss = loss / n_terms

        # Enqueue current batch
        self._enqueue(proj, count_class)
        return loss


# ── AEC module (wraps CountHeadV1 for C1/C2) ─────────────────────────────────
class AdaptiveEventCardinality(nn.Module):
    """
    AEC-CE module for C1 (no contrastive) and C2 (+ contrastive).

    Inputs come from EventInterfaceV1:
        text_tokens  (B, L_txt, D_txt) + text_mask (B, L_txt)
        event_feat   (B, M, 256)       + event_mask (B, M)

    Usage:
        variant="C1"  → no contrastive
        variant="C2"  → contrastive weight 0.1
    """

    def __init__(
        self,
        text_dim: int = 512,
        event_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = NUM_COUNT_CLASSES,
        variant: str = "C1",       # "C1" or "C2"
        contrastive_weight: float = 0.1,
        tau_count: float = 1.0,    # temperature for inference softmax
        seed: int = 2024,
        count_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert variant in ("C1", "C2", "G0", "G0-Con"), f"Unknown variant {variant}"
        self.variant = variant
        self.contrastive_weight = contrastive_weight
        self.tau_count = tau_count

        # Shared count head
        self.count_head = CountHeadV1(
            text_dim=text_dim, set_dim=event_dim,
            hidden_dim=hidden_dim, num_classes=num_classes,
        )
        # Initialise with isolated RNG fork
        init_count_head_isolated(self.count_head, seed=seed, rng_key="CountHeadV1")

        # Count contrastive (C2 / G0-Con only)
        self.use_contrastive = variant in ("C2", "G0-Con")
        if self.use_contrastive:
            self.contrastive_head = CountContrastiveHead(
                feat_dim=hidden_dim, num_classes=num_classes
            )

        # Class weights buffer (for loss)
        if count_class_weights is not None:
            self.register_buffer("class_weights", count_class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_classes))

    def forward(
        self,
        text_feat: torch.Tensor,    # (B, L, D_txt) or (B, D_txt) already pooled
        text_mask: torch.Tensor,    # (B, L) bool True=valid (ignored if text_feat is 2D)
        set_feat: torch.Tensor,     # (B, K, D_set) or (B, D_set) already pooled
        set_mask: torch.Tensor,     # (B, K) bool True=valid (ignored if set_feat is 2D)
        count_class: Optional[torch.Tensor] = None,  # (B,) int64, for loss
        audit_counter: Optional[list] = None,
    ) -> dict:
        # Pool if needed
        if text_feat.dim() == 3:
            text_mean = masked_mean(text_feat, text_mask, audit_counter)  # (B, D_txt)
        else:
            text_mean = text_feat

        if set_feat.dim() == 3:
            set_mean = masked_mean(set_feat, set_mask, audit_counter)     # (B, D_set)
        else:
            set_mean = set_feat

        # CountHeadV1
        g = self.count_head.encode(text_mean, set_mean)   # (B, hidden_dim)
        count_logits = self.count_head.classifier(g)       # (B, 5)
        count_probs = torch.softmax(count_logits / self.tau_count, dim=-1)   # (B, 5)

        out = dict(count_logits=count_logits, count_probs=count_probs, g=g)

        # Compute loss if labels provided
        if count_class is not None:
            # Weighted CE loss
            loss_count = F.cross_entropy(
                count_logits,
                count_class,
                weight=self.class_weights.to(count_logits.device),
            )
            out["loss_count"] = loss_count

            if self.use_contrastive:
                loss_con = self.contrastive_head(g, count_class)
                out["loss_count_con"] = loss_con
                out["loss_count_total"] = loss_count + self.contrastive_weight * loss_con
            else:
                out["loss_count_total"] = loss_count

        return out


# ── Effective-number class weights ────────────────────────────────────────────
def compute_effective_number_weights(
    class_counts: dict,  # {label (int): count (int)}
    num_classes: int = NUM_COUNT_CLASSES,
    beta_factor: float = 0.9999,
    w_min: float = COUNT_W_MIN,
    w_max: float = COUNT_W_MAX,
) -> torch.Tensor:
    """
    Compute effective-number weighting (§8) clipped to [w_min, w_max].
    Returns (num_classes,) float tensor.
    """
    weights = []
    for c in range(num_classes):
        n_c = class_counts.get(c, 0)
        if n_c == 0:
            w = 0.0
        else:
            eff = (1 - beta_factor ** n_c) / (1 - beta_factor)
            w = 1.0 / eff
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.float32)
    non_zero = weights > 0
    if non_zero.any():
        weights = weights / weights[non_zero].mean()
    weights[~non_zero] = w_max
    weights = weights.clamp(w_min, w_max)
    return weights


# ── Selection rules (§9.6) ────────────────────────────────────────────────────
def select_events_from_aec(
    count_probs: torch.Tensor,     # (5,) or (B, 5)
    mode_score: torch.Tensor,      # (M,) or (B, M) – sigmoid(event) * sigmoid(quality)
    event_mask: torch.Tensor,      # (M,) or (B, M) bool
    tau_mode: float = 0.5,
    top_n_clip: int = 10,
) -> list:
    """
    AEC selection (§9.6 / §15.2 count rules).
    Returns list of selected mode indices (0-indexed within M).

    pred_count = argmax P_count
    - 0 → empty set
    - 1/2/3 → Top-N by mode_score among valid modes
    - 4+ → threshold tau_mode on sigmoid(event_logit), min 4, max top_n_clip
    """
    squeeze = (count_probs.dim() == 1)
    if squeeze:
        count_probs = count_probs.unsqueeze(0)
        mode_score = mode_score.unsqueeze(0)
        event_mask = event_mask.unsqueeze(0)

    B = count_probs.shape[0]
    results = []

    for b in range(B):
        pred_count = int(count_probs[b].argmax().item())

        if pred_count == 0:
            results.append([])
            continue

        valid_idx = torch.where(event_mask[b])[0]
        if valid_idx.numel() == 0:
            results.append([])
            continue

        scores = mode_score[b, valid_idx]

        if pred_count in (1, 2, 3):
            n = min(pred_count, valid_idx.numel())
            _, top_local = scores.topk(n, largest=True)
            selected = valid_idx[top_local].tolist()
        else:
            # pred_count == 4 (class "4+")
            # threshold tau_mode, min 4, max top_n_clip
            above = (scores >= tau_mode).nonzero(as_tuple=False).squeeze(-1)
            if above.numel() < 4:
                # fallback to Top-4
                n = min(4, valid_idx.numel())
                _, top_local = scores.topk(n, largest=True)
                above = top_local
            # cap at top_n_clip
            if above.numel() > top_n_clip:
                top_scores = scores[above]
                _, keep = top_scores.topk(top_n_clip, largest=True)
                above = above[keep]
            selected = valid_idx[above].tolist()

        results.append(selected)

    if squeeze:
        return results[0]
    return results


# Need math for log in contrastive
import math
