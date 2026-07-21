# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn.functional as F
from torch import nn

from models.flash_vtg_gmr.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from models.flash_vtg_gmr.position_encoding import build_position_encoding, PositionEmbeddingSine
import math
from nncore.nn import build_model as build_adapter
from models.flash_vtg_gmr.blocks.generator import PointGenerator

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res


def masked_topk_indices(scores: torch.Tensor, valid_mask: torch.Tensor, k: int) -> torch.Tensor:
    """Per-sample top-k indices with every invalid row ranked after valid rows."""
    if scores.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("scores/valid_mask must have the same shape and a boolean mask")
    masked_scores = scores.masked_fill(~valid_mask, float("-inf"))
    return masked_scores.sort(dim=1, descending=True).indices[:, :k]

class ConfidenceScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=1, num_mlp_layers=3):
        super(ConfidenceScorer, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_conv_layers):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            else:
                self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            self.activations.append(nn.ReLU(inplace=False))

        self.fc = MLP(out_channels, out_channels // 2, 1, num_layers=num_mlp_layers)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)

        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)

        x = x.squeeze(2).permute(0, 2, 1)
        x = self.fc(x)

        return x

class FlashVTG(nn.Module):
    """ FlashVTG. """

    def __init__(self, transformer, position_embed, txt_position_embed, n_input_proj, input_dropout, txt_dim, vid_dim, aud_dim=0, use_txt_pos=False,
                strides=(1, 2, 4, 8),
                buffer_size=2048,
                max_num_moment=50,
                merge_cls_sal=True,
                pyramid_cfg=None,
                pooling_cfg=None,
                coord_head_cfg=None,
                args=None):
        """ Initializes the model."""
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.PositionEmbeddingSine = PositionEmbeddingSine(hidden_dim, normalize=True)

        # input projection
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # set up dummy token
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        # build muti-scale pyramid
        self.pyramid = build_adapter(pyramid_cfg, hidden_dim, strides)

        self.pooling = build_adapter(pooling_cfg, hidden_dim)
        self.conf_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.class_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.coord_head = build_adapter(coord_head_cfg, hidden_dim, 2)
        self.generator = PointGenerator(strides, buffer_size)
        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.args = args
        self.x = nn.Parameter(torch.tensor(0.5))
        # Optional: existence head for GMR (positive/negative classification)
        self.use_exist_head = bool(getattr(args, "use_exist_head", False))
        self.exist_pool = str(getattr(args, "exist_pool", "mean"))
        if self.use_exist_head:
            # Lite head: concat(query_emb, pooled_video_emb) -> logit
            self.exist_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        if getattr(args, "enable_adapter", False):
            from models.flash_vtg_gmr.event_adapter import ProposalToEventAdapter
            self.event_adapter = ProposalToEventAdapter(
                feat_dim=hidden_dim,
                num_modes=10,
                variant=getattr(args, "adapter_variant", "P0"),
            )
        if getattr(args, "enable_aec", False):
            from models.flash_vtg_gmr.event_cardinality import AdaptiveEventCardinality
            self.aec = AdaptiveEventCardinality(
                text_dim=getattr(args, "t_feat_dim", 512),
                event_dim=hidden_dim,
                variant=getattr(args, "aec_variant", "C1"),
                seed=getattr(args, "seed", 2024),
            )

    def train(self, mode: bool = True):
        """Keep frozen B0/P0 modules in eval mode during downstream training."""
        super().train(mode)
        if mode and bool(getattr(self.args, "freeze_backbone", False)):
            for name, module in self.named_children():
                if name not in {"event_adapter", "aec"}:
                    module.eval()
        if mode and bool(getattr(self.args, "freeze_adapter", False)) and hasattr(self, "event_adapter"):
            self.event_adapter.eval()
        return self


    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets=None):
        if vid is not None:
            _count = [v.count('_') for v in vid]
            # Only QVHighlights video IDs use the conventional "_start_end" suffix.
            # Soccer-GMR IDs may contain underscores that are part of the actual ID.
            is_qv_format = False
            for p in [getattr(self.args, "train_path", ""), getattr(self.args, "eval_path", "")]:
                if isinstance(p, str) and (("highlight_" in p) or ("qvhighlight" in p)):
                    is_qv_format = True
                    break
            if self.args.dset_name == 'hl' and is_qv_format:
                _position_to_cut = [find_nth(v, '_', _count[i]-1) for i, v in enumerate(vid)]
                ori_vid = [v[:_position_to_cut[i]] for i, v in enumerate(vid)]
            else:
                ori_vid = [v for v in vid]

        # Project inputs to the same hidden dimension
        src_vid = self.input_vid_proj(src_vid)
        raw_src_txt = src_txt
        src_txt = self.input_txt_proj(src_txt)
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)

        # Insert dummy token in front of txt
        txt_dummy = self.dummy_rep_token.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(src_txt.shape[0], 1, 1)
        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)


        mask_txt = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)

        pos_dummy = self.dummy_rep_pos.reshape([1, self.args.num_dummies, self.hidden_dim]).repeat(pos_txt.shape[0], 1, 1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)
        src_txt_dummy = src_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2) # (L, batch_size, d)

        memory = self.txtproj_encoder(src_txt_dummy, src_key_padding_mask=~(src_txt_mask_dummy.bool()), pos=pos_txt_dummy)
        dummy_token = memory[:self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies]).to(src_txt_mask.device).repeat(src_txt_mask.shape[0], 1)
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        src = torch.cat([src_vid, src_txt_dummy], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()  # (bsz, L_vid+L_txt)
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)

        video_length = src_vid.shape[1]

        video_emb, video_msk, pos_embed, attn_weights, saliency_scores = self.transformer(src, ~mask, pos, video_length=video_length, saliency_proj1=self.saliency_proj1, saliency_proj2=self.saliency_proj2)

        video_emb = video_emb.permute(1, 0, 2)  # (L, batch_size, d) -> (batch_size, L, d)
        video_msk = (~video_msk).int()
        pymid, pymid_msk = self.pyramid(
            video_emb, video_msk, return_mask=(self.training == True or getattr(self.args, "enable_adapter", False) or getattr(self.args, "enable_aec", False))
        )
        point = self.generator(pymid)

        with torch.autocast("cuda", enabled=False):
            video_emb = video_emb.float()
            query_emb = self.pooling(src_txt.float(), src_txt_mask)

            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)
            out_conf = torch.cat(pymid, dim=1)
            out_conf = self.conf_head(out_conf)
            out_class = self.x*out_class+(1-self.x)*out_conf

            if self.coord_head is not None:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)
            else:
                out_coord = None

            bs, t = src_vid.shape[0], src_vid.shape[1]
            output = dict(_avg_factor=bs)
            output["saliency_scores"] = saliency_scores
            output["t2vattnvalues"] = (attn_weights[:,:,self.args.num_dummies:] * (src_txt_mask.unsqueeze(1).repeat(1, video_length, 1))).sum(2)
            output["t2vattnvalues"] = torch.clamp(output["t2vattnvalues"], 0, 1)
            output["video_msk"] = video_msk

            # Candidate extraction (Part 2 §4)
            if getattr(self.args, "enable_adapter", False) or getattr(self.args, "enable_aec", False) or getattr(self.args, "variant", None) == "G0-Threshold":
                B = video_emb.shape[0]
                K = 50
                # Decode spans normalized [0, 1]
                pts_center = point[:, 0].unsqueeze(0)  # (1, total_points)
                pts_stride = point[:, 3].unsqueeze(0)  # (1, total_points)
                start_sec = (pts_center - out_coord[:, :, 0] * pts_stride) * self.args.clip_length
                end_sec = (pts_center + out_coord[:, :, 1] * pts_stride) * self.args.clip_length
                spans_sec = torch.stack([start_sec, end_sec], dim=-1)  # (B, total_points, 2)
                dur_sec = video_msk.sum(dim=1, keepdim=True) * self.args.clip_length  # (B, 1)
                dur_sec = dur_sec.clamp(min=1e-6)
                spans_norm = spans_sec / dur_sec.unsqueeze(-1)  # (B, total_points, 2)
                spans_norm = spans_norm.clamp(0.0, 1.0)

                # Find top-K based on out_class.  Invalid pyramid positions must
                # lose before sorting, otherwise padding can displace valid rows.
                scores = out_class.squeeze(-1)  # (B, total_points)
                if pymid_msk:
                    pymid_msk_cat = torch.cat(pymid_msk, dim=1).bool()
                else:
                    pymid_msk_cat = torch.ones_like(scores, dtype=torch.bool)
                candidate_topk_idx = masked_topk_indices(scores, pymid_msk_cat, K)

                # Gather features, mask, span, logit, point, scale
                pymid_cat = torch.cat(pymid, dim=1)  # (B, total_points, 256)
                candidate_feat = torch.gather(pymid_cat, 1, candidate_topk_idx.unsqueeze(-1).expand(-1, -1, 256))

                candidate_mask = torch.gather(pymid_msk_cat, 1, candidate_topk_idx)

                candidate_span = torch.gather(spans_norm, 1, candidate_topk_idx.unsqueeze(-1).expand(-1, -1, 2))
                candidate_logit = torch.gather(scores, 1, candidate_topk_idx)  # (B, K)

                point_expanded = point.unsqueeze(0).expand(B, -1, -1)  # (B, total_points, 4)
                candidate_point = torch.gather(point_expanded, 1, candidate_topk_idx.unsqueeze(-1).expand(-1, -1, 4))

                scale_expanded = point[:, 3].unsqueeze(0).expand(B, -1)  # (B, total_points)
                candidate_scale = torch.gather(scale_expanded, 1, candidate_topk_idx)  # (B, K)

                query_global = query_emb.squeeze(1)  # (B, 256)

                output["candidate_feat"] = candidate_feat
                output["candidate_mask"] = candidate_mask
                output["candidate_span"] = candidate_span
                output["candidate_logit"] = candidate_logit
                output["candidate_topk_idx"] = candidate_topk_idx
                output["candidate_point"] = candidate_point
                output["candidate_scale"] = candidate_scale
                output["query_global"] = query_global

                if getattr(self.args, "enable_adapter", False):
                    # Run P0 adapter
                    adapter_out = self.event_adapter(
                        candidate_feat,
                        candidate_mask,
                        candidate_span,
                        candidate_logit,
                        candidate_scale,
                        query_global,
                    )
                    output.update(adapter_out)

                    # Build EventInterfaceV1 and expose it
                    from models.flash_vtg_gmr.event_adapter import p0_inference
                    iface = p0_inference(
                        adapter_out,
                        query_global,
                        b0_sha=getattr(self.args, "baseline_checkpoint_sha256", None),
                        p0_sha=getattr(self.args, "public_p0_checkpoint_sha256", None),
                        fm_sha=getattr(self.args, "feature_manifest_sha256", None),
                    )
                    output["event_interface"] = iface

                if getattr(self.args, "enable_aec", False):
                    aec_variant = getattr(self.args, "aec_variant", "C1")
                    if aec_variant in ("G0", "G0-Con"):
                        # AEC runs on raw candidates
                        aec_out = self.aec(
                            text_feat=raw_src_txt,
                            text_mask=src_txt_mask,
                            set_feat=candidate_feat,
                            set_mask=candidate_mask,
                            count_class=targets.get("count_class") if targets is not None else None,
                        )
                    else:
                        # AEC runs on event modes from adapter
                        event_feat = adapter_out["event_feat"]
                        event_mask = adapter_out["event_mask"]
                        aec_out = self.aec(
                            text_feat=raw_src_txt,
                            text_mask=src_txt_mask,
                            set_feat=event_feat,
                            set_mask=event_mask,
                            count_class=targets.get("count_class") if targets is not None else None,
                        )
                    output["pred_count_logits"] = aec_out["count_logits"]
                    output["pred_count_probs"] = aec_out["count_probs"]
                    if "loss_count" in aec_out:
                        output["loss_count"] = aec_out["loss_count"]
                    if "loss_count_con" in aec_out:
                        output["loss_count_con"] = aec_out["loss_count_con"]
                    if "loss_count_total" in aec_out:
                        output["loss_count_total"] = aec_out["loss_count_total"]

            if self.use_exist_head:
                vmask = video_msk.float()  # (bsz, L_vid), 1=valid
                if self.exist_pool == "max":
                    video_pooled = video_emb.masked_fill(vmask.unsqueeze(-1) == 0, float("-inf")).max(dim=1).values
                else:
                    denom = vmask.sum(dim=1, keepdim=True).clamp(min=1.0)
                    video_pooled = (video_emb * vmask.unsqueeze(-1)).sum(dim=1) / denom
                query_exist = query_emb.float()
                if query_exist.ndim == 3:
                    # pooling adapters may return (B, 1, D); reduce to (B, D) for existence head.
                    query_exist = query_exist.squeeze(1) if query_exist.shape[1] == 1 else query_exist.mean(dim=1)
                exist_inp = torch.cat([query_exist, video_pooled.float()], dim=-1)
                output["pred_exist_logits"] = self.exist_head(exist_inp).squeeze(-1)

            if self.training == False and targets is not None and "span_labels" in targets:
                output["point"] = point
                output["video_emb"] = video_emb
                output["query_emb"] = query_emb
                output["pymid_msk"] = pymid_msk
                output["out_class"] = out_class.clone()
                output["out_coord"] = out_coord

            if self.training == True:

                output["point"] = point
                output["video_emb"] = video_emb
                output["query_emb"] = query_emb
                output["pymid_msk"] = pymid_msk
                output["out_class"] = out_class
                output["out_coord"] = out_coord

                boundarys = []
                out_class = out_class.sigmoid()
                for idx, boundary in enumerate(out_coord):
                    boundary = boundary.clone()

                    boundary[:, 0] = boundary[:, 0] * -1
                    boundary = boundary * point[:, 3, None].repeat(1, 2)
                    boundary = boundary + point[:, 0, None].repeat(1, 2)
                    boundary = boundary / (1/self.args.clip_length)
                    boundary = torch.cat((boundary, out_class[idx]), dim=-1)

                    _, inds = out_class[idx, :, 0].sort(descending=True)
                    boundary = boundary[inds[:]]
                    boundarys.append(boundary)

                boundarys = torch.stack(boundarys, dim=0)
                output["pred_spans"] = boundarys


            if self.training == False:
                assert bs == 1, "batch size larger than 1 is not supported for inference"
                out_class = out_class.sigmoid()

                output["_out"] = dict(label=targets.get("label", [None])[0])
                output["_out"]["video_msk"] = video_msk
                output["_out"]["saliency"] = saliency_scores[0]
                if self.use_exist_head and ("pred_exist_logits" in output):
                    output["_out"]["pred_exist_score"] = torch.sigmoid(output["pred_exist_logits"]).detach().cpu()

                if self.coord_head is not None:
                    boundary = out_coord[0]
                    boundary[:, 0] *= -1
                    boundary *= point[:, 3, None].repeat(1, 2)
                    boundary += point[:, 0, None].repeat(1, 2)
                    boundary /= 1/self.args.clip_length
                    boundary = torch.cat((boundary, out_class[0]), dim=-1)

                    _, inds = out_class[0, :, 0].sort(descending=True)
                    boundary = boundary[inds[: self.max_num_moment]]

                    output["_out"]["boundary"] = boundary

        if self.training == True and self.args.use_neg:
            ### Neg Pairs ###
            neg_vid = ori_vid[1:] + ori_vid[:1]
            real_neg_mask = torch.Tensor(element_wise_list_equal(ori_vid, neg_vid)).to(src_txt_dummy.device)
            real_neg_mask = real_neg_mask == False
            if real_neg_mask.sum() != 0:

                src_txt_dummy_neg = torch.cat([src_txt_dummy[1:], src_txt_dummy[0:1]], dim=0)
                src_txt_mask_dummy_neg = torch.cat([src_txt_mask_dummy[1:], src_txt_mask_dummy[0:1]], dim=0)
                src_dummy_neg = torch.cat([src_vid, src_txt_dummy_neg], dim=1)
                mask_dummy_neg = torch.cat([src_vid_mask, src_txt_mask_dummy_neg], dim=1).bool()
                pos_neg = pos.clone()

                mask_dummy_neg = mask_dummy_neg[real_neg_mask]
                src_dummy_neg = src_dummy_neg[real_neg_mask]
                pos_neg = pos_neg[real_neg_mask]
                src_txt_mask_dummy_neg = src_txt_mask_dummy_neg[real_neg_mask]

                memory_neg, video_msk, pos_embed, attn_weights_neg, saliency_scores_neg = self.transformer(src_dummy_neg, ~mask_dummy_neg, pos_neg, video_length=video_length, saliency_proj1=self.saliency_proj1, saliency_proj2=self.saliency_proj2)

                output["saliency_scores_neg"] = saliency_scores_neg
                output["src_txt_mask_neg"] = src_txt_mask_dummy_neg

                output["t2vattnvalues_neg"] = (attn_weights_neg[:, :, self.args.num_dummies:] * (src_txt_mask_dummy_neg[:, self.args.num_dummies:].unsqueeze(1).repeat(1, video_length, 1))).sum(2)
                output["t2vattnvalues_neg"] = torch.clamp(output["t2vattnvalues_neg"], 0, 1)
            else:
                output["saliency_scores_neg"] = None
                output["t2vattnvalues_neg"] = None
            output["real_neg_mask"] = real_neg_mask
            output["dummy_tokens"] = dummy_token
        else:
            output["saliency_scores_neg"] = None
            output["t2vattnvalues_neg"] = None
            output["real_neg_mask"] = None
            output["dummy_tokens"] = dummy_token

        return output

class SetCriterion(nn.Module):
    """ This class computes the loss."""

    def __init__(self, weight_dict, eos_coef, losses, saliency_margin=1, args=None):
        """ Create the criterion."""
        super().__init__()
        self.args=args
        self.weight_dict = weight_dict
        self.losses = losses
        self.saliency_margin = saliency_margin
        self.device = args.device

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1

        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.l2_criterion = torch.nn.MSELoss().to(self.args.device)
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none').to(self.args.device)
        self.bce_criterion = nn.BCELoss(reduction='none')
        self.SampledNCELoss = SampledNCELoss().to(self.args.device)
        from nncore.nn import build_loss
        self.loss=build_loss(args.cfg.model.loss_cfg)
        # Exclude the highlight-detection saliency branch in MR-only mode.
        if getattr(self.args, "mr_only", False) and hasattr(self.loss, "_loss_sal"):
            self.loss._loss_sal = None

    def norm(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def loss_labels(self, outputs, targets, log=True):
        sal_score = targets["saliency_all_labels"]
        conf = outputs["out_class"][:, :sal_score.shape[1], 0]

        norm_sal_score = self.norm(sal_score)
        norm_conf = self.norm(conf)
        losses = F.mse_loss(norm_sal_score, norm_conf)
        return {"loss_label": losses}

    def loss_exist(self, outputs, targets, log=True):
        if ("exist_label" not in targets) or ("pred_exist_logits" not in outputs):
            return {"loss_exist": 0.0}
        logits = outputs["pred_exist_logits"]
        labels = targets["exist_label"].float()
        if logits.ndim != 1:
            logits = logits.view(-1)
        if labels.ndim != 1:
            labels = labels.view(-1)
        return {"loss_exist": F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")}

    def loss_saliency(self, outputs, targets, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        # Neg pair loss
        if outputs["saliency_scores_neg"] is not None: ## When batch size is not 1 (negative pair exists)
            vid_token_mask = outputs["video_msk"]
            real_neg_mask = outputs["real_neg_mask"]
            saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
            loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat([saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair * 0.
            else:
                loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair

            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_msk"]
            # Neg pair loss

            if outputs["t2vattnvalues_neg"] is not None:
                saliency_scores_neg = outputs["t2vattnvalues_neg"].clone()  # (N, L)
                loss_neg_pair_attn = (- torch.log(1. - saliency_scores_neg) * (vid_token_mask[real_neg_mask])).sum(dim=1).mean()

            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            # real neg
            realneg_saliency_scores = torch.cat([saliency_scores[real_neg_mask], saliency_scores_neg], dim=1)
            realneg_saliency_contrast_label = torch.cat(
                [saliency_contrast_label[real_neg_mask], torch.zeros_like(saliency_contrast_label)[real_neg_mask]], dim=1)
            realneg_vid_token_mask = vid_token_mask[real_neg_mask].repeat([1, 2])
            realneg_saliency_scores = realneg_vid_token_mask * realneg_saliency_scores + (
                        1. - realneg_vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive_attn = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(realneg_saliency_contrast_label > 100)  # no drop
                pos_mask = (realneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = realneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * realneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive_attn = loss_rank_contrastive_attn + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive_attn / 12

            false_neg_mask = ~(real_neg_mask)
            if false_neg_mask.sum() != 0:
                if false_neg_mask.sum() == 1:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask].unsqueeze(0)
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask].unsqueeze(0)
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3
                else:
                    falseneg_saliency_scores = saliency_scores[false_neg_mask]
                    falseneg_saliency_contrast_label = saliency_contrast_label[false_neg_mask]
                    falseneg_vid_token_mask = vid_token_mask[false_neg_mask]
                    falseneg_saliency_scores = falseneg_vid_token_mask * falseneg_saliency_scores + (1. - falseneg_vid_token_mask) * -1e+3

                tau = 0.5
                falseneg_loss_rank_contrastive = 0.
                for rand_idx in range(1, 12):
                    drop_mask = ~(falseneg_saliency_contrast_label > 100)  # no drop
                    pos_mask = (falseneg_saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                    if torch.sum(pos_mask) == 0:  # no positive sample
                        continue
                    else:
                        batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                    # drop higher ranks
                    cur_saliency_scores = falseneg_saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                    # numerical stability
                    logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                    # softmax
                    exp_logits = torch.exp(logits)
                    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                    mean_log_prob_pos = (pos_mask * log_prob * falseneg_vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                    loss = - mean_log_prob_pos * batch_drop_mask
                    falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive + loss.mean()
                falseneg_loss_rank_contrastive = falseneg_loss_rank_contrastive / 12
                loss_rank_contrastive += falseneg_loss_rank_contrastive

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            if self.args.dset_name in ['youtube_uni']:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn * 0 + loss_saliency_attn
            else:
                loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_neg_pair_attn + loss_saliency_attn
            loss_saliency = loss_saliency + (loss_saliency_attn * self.args.lw_wattn)

        else: ## when batch size == 1
            vid_token_mask = outputs["video_msk"]
            saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive = loss_rank_contrastive / 12

            saliency_scores = outputs["saliency_scores"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

            loss_saliency = loss_saliency + loss_rank_contrastive
            ########### Saliency loss to t2v attn weights ##############
            """higher scores for positive clips"""
            vid_token_mask = outputs["video_msk"]
            saliency_scores = outputs["t2vattnvalues"].clone()  # (N, L)
            saliency_contrast_label = targets["saliency_all_labels"]

            saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

            tau = 0.5
            loss_rank_contrastive = 0.
            for rand_idx in range(1, 12):
                drop_mask = ~(saliency_contrast_label > 100)  # no drop
                pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx
                if torch.sum(pos_mask) == 0:  # no positive sample
                    continue
                else:
                    batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

                # drop higher ranks
                cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
                # numerical stability
                logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
                # softmax
                exp_logits = torch.exp(logits)
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

                mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
                loss = - mean_log_prob_pos * batch_drop_mask
                loss_rank_contrastive = loss_rank_contrastive + loss.mean()
            loss_rank_contrastive_attn = loss_rank_contrastive / 12

            saliency_scores = outputs["t2vattnvalues"]  # (N, L)
            pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
            neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
            num_pairs = pos_indices.shape[1]  # typically 2 or 4
            batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
            pos_scores = torch.stack(
                [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            neg_scores = torch.stack(
                [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
            loss_saliency_attn = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                            / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale
            saliency_binary_label = torch.clamp(targets["saliency_all_labels"], 0, 1)
            logits = saliency_scores.reshape(-1)
            labels_x = saliency_binary_label.reshape(-1)
            BCEcriterion = nn.BCELoss()
            bceloss = BCEcriterion(logits, labels_x)

            loss_saliency_attn = loss_rank_contrastive_attn + bceloss + loss_saliency_attn
            loss_saliency += (loss_saliency_attn * self.args.lw_wattn)
        return {"loss_saliency": loss_saliency}

    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "saliency": self.loss_saliency,
            "exist": self.loss_exist,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, **kwargs)

    def extract_relevant_windows(self, data_list):
        all_windows = [instance['relevant_windows'] for instance in data_list]
        max_len = max(len(windows) for windows in all_windows)

        padded_windows = []
        for windows in all_windows:
            new_windows = windows.copy()
            while len(new_windows) < max_len:
                new_windows.append([float('inf'), float('inf')])
            padded_windows.append(new_windows)

        result_tensor = torch.tensor(padded_windows, dtype=torch.float32)

        return result_tensor

    def forward(self, batch, outputs, targets):
        """ This performs the loss computation."""
        losses = {}

        # If existence labels exist, apply MR losses only on positives to avoid NaN / invalid supervision.
        exist_labels = targets.get("exist_label", None)
        pos_mask = None
        if exist_labels is not None:
            pos_mask = (exist_labels > 0.5)

        part2_losses = {}
        # 1. P0 Adapter losses: L_event, L_quality, L_span (if P0-R)
        if getattr(self.args, "enable_adapter", False) and not getattr(self.args, "freeze_adapter", False):
            from models.flash_vtg_gmr.event_adapter import compute_adapter_losses
            adapter_targets = []
            for meta in targets["label"]:
                # Candidate/event spans are normalized by D_grid=T*clip_length.
                # GT must use the same coordinate system (Part 1 temporal contract).
                duration = float(meta.get("D_grid", meta["duration"]))
                rel_wins = meta.get("relevant_windows", [])
                if rel_wins is None:
                    rel_wins = []
                is_null = len(rel_wins) == 0
                norm_wins = [[w[0] / duration, w[1] / duration] for w in rel_wins]
                adapter_targets.append({
                    "is_null": is_null,
                    "relevant_windows_norm": norm_wins
                })
            
            adapter_losses = compute_adapter_losses(
                event_logit=outputs["event_logit"],
                quality_logit=outputs["quality_logit"],
                event_span=outputs["event_span"],
                event_mask=outputs["event_mask"],
                targets=adapter_targets,
                variant=getattr(self.args, "adapter_variant", "P0"),
            )
            if "loss_event" in adapter_losses:
                part2_losses["loss_event"] = adapter_losses["loss_event"]
            if "loss_quality" in adapter_losses:
                part2_losses["loss_quality"] = adapter_losses["loss_quality"]
            if "loss_span" in adapter_losses:
                part2_losses["loss_span"] = adapter_losses["loss_span"]

        # 2. AEC losses: L_count, L_count_con
        if getattr(self.args, "enable_aec", False):
            if "loss_count" in outputs:
                part2_losses["loss_count"] = outputs["loss_count"]
            if "loss_count_con" in outputs:
                part2_losses["loss_count_con"] = outputs["loss_count_con"]

        def _filter_dict_by_mask(d, m):
            def _filter_value_by_mask(v):
                if torch.is_tensor(v):
                    if v.ndim >= 1 and v.shape[0] == m.shape[0]:
                        return v[m.to(v.device)]
                    return v
                if isinstance(v, list):
                    return [_filter_value_by_mask(e) for e in v]
                if isinstance(v, tuple):
                    return tuple(_filter_value_by_mask(e) for e in v)
                if isinstance(v, dict):
                    return {kk: _filter_value_by_mask(vv) for kk, vv in v.items()}
                return v

            filtered = {}
            keep = torch.where(m)[0].tolist()
            for k, v in d.items():
                if k == "span_labels" and isinstance(v, list) and len(v) == m.shape[0]:
                    filtered[k] = [v[index] for index in keep]
                else:
                    filtered[k] = _filter_value_by_mask(v)
            return filtered

        # build nncore-loss inputs
        if pos_mask is None or bool(pos_mask.all()):
            meta_for_loss = batch[0]
            outputs_for_loss = outputs
            targets_for_loss = targets
        else:
            meta_for_loss = [batch[0][i] for i in torch.where(pos_mask)[0].tolist()]
            outputs_for_loss = _filter_dict_by_mask(outputs, pos_mask)
            targets_for_loss = _filter_dict_by_mask(targets, pos_mask)

        if pos_mask is None or pos_mask.any():
            new_outputs = {}
            if len(meta_for_loss) > 0 and meta_for_loss[0].get("relevant_windows", None) is not None:
                new_outputs["boundary"] = self.extract_relevant_windows(meta_for_loss).to(self.device)
            else:
                new_outputs["boundary"] = None
            if "saliency_all_labels" in targets_for_loss:
                new_outputs["saliency"] = targets_for_loss["saliency_all_labels"]
            if "saliency_pos_labels" in targets_for_loss:
                new_outputs["pos_clip"] = targets_for_loss["saliency_pos_labels"][:, 0].unsqueeze(1)
            new_outputs["label"] = meta_for_loss
            if "fps" in targets_for_loss:
                new_outputs["fps"] = targets_for_loss["fps"]
            new_outputs.update(outputs_for_loss)

            losses = self.loss(new_outputs, outputs_for_loss)
        else:
            # All-negative batch: legacy MR losses are skipped.  Part 2 does not
            # construct the legacy existence head, so use an available model
            # tensor only as a device/dtype-preserving zero anchor.
            anchor = outputs.get("pred_exist_logits", outputs.get("candidate_logit", outputs["saliency_scores"]))
            losses = {k: anchor.sum() * 0.0 for k in self.weight_dict.keys() if k != "loss_exist"}

        # Compute auxiliary losses (saliency/labels/exist)
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        losses.update(part2_losses)
        return losses

class Parameter(nn.Parameter):
    """
    An :obj:`nn.Parameter` class that supports multiple inputs initializes the
    parameters using a scaled normal distribution.
    """

    def __new__(cls, *args, requires_grad=True, **kwargs):
        if torch.is_tensor(args[0]):
            data = args[0]
        elif isinstance(args[0], float):
            data = torch.Tensor([args[0]])
        elif isinstance(args[0], (list, tuple)):
            data = torch.randn(args[0], **kwargs) / args[0][-1]**0.5
        else:
            data = torch.randn(args, **kwargs) / args[-1]**0.5

        return torch.Tensor._make_subclass(cls, data, requires_grad)

class SampledNCELoss(nn.Module):

    def __init__(self,
                 temperature=0.07,
                 max_scale=100,
                 learnable=False,
                 direction=('row', 'col')):
        super(SampledNCELoss, self).__init__()

        scale = torch.Tensor([math.log(1 / temperature)])

        if learnable:
            self.scale = Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        self.temperature = temperature
        self.max_scale = max_scale
        self.learnable = learnable
        self.direction = (direction, ) if isinstance(direction, str) else direction

    def extra_repr(self):
        return ('temperature={}, max_scale={}, learnable={}, direction={}, loss_weight={}'
                .format(self.temperature, self.max_scale, self.learnable, self.direction,
                        self.loss_weight))

    def forward(self, video_emb, query_emb, video_msk, saliency, pos_clip):
        batch_inds = torch.arange(video_emb.size(0), device=video_emb.device)

        pos_scores = saliency[batch_inds, pos_clip].unsqueeze(-1)
        loss_msk = (saliency <= pos_scores) * video_msk

        scale = self.scale.exp().clamp(max=self.max_scale)
        i_sim = F.cosine_similarity(video_emb, query_emb, dim=-1) * scale
        i_sim = i_sim + torch.where(loss_msk > 0, .0, float('-inf'))

        loss = 0

        if 'row' in self.direction:
            i_met = F.log_softmax(i_sim, dim=1)[batch_inds, pos_clip]
            loss = loss - i_met.sum() / i_met.size(0)

        if 'col' in self.direction:
            j_sim = i_sim.t()
            j_met = F.log_softmax(j_sim, dim=1)[pos_clip, batch_inds]
            loss = loss - j_met.sum() / j_met.size(0)

        return loss

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=False)
        return x  # (N, L, D)


def build_model1(args):
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = FlashVTG(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
        strides=args.cfg.model.strides,
        buffer_size=args.cfg.model.buffer_size,
        max_num_moment=args.cfg.model.max_num_moment,
        pyramid_cfg=args.cfg.model.pyramid_cfg,
        pooling_cfg=args.cfg.model.pooling_cfg,
        coord_head_cfg=args.cfg.model.coord_head_cfg,
        args=args
    )

    weight_dict = {
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency,
        "loss_reg": args.lw_reg,
        "loss_cls": args.lw_cls,
        "loss_sal": args.lw_sal,
    }

    # Formal Part 2 freezes B0 and optimizes only its preregistered new-module
    # objective.  Legacy localization terms are neither weighted nor allowed to
    # influence checkpoint loss accounting.
    if getattr(args, "variant", None) is not None:
        weight_dict = {}

    if getattr(args, "enable_adapter", False) and not getattr(args, "freeze_adapter", False):
        weight_dict["loss_event"] = 1.0
        weight_dict["loss_quality"] = 1.0
        if getattr(args, "adapter_variant", "P0") == "P0-R":
            weight_dict["loss_span"] = 1.0

    if getattr(args, "enable_aec", False):
        weight_dict["loss_count"] = 1.0
        if getattr(args, "aec_variant", "C1") in ("C2", "G0-Con"):
            weight_dict["loss_count_con"] = 0.1

    # Retain only localization losses in MR-only mode.
    if getattr(args, "mr_only", False):
        weight_dict.pop("loss_label", None)
        weight_dict.pop("loss_saliency", None)
        weight_dict.pop("loss_sal", None)
        losses = []
    else:
        losses = ["saliency", "labels"]

    if bool(getattr(args, "use_exist_head", False)):
        weight_dict["loss_exist"] = float(getattr(args, "exist_loss_coef", 1.0))
        losses = list(losses) + ["exist"]

    criterion = SetCriterion(
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, saliency_margin=args.saliency_margin, args=args
    )
    criterion.to(device)
    return model, criterion
