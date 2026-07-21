"""Validation-only threshold and count-temperature calibration for Part 2."""

from __future__ import annotations

import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval.event_set_metrics import compute_event_set_metrics, temporal_iou
from training.flash_vtg_gmr.contracts import sha256_file
from training.flash_vtg_gmr.dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from training.flash_vtg_gmr.inference import setup_model


def _positive_map(pairs: list[tuple[list[list[float]], list[list[float]]]]) -> float:
    """Deterministic positive-query mAP tie-break over tIoU 0.50:0.95."""
    values = []
    for predictions, ground_truth in pairs:
        if not ground_truth:
            continue
        threshold_aps = []
        for threshold in [value / 100.0 for value in range(50, 100, 5)]:
            matched = set()
            precision_sum = 0.0
            true_positives = 0
            for rank, prediction in enumerate(predictions, start=1):
                candidates = [
                    (temporal_iou(prediction, gt), index)
                    for index, gt in enumerate(ground_truth)
                    if index not in matched
                ]
                best_iou, best_index = max(candidates, default=(0.0, -1))
                if best_iou >= threshold:
                    matched.add(best_index)
                    true_positives += 1
                    precision_sum += true_positives / rank
            threshold_aps.append(precision_sum / max(len(ground_truth), 1))
        values.append(sum(threshold_aps) / len(threshold_aps))
    return float(sum(values) / len(values)) if values else 0.0


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Fit one positive scalar temperature by validation cross entropy."""
    best = (float("inf"), 1.0)
    # Log-spaced grid is stable and deterministic; temperature never changes argmax.
    for index in range(161):
        log_temperature = math.log(0.05) + index * (math.log(5.0) - math.log(0.05)) / 160
        temperature = math.exp(log_temperature)
        nll = float(torch.nn.functional.cross_entropy(logits / temperature, labels).item())
        best = min(best, (nll, temperature))
    return best[1], best[0]


def _decode(
    output: dict,
    meta: dict,
    variant: str,
    tau: float,
    temperature: float,
) -> list[list[float]]:
    duration = float(meta.get("D_grid", meta["duration"]))
    if variant == "G0-Threshold":
        spans = output["candidate_span"][0]
        event_scores = torch.sigmoid(output["candidate_logit"][0])
        mask = output["candidate_mask"][0]
        rows = [
            [float(spans[index, 0]) * duration, float(spans[index, 1]) * duration, float(event_scores[index])]
            for index in torch.where(mask)[0].tolist()
            if float(event_scores[index]) >= tau
        ]
        return sorted(rows, key=lambda row: row[2], reverse=True)

    probabilities = torch.softmax(output["pred_count_logits"][0] / temperature, dim=-1)
    count_class = int(probabilities.argmax().item())
    if count_class == 0:
        return []
    if variant in {"G0", "G0-Con"}:
        spans = output["candidate_span"][0]
        activity = torch.sigmoid(output["candidate_logit"][0])
        rank_score = activity
        mask = output["candidate_mask"][0]
    else:
        spans = output["event_span"][0]
        activity = torch.sigmoid(output["event_logit"][0])
        rank_score = activity * torch.sigmoid(output["quality_logit"][0])
        mask = output["event_mask"][0]
    valid = [(index, float(rank_score[index]), float(activity[index])) for index in torch.where(mask)[0].tolist()]
    valid.sort(key=lambda row: row[1], reverse=True)
    if count_class in {1, 2, 3}:
        selected = valid[:count_class]
    else:
        selected = [row for row in valid if row[2] >= tau]
        if len(selected) < 4:
            selected = valid[: min(4, len(valid))]
        else:
            selected = selected[:10]
    return [
        [float(spans[index, 0]) * duration, float(spans[index, 1]) * duration, score]
        for index, score, _ in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_manifest_index", required=True)
    parser.add_argument("--feature_manifest")
    parser.add_argument("--baseline_index")
    parser.add_argument("--raw_threshold_calibration")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variant", choices=["G0-Threshold", "G0", "G0-Con", "C1", "C2"])
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    opt = checkpoint["opt"]
    checkpoint_variant = getattr(opt, "variant", None)
    variant = args.variant or checkpoint_variant
    if variant is None:
        raise ValueError("--variant is required when calibrating a B0 checkpoint")
    if checkpoint_variant is not None and args.variant is not None and checkpoint_variant != args.variant:
        raise ValueError(f"Checkpoint variant {checkpoint_variant} != requested {args.variant}")
    opt.variant = variant
    opt.device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    opt.resume = args.checkpoint
    opt.resume_all = False
    opt.data_manifest_index = args.data_manifest_index
    if args.feature_manifest:
        opt.feature_manifest = args.feature_manifest
    if args.baseline_index:
        opt.baseline_index = args.baseline_index
    if variant == "G0-Threshold":
        opt.init_backbone_ckpt = args.checkpoint
        # B0 contains the legacy existence head, which Part 2 deliberately does
        # not construct.  The audited partial-init path above loads all usable
        # B0 tensors and excludes that head; strict resume is not appropriate.
        opt.resume = None
    opt.eval_split_name = "val"

    with open(args.data_manifest_index, "r", encoding="utf-8") as handle:
        manifest_index = json.load(handle)
    split_record = manifest_index["data_manifests"]["val"]
    opt.eval_path = split_record["path"]
    dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type=opt.q_feat_type,
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=-1,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        mr_only=opt.mr_only,
        keep_empty_gt=True,
        strict_data_contract=True,
        require_text_mask=True,
        text_store_length=opt.text_store_length,
        legacy_text_mask=False,
        legacy_gt_sampling=False,
        seed=opt.seed,
        split="val",
    )
    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
    )
    model, _, _, _ = setup_model(opt)
    model.eval()
    outputs = []
    metas = []
    count_logits = []
    count_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="validation calibration forward"):
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            targets = targets or {}
            targets["label"] = batch[0]
            targets["fps"] = torch.full((1,), 1 / opt.clip_length, device=opt.device)
            count_class = min(len(batch[0][0].get("relevant_windows") or []), 4)
            targets["count_class"] = torch.tensor([count_class], device=opt.device)
            raw = model(**model_inputs, targets=targets)
            outputs.append({
                key: value.detach().cpu()
                for key, value in raw.items()
                if key in {
                    "candidate_span", "candidate_logit", "candidate_mask",
                    "event_span", "event_logit", "quality_logit", "event_mask",
                    "pred_count_logits",
                }
            })
            metas.append(batch[0][0])
            if "pred_count_logits" in raw:
                count_logits.append(raw["pred_count_logits"].detach().cpu())
                count_labels.append(count_class)

    temperature, validation_nll = 1.0, None
    if count_logits:
        temperature, validation_nll = _fit_temperature(
            torch.cat(count_logits, dim=0), torch.tensor(count_labels, dtype=torch.long)
        )

    raw_calibration_sha = None
    if variant in {"G0", "G0-Con"}:
        if not args.raw_threshold_calibration:
            raise ValueError(f"{variant} requires --raw_threshold_calibration from same-seed G0-Threshold")
        with open(args.raw_threshold_calibration, "r", encoding="utf-8") as handle:
            raw_calibration = json.load(handle)
        if raw_calibration.get("variant") != "G0-Threshold" or int(raw_calibration.get("seed", -1)) != int(opt.seed):
            raise ValueError("G0-Threshold calibration variant/seed mismatch")
        if raw_calibration.get("baseline_checkpoint_sha256") != getattr(opt, "baseline_checkpoint_sha256", None):
            raise ValueError("G0-Threshold calibration B0 hash mismatch")
        tau_values = [float(raw_calibration["tau_raw"])]
        raw_calibration_sha = sha256_file(args.raw_threshold_calibration)
    else:
        tau_values = [value / 100.0 for value in range(1, 100)]

    best_key = (-1.0, -1.0, -1.0)
    best_tau = tau_values[0]
    for tau in tau_values:
        decoded = [_decode(output, meta, variant, tau, temperature) for output, meta in zip(outputs, metas)]
        pairs = [
            ([window[:2] for window in pred], meta.get("relevant_windows") or [])
            for pred, meta in zip(decoded, metas)
        ]
        set_success = float(compute_event_set_metrics(pairs)["SetSuccess"])
        positive_map = _positive_map(pairs)
        # Final tie-break prefers the lower threshold for determinism.
        key = (set_success, positive_map, -tau)
        if key > best_key:
            best_key = key
            best_tau = tau

    result = {
        "schema_version": "hiea2m.count-calibration.v1",
        "variant": variant,
        "seed": int(opt.seed),
        "split": "val",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "baseline_checkpoint_sha256": getattr(opt, "baseline_checkpoint_sha256", None),
        "feature_manifest_sha256": sha256_file(opt.feature_manifest),
        "data_manifest_index_sha256": sha256_file(args.data_manifest_index),
        "validation_manifest_sha256": split_record["sha256"],
        "raw_threshold_calibration_sha256": raw_calibration_sha,
        "T_count": temperature,
        "validation_count_nll": validation_nll,
        "SetSuccess@0.5": best_key[0],
        "positive_query_mAP_tiebreak": best_key[1],
    }
    if variant in {"G0-Threshold", "G0", "G0-Con"}:
        result["tau_raw"] = best_tau
    else:
        result["tau_mode"] = best_tau
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
