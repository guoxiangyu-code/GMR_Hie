import pprint
import sys
import json
from tqdm import tqdm, trange
import numpy as np
import os
from collections import defaultdict
from models.flash_vtg_gmr.utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from training.flash_vtg_gmr.config import TestOptions
from training.flash_vtg_gmr.dataset import (
    StartEndDataset,
    start_end_collate,
    prepare_batch_inputs,
)
from training.flash_vtg_gmr.postprocessing import PostProcessorDETR
from models.flash_vtg_gmr.standalone_eval.eval import eval_submission
from eval.metrics import compute_gmr_cls
from eval.normalization import load_ts_window_cfg, normalize_ground_truth
from models.flash_vtg_gmr.utils.basic_utils import save_jsonl, save_json
from training.flash_vtg_gmr.reproducibility import configure_runtime, restore_rng_state
from training.flash_vtg_gmr.contracts import sha256_file

import nncore
from nncore.ops import temporal_iou

import logging

logger = logging.getLogger(__name__)


def _compute_legacy_cls_summary(submission, ground_truth, threshold):
    """Map the maintained GMR-CLS evaluator to the legacy training log fields."""
    threshold = float(threshold)
    cls_metrics = compute_gmr_cls(
        submission,
        ground_truth,
        thresholds=(threshold,),
    )
    confusion = cls_metrics["per_threshold"][str(threshold)]
    tp, tn = confusion["TP"], confusion["TN"]
    fp, fn = confusion["FP"], confusion["FN"]
    tpr = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    tnr = 100.0 * tn / (tn + fp) if tn + fp else 0.0
    return {
        "TPR": round(tpr, 2),
        "TNR": round(tnr, 2),
        "BalancedAcc": round((tpr + tnr) / 2.0, 2),
    }
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def clamp_prediction_to_decode_duration(prediction):
    decode_duration = float(prediction.pop("_decode_duration"))
    clamped = []
    for window in prediction["pred_relevant_windows"]:
        start = max(0.0, min(float(window[0]), decode_duration))
        end = max(0.0, min(float(window[1]), decode_duration))
        if end <= start:
            continue
        clamped.append([start, end, *window[2:]])
    prediction["pred_relevant_windows"] = clamped
    if prediction.get("variant") == "G0-Threshold":
        prediction["selected_count"] = len(clamped)
    return prediction


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms, nms_type):
    mr_res_after_nms = []
    for e in mr_res:
        bnd = torch.tensor(e["pred_relevant_windows"])
        for i in range(bnd.size(0)):
            max_idx = bnd[i:, -1].argmax(dim=0)
            bnd = nncore.swap_element(bnd, i, max_idx + i)
            iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

            if nms_type == 'normal':
                bnd[i + 1:, -1][iou >= nms_thd] = 0
            elif nms_type == 'linear':
                bnd[i + 1:, -1] *= 1 - iou
            else:
                raise ValueError(f"Unknown nms_type: {nms_type}")

        _, inds = bnd[:, -1].sort(descending=True)
        bnd = bnd[inds]
        e["pred_relevant_windows"] = bnd.tolist()

        mr_res_after_nms.append(e)
    return mr_res_after_nms


def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)

    shared_qids = set()
    gt_aligned = []

    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission(
            submission,
            gt_data,
            verbose=opt.debug,
            match_number=not opt.debug,
            full_only=opt.eval_full_only,
            mr_only=opt.mr_only,
        )
        if getattr(opt, "use_exist_head", False):
            from models.flash_vtg_gmr.utils.basic_utils import load_jsonl

            gt_raw = load_jsonl(opt.eval_path)
            ts_cfg = load_ts_window_cfg(None)
            gt, _ = normalize_ground_truth(gt_raw, ts_cfg, drop_empty_gt=False)
            from eval.eval_main import validate_qid_coverage

            validate_qid_coverage(submission, gt)
            shared_qids = {e["qid"] for e in gt}
            submission_aligned = submission
            gt_aligned = gt

            pred_score_thd_for_cls = float(getattr(opt, "pred_score_thd_for_cls", 0.5))
            cls_metrics = _compute_legacy_cls_summary(
                submission_aligned,
                gt_aligned,
                threshold=pred_score_thd_for_cls,
            )
            metrics["brief"]["GMR-TPR"] = cls_metrics["TPR"]
            metrics["brief"]["GMR-TNR"] = cls_metrics["TNR"]
            metrics["brief"]["GMR-BalancedAcc"] = cls_metrics["BalancedAcc"]

        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [
            submission_path,
        ]

    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd))
        submission_after_nms = post_processing_mr_nms(
            submission,
            nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms,
            max_after_nms=opt.max_after_nms,
            nms_type=opt.nms_type,
        )

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(
            ".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd)
        )
        save_jsonl(submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_submission(
                submission_after_nms,
                gt_data,
                verbose=opt.debug,
                match_number=not opt.debug,
                full_only=opt.eval_full_only,
                mr_only=opt.mr_only,
            )
            if getattr(opt, "use_exist_head", False):
                submission_after_nms_aligned = [e for e in submission_after_nms if e.get("qid") in shared_qids]
                pred_score_thd_for_cls = float(getattr(opt, "pred_score_thd_for_cls", 0.5))
                cls_metrics_nms = _compute_legacy_cls_summary(
                    submission_after_nms_aligned,
                    gt_aligned,
                    threshold=pred_score_thd_for_cls,
                )
                metrics_nms["brief"]["GMR-TPR"] = cls_metrics_nms["TPR"]
                metrics_nms["brief"]["GMR-TNR"] = cls_metrics_nms["TNR"]
                metrics_nms["brief"]["GMR-BalancedAcc"] = cls_metrics_nms["BalancedAcc"]
            save_metrics_nms_path = submission_nms_path.replace(
                ".jsonl", "_metrics.json"
            )
            save_json(
                metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False
            )
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [
                submission_nms_path,
            ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths

# for HL
@torch.no_grad()
def compute_hl_results(
    model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None
):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5  # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            bsz = int(model_inputs["src_vid"].shape[0])
            targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)
        else:
            targets = {}

        outputs = model(**model_inputs, targets=targets)

        preds = outputs["saliency_scores"].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta["label"]  # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"

            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred = pred.cpu()
                    cur_pred = pred[: len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, 0.0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)

            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[: len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)

                cur_label = torch.Tensor(label).squeeze()[inds].tolist()

                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)

            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters

# for MR
def select_predictions_for_inference(outputs, opt, meta):
    # outputs: output dict from model
    # opt: argparse Namespace
    # meta: sample metadata dict (contains qid, duration, etc.)
    # Returns: list of [start, end, score] in seconds
    
    variant = getattr(opt, "variant", None)
    duration = float(meta.get("D_grid", meta["duration"]))
    
    if variant is None:
        return outputs["_out"]["boundary"].tolist()

    # AEC / P0 / G0 variants
    tau_mode = 0.5
    tau_raw = 0.5
    
    calib = {}
    if getattr(opt, "count_calibration", None) is not None:
        import json
        with open(opt.count_calibration, "r", encoding="utf-8") as f:
            calib = json.load(f)
        tau_mode = float(calib.get("tau_mode", 0.5))
        tau_raw = float(calib.get("tau_raw", 0.5))

    if variant == "G0-Threshold":
        cand_span = outputs["candidate_span"][0]  # (K, 2)
        cand_logit = outputs["candidate_logit"][0]  # (K,)
        cand_mask = outputs["candidate_mask"][0]  # (K,)
        
        scores = torch.sigmoid(cand_logit)  # (K,)
        selected = []
        for idx in range(cand_span.shape[0]):
            if cand_mask[idx] and scores[idx] >= tau_raw:
                s, e = cand_span[idx].tolist()
                selected.append([s * duration, e * duration, float(scores[idx])])
        selected.sort(key=lambda x: x[2], reverse=True)
        return selected

    elif variant in ("G0", "G0-Con"):
        temperature = float(calib.get("T_count", 1.0))
        count_probs = torch.softmax(outputs["pred_count_logits"][0] / temperature, dim=-1)
        pred_count = int(count_probs.argmax().item())
        
        if pred_count == 0:
            return []
            
        cand_span = outputs["candidate_span"][0]  # (K, 2)
        cand_logit = outputs["candidate_logit"][0]  # (K,)
        cand_mask = outputs["candidate_mask"][0]  # (K,)
        
        scores = torch.sigmoid(cand_logit)  # (K,)
        valid_indices = torch.where(cand_mask)[0].tolist()
        
        valid_candidates = [(idx, float(scores[idx])) for idx in valid_indices]
        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if pred_count in (1, 2, 3):
            keep_n = min(pred_count, len(valid_candidates))
            chosen = valid_candidates[:keep_n]
        else:
            above_thd = [x for x in valid_candidates if x[1] >= tau_raw]
            if len(above_thd) < 4:
                chosen = valid_candidates[:min(4, len(valid_candidates))]
            else:
                chosen = above_thd[:min(10, len(above_thd))]
                
        res = []
        for idx, score in chosen:
            s, e = cand_span[idx].tolist()
            res.append([s * duration, e * duration, score])
        return res

    elif variant in ("P0", "P0-R", "P0-AllK"):
        event_span = outputs["event_span"][0]  # (M, 2)
        event_logit = outputs["event_logit"][0]  # (M,)
        quality_logit = outputs["quality_logit"][0]  # (M,)
        event_mask = outputs["event_mask"][0]  # (M,)
        
        event_score = torch.sigmoid(event_logit)
        qual_score = torch.sigmoid(quality_logit)
        mode_score = event_score * qual_score
        
        chosen = []
        for idx in range(event_span.shape[0]):
            if event_mask[idx] and event_score[idx] >= 0.5:
                chosen.append((idx, float(mode_score[idx])))
        chosen.sort(key=lambda x: x[1], reverse=True)
        
        res = []
        for idx, score in chosen:
            s, e = event_span[idx].tolist()
            res.append([s * duration, e * duration, score])
        return res

    elif variant in ("C1", "C2"):
        temperature = float(calib.get("T_count", 1.0))
        count_probs = torch.softmax(outputs["pred_count_logits"][0] / temperature, dim=-1)
        pred_count = int(count_probs.argmax().item())
        
        if pred_count == 0:
            return []
            
        event_span = outputs["event_span"][0]  # (M, 2)
        event_logit = outputs["event_logit"][0]  # (M,)
        quality_logit = outputs["quality_logit"][0]  # (M,)
        event_mask = outputs["event_mask"][0]  # (M,)
        
        event_score = torch.sigmoid(event_logit)
        qual_score = torch.sigmoid(quality_logit)
        mode_score = event_score * qual_score
        
        valid_indices = torch.where(event_mask)[0].tolist()
        valid_modes = [(idx, float(mode_score[idx]), float(event_score[idx])) for idx in valid_indices]
        valid_modes.sort(key=lambda x: x[1], reverse=True)
        
        if pred_count in (1, 2, 3):
            keep_n = min(pred_count, len(valid_modes))
            chosen = valid_modes[:keep_n]
        else:
            above_thd = [x for x in valid_modes if x[2] >= tau_mode]
            if len(above_thd) < 4:
                chosen = valid_modes[:min(4, len(valid_modes))]
            else:
                chosen = above_thd[:min(10, len(above_thd))]
                
        res = []
        for idx, score, _ in chosen:
            s, e = event_span[idx].tolist()
            res.append([s * duration, e * duration, score])
        return res

    return []

@torch.no_grad()
def compute_mr_results(
    model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None
):
    model.eval()
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        if targets is not None:
            targets["label"] = batch[0]
            bsz = int(model_inputs["src_vid"].shape[0])
            targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)
            counts = []
            for meta in batch[0]:
                rel_wins = meta.get("relevant_windows", [])
                if rel_wins is None:
                    rel_wins = []
                counts.append(min(len(rel_wins), 4))
            targets["count_class"] = torch.tensor(counts, dtype=torch.long, device=opt.device)
        else:
            targets = {}
        outputs = model(**model_inputs, targets=targets)

        # Optional existence calibration (GMR): softly suppress window scores for negatives
        pred_exist_scores = None
        if getattr(opt, "use_exist_head", False) and ("pred_exist_logits" in outputs):
            pred_exist_scores = torch.sigmoid(outputs["pred_exist_logits"]).detach().cpu()
            thd = float(getattr(opt, "exist_gate_thd", 0.5))
            mult = torch.where(pred_exist_scores >= thd, torch.ones_like(pred_exist_scores), pred_exist_scores)

        boundary_out = outputs.get("_out", {}).get("boundary", None)
        if pred_exist_scores is not None and boundary_out is not None:
            # Boundary decoding currently assumes an inference batch size of one.
            boundary_out = boundary_out.clone()
            boundary_out[:, 2] = boundary_out[:, 2] * float(mult[0])

        if opt.span_loss_type == "l1":
            _bnd = boundary_out if boundary_out is not None else (outputs.get("_out", {}).get("boundary") if "_out" in outputs else None)
            scores = _bnd[:, 2] if _bnd is not None else None
            pred_spans = _bnd[:, :2].unsqueeze(0) if _bnd is not None else None
            _saliency_scores = outputs["_out"]["saliency"].unsqueeze(0) if "_out" in outputs else None

            saliency_scores = []
            if _saliency_scores is not None:
                valid_vid_lengths = outputs["_out"]["video_msk"].sum(1).cpu().tolist()
                for j in range(len(valid_vid_lengths)):
                    ss = _saliency_scores[j, : int(valid_vid_lengths[j])].tolist()
                    ss = [float(f"{e:.3f}") for e in ss]
                    saliency_scores.append(ss)
        else:
            bsz, n_queries = outputs["pred_spans"].shape[
                :2
            ]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(
                bsz, n_queries, 2, opt.max_v_l
            )
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(
                -1
            )  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions
        for idx, meta in enumerate(query_meta):
            if getattr(opt, "variant", None) is not None:
                cur_ranked_preds = select_predictions_for_inference(outputs, opt, meta)
                cur_ranked_preds = [
                    [float(f"{e[0]:.3f}"), float(f"{e[1]:.3f}"), float(f"{e[2]:.3f}")]
                    for e in cur_ranked_preds
                ]
                decode_duration = float(meta.get("D_decode", meta["duration"]))
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                    _decode_duration=decode_duration,
                    variant=getattr(opt, "variant", None),
                )
                if "pred_count_logits" in outputs:
                    temperature = 1.0
                    if getattr(opt, "count_calibration", None):
                        with open(opt.count_calibration, "r", encoding="utf-8") as handle:
                            temperature = float(json.load(handle).get("T_count", 1.0))
                    count_probs = torch.softmax(outputs["pred_count_logits"][0] / temperature, dim=-1)
                    pred_count = int(count_probs.argmax().item())
                    cur_query_pred["pred_count"] = pred_count
                    cur_query_pred["pred_count_probs"] = [float(x) for x in count_probs.tolist()]
                    cur_query_pred["pred_exist_score"] = float(1.0 - count_probs[0])
                
                # Save oracle modes and raw proposals for diagnostics
                if "event_span" in outputs:
                    event_span = outputs["event_span"][0]
                    event_mask = outputs["event_mask"][0]
                    event_logits = outputs["event_logit"][0]
                    quality_logits = outputs["quality_logit"][0]
                    duration = float(meta.get("D_grid", meta["duration"]))
                    modes = []
                    for m_i in range(event_span.shape[0]):
                        if event_mask[m_i]:
                            s, e = event_span[m_i].tolist()
                            modes.append([
                                float(f"{s * duration:.3f}"),
                                float(f"{e * duration:.3f}"),
                                float(event_logits[m_i]),
                                float(quality_logits[m_i]),
                            ])
                    cur_query_pred["oracle_mode_windows"] = modes

                if "candidate_span" in outputs:
                    cand_span = outputs["candidate_span"][0]
                    cand_mask = outputs["candidate_mask"][0]
                    cand_logits = outputs["candidate_logit"][0]
                    duration = float(meta.get("D_grid", meta["duration"]))
                    cands = []
                    for c_i in range(cand_span.shape[0]):
                        if cand_mask[c_i]:
                            s, e = cand_span[c_i].tolist()
                            cands.append([
                                float(f"{s * duration:.3f}"),
                                float(f"{e * duration:.3f}"),
                                float(cand_logits[c_i]),
                            ])
                    cur_query_pred["raw_proposal_windows"] = cands
                mr_res.append(cur_query_pred)
            else:
                spans_src = boundary_out if boundary_out is not None else outputs["_out"]["boundary"]
                decode_duration = float(meta.get("D_decode", meta["duration"]))
                spans = torch.clamp(spans_src, 0, decode_duration)
                cur_ranked_preds = spans.tolist()
                cur_ranked_preds = [
                    [float(f"{e:.3f}") for e in row] for row in cur_ranked_preds
                ]
                cur_query_pred = dict(
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    pred_relevant_windows=cur_ranked_preds,
                    _decode_duration=decode_duration,
                )
                if not getattr(opt, "mr_only", False):
                    cur_query_pred["pred_saliency_scores"] = saliency_scores[idx]
                if pred_exist_scores is not None:
                    cur_query_pred["pred_exist_score"] = float(f"{float(pred_exist_scores[idx]):.3f}")
                mr_res.append(cur_query_pred)

        if criterion is not None:
            loss_dict = criterion(batch, outputs, targets)
            loss_dict = {k: v for k, v in loss_dict.items() if "loss" in k}
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
            )
            loss_dict["loss_overall"] = float(losses)
        else:
            loss_dict = {k: v for k, v in outputs.items() if 'loss' in k}
            losses = sum(loss_dict.values())
            loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v)
            )

    if write_tb and len(loss_meters) != 1:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    if opt.dset_name in ["hl"]:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=150,
            min_w_l=2,
            max_w_l=150,
            move_window_method="left",
            process_func_names=("clip_ts", "round_multiple"),
        )
    elif opt.dset_name in ["charadesSTA"]:
        if opt.v_feat_dim == 4096:  # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=360,
                min_w_l=12,
                max_w_l=360,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length,
                min_ts_val=0,
                max_ts_val=150,
                min_w_l=2,
                max_w_l=60,
                move_window_method="left",
                process_func_names=("clip_ts", "round_multiple"),
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length,
            min_ts_val=0,
            max_ts_val=50000,
            min_w_l=0,
            max_w_l=50000,
            move_window_method="left",
            process_func_names=(["round_multiple"]),
        )

    mr_res = post_processor(mr_res)
    for prediction in mr_res:
        clamp_prediction_to_decode_duration(prediction)
    return mr_res, loss_meters


def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(
        model, eval_loader, opt, epoch_i, criterion, tb_writer
    )  # list(dict)
    return eval_res, eval_loss_meters


def eval_epoch(
    model,
    eval_dataset,
    opt,
    save_submission_filename,
    epoch_i=None,
    criterion=None,
    tb_writer=None,
):
    logger.info("Generate submissions")
    model.eval()
    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    if opt.dset_name == "tacos":
        shuffle = True
    else:
        shuffle = False

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=shuffle,
        pin_memory=opt.pin_memory,
    )

    # tvsum
    if opt.dset_name in ["tvsum", "youtube_uni"]:
        metrics, eval_loss_meters = compute_hl_results(
            model, eval_loader, opt, epoch_i, criterion, tb_writer
        )

        # to match original save format
        submission = [{"brief": metrics}]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(
            model, eval_loader, opt, epoch_i, criterion, tb_writer
        )

        if opt.dset_name in ["charadesSTA", "tacos", "nlq"]:
            new_submission = []
            for s in submission:
                s.pop("pred_saliency_scores", None)
                new_submission.append(s)
            submission = new_submission

        if getattr(opt, "variant", None) is not None:
            submission_path = os.path.join(opt.results_dir, save_submission_filename)
            save_jsonl(submission, submission_path)
            if opt.eval_split_name != "val":
                return None, None, eval_loss_meters, [submission_path]
            from eval.eval_main import evaluate_gmr
            metrics = evaluate_gmr(
                submission,
                eval_dataset.data,
                map_num_workers=max(1, int(opt.num_workers)),
                verbose=False,
            )
            metrics["brief"]["MR-full-mAP"] = metrics["brief"]["mAP"]
            from eval.event_set_metrics import compute_submission_diagnostics
            diagnostics = compute_submission_diagnostics(submission, eval_dataset.data)
            metrics["diagnostics"] = diagnostics
            metrics["brief"].update({
                "SetSuccess@0.5": diagnostics["SetSuccess@0.5"],
                "Count-Acc-5": diagnostics["Count-Acc-5"],
                "Selected-FullCoverage@0.5": diagnostics["Selected-FullCoverage@0.5"],
                "DuplicateRate@0.5": diagnostics["DuplicateRate@0.5"],
            })
            coverage = float(diagnostics["Selected-FullCoverage@0.5"])
            duplicate_complement = 1.0 - float(diagnostics["DuplicateRate@0.5"])
            if np.isfinite(coverage) and coverage + duplicate_complement > 0:
                adapter_score = 2.0 * coverage * duplicate_complement / (coverage + duplicate_complement)
            else:
                adapter_score = 0.0
            metrics["brief"]["AdapterScore"] = adapter_score
            if getattr(opt, "variant", None) in {"P0", "P0-R", "P0-AllK"}:
                val_adapter_loss = sum(
                    float(eval_loss_meters[name].avg)
                    for name in (
                        ("loss_event", "loss_quality", "loss_span")
                        if getattr(opt, "variant", None) == "P0-AllK"
                        else ("loss_event", "loss_quality")
                    )
                )
                if not np.isfinite(val_adapter_loss):
                    raise FloatingPointError(
                        f"Non-finite validation adapter loss: {val_adapter_loss}"
                    )
                metrics["brief"]["Val-Adapter-Loss"] = val_adapter_loss
                # Checkpoint keys are lexicographically maximised.
                metrics["brief"]["AdapterTieBreak"] = -val_adapter_loss
            metrics_path = submission_path.replace(".jsonl", "_metrics.json")
            save_json(metrics, metrics_path, save_pretty=True, sort_keys=False)
            metrics_nms = None
            latest_file_paths = [submission_path, metrics_path]
        else:
            metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
                submission, opt, eval_dataset.data, save_submission_filename
            )
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    
    # Configure option flags automatically based on variant
    variant = getattr(opt, "variant", None)
    if variant is not None:
        # Part 2 has exactly one empty-set rule; the legacy existence head/gate
        # is neither constructed nor used.
        opt.use_exist_head = False
        if variant in ("G0", "G0-Con"):
            opt.enable_aec = True
            opt.aec_variant = variant
            opt.freeze_backbone = True
            opt.enable_adapter = False
        elif variant in ("P0", "P0-R", "P0-AllK"):
            opt.enable_adapter = True
            opt.adapter_variant = variant
            opt.freeze_backbone = True
            opt.enable_aec = False
        elif variant in ("C1", "C2"):
            opt.enable_adapter = True
            opt.adapter_variant = "P0"
            opt.enable_aec = True
            opt.aec_variant = variant
            opt.freeze_backbone = True
            opt.freeze_adapter = True

    if getattr(opt, "feature_manifest", None):
        opt.feature_manifest_sha256 = sha256_file(opt.feature_manifest)

    # During public-P0 inference the checkpoint itself is the immutable
    # EventInterface producer.  Stamp its hash before constructing the model so
    # every emitted EventInterfaceV1 carries the complete provenance tuple.
    if variant in {"P0", "P0-R", "P0-AllK"} and getattr(opt, "resume", None):
        opt.public_p0_checkpoint_sha256 = sha256_file(opt.resume)

    baseline_record = None
    if variant is not None:
        if not getattr(opt, "baseline_index", None):
            raise ValueError("Part 2 requires --baseline_index")
        with open(opt.baseline_index, "r", encoding="utf-8") as handle:
            baseline_index = json.load(handle)
        if baseline_index.get("variant") != "B0":
            raise ValueError("Part 2 requires a finalized B0 baseline index")
        # The baseline identity comes from the verified index, never from the
        # historical parser default embedded in an older opt.json.
        opt.baseline_variant = "B0"
        seed_record = baseline_index.get("runs", {}).get(str(opt.seed))
        if seed_record is None:
            raise ValueError(f"Baseline index has no seed {opt.seed}")
        baseline_record = seed_record["checkpoint"]
        if sha256_file(baseline_record["path"]) != baseline_record["sha256"]:
            raise ValueError("Baseline checkpoint hash mismatch")
        if sha256_file(opt.feature_manifest) != baseline_index["feature_manifest"]["sha256"]:
            raise ValueError("Baseline/feature manifest hash mismatch")
        if sha256_file(opt.data_manifest_index) != baseline_index["data_manifest_index"]["sha256"]:
            raise ValueError("Baseline/data manifest index hash mismatch")
        opt.baseline_checkpoint_sha256 = baseline_record["sha256"]

        calibration_path = getattr(opt, "count_calibration", None)
        if calibration_path:
            with open(calibration_path, "r", encoding="utf-8") as handle:
                calibration = json.load(handle)
            if calibration.get("variant") != variant or int(calibration.get("seed", -1)) != int(opt.seed):
                raise ValueError("Count calibration variant/seed mismatch")
            if calibration.get("split") != "val":
                raise ValueError("Count calibration must be fitted on validation")
            if calibration.get("baseline_checkpoint_sha256") != opt.baseline_checkpoint_sha256:
                raise ValueError("Count calibration B0 hash mismatch")
            if calibration.get("feature_manifest_sha256") != opt.feature_manifest_sha256:
                raise ValueError("Count calibration feature hash mismatch")
            if getattr(opt, "resume", None) and calibration.get("checkpoint_sha256") != sha256_file(opt.resume):
                raise ValueError("Count calibration checkpoint hash mismatch")
            if variant in {"G0", "G0-Con"} and not calibration.get("raw_threshold_calibration_sha256"):
                raise ValueError("G0/G0-Con calibration must bind frozen G0-Threshold calibration")
        elif hasattr(opt, "eval_results_dir") and variant in {"G0-Threshold", "G0", "G0-Con", "C1", "C2"}:
            raise ValueError(f"{variant} inference requires --count_calibration")
        if variant in {"P0", "P0-R", "P0-AllK"} and calibration_path:
            raise ValueError("P0 inference must not use count calibration")

    from models.flash_vtg_gmr.model import build_model1
    model, criterion = build_model1(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    # Load backbone checkpoint (B0) if provided
    if getattr(opt, "init_backbone_ckpt", None) is not None:
        if baseline_record is not None:
            if os.path.realpath(opt.init_backbone_ckpt) != os.path.realpath(baseline_record["path"]):
                raise ValueError("--init_backbone_ckpt is not the indexed checkpoint for this seed")
            if sha256_file(opt.init_backbone_ckpt) != baseline_record["sha256"]:
                raise ValueError("--init_backbone_ckpt hash mismatch")
        logger.info(f"Load backbone checkpoint from {opt.init_backbone_ckpt}")
        ckpt = torch.load(opt.init_backbone_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict"))
        backbone_state = {k: v for k, v in state.items() if not k.startswith("event_adapter") and not k.startswith("aec")}
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        illegal_missing = [key for key in missing if not key.startswith(("event_adapter.", "aec."))]
        illegal_unexpected = [key for key in unexpected if not key.startswith("exist_head.")]
        if illegal_missing or illegal_unexpected:
            raise RuntimeError(
                f"Invalid B0 partial init: missing={illegal_missing}, unexpected={illegal_unexpected}"
            )
        logger.info(f"Backbone loaded. Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys")

    # Load adapter checkpoint (public P0) if provided
    if getattr(opt, "adapter_ckpt", None) is not None:
        logger.info(f"Load adapter checkpoint from {opt.adapter_ckpt}")
        ckpt = torch.load(opt.adapter_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict"))
        ckpt_seed = int(ckpt.get("seed", getattr(ckpt.get("opt", None), "seed", -1)))
        if ckpt_seed != int(opt.seed):
            raise ValueError(f"Public P0 seed mismatch: {ckpt_seed} != {opt.seed}")
        expected_b0 = ckpt.get("baseline_checkpoint_sha256")
        if expected_b0 != opt.baseline_checkpoint_sha256:
            raise ValueError("Public P0 was not built from the indexed same-seed B0")
        expected_feature = ckpt.get("feature_manifest_sha256")
        if expected_feature != opt.feature_manifest_sha256:
            raise ValueError("Public P0 feature manifest hash mismatch")
        # Public P0 is a complete checkpoint.  Load B0 + adapter together and
        # leave only the newly constructed AEC keys missing.
        p0_state = {k: v for k, v in state.items() if not k.startswith("aec.")}
        missing, unexpected = model.load_state_dict(p0_state, strict=False)
        illegal_missing = [key for key in missing if not key.startswith("aec.")]
        if illegal_missing or unexpected:
            raise RuntimeError(f"Invalid public P0 checkpoint: missing={illegal_missing}, unexpected={unexpected}")
        opt.public_p0_checkpoint_sha256 = sha256_file(opt.adapter_ckpt)
        logger.info("Loaded complete B0+P0 checkpoint; only AEC remains newly initialized")

    # Freezing parameters
    if getattr(opt, "freeze_backbone", False):
        logger.info("Freezing backbone parameters")
        for n, p in model.named_parameters():
            if not n.startswith("event_adapter") and not n.startswith("aec"):
                p.requires_grad = False

    if getattr(opt, "freeze_adapter", False):
        logger.info("Freezing adapter parameters")
        for n, p in model.named_parameters():
            if n.startswith("event_adapter"):
                p.requires_grad = False

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad],
            "lr": opt.lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop, gamma=0.5)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu", weights_only=False)
        from collections import OrderedDict
        state = checkpoint.get("model", checkpoint.get("state_dict"))
        if state is None:
            raise KeyError("Checkpoint must contain 'model' or 'state_dict'")
        if variant == "G0-Threshold":
            # The indexed Part 1 B0 predates the Part 2 single-empty-set rule
            # and contains a legacy existence head.  G0-Threshold reuses only
            # the localization backbone; no other resume path gets this
            # narrowly-scoped compatibility exception.
            state = {key: value for key, value in state.items() if not key.startswith("exist_head.")}
        if any(k.startswith("module.") for k in state.keys()):
            new_state_dict = OrderedDict()
            for k, v in state.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(state, strict=True)
        if opt.resume_all:
            if getattr(opt, "strict_data_contract", False) and checkpoint.get("checkpoint_boundary") != "epoch":
                raise ValueError("Strict resume only supports checkpoints saved at an epoch boundary")
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            opt.start_epoch = checkpoint["epoch"] + 1
            opt._resume_rng_state = checkpoint.get("reproducibility_state")
            opt._resume_training_state = checkpoint.get("training_state")
            restore_rng_state(opt._resume_rng_state)
    else:
        logger.warning(
            "If you intend to evaluate the model, please specify --resume with ckpt path"
        )

    return model, criterion, optimizer, lr_scheduler


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    opt.cfg = nncore.Config.from_file(opt.config)

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")

    configure_runtime(repro_check=bool(getattr(opt, "repro_check", False)))

    assert opt.eval_path is not None
    if opt.eval_split_name == "val":
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
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
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        mr_only=opt.mr_only,
        keep_empty_gt=(getattr(opt, "variant", None) is not None or bool(getattr(opt, "use_exist_head", False))),
        strict_data_contract=getattr(opt, "strict_data_contract", False),
        require_text_mask=getattr(opt, "require_text_mask", False),
        text_store_length=getattr(opt, "text_store_length", 77),
        legacy_text_mask=getattr(opt, "legacy_text_mask", False),
        legacy_gt_sampling=getattr(opt, "legacy_gt_sampling", False),
        seed=getattr(opt, "seed", 2024),
        split=opt.eval_split_name,
    )
    model, criterion, _, _ = setup_model(opt)
    # A resolved, immutable inference configuration is part of every formal
    # run record.  This is especially important for calibration-only
    # G0-Threshold, which deliberately resumes a B0 checkpoint.
    resolved_opt = {}
    for key, value in vars(opt).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            resolved_opt[key] = value
        elif isinstance(value, (list, tuple)):
            resolved_opt[key] = list(value)
        elif isinstance(value, dict):
            resolved_opt[key] = value
        else:
            resolved_opt[key] = str(value)
    save_json(
        resolved_opt,
        os.path.join(opt.results_dir, "resolved_inference_opt.json"),
        save_pretty=True,
        sort_keys=True,
    )
    save_submission_filename = "hl_{}_submission.jsonl".format(opt.eval_split_name)

    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = eval_epoch(
            model, eval_dataset, opt, save_submission_filename, criterion=criterion
        )
    if opt.eval_split_name == "val":
        logger.info(
            "metrics_no_nms {}".format(
                pprint.pformat(metrics_no_nms["brief"], indent=4)
            )
        )
    if metrics_nms is not None:
        logger.info(
            "metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4))
        )


if __name__ == "__main__":
    start_inference()
