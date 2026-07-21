import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from training.flash_vtg_gmr.inference import setup_model, start_end_collate, prepare_batch_inputs
from training.flash_vtg_gmr.dataset import StartEndDataset
from tests.test_event_set_metrics import compute_event_set_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--data_manifest_index", required=True, help="Data manifest index")
    parser.add_argument("--split", default="val", help="Split name to calibrate on")
    parser.add_argument("--output", required=True, help="Output calibration JSON path")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    # Load checkpoint to extract original options
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    opt = checkpoint["opt"]
    
    # Override paths and device
    opt.device = args.device
    opt.device = torch.device(f"cuda:{opt.device}" if opt.device >= 0 else "cpu")
    opt.resume = args.checkpoint
    opt.data_manifest_index = args.data_manifest_index
    opt.eval_split_name = args.split
    
    # Build dataset config and create dataset
    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.eval_path if args.split == "val" else opt.train_path, # will override below
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
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        mr_only=opt.mr_only,
        keep_empty_gt=True,
        strict_data_contract=getattr(opt, "strict_data_contract", False),
        require_text_mask=getattr(opt, "require_text_mask", False),
        text_store_length=getattr(opt, "text_store_length", 77),
        legacy_text_mask=getattr(opt, "legacy_text_mask", False),
        legacy_gt_sampling=getattr(opt, "legacy_gt_sampling", False),
        seed=opt.seed,
    )
    
    # Load canonical manifest paths from data_manifest_index
    with open(args.data_manifest_index, "r", encoding="utf-8") as f:
        manifest_index = json.load(f)
    split_path = manifest_index["data_manifests"][args.split]["path"]
    dataset_config["data_path"] = split_path
    
    val_dataset = StartEndDataset(**dataset_config)
    val_loader = DataLoader(
        val_dataset,
        collate_fn=start_end_collate,
        batch_size=1, # always 1 for evaluation
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
    )

    model, _, _, _ = setup_model(opt)
    model.eval()

    # Collect raw outputs from the validation set
    collected_outputs = []
    collected_metas = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Running forward pass on validation"):
            meta = batch[0][0]
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device)
            if targets is not None:
                targets["label"] = batch[0]
                bsz = int(model_inputs["src_vid"].shape[0])
                targets["fps"] = torch.full((bsz,), 1 / opt.clip_length, device=opt.device)
            else:
                targets = {}
                
            outputs = model(**model_inputs, targets=targets)
            
            # Save relevant tensors in CPU memory
            item_out = {}
            for k in ["candidate_span", "candidate_logit", "candidate_mask", 
                      "event_span", "event_logit", "quality_logit", "event_mask",
                      "pred_count_logits", "pred_count_probs"]:
                if k in outputs and outputs[k] is not None:
                    item_out[k] = outputs[k].cpu()
            
            collected_outputs.append(item_out)
            collected_metas.append(meta)

    # Helper function to compute SetSuccess for a given set of parameters
    def evaluate_calibration(param_dict):
        predictions = []
        for out, meta in zip(collected_outputs, collected_metas):
            duration = float(meta["duration"])
            variant = getattr(opt, "variant", None)
            
            # Implement the selection logic with local calibration params
            res = []
            if variant == "G0-Threshold":
                tau_raw = param_dict["tau_raw"]
                cand_span = out["candidate_span"][0]
                cand_logit = out["candidate_logit"][0]
                cand_mask = out["candidate_mask"][0]
                scores = torch.sigmoid(cand_logit)
                
                chosen = []
                for idx in range(cand_span.shape[0]):
                    if cand_mask[idx] and scores[idx] >= tau_raw:
                        s, e = cand_span[idx].tolist()
                        chosen.append([s * duration, e * duration, float(scores[idx])])
                chosen.sort(key=lambda x: x[2], reverse=True)
                res = chosen
                
            elif variant in ("G0", "G0-Con"):
                tau_raw = param_dict["tau_raw"]
                T_count = param_dict.get("T_count", 1.0)
                
                # Apply T_count scaling to logits
                logits = out["pred_count_logits"][0] / T_count
                probs = torch.softmax(logits, dim=-1)
                pred_count = int(probs.argmax().item())
                
                if pred_count > 0:
                    cand_span = out["candidate_span"][0]
                    cand_logit = out["candidate_logit"][0]
                    cand_mask = out["candidate_mask"][0]
                    scores = torch.sigmoid(cand_logit)
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
                    for idx, score in chosen:
                        s, e = cand_span[idx].tolist()
                        res.append([s * duration, e * duration, score])
                        
            elif variant in ("C1", "C2"):
                tau_mode = param_dict["tau_mode"]
                T_count = param_dict.get("T_count", 1.0)
                
                logits = out["pred_count_logits"][0] / T_count
                probs = torch.softmax(logits, dim=-1)
                pred_count = int(probs.argmax().item())
                
                if pred_count > 0:
                    event_span = out["event_span"][0]
                    event_logit = out["event_logit"][0]
                    quality_logit = out["quality_logit"][0]
                    event_mask = out["event_mask"][0]
                    
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
                    for idx, score, _ in chosen:
                        s, e = event_span[idx].tolist()
                        res.append([s * duration, e * duration, score])

            # predictions format: list of [start, end]
            pred_windows = [w[:2] for w in res]
            gt_windows = meta.get("relevant_windows", [])
            if gt_windows is None:
                gt_windows = []
            # Normalize GT to seconds (just in case they are not)
            predictions.append((pred_windows, gt_windows))

        metrics = compute_event_set_metrics(predictions, theta=0.5)
        return metrics["SetSuccess"]

    # Search parameters
    best_params = {}
    variant = getattr(opt, "variant", None)
    
    if variant == "G0-Threshold":
        # Search tau_raw in [0.01, 0.99]
        best_score = -1.0
        best_tau = 0.5
        for tau_int in range(1, 100):
            tau = tau_int / 100.0
            score = evaluate_calibration({"tau_raw": tau})
            if score > best_score:
                best_score = score
                best_tau = tau
        best_params = {"tau_raw": best_tau, "SetSuccess": best_score}
        print(f"Calibrated G0-Threshold: {best_params}")

    elif variant in ("G0", "G0-Con"):
        # We need tau_raw and T_count.
        # But wait! "G0-Threshold validation-only single threshold" should be reused for tau_raw.
        # Let's search over T_count (0.1 to 3.0) and tau_raw (0.01 to 0.99)
        best_score = -1.0
        best_tau = 0.5
        best_T = 1.0
        for T_int in range(1, 31):
            T = T_int / 10.0
            for tau_int in range(1, 100, 5): # coarse search for speed
                tau = tau_int / 100.0
                score = evaluate_calibration({"tau_raw": tau, "T_count": T})
                if score > best_score:
                    best_score = score
                    best_tau = tau
                    best_T = T
        # Fine tuning around best
        for tau_int in range(max(1, int(best_tau*100)-5), min(100, int(best_tau*100)+5)):
            tau = tau_int / 100.0
            score = evaluate_calibration({"tau_raw": tau, "T_count": best_T})
            if score > best_score:
                best_score = score
                best_tau = tau
        best_params = {"tau_raw": best_tau, "T_count": best_T, "SetSuccess": best_score}
        print(f"Calibrated {variant}: {best_params}")

    elif variant in ("C1", "C2"):
        # We need tau_mode and T_count
        best_score = -1.0
        best_tau = 0.5
        best_T = 1.0
        for T_int in range(1, 31):
            T = T_int / 10.0
            for tau_int in range(1, 100, 5):
                tau = tau_int / 100.0
                score = evaluate_calibration({"tau_mode": tau, "T_count": T})
                if score > best_score:
                    best_score = score
                    best_tau = tau
                    best_T = T
        # Fine tuning
        for tau_int in range(max(1, int(best_tau*100)-5), min(100, int(best_tau*100)+5)):
            tau = tau_int / 100.0
            score = evaluate_calibration({"tau_mode": tau, "T_count": best_T})
            if score > best_score:
                best_score = score
                best_tau = tau
        best_params = {"tau_mode": best_tau, "T_count": best_T, "SetSuccess": best_score}
        print(f"Calibrated {variant}: {best_params}")

    # Write calibration parameters
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved calibration to {args.output}")

if __name__ == "__main__":
    main()
