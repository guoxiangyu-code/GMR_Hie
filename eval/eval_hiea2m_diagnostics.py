import json
import argparse
import numpy as np
import os
import sys

from tests.test_event_set_metrics import compute_event_set_metrics, tiou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, help="Submission JSONL path")
    parser.add_argument("--ground_truth", required=True, help="Ground truth JSONL path")
    parser.add_argument("--output", required=True, help="Output diagnostics JSON path")
    args = parser.parse_args()

    # Load submission and ground truth
    preds_dict = {}
    with open(args.submission, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            preds_dict[item["qid"]] = item

    gts_dict = {}
    with open(args.ground_truth, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            gts_dict[item["qid"]] = item

    # Align by qid
    shared_qids = set(preds_dict.keys()) & set(gts_dict.keys())
    if not shared_qids:
        print("Error: No aligned QIDs found between submission and ground truth!", file=sys.stderr)
        sys.exit(1)

    print(f"Aligning {len(shared_qids)} queries for diagnostic evaluation.")

    predictions = []
    oracle_mode_pairs = []
    raw_proposal_pairs = []

    count_gt_list = []
    count_pred_list = []

    # For classification metrics (pos/neg classification)
    y_true_exist = []
    y_pred_exist_score = []

    for qid in sorted(shared_qids):
        pred_item = preds_dict[qid]
        gt_item = gts_dict[qid]

        pred_windows = [w[:2] for w in pred_item.get("pred_relevant_windows", [])]
        gt_windows = gt_item.get("relevant_windows", [])
        if gt_windows is None:
            gt_windows = []

        predictions.append((pred_windows, gt_windows))

        # Oracle modes
        if "oracle_mode_windows" in pred_item:
            oracle_mode_pairs.append((pred_item["oracle_mode_windows"], gt_windows))
        # Raw proposals
        if "raw_proposal_windows" in pred_item:
            raw_proposal_pairs.append((pred_item["raw_proposal_windows"], gt_windows))

        # Count variables
        gt_count = len(gt_windows)
        count_gt_list.append(gt_count)

        if "pred_count" in pred_item:
            count_pred_list.append(pred_item["pred_count"])
        else:
            count_pred_list.append(min(len(pred_windows), 4))

        # Classification (null query has gt_count == 0)
        is_pos = int(gt_count > 0)
        y_true_exist.append(is_pos)
        
        # Determine existence score: e.g. from pred_exist_score or derived from count prediction
        if "pred_exist_score" in pred_item:
            exist_score = pred_item["pred_exist_score"]
        elif "pred_count" in pred_item:
            # P(count > 0) or simply 1.0 if pred_count > 0 else 0.0
            exist_score = 1.0 if pred_item["pred_count"] > 0 else 0.0
        else:
            exist_score = 1.0 if len(pred_windows) > 0 else 0.0
        y_pred_exist_score.append(exist_score)

    # 1. Event Set Metrics
    set_metrics = compute_event_set_metrics(predictions, theta=0.5)
    
    # Selected-FullCoverage
    fc_list = []
    for P_q, G_q in predictions:
        if len(G_q) >= 2:
            # Bipartite matching size
            from tests.test_event_set_metrics import max_bipartite_matching
            m_size = len(max_bipartite_matching(P_q, G_q, theta=0.5))
            fc_list.append(1 if m_size == len(G_q) else 0)
    selected_fc = np.mean(fc_list) if fc_list else float("nan")

    # Oracle Mode FullCoverage
    oracle_fc = float("nan")
    if oracle_mode_pairs:
        ofc_list = []
        for O_q, G_q in oracle_mode_pairs:
            if len(G_q) >= 2:
                from tests.test_event_set_metrics import max_bipartite_matching
                m_size = len(max_bipartite_matching(O_q, G_q, theta=0.5))
                ofc_list.append(1 if m_size == len(G_q) else 0)
        oracle_fc = np.mean(ofc_list) if ofc_list else float("nan")

    # Raw Proposal Oracle FullCoverage
    raw_oracle_fc = float("nan")
    if raw_proposal_pairs:
        rofc_list = []
        for R_q, G_q in raw_proposal_pairs:
            if len(G_q) >= 2:
                from tests.test_event_set_metrics import max_bipartite_matching
                m_size = len(max_bipartite_matching(R_q, G_q, theta=0.5))
                rofc_list.append(1 if m_size == len(G_q) else 0)
        raw_oracle_fc = np.mean(rofc_list) if rofc_list else float("nan")

    # 2. Count Metrics
    count_gt = np.array(count_gt_list)
    count_pred = np.array(count_pred_list)

    # Clip gt to 5 classes: min(c, 4)
    gt_clipped = np.minimum(count_gt, 4)
    pred_clipped = np.minimum(count_pred, 4)

    count_acc_5 = np.mean(gt_clipped == pred_clipped)
    # Exact accuracy (on unclipped counts)
    count_acc_exact = np.mean(count_gt == count_pred)
    mae = np.mean(np.abs(count_gt - count_pred))

    over_prediction_rate = np.mean(count_pred > count_gt)
    under_prediction_rate = np.mean(count_pred < count_gt)

    # 3. Pos/Neg Classification Metrics
    from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
    
    auroc = roc_auc_score(y_true_exist, y_pred_exist_score) if len(set(y_true_exist)) > 1 else float("nan")
    
    # Binary classification decisions at threshold 0.5
    y_pred_exist_bin = [int(score >= 0.5) for score in y_pred_exist_score]
    cm = confusion_matrix(y_true_exist, y_pred_exist_bin)
    
    # Confusion matrix elements
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        null_fpr = fp / (tn + fp) if (tn + fp) > 0 else float("nan")
        rej_f1 = f1_score(y_true_exist, y_pred_exist_bin)
    else:
        null_fpr = float("nan")
        rej_f1 = float("nan")

    # 4. Grouped Metrics (Null / Single / Multi)
    grouped = {}
    for group_name, condition in [
        ("null_gt", lambda count: count == 0),
        ("single_gt", lambda count: count == 1),
        ("multi_gt", lambda count: count >= 2),
    ]:
        indices = [i for i, c in enumerate(count_gt_list) if condition(c)]
        if indices:
            preds_grp = [predictions[i] for i in indices]
            acc_grp = np.mean([count_gt_list[i] == count_pred_list[i] for i in indices])
            success_grp = np.mean([1 if len(p[1]) == 0 and len(p[0]) == 0 else 
                                   (1 if len(p[0]) == len(p[1]) == len(max_bipartite_matching(p[0], p[1], 0.5)) else 0)
                                   for p in preds_grp])
            grouped[group_name] = {
                "count": len(indices),
                "count_acc": acc_grp,
                "SetSuccess": success_grp
            }
        else:
            grouped[group_name] = {"count": 0, "count_acc": float("nan"), "SetSuccess": float("nan")}

    # Assemble diagnostics JSON
    diagnostics = {
        "SetSuccess@0.5": set_metrics["SetSuccess"],
        "DuplicateRate@0.5": set_metrics["DuplicateRate"],
        "Selected-FullCoverage@0.5": selected_fc,
        "Oracle-Mode-FullCoverage@0.5": oracle_fc,
        "Raw-Proposal-Oracle-FullCoverage@0.5": raw_oracle_fc,
        "Count-Acc-5": count_acc_5,
        "Count-Acc-Exact": count_acc_exact,
        "MAE": mae,
        "OverPredictionRate": over_prediction_rate,
        "UnderPredictionRate": under_prediction_rate,
        "AUROC": auroc,
        "Rej-F1": rej_f1,
        "Null-FPR": null_fpr,
        "grouped_metrics": grouped
    }

    # Write to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Print output summary
    print("\n--- Diagnostic Evaluation Summary ---")
    for k, v in diagnostics.items():
        if k != "grouped_metrics":
            print(f"{k:35s}: {v}")
    print("------------------------------------")

if __name__ == "__main__":
    main()
