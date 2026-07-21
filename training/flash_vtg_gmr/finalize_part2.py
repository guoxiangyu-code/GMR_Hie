import os
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/adapters/part2_index.json", help="Path to Part 2 index JSON")
    args = parser.parse_args()

    # The 6 required variants and 2 seeds
    required_variants = ["G0-Threshold", "G0", "G0-Con", "P0", "C1", "C2"]
    required_seeds = [2024, 2025]

    if not os.path.exists(args.index):
        print(f"Error: Part 2 index file {args.index} does not exist. Creating a placeholder...", file=sys.stderr)
        placeholder = {
            "status": "INCOMPLETE",
            "runs": {}
        }
        os.makedirs(os.path.dirname(args.index), exist_ok=True)
        with open(args.index, "w") as f:
            json.dump(placeholder, f, indent=2)

    with open(args.index, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", {})
    all_ok = True
    report = {}

    for seed in required_seeds:
        seed_str = str(seed)
        report[seed_str] = {}
        if seed_str not in runs:
            runs[seed_str] = {}

        for var in required_variants:
            report[seed_str][var] = {"status": "MISSING"}
            var_data = runs[seed_str].get(var, {})
            
            ckpt_path = var_data.get("checkpoint_path", "")
            eval_path = var_data.get("evaluation_path", "")
            diag_path = var_data.get("diagnostics_path", "")

            # Check files on disk
            ckpt_ok = bool(ckpt_path and os.path.exists(ckpt_path))
            eval_ok = bool(eval_path and os.path.exists(eval_path))
            diag_ok = bool(diag_path and os.path.exists(diag_path))

            if var == "G0-Threshold":
                # G0-Threshold is validation only, might not have checkpoint_path
                # But must have evaluation_path and diagnostics_path
                if eval_ok and diag_ok:
                    report[seed_str][var] = {
                        "status": "VERIFIED",
                        "evaluation_path": eval_path,
                        "diagnostics_path": diag_path
                    }
                else:
                    report[seed_str][var] = {
                        "status": "FAILED",
                        "reason": f"Missing files (eval_ok={eval_ok}, diag_ok={diag_ok})"
                    }
                    all_ok = False
            else:
                if ckpt_ok and eval_ok and diag_ok:
                    report[seed_str][var] = {
                        "status": "VERIFIED",
                        "checkpoint_path": ckpt_path,
                        "evaluation_path": eval_path,
                        "diagnostics_path": diag_path
                    }
                else:
                    report[seed_str][var] = {
                        "status": "FAILED",
                        "reason": f"Missing files (ckpt_ok={ckpt_ok}, eval_ok={eval_ok}, diag_ok={diag_ok})"
                    }
                    all_ok = False

    # Update index status
    data["status"] = "COMPLETE" if all_ok else "INCOMPLETE"
    data["runs"] = runs
    data["report"] = report

    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("\n--- Part 2 Verification Report ---")
    print(json.dumps(report, indent=2))
    print(f"Final status: {data['status']}")
    print("----------------------------------")

    if not all_ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
