import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to index JSON file")
    parser.add_argument("--seed", required=True, help="Seed value")
    args = parser.parse_args()

    with open(args.index, "r", encoding="utf-8") as f:
        data = json.load(f)

    runs = data.get("runs", {})
    seed_str = str(args.seed)

    if seed_str not in runs:
        print(f"Error: seed {seed_str} not found in runs", file=sys.stderr)
        sys.exit(1)

    run_data = runs[seed_str]
    # Check different potential paths
    if isinstance(run_data, dict):
        if "checkpoint" in run_data:
            ckpt = run_data["checkpoint"]
            if isinstance(ckpt, dict) and "path" in ckpt:
                print(ckpt["path"])
                return
            elif isinstance(ckpt, str):
                print(ckpt)
                return
        if "checkpoint_path" in run_data:
            print(run_data["checkpoint_path"])
            return

    # Fallback / string
    if isinstance(run_data, str):
        print(run_data)
        return

    print(f"Error: could not find checkpoint path for seed {seed_str} in {args.index}", file=sys.stderr)
    sys.exit(1)

if __name__ == "__main__":
    main()
