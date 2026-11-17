"""Run the paper tasks and seeds with resumable subprocesses."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


TASKS = {
    "PHM2009": {
        "T0": "[[4,8,12],0]",
        "T1": "[[0,8,12],4]",
        "T2": "[[0,4,12],8]",
        "T3": "[[0,4,8],12]",
    },
    "PU": {
        "T0": "[[1,2,3],0]",
        "T1": "[[0,2,3],1]",
        "T2": "[[0,1,3],2]",
        "T3": "[[0,1,2],3]",
    },
}


def parse_seeds(value):
    seeds = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(part))
    return sorted(set(seeds))


def completed_runs(output_root):
    completed = set()
    for path in output_root.glob("*/summary.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            completed.add((item["data_name"], item["task_name"], int(item["seed"])))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("41-50"))
    parser.add_argument("--pu_data_root", default="datasets/PU")
    parser.add_argument(
        "--phm_data_root",
        default="datasets/PHM2009/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt",
    )
    parser.add_argument("--output_root", type=Path, default=Path("results/runs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lambda_domain", type=float, default=0.0)
    parser.add_argument("--lambda_metric", type=float, default=0.2)
    parser.add_argument("--lambda_mixup", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed = completed_runs(output_root)

    for dataset in args.datasets:
        for task_name, transfer_task in TASKS[dataset].items():
            for seed in args.seeds:
                key = (dataset, task_name, seed)
                if key in completed:
                    print(f"[skip] {dataset} {task_name} seed={seed}")
                    continue
                command = [
                    sys.executable, str(root / "main.py"),
                    "--data_name", dataset,
                    "--task_name", task_name,
                    "--transfer_task", transfer_task,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--device", args.device,
                    "--lambda_domain", str(args.lambda_domain),
                    "--lambda_metric", str(args.lambda_metric),
                    "--lambda_mixup", str(args.lambda_mixup),
                    "--pu_data_root", args.pu_data_root,
                    "--phm_data_root", args.phm_data_root,
                    "--output_root", str(output_root),
                ]
                print("[run]", " ".join(command))
                if not args.dry_run:
                    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
