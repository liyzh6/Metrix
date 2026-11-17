"""Command-line entry point for the standalone CC implementation."""

import argparse
import ast
import json
import logging
from datetime import datetime
from pathlib import Path

from train import CCTrainer, set_seed


def parse_transfer_task(value):
    try:
        task = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid transfer task: {value}") from error
    if not isinstance(task, (list, tuple)) or len(task) != 2:
        raise argparse.ArgumentTypeError("transfer_task must look like [[0,1,2],3]")
    sources, target = task
    if not isinstance(sources, (list, tuple)) or len(sources) != 3:
        raise argparse.ArgumentTypeError("transfer_task must contain exactly three sources")
    return [list(map(int, sources)), int(target)]


def infer_task_name(data_name, target):
    if data_name == "PU":
        return f"T{target}"
    return {0: "T0", 4: "T1", 8: "T2", 12: "T3"}.get(target, f"target{target}")


def build_parser():
    parser = argparse.ArgumentParser(description="Standalone CC fault-diagnosis trainer")
    parser.add_argument("--data_name", choices=("PU", "PHM2009"), default="PU")
    parser.add_argument("--transfer_task", type=parse_transfer_task, default=[[1, 2, 3], 0])
    parser.add_argument("--task_name", default=None)
    parser.add_argument("--pu_data_root", default="datasets/PU")
    parser.add_argument(
        "--phm_data_root",
        default="datasets/PHM2009/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt",
    )
    parser.add_argument("--normalization", choices=("mean-std", "-1-1"), default="mean-std")
    parser.add_argument("--num_train_samples", type=int, default=200)
    parser.add_argument("--num_test_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--save_every", type=int, default=10,
                        help="save a periodic checkpoint every N epochs; 0 disables periodic saves")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--steps", type=lambda value: [int(x) for x in value.split(",")], default=[20, 40])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda_domain", type=float, default=0.0)
    parser.add_argument("--lambda_metric", type=float, default=0.2)
    parser.add_argument("--lambda_mixup", type=float, default=0.5)
    parser.add_argument("--metric_margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", default="results/runs")
    return parser


def configure_logging(run_dir):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    for handler in (
        logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)


def main():
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.num_train_samples <= 0 or args.num_test_samples <= 0:
        raise ValueError("sample counts must be positive")

    target = args.transfer_task[1]
    args.task_name = args.task_name or infer_task_name(args.data_name, target)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{args.data_name}_{args.task_name}_S{args.seed}_{stamp}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    configure_logging(run_dir)
    set_seed(args.seed)

    if args.data_name == "PU":
        from data.PU_bearing import build_datasets
        datasets, num_classes = build_datasets(
            args.transfer_task, args.num_train_samples, args.num_test_samples,
            args.normalization, args.pu_data_root,
        )
    else:
        from data.PHM_2009_Geatbox import build_datasets
        datasets, num_classes = build_datasets(
            args.transfer_task, args.num_train_samples, args.num_test_samples,
            args.normalization, args.phm_data_root,
        )

    args.num_classes = num_classes
    config = vars(args).copy()
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("run_dir=%s", run_dir.resolve())
    logging.info("config=%s", json.dumps(config, ensure_ascii=False))
    summary = CCTrainer(args, datasets, num_classes, run_dir).run()
    logging.info("summary=%s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
