"""Plot target accuracy and training loss for completed Metrix runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for data_name in ("PHM2009", "PU"):
        runs = []
        for history in sorted(args.runs_root.glob("*/history.csv")):
            config_path = history.parent / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("data_name") != data_name or int(config.get("seed", -1)) != args.seed:
                continue
            with history.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                runs.append((config.get("task_name", history.parent.name), rows))

        runs.sort(key=lambda item: item[0])
        for field, title, ylabel, filename in (
            ("target_accuracy", f"{data_name} target accuracy", "Accuracy",
             f"{data_name}_target_accuracy.png"),
            ("train_total", f"{data_name} training loss", "Loss",
             f"{data_name}_training_loss.png"),
        ):
            if not runs:
                continue
            figure, axis = plt.subplots(figsize=(7.2, 5), dpi=220)
            for task, rows in runs:
                axis.plot(
                    [int(row["epoch"]) + 1 for row in rows],
                    [float(row[field]) for row in rows],
                    linewidth=2,
                    label=task,
                )
            axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
            if field == "target_accuracy":
                axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.25)
            axis.legend(title="Task", ncol=2)
            figure.tight_layout()
            path = args.output_dir / filename
            figure.savefig(path, bbox_inches="tight")
            plt.close(figure)
            print(f"saved: {path}")


if __name__ == "__main__":
    main()
