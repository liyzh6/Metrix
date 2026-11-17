"""Aggregate standalone CC run summaries into CSV reports."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize(items, group_fields):
    groups = defaultdict(list)
    for item in items:
        groups[tuple(item[field] for field in group_fields)].append(item)

    rows = []
    for key, group in sorted(groups.items()):
        best = [float(item["best_accuracy"]) for item in group]
        row = dict(zip(group_fields, key))
        row.update({
            "runs": len(group),
            "mean_best_accuracy": statistics.mean(best),
            "std_best_accuracy": statistics.pstdev(best) if len(best) > 1 else 0.0,
            "minimum_best_accuracy": min(best),
            "maximum_best_accuracy": max(best),
            "mean_final_accuracy": statistics.mean(float(item["final_accuracy"]) for item in group),
            "mean_last10_accuracy": statistics.mean(float(item["last10_average"]) for item in group),
            "mean_runtime_seconds": statistics.mean(float(item["runtime_seconds"]) for item in group),
        })
        if group_fields == ["data_name"]:
            classes = int(group[0]["num_classes"])
            chance = 1.0 / classes
            row["num_classes"] = classes
            row["chance_accuracy"] = chance
            row["chance_adjusted_accuracy"] = (
                row["mean_best_accuracy"] - chance
            ) / (1.0 - chance)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate CC experiment results")
    parser.add_argument("--runs_dir", default="results/runs")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir) if args.output_dir else runs_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(runs_dir.glob("*/summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        item["summary_path"] = str(path.resolve())
        summaries.append(item)
    if not summaries:
        raise SystemExit(f"No summary.json files found below {runs_dir}")

    summaries.sort(key=lambda item: (item["data_name"], item["task_name"], item["seed"]))
    write_csv(output_dir / "All-Runs.csv", summaries)
    write_csv(output_dir / "Task-Summary.csv", summarize(summaries, ["data_name", "task_name"]))
    write_csv(output_dir / "Dataset-Summary.csv", summarize(summaries, ["data_name"]))
    print(f"Aggregated {len(summaries)} runs into {output_dir.resolve()}")


if __name__ == "__main__":
    main()
