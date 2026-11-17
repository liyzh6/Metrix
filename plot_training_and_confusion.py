"""Plot training curves and target confusion matrices from metrix run folders."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classifier import Classifier
from extractor import FeatureExtractor


def args():
    p = argparse.ArgumentParser()
    default_root = ROOT / "results" / "runs"
    p.add_argument("--runs-root", type=Path, default=default_root)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42,
                   help="Only plot runs with this random seed.")
    return p.parse_args()


def load_history(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {
        "epoch": np.array([int(r["epoch"]) + 1 for r in rows]),
        "accuracy": np.array([float(r.get("train_accuracy", r["target_accuracy"])) for r in rows]),
        "train_loss": np.array([float(r["train_total"]) for r in rows]),
    }


def discover(root, seed=None):
    runs = []
    for d in sorted(p.parent for p in root.glob("*/config.json")):
        if not d.is_dir():
            continue
        required = [d / "config.json", d / "history.csv", d / "last_model.pt"]
        if all(p.is_file() for p in required):
            config = json.loads((d / "config.json").read_text(encoding="utf-8"))
            if seed is not None and int(config.get("seed", -1)) != seed:
                continue
            runs.append((d, config, load_history(d / "history.csv")))
    return runs


def plot_curves(runs, output):
    """Write one accuracy figure and one training-loss figure per dataset.

    Each figure contains the four task curves T0--T3.  Target-domain loss is
    intentionally omitted; ``train_total`` is the only loss shown.
    """
    paths = []
    for data_name in sorted({c["data_name"] for _, c, _ in runs}):
        selected = sorted(
            [(d, c, h) for d, c, h in runs if c["data_name"] == data_name],
            key=lambda item: item[1].get("task_name", item[0].name),
        )
        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
        for _, config, h in selected:
            task = config.get("task_name", "task")
            ax.plot(h["epoch"], h["accuracy"], lw=2.0, label=task)
        ax.set_title(f"{data_name} training accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(title="Task", ncol=2)
        fig.tight_layout()
        path = output / f"{data_name}_training_accuracy_four_tasks.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
        for _, config, h in selected:
            task = config.get("task_name", "task")
            ax.plot(h["epoch"], h["train_loss"], lw=2.0, label=task)
        ax.set_title(f"{data_name} training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(alpha=0.25)
        ax.legend(title="Task", ncol=2)
        fig.tight_layout()
        path = output / f"{data_name}_training_loss_four_tasks.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def build_dataset(config):
    transfer = config["transfer_task"]
    norm = config.get("normalization", "mean-std")
    ntrain = int(config.get("num_train_samples", 200))
    ntest = int(config.get("num_test_samples", 50))
    if config["data_name"] == "PU":
        from data.PU_bearing import build_datasets
        return build_datasets(transfer, ntrain, ntest, norm, config["pu_data_root"])[0][3]
    from data.PHM_2009_Geatbox import build_datasets
    return build_datasets(transfer, ntrain, ntest, norm, config["phm_data_root"])[0][3]


@torch.no_grad()
def predict(run_dir, config, device, batch_size):
    checkpoint = torch.load(run_dir / "last_model.pt", map_location=device, weights_only=False)
    extractor = FeatureExtractor().to(device)
    classifier = Classifier(int(config["num_classes"])).to(device)
    extractor.load_state_dict(checkpoint["extractor"])
    classifier.load_state_dict(checkpoint["classifier"])
    extractor.eval(); classifier.eval()
    loader = DataLoader(build_dataset(config), batch_size=batch_size, shuffle=False, num_workers=0)
    true, pred = [], []
    for x, y in loader:
        logits = classifier(extractor(x.to(device)))
        true.append(y.numpy())
        pred.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(true), np.concatenate(pred), int(config["num_classes"])


def plot_confusion(run_dir, config, y_true, y_pred, nclasses, output):
    cm = np.zeros((nclasses, nclasses), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    np.savetxt(output / f"{config['data_name']}_{config.get('task_name','task')}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{config['data_name']} {config.get('task_name', '')} confusion matrix (last epoch)")
    ax.set_xticks(range(nclasses)); ax.set_yticks(range(nclasses))
    threshold = cm.max() / 2 if cm.max() else 0
    # Annotate every cell, including the 14-by-14 PU matrix.
    fontsize = 7 if nclasses <= 10 else 5
    for i in range(nclasses):
        for j in range(nclasses):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=fontsize)
    fig.tight_layout()
    path = output / f"{config['data_name']}_{config.get('task_name','task')}_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    a = args()
    device = torch.device(a.device if a.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    runs = discover(a.runs_root, a.seed)
    if not runs:
        raise FileNotFoundError(f"No complete runs with last_model.pt below {a.runs_root}")
    output = a.output_dir or (a.runs_root / "figures")
    output.mkdir(parents=True, exist_ok=True)
    print("runs:", ", ".join(d.name for d, _, _ in runs))
    print("curves:", ", ".join(str(p) for p in plot_curves(runs, output)))
    for d, config, _ in runs:
        y_true, y_pred, nclasses = predict(d, config, device, a.batch_size)
        path = plot_confusion(d, config, y_true, y_pred, nclasses, output)
        print(f"{config['data_name']} {config.get('task_name')}: accuracy={(y_true == y_pred).mean():.4f}; confusion={path}")


if __name__ == "__main__":
    main()
