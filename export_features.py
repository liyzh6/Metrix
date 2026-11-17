"""Export PHM2009 or PU T1 representations from a saved final checkpoint.

Examples (run from the repository root)::

    python export_features.py --data_name PHM2009 --task_name T0 \
        --run_dir results/runs/PHM2009_T0_S42_...
    python export_features.py --data_name PU --task_name T0 \
        --run_dir results/runs/PU_T0_S42_...

The resulting MAT file contains ``features``, ``labels`` and ``domains``.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import DataLoader

# Allow this file to be run either as ``python export_features.py`` from metrix
# or as ``python metrix/export_features.py`` from the repository root.
METRIX_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = METRIX_ROOT.parent
if str(METRIX_ROOT) not in sys.path:
    sys.path.insert(0, str(METRIX_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor import FeatureExtractor as MetrixFeatureExtractor


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_name", choices=("PHM2009", "PU"), required=True)
    parser.add_argument("--task_name", default="T0")
    parser.add_argument("--run_dir", type=Path, default=None,
                        help="directory containing last_model.pt and config.json")
    parser.add_argument("--runs_root", type=Path, default=METRIX_ROOT / "results" / "runs")
    parser.add_argument("--output_dir", type=Path, default=METRIX_ROOT / "results" / "features")
    parser.add_argument("--pu_data_root", default="datasets/PU")
    parser.add_argument(
        "--phm_data_root",
        default="datasets/PHM2009/PHM2009 Gearbox/PHM_Society_2009_Competition_Expanded_txt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def _find_run_dir(args):
    if args.run_dir is not None:
        return args.run_dir
    candidates = []
    for directory in args.runs_root.glob(f"{args.data_name}_{args.task_name}_S*"):
        if (directory / "last_model.pt").is_file() and (directory / "config.json").is_file():
            candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            f"No run with last_model.pt and config.json found below {args.runs_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_config(run_dir):
    path = run_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _task_name_from_config(config):
    if config.get("task_name"):
        return str(config["task_name"])
    transfer_task = config.get("transfer_task", [[], None])
    target = transfer_task[1]
    if config.get("data_name") == "PU":
        return f"T{target}"
    return {0: "T0", 4: "T1", 8: "T2", 12: "T3"}.get(target, f"target{target}")


def _normalise_state_dict(state_dict):
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def _build_extractor(checkpoint, device):
    state_dict = _normalise_state_dict(checkpoint["extractor"])
    # metrix/train.py uses ``network.*`` keys; utils/train.py uses the
    # original project model with ``conv1.*`` keys.
    if any(key.startswith("network.") for key in state_dict):
        extractor = MetrixFeatureExtractor()
    else:
        from models.extractor import Feature_extractor
        extractor = Feature_extractor()
    extractor.load_state_dict(state_dict)
    extractor.to(device)
    extractor.eval()
    return extractor


def _build_datasets(config, args):
    transfer_task = config.get("transfer_task")
    if not isinstance(transfer_task, (list, tuple)) or len(transfer_task) != 2:
        raise ValueError("config.json must contain transfer_task=[[source...], target]")
    sources, target = transfer_task
    if len(sources) != 3:
        raise ValueError("T1 export requires exactly three source domains")
    normalisation = config.get("normalization", config.get("normalizetype", "mean-std"))
    train_samples = int(config.get("num_train_samples", 200))
    test_samples = int(config.get("num_test_samples", 50))
    if config["data_name"] == "PU":
        from data.PU_bearing import build_datasets
        return build_datasets(
            [list(map(int, sources)), int(target)], train_samples, test_samples,
            normalisation, args.pu_data_root,
        ), [int(value) for value in sources] + [int(target)]
    from data.PHM_2009_Geatbox import build_datasets
    return build_datasets(
        [list(map(int, sources)), int(target)], train_samples, test_samples,
        normalisation, args.phm_data_root,
    ), [int(value) for value in sources] + [int(target)]


def export_features(args):
    run_dir = _find_run_dir(args)
    config = _load_config(run_dir)
    if config.get("data_name") != args.data_name:
        raise ValueError(f"Run data_name={config.get('data_name')!r} does not match {args.data_name!r}")
    config_task_name = _task_name_from_config(config)
    if config_task_name != args.task_name:
        raise ValueError(f"Run task_name={config_task_name!r} does not match {args.task_name!r}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint_path = run_dir / "last_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Final checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    extractor = _build_extractor(checkpoint, device)
    (datasets, _num_classes), domain_ids = _build_datasets(config, args)

    feature_blocks, label_blocks, domain_blocks = [], [], []
    with torch.no_grad():
        for dataset, domain_id in zip(datasets, domain_ids):
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            domain_features, domain_labels = [], []
            for inputs, labels in loader:
                output = extractor(inputs.to(device, non_blocking=True))
                domain_features.append(output.cpu().numpy())
                domain_labels.append(labels.numpy())
            features = np.concatenate(domain_features, axis=0)
            labels = np.concatenate(domain_labels, axis=0).astype(np.int64, copy=False)
            feature_blocks.append(features)
            label_blocks.append(labels)
            domain_blocks.append(np.full(labels.shape, domain_id, dtype=np.int64))

    features = np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
    labels = np.concatenate(label_blocks, axis=0)
    domains = np.concatenate(domain_blocks, axis=0)
    if not np.isfinite(features).all():
        raise ValueError("Extracted features contain NaN or Inf")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.data_name}_{args.task_name}_features.mat"
    scipy.io.savemat(output_path, {
        "features": features,
        "labels": labels.reshape(-1, 1),
        "domains": domains.reshape(-1, 1),
    })
    print(json.dumps({
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "output": str(output_path.resolve()),
        "samples": int(features.shape[0]),
        "feature_dimension": int(features.shape[1]),
        "domains": sorted(set(domains.tolist())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    export_features(_parse_args())
