# Metrix

Official PyTorch implementation of **Metrix**, a multi-source domain
generalization method for rotating-machinery fault diagnosis. Metrix combines:

1. three-source Dirichlet Mixup in the input space; and
2. a compactness-separation metric objective in the feature space.

The default objective is

```text
L = L_CE + 0.2 * L_metric + 0.5 * L_mixup
```

The class-conditional domain-center term is implemented for controlled
experiments but disabled in the reported default configuration.

## Repository structure

```text
.
├── main.py                    # command-line training entry point
├── train.py                   # training and target-domain evaluation
├── extractor.py               # 1-D CNN feature extractor
├── classifier.py              # fault classifier
├── loss.py                    # domain and compactness-separation losses
├── mixup.py                   # three-source Dirichlet Mixup
├── data/
│   ├── PU_bearing.py          # Paderborn University loader
│   └── PHM_2009_Geatbox.py    # PHM 2009 gearbox loader
├── results/aggregate.py       # aggregate repeated runs
├── scripts/run_experiments.py # reproduce all tasks and seeds
└── tests/test_core.py         # data-independent core checks
```

The spelling `Geatbox` is retained in the filename for compatibility with the
original experiment code.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/liyzh6/Metrix.git
cd Metrix
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, follow the installation command provided by
the official PyTorch website instead of installing the generic wheel first.

## Datasets

The datasets are not redistributed by this repository. Obtain them from their
respective owners and pass their locations with command-line options.

### Paderborn University bearing dataset

Expected class folders are `KA04`, `KA15`, ..., `KB24`. Each class folder must
contain the original MATLAB files, for example:

```text
<PU_ROOT>/KA04/KA04/N09_M07_F10_KA04_1.mat
```

If the duplicate class directory is absent, `<PU_ROOT>/KA04/...` is also
accepted.

### PHM 2009 gearbox dataset

Expected layout:

```text
<PHM_ROOT>/spur 1/spur 1_30hz_High_1/spur 1_30hz_High_1.txt
```

Only the first signal column is used. Signals are split into non-overlapping
windows of 1,024 points and normalized per window.

## Single-run examples

PU Task 0:

```bash
python main.py \
  --data_name PU \
  --transfer_task "[[1,2,3],0]" \
  --task_name T0 \
  --pu_data_root /path/to/PU \
  --seed 42
```

PHM2009 Task 0:

```bash
python main.py \
  --data_name PHM2009 \
  --transfer_task "[[4,8,12],0]" \
  --task_name T0 \
  --phm_data_root /path/to/PHM_Society_2009_Competition_Expanded_txt \
  --seed 42
```

The default reported configuration is equivalent to:

```bash
python main.py --lambda_domain 0 --lambda_metric 0.2 --lambda_mixup 0.5
```

Each run writes `config.json`, `history.csv`, `summary.json`, checkpoints, and a
training log below `results/runs/`.

## Reproducing all tasks

Run both datasets, four tasks, and seeds 41--50 sequentially:

```bash
python scripts/run_experiments.py \
  --datasets PHM2009 PU \
  --seeds 41-50 \
  --pu_data_root /path/to/PU \
  --phm_data_root /path/to/PHM_Society_2009_Competition_Expanded_txt
```

Use `--dry_run` to print commands without starting training. Completed
dataset/task/seed combinations are detected from `summary.json` files and
skipped when the script is resumed.

Aggregate results:

```bash
python results/aggregate.py --runs_dir results/runs --output_dir results/summary
```

Plot the seed-42 histories:

```bash
python plot_history_curves.py --runs_root results/runs --seed 42
```

## Ablations

Only change the loss coefficients:

```bash
# classification only
python main.py --lambda_domain 0 --lambda_metric 0 --lambda_mixup 0

# metric only
python main.py --lambda_domain 0 --lambda_metric 0.2 --lambda_mixup 0

# Dirichlet Mixup only
python main.py --lambda_domain 0 --lambda_metric 0 --lambda_mixup 0.5

# full Metrix
python main.py --lambda_domain 0 --lambda_metric 0.2 --lambda_mixup 0.5
```

## Evaluation protocol

The implementation reports target-domain accuracy every epoch. For consistency
with the original experiments, `best_model.pt` is selected using target-domain
accuracy. This means the target domain participates in checkpoint selection;
comparisons should use the same protocol, and papers should disclose it.

## Tests

The core test does not require either dataset:

```bash
python -m unittest tests.test_core
```

## License

This project is released under the [MIT License](LICENSE).
