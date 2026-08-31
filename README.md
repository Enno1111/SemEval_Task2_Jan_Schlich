# Adding Temporal Dependencies and User-Aware Context to a Joint Prediction of Valence and Arousal from Self-Annotated Text

Code and results for the Bachelor thesis of the same name.
SemEval 2026 Task 2, Subtask 1 — Data and Web Science Group, University of Mannheim.

Jan Schlich · Supervisor: Prof. Dr. Ponzetto

---

## What this repository contains

Four experiments build on one another. Each adds one input signal to the same
dual-head transformer regressor, so that the configurations differ only in the
information supplied at the input and remain directly comparable.

| Folder | Experiment | Injected signal | Thesis |
|---|---|---|---|
| `1.1_ClassificationBaseline` | 1 — output formulation | — | §4.2 |
| `1.2_RegressionBaseline` | 1 — output formulation | — | §4.2 |
| `2.1_Temporal` | 2 — temporal features | date prefix | §4.3 |
| `2.2_UserID` | 3 — user-specific features | user identifier | §4.4 |
| `3_Regression_Combined` | 4 — combined | both | §4.5 |

Experiment 1 compares a classification against a regression formulation and
determines the architecture used by all later experiments. Experiment 4 inherits
that architecture unchanged and therefore performs no search of its own, which is
why it has no `ablation.py`.

---

## Layout of an experiment folder

```
<experiment>/
├── model.py                   training, configuration constants at the top
├── predict.py                 inference on the held-out set
├── evaluation.py              scoring of a single prediction file
├── run.py                     trains over three seeds, reports scores per group
├── ablation.py                sequential hyperparameter search
├── group_error_analysis.py    residual bias and error per evaluation group
├── counterfactual_ablation.py replaces injected tokens with placeholders
├── attention_analysis.py      attention mass on injected tokens, final layer
├── head_analysis.py           the same, resolved by layer and head
├── offset_analysis.py         within- vs between-user variance of the ablation effect
├── *.sh                       matching SLURM job scripts
└── results/
    ├── predictions.csv        predictions of the last seed
    ├── *.csv, *.png           analysis output
    └── logs/                  stdout and stderr of every job
```

Not every folder has every script: the explainability probes exist only where
there is an injected signal to probe.

## Shared files

- `data/` — training and held-out set of the shared task
- `eval.py` — official metric of the task organisers, used unmodified

Model checkpoints are written to `../models/` and are not tracked, being too
large for the repository.

---

## Running an experiment

All scripts expect to be run from inside their own experiment folder, since
their output paths are relative:

```bash
cd 2.1_Temporal
python run.py                  # locally
sbatch run.sh                  # on the cluster
```

Order matters. `run.py` produces `results/predictions.csv`, which the analysis
scripts read; running one of them first will fail on a missing file.

```bash
sbatch run_ablation.sh          # 1. hyperparameter search — writes to the log only
#    transfer the winning values into model.py by hand
sbatch run.sh                   # 2. three seeds, scores per evaluation group
sbatch group_error_analysis.sh  # 3. residual analysis
sbatch counterfactual_ablation.sh
sbatch attention_analysis.sh
sbatch head_analysis.sh
```

The ablation reports validation loss on the internal split; the selection is not
automated, the winning values are set manually in `model.py`.

### Environment

Python 3.9 with PyTorch, Transformers, pandas, scipy and matplotlib. On the
cluster the jobs activate a virtual environment at `~/venv`; adjust the
`source` line in the `.sh` files if yours lives elsewhere.

---

## Configuration

Every experiment is configured through constants at the top of its `model.py` —
encoder, pooling strategy, head dimension, learning rate, batch size, dropout,
and where applicable `TEMPORAL_MODE`, `MIN_USER_TEXTS` and `USER_ID_LENGTH`.
The values committed here are those reported in the thesis.

`MIN_USER_TEXTS` is set to the same value in all five folders, including those
without a user-identifier feature. There it has no effect on the model and only
defines the evaluation groups, which keeps them comparable across experiments.

## Evaluation groups

`run.py` disaggregates every score across seven groups: `overall`, `seen` and
`unseen` users, within `seen` the split into `seen_own_id` and `seen_unknown`,
and the two text types `is_words` and `not_is_words`. The definitions live in
`run.py` and are imported by the analysis scripts, so the groups are identical
everywhere.

## Reproducibility

Training runs with a fixed seed and deterministic cuBLAS reductions. The
single-signal configurations reproduce exactly; the combined configuration does
not, because of a non-deterministic attention kernel. Group scores there shift
by up to 0.02 between runs, and the explainability probes — which operate on a
single checkpoint — shift correspondingly. This is discussed in the Limitations
chapter of the thesis.
