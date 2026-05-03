# PROJET-DL

Respiratory sound classification on ICBHI 2017 with a reproducible deep-learning workflow.

## Goals

- Reproduce a strong baseline for `normal / crackle / wheeze / both`
- Optimize sensitivity (recall) while keeping specificity stable
- Compare results against the AST + SAM reference paper

## Professional Project Layout

```text
PROJET-DL/
├── configs/
│   ├── preprocess.yaml
│   └── train_baseline.yaml
├── data/
│   ├── README.md
│   ├── ICBHI_challenge_train_test.txt
│   └── ICBHI_final_database/          # local only (not committed)
├── checkpoints/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── scripts/
│   └── validate_icbhi.py
├── src/
│   └── projet_dl/
│       ├── constants.py
│       ├── cli/
│       │   ├── preprocess.py
│       │   ├── train.py
│       │   └── evaluate.py
│       ├── data/
│       │   └── icbhi.py
│       ├── features/
│       │   └── audio.py
│       ├── preprocessing/
│       │   └── pipeline.py
│       ├── models/
│       │   └── simple_ast.py
│       ├── training/
│       │   ├── losses.py
│       │   ├── sam.py
│       │   └── pipeline.py
│       ├── evaluation/
│       │   └── pipeline.py
│       └── utils/
│           └── metrics.py
├── preprocess.py                        # compatibility wrapper
├── train.py                             # compatibility wrapper
├── evaluate.py                          # compatibility wrapper
├── setup.bat
├── activate.bat
├── preprocess.bat
├── train.bat
├── evaluate.bat
└── requirements.txt
```

## Why This Structure Is Better

- Clear separation of concerns (data, features, model, training, evaluation)
- Reusable package code under `src/projet_dl`
- CLI entrypoints for clean scripts and automation
- Compatibility wrappers keep your old commands working
- Ready for experiments, ablations, and report reproducibility

## Windows PowerShell Workflow

### 1) Environment setup

```powershell
cd C:\Users\omarc\Desktop\DEVOPS\PROJET-DL
.\setup.bat
```

### 2) Validate dataset

```powershell
python scripts/validate_icbhi.py --data_dir data\ICBHI_final_database --split_file data\ICBHI_challenge_train_test.txt
```

### 3) Preprocess

```powershell
.\preprocess.bat
```

### 4) Train baseline

```powershell
.\train.bat
```

### 5) Evaluate

```powershell
.\evaluate.bat
```

## Direct CLI Commands (Optional)

```powershell
python -m projet_dl.cli.preprocess --data_dir data\ICBHI_final_database --split_file data\ICBHI_challenge_train_test.txt --output data\preprocessed_data.npz
python -m projet_dl.cli.train --data data\preprocessed_data.npz --epochs 20 --batch_size 8 --lr 1e-5 --output checkpoints\baseline_model.pth
python -m projet_dl.cli.evaluate --model checkpoints\baseline_model.pth --data data\preprocessed_data.npz --output results\baseline_eval
```

If `python -m projet_dl...` is not found, keep using the wrapper scripts (`preprocess.py`, `train.py`, `evaluate.py`) from the repository root.

## References

- Paper: https://arxiv.org/abs/2512.22564
- Reference code: https://github.com/Atakanisik/ICBHI-AST-SAM
- Dataset: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
