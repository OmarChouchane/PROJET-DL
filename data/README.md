# ICBHI 2017 dataset placement and validation

This folder contains instructions to prepare the ICBHI 2017 dataset for the project.

Steps to acquire the data (manual due to licensing):

1. Download the dataset from the official ICBHI Challenge page:
   https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
2. Extract the downloaded zip. You should obtain a folder named `ICBHI_final_database` containing `.wav` and `.txt` annotation files.
3. Download the official split file `ICBHI_challenge_train_test.txt` from the challenge page and place it directly under `data/`.
4. The final layout should be:

```
data/
├── ICBHI_final_database/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ... (920 files)
└── ICBHI_challenge_train_test.txt
```

## Validation helper

After placing the files, run the validation script to check the dataset organization and basic integrity:

```bash
python scripts/validate_icbhi.py --data_dir data/ICBHI_final_database --split_file data/ICBHI_challenge_train_test.txt
```

If the validation reports missing annotation files or unexpected counts, re-check the extracted archive and placement.

## Notes

- Do not commit the raw dataset files to the repository (license restrictions). Keep them under `data/` locally.
- The `preprocess.py` script in the repo will expect this layout. See its docstring for parameters.
