#!/usr/bin/env python3
"""Validate basic structure of the ICBHI 2017 dataset.

Checks:
- data directory exists
- counts of .wav and .txt files
- for each .wav there is a matching .txt annotation
- presence of split file

Usage:
    python scripts/validate_icbhi.py --data_dir data/ICBHI_final_database --split_file data/ICBHI_challenge_train_test.txt
"""
from pathlib import Path
import argparse
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Path to ICBHI_final_database folder")
    p.add_argument("--split_file", required=True, help="Path to ICBHI_challenge_train_test.txt")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    split_file = Path(args.split_file)

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"ERROR: data_dir not found: {data_dir}")
        sys.exit(2)
    if not split_file.exists():
        print(f"ERROR: split_file not found: {split_file}")
        sys.exit(2)

    wavs = list(data_dir.rglob("*.wav"))
    txts = list(data_dir.rglob("*.txt"))

    print(f"Found {len(wavs)} .wav files and {len(txts)} .txt annotation files under {data_dir}")

    # check 1-to-1 correspondence by stem
    wav_stems = {p.stem for p in wavs}
    txt_stems = {p.stem for p in txts}

    missing_txt = sorted(wav_stems - txt_stems)
    missing_wav = sorted(txt_stems - wav_stems)

    if missing_txt:
        print(f"WARNING: {len(missing_txt)} .wav files have no matching .txt (showing up to 10):")
        for s in missing_txt[:10]:
            print("  ", s)
    else:
        print("All .wav files have matching .txt annotations.")

    if missing_wav:
        print(f"WARNING: {len(missing_wav)} .txt files have no matching .wav (showing up to 10):")
        for s in missing_wav[:10]:
            print("  ", s)

    # quick check of split file lines
    with open(split_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"Split file has {len(lines)} non-empty lines.")

    # success code
    if missing_txt:
        print("Validation completed with warnings.")
        sys.exit(1)
    else:
        print("Validation successful.")
        sys.exit(0)


if __name__ == "__main__":
    main()
