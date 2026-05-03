from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from projet_dl.constants import LABEL_MAP, LABEL_INV
from projet_dl.data.icbhi import load_split_file, parse_annotation_file
from projet_dl.features.audio import apply_cyclic_padding, extract_log_mel


def run_preprocessing(data_dir, split_file, output, sr, duration, n_mels, n_fft, hop_length):
    import librosa

    data_dir = Path(data_dir)
    split_file = Path(split_file)
    output = Path(output)

    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_file not found: {split_file}")

    train_files, test_files = load_split_file(split_file)
    wav_files = sorted(data_dir.glob("*.wav"))

    target_length = int(sr * duration)

    train_specs, train_labels, train_metadata = [], [], []
    test_specs, test_labels, test_metadata = [], [], []
    label_counts = defaultdict(lambda: defaultdict(int))

    for wav_path in tqdm(wav_files, desc="preprocess"):
        base = wav_path.stem
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
        cycles = parse_annotation_file(txt_path)

        if base in train_files:
            split_name = "train"
        elif base in test_files:
            split_name = "test"
        else:
            continue

        for start_s, end_s, label in cycles:
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            segment = audio[start_sample:end_sample]
            if len(segment) == 0:
                continue

            padded = apply_cyclic_padding(segment, target_length)
            spec = extract_log_mel(padded, sr, n_mels, n_fft, hop_length)
            label_id = LABEL_MAP[label]
            metadata = {
                "wav_file": wav_path.name,
                "start_s": start_s,
                "end_s": end_s,
                "label": label,
            }

            if split_name == "train":
                train_specs.append(spec)
                train_labels.append(label_id)
                train_metadata.append(metadata)
            else:
                test_specs.append(spec)
                test_labels.append(label_id)
                test_metadata.append(metadata)

            label_counts[split_name][label] += 1

    output.parent.mkdir(parents=True, exist_ok=True)

    train_specs_arr = np.array(train_specs, dtype=np.float32)
    train_labels_arr = np.array(train_labels, dtype=np.int64)
    test_specs_arr = np.array(test_specs, dtype=np.float32)
    test_labels_arr = np.array(test_labels, dtype=np.int64)

    np.savez_compressed(
        output,
        train_specs=train_specs_arr,
        train_labels=train_labels_arr,
        test_specs=test_specs_arr,
        test_labels=test_labels_arr,
        train_metadata=np.array(train_metadata, dtype=object),
        test_metadata=np.array(test_metadata, dtype=object),
        label_map=np.array([LABEL_MAP], dtype=object),
        config=np.array(
            [{
                "sr": sr,
                "duration": duration,
                "n_mels": n_mels,
                "n_fft": n_fft,
                "hop_length": hop_length,
            }],
            dtype=object,
        ),
    )

    return {
        "train_samples": int(len(train_labels_arr)),
        "test_samples": int(len(test_labels_arr)),
        "shape": tuple(train_specs_arr.shape[1:]) if len(train_specs_arr) else (),
        "label_counts": {
            "train": dict(label_counts["train"]),
            "test": dict(label_counts["test"]),
        },
        "labels": LABEL_INV,
    }
