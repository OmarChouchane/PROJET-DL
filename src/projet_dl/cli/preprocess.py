import argparse

from projet_dl.preprocessing.pipeline import run_preprocessing


def build_parser():
    p = argparse.ArgumentParser(description="Preprocess ICBHI dataset into spectrogram tensors")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--output", default="data/preprocessed_data.npz")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--n_mels", type=int, default=64)
    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--hop_length", type=int, default=160)
    return p


def main():
    args = build_parser().parse_args()
    summary = run_preprocessing(
        data_dir=args.data_dir,
        split_file=args.split_file,
        output=args.output,
        sr=args.sr,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    print("[preprocess] done")
    print(summary)


if __name__ == "__main__":
    main()
