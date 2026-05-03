import argparse

from projet_dl.evaluation.pipeline import run_evaluation


def build_parser():
    p = argparse.ArgumentParser(description="Evaluate trained model on test set")
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="results/baseline_eval")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default=None)
    return p


def main():
    args = build_parser().parse_args()
    summary = run_evaluation(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("[evaluate] done")
    print(summary)


if __name__ == "__main__":
    main()
