import argparse

from projet_dl.training.pipeline import run_training


def build_parser():
    p = argparse.ArgumentParser(description="Train baseline AST model")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--output", default="checkpoints/baseline_model.pth")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_sam", action="store_true")
    p.add_argument("--no_focal", action="store_true")
    p.add_argument("--oversample", action="store_true", help="Use WeightedRandomSampler to oversample minority classes")
    p.add_argument("--weight_factor", type=float, default=1.0, help="Multiply computed class weights by this factor for minority boosting")
    return p


def main():
    args = build_parser().parse_args()
    summary = run_training(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_sam=not args.no_sam,
        use_focal=not args.no_focal,
        seed=args.seed,
        oversample=args.oversample,
        weight_factor=args.weight_factor,
        device=args.device,
    )
    print("[train] done")
    print(summary)


if __name__ == "__main__":
    main()
