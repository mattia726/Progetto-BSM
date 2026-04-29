"""Train and evaluate the Bayesian MNIST classifier."""

import argparse
from pathlib import Path

import torch

from bnn.mnist import (
    BayesianLinear,
    BayesianMLP,
    MNIST_MEAN,
    MNIST_STD,
    build_dataloaders,
    predict_probabilities,
    run_epoch,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bayesian neural network on MNIST with PyTorch.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where MNIST will be stored.")
    parser.add_argument("--download", action="store_true", help="Download MNIST into --data-dir if it is absent.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, default=512, help="Evaluation batch size.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--hidden-dim", type=int, default=400, help="Hidden units in each Bayesian layer.")
    parser.add_argument("--prior-sigma", type=float, default=1.0, help="Standard deviation of the Gaussian prior.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--train-samples", type=int, default=1, help="Weight samples per training batch.")
    parser.add_argument("--test-samples", type=int, default=10, help="Weight samples for Bayesian prediction.")
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers. Keep 0 on Windows if needed.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional path to save model weights.")
    return parser.parse_args()


def main() -> None:
    """Entry point for training and evaluating the Bayesian MNIST model."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )
    model = BayesianMLP(hidden_dim=args.hidden_dim, prior_sigma=args.prior_sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            dataset_size=train_size,
            optimizer=optimizer,
            num_samples=args.train_samples,
        )
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            dataset_size=test_size,
            optimizer=None,
            num_samples=args.test_samples,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} | train acc {train_metrics['accuracy']:.4%} | "
            f"test loss {test_metrics['loss']:.4f} | test acc {test_metrics['accuracy']:.4%}"
        )
    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model=model,
            save_path=args.save_path,
            hidden_dim=args.hidden_dim,
            prior_sigma=args.prior_sigma,
        )
        print(f"Saved weights to {args.save_path}")


if __name__ == "__main__":
    main()
