"""CLI entrypoint for Monte Carlo dropout regression on disjoint intervals."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .bnn_regression_data import (
    DEFAULT_OBSERVED_INTERVALS,
    apply_preset,
    build_interior_gap_guide_dataset,
    build_outside_interval_guide_dataset,
    build_regression_dataset,
    complement_intervals,
    intervals_to_string,
    parse_float_list,
    parse_hidden_dims,
    parse_intervals,
    resolve_regression_target,
    RegressionStandardizer,
    sample_inputs_from_intervals,
    sample_targets_with_interval_noise,
    set_seed,
    split_tensor_dataset,
    validate_intervals_in_domain,
)
from .bnn_regression_eval import compute_interval_coverage, plot_predictions, print_generated_coverage_results, summarize_region_uncertainty


def parse_args() -> argparse.Namespace:
    """Define the command line interface."""

    parser = argparse.ArgumentParser(
        description="Feedforward regression with Monte Carlo dropout on disjoint intervals."
    )
    parser.add_argument(
        "--preset",
        choices=["paper-figure5"],
        default=None,
        help="Optional preset that overrides the target function and interval/domain layout.",
    )
    parser.add_argument(
        "--data-from-checkpoint",
        type=Path,
        default=None,
        help="Optional Bayes-by-Backprop regression checkpoint whose saved observed/train/guide data should be reused exactly.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=[256, 256, 256],
        help="Hidden layer sizes, e.g. 128,128.",
    )
    parser.add_argument("--activation", choices=["gelu", "relu"], default="relu", help="Hidden-layer activation.")
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout probability applied after each hidden layer during training.",
    )
    parser.add_argument(
        "--test-dropout-rate",
        type=float,
        default=None,
        help="Optional dropout probability to use during Monte Carlo prediction. Defaults to --dropout-rate.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=64,
        help="Monte Carlo dropout passes used to estimate validation MSE.",
    )
    parser.add_argument("--test-samples", type=int, default=400, help="Monte Carlo dropout passes used for predictive summaries.")
    parser.add_argument("--train-points", type=int, default=192, help="Number of training points sampled from observed intervals.")
    parser.add_argument(
        "--target-function",
        choices=["oscillatory", "paper"],
        default="oscillatory",
        help="Synthetic regression target to fit.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of observed samples held out for validation and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=80,
        help="Stop when validation MSE has not improved for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-MSE improvement required to reset early stopping patience.",
    )
    parser.add_argument("--noise-std", type=float, default=0.02, help="Noise scale used to generate observations.")
    parser.add_argument(
        "--observed-interval-noise-stds",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated noise standard deviations, one per observed interval.",
    )
    parser.add_argument("--domain-min", type=float, default=-5.0, help="Minimum x-value of the evaluation domain.")
    parser.add_argument("--domain-max", type=float, default=4.5, help="Maximum x-value of the evaluation domain.")
    parser.add_argument(
        "--observed-intervals",
        type=parse_intervals,
        default=DEFAULT_OBSERVED_INTERVALS,
        help="Comma-separated disjoint intervals such as -4:-2,-0.5:0.75,1.75:3.5.",
    )
    parser.add_argument(
        "--guide-points-outside-intervals",
        type=int,
        default=0,
        help="Sparse extra observations sampled from gaps outside the main observed intervals and always kept in training.",
    )
    parser.add_argument(
        "--guide-region-mode",
        choices=["all-gaps", "outer-only"],
        default="all-gaps",
        help="Sample guide observations from all missing gaps or only from the outer extrapolation gaps.",
    )
    parser.add_argument(
        "--guide-points-interior-gaps",
        type=int,
        default=0,
        help="Additional sparse guide observations sampled only from interior gaps between observed intervals.",
    )
    parser.add_argument("--grid-size", type=int, default=500, help="Points used to evaluate the predictive curve.")
    parser.add_argument(
        "--coverage-eval-points",
        type=int,
        default=0,
        help="Optional number of fresh generated noisy points used for empirical interval coverage evaluation.",
    )
    parser.add_argument(
        "--coverage-eval-samples",
        type=int,
        default=1000,
        help="Monte Carlo dropout samples used for generated coverage evaluation.",
    )
    parser.add_argument(
        "--coverage-eval-seed",
        type=int,
        default=42,
        help="Random seed used for generated coverage evaluation.",
    )
    parser.add_argument(
        "--show-coverage-points",
        action="store_true",
        help="Overlay the generated coverage-evaluation points on the plot.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="How often to print training progress.")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("outputs") / "regression" / "plots" / "mc_dropout_regression.png",
        help="Path for the predictive uncertainty plot.",
    )
    parser.add_argument(
        "--plot-quantiles",
        choices=["function", "observation"],
        default="function",
        help="Which predictive quantiles to draw. For MC dropout both choices coincide.",
    )
    parser.add_argument(
        "--hide-summary-box",
        action="store_true",
        help="Omit the plot textbox that reports mean IQR and mean 95%% widths.",
    )
    parser.add_argument(
        "--shade-observed-intervals",
        action="store_true",
        help="Shade the input intervals that contain training observations.",
    )
    return parser.parse_args()


def rows_to_counter(inputs: torch.Tensor, targets: torch.Tensor) -> Counter[tuple[float, float]]:
    """Convert paired one-dimensional inputs and targets to a multiset of rows."""

    flat_inputs = inputs.squeeze(-1).tolist()
    flat_targets = targets.squeeze(-1).tolist()
    return Counter((float(input_value), float(target_value)) for input_value, target_value in zip(flat_inputs, flat_targets))


def subtract_paired_rows(
    source_inputs: torch.Tensor,
    source_targets: torch.Tensor,
    rows_to_remove: Counter[tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove a multiset of exact (input, target) rows from a paired dataset."""

    kept_indices: list[int] = []
    remaining_to_remove = rows_to_remove.copy()

    for index, (input_value, target_value) in enumerate(
        zip(source_inputs.squeeze(-1).tolist(), source_targets.squeeze(-1).tolist())
    ):
        key = (float(input_value), float(target_value))
        if remaining_to_remove[key] > 0:
            remaining_to_remove[key] -= 1
            continue
        kept_indices.append(index)

    if any(count > 0 for count in remaining_to_remove.values()):
        raise ValueError("Could not exactly recover the validation subset from the checkpointed data.")

    if not kept_indices:
        empty_inputs = source_inputs.new_empty((0, source_inputs.size(1)))
        empty_targets = source_targets.new_empty((0, source_targets.size(1)))
        return empty_inputs, empty_targets

    keep_tensor = torch.tensor(kept_indices, dtype=torch.long)
    return source_inputs[keep_tensor], source_targets[keep_tensor]


def interval_membership_mask(inputs: torch.Tensor, intervals: Sequence[tuple[float, float]]) -> torch.Tensor:
    """Return a mask for inputs that fall inside any observed interval."""

    flat_inputs = inputs.squeeze(-1)
    mask = torch.zeros_like(flat_inputs, dtype=torch.bool)
    for left, right in intervals:
        mask |= (flat_inputs >= left) & (flat_inputs <= right)
    return mask


def load_data_from_checkpoint(
    checkpoint_path: Path,
) -> Dict[str, torch.Tensor | str | float | list[tuple[float, float]] | list[float] | None]:
    """Load the exact saved data tensors and metadata from a regression checkpoint."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    observed_inputs = checkpoint.get("observed_inputs")
    observed_targets = checkpoint.get("observed_targets")
    train_inputs = checkpoint.get("train_inputs")
    train_targets = checkpoint.get("train_targets")
    guide_inputs = checkpoint.get("guide_inputs")
    guide_targets = checkpoint.get("guide_targets")

    if observed_inputs is None or observed_targets is None or train_inputs is None or train_targets is None:
        raise ValueError("The checkpoint does not contain the saved observed/train tensors needed for MC-dropout reuse.")

    observed_intervals = [tuple(interval) for interval in checkpoint["observed_intervals"]]
    train_observed_mask = interval_membership_mask(train_inputs, observed_intervals)
    train_observed_inputs = train_inputs[train_observed_mask]
    train_observed_targets = train_targets[train_observed_mask]
    validation_inputs, validation_targets = subtract_paired_rows(
        source_inputs=observed_inputs,
        source_targets=observed_targets,
        rows_to_remove=rows_to_counter(train_observed_inputs, train_observed_targets),
    )

    if validation_inputs.size(0) == 0:
        raise ValueError("The checkpointed data leaves no validation observations after reconstruction.")

    return {
        "target_function": str(checkpoint["target_function"]),
        "domain_min": float(checkpoint["domain_min"]),
        "domain_max": float(checkpoint["domain_max"]),
        "observed_intervals": observed_intervals,
        "preset": checkpoint.get("preset"),
        "noise_std": float(checkpoint.get("noise_std", 0.02)),
        "observed_interval_noise_stds": checkpoint.get("observed_interval_noise_stds"),
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "validation_inputs": validation_inputs,
        "validation_targets": validation_targets,
        "observed_inputs": observed_inputs,
        "observed_targets": observed_targets,
        "guide_inputs": guide_inputs,
        "guide_targets": guide_targets,
        "guide_points_outside_intervals": int(checkpoint.get("guide_points_outside_intervals", 0)),
        "guide_points_interior_gaps": int(checkpoint.get("guide_points_interior_gaps", 0)),
        "guide_region_mode": str(checkpoint.get("guide_region_mode", "all-gaps")),
    }


def build_activation(name: str) -> nn.Module:
    """Return the requested hidden-layer activation."""

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


class DropoutRegressor(nn.Module):
    """Feedforward regressor with dropout after each hidden layer."""

    def __init__(self, hidden_dims: list[int], activation: str, dropout_rate: float) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = 1
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(build_activation(activation))
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear layers with Xavier weights."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(inputs)
        return self.output(hidden)


def run_epoch(
    model: DropoutRegressor,
    loader: DataLoader,
    device: torch.device,
    scaler: RegressionStandardizer,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    """Run one train or evaluation epoch with standard dropout behavior."""

    is_training = optimizer is not None
    model.train(mode=is_training)
    totals = {"loss": 0.0, "mse": 0.0, "examples": 0.0}

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            predictions = model(inputs)
            loss = F.mse_loss(predictions, targets)
            if is_training:
                loss.backward()
                optimizer.step()

        original_predictions = scaler.inverse_targets(predictions.detach().cpu())
        original_targets = scaler.inverse_targets(targets.detach().cpu())
        mse = F.mse_loss(original_predictions, original_targets)

        totals["loss"] += loss.item() * batch_size
        totals["mse"] += mse.item() * batch_size
        totals["examples"] += batch_size

    total_examples = totals["examples"]
    return {
        "loss": totals["loss"] / total_examples,
        "mse": totals["mse"] / total_examples,
    }


def configure_dropout_for_testing(model: nn.Module, dropout_rate: float) -> list[tuple[nn.Module, float]]:
    """Activate dropout layers in eval mode and temporarily override their rate."""

    original_rates: list[tuple[nn.Module, float]] = []
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            original_rates.append((module, float(module.p)))
            module.p = dropout_rate
            module.train()
    return original_rates


@torch.no_grad()
def sample_mc_predictions(
    model: DropoutRegressor,
    inputs: torch.Tensor,
    num_samples: int,
    device: torch.device,
    dropout_rate: float,
) -> torch.Tensor:
    """Draw multiple predictions using different dropout masks at test time."""

    if num_samples <= 0:
        raise ValueError("The number of Monte Carlo dropout samples must be positive.")
    if not 0.0 < dropout_rate < 1.0:
        raise ValueError("The Monte Carlo dropout rate must be in the open interval (0, 1).")

    inputs = inputs.to(device)
    was_training = model.training
    model.eval()
    original_dropout_rates = configure_dropout_for_testing(model, dropout_rate=dropout_rate)

    samples = []
    for _ in range(num_samples):
        samples.append(model(inputs).squeeze(-1))

    for module, original_rate in original_dropout_rates:
        module.p = original_rate
    model.train(mode=was_training)
    return torch.stack(samples, dim=0)


@torch.no_grad()
def evaluate_regression(
    model: DropoutRegressor,
    loader: DataLoader,
    device: torch.device,
    scaler: RegressionStandardizer,
    num_samples: int,
    test_dropout_rate: float,
) -> Dict[str, float]:
    """Evaluate validation MSE using Monte Carlo dropout predictive means."""

    totals = {"mse": 0.0, "predictive_std": 0.0, "examples": 0.0}

    for inputs, targets in loader:
        targets = targets.to(device)
        batch_size = inputs.size(0)

        samples = sample_mc_predictions(
            model=model,
            inputs=inputs,
            num_samples=num_samples,
            device=device,
            dropout_rate=test_dropout_rate,
        )
        predictive_mean = samples.mean(dim=0).unsqueeze(-1).cpu()
        predictive_std = scaler.scale_target_std(samples.std(dim=0, unbiased=False).unsqueeze(-1).cpu())

        original_mean = scaler.inverse_targets(predictive_mean)
        original_targets = scaler.inverse_targets(targets.detach().cpu())
        mse = F.mse_loss(original_mean, original_targets)

        totals["mse"] += mse.item() * batch_size
        totals["predictive_std"] += predictive_std.mean().item() * batch_size
        totals["examples"] += batch_size

    total_examples = totals["examples"]
    return {
        "mse": totals["mse"] / total_examples,
        "predictive_std": totals["predictive_std"] / total_examples,
    }


@torch.no_grad()
def predict_distribution(
    model: DropoutRegressor,
    inputs: torch.Tensor,
    num_samples: int,
    device: torch.device,
    scaler: RegressionStandardizer,
    test_dropout_rate: float,
) -> Dict[str, torch.Tensor]:
    """Estimate predictive quantiles with Monte Carlo dropout."""

    samples = sample_mc_predictions(
        model=model,
        inputs=inputs,
        num_samples=num_samples,
        device=device,
        dropout_rate=test_dropout_rate,
    )
    function_samples_original = scaler.inverse_targets(samples.unsqueeze(-1).cpu()).squeeze(-1)
    mean = scaler.inverse_targets(samples.mean(dim=0).unsqueeze(-1).cpu()).squeeze(-1)
    epistemic_std = scaler.scale_target_std(samples.std(dim=0, unbiased=False).unsqueeze(-1).cpu()).squeeze(-1)
    zeros = torch.zeros_like(epistemic_std)

    function_median = torch.quantile(function_samples_original, q=0.5, dim=0)
    function_q25 = torch.quantile(function_samples_original, q=0.25, dim=0)
    function_q75 = torch.quantile(function_samples_original, q=0.75, dim=0)
    function_q025 = torch.quantile(function_samples_original, q=0.025, dim=0)
    function_q975 = torch.quantile(function_samples_original, q=0.975, dim=0)

    return {
        "means": samples,
        "function_samples": function_samples_original,
        "predictive_samples": function_samples_original,
        "mean": mean,
        "function_median": function_median,
        "function_q25": function_q25,
        "function_q75": function_q75,
        "function_q025": function_q025,
        "function_q975": function_q975,
        "observation_median": function_median,
        "observation_q25": function_q25,
        "observation_q75": function_q75,
        "observation_q025": function_q025,
        "observation_q975": function_q975,
        "median": function_median,
        "q25": function_q25,
        "q75": function_q75,
        "epistemic_std": epistemic_std,
        "aleatoric_std": zeros,
        "predictive_std": epistemic_std,
    }


@torch.no_grad()
def evaluate_generated_coverage(
    model: DropoutRegressor,
    scaler: RegressionStandardizer,
    target_fn,
    observed_intervals: Sequence[tuple[float, float]],
    domain_min: float,
    domain_max: float,
    noise_std: float,
    observed_interval_noise_stds: Sequence[float] | None,
    num_points: int,
    predictive_samples: int,
    seed: int,
    device: torch.device,
    test_dropout_rate: float,
) -> tuple[Dict[str, Dict[str, float]], Dict[str, torch.Tensor]]:
    """Estimate empirical interval coverage on fresh generated noisy points."""

    if num_points <= 0:
        raise ValueError("The number of generated coverage points must be positive.")
    if predictive_samples <= 0:
        raise ValueError("The number of predictive samples for coverage must be positive.")

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        full_inputs = domain_min + (domain_max - domain_min) * torch.rand((num_points, 1), generator=generator)
        full_targets = sample_targets_with_interval_noise(
            inputs=full_inputs,
            target_fn=target_fn,
            default_noise_std=noise_std,
            intervals=observed_intervals,
            interval_noise_stds=observed_interval_noise_stds,
        )
        full_summary = predict_distribution(
            model=model,
            inputs=scaler.transform_inputs(full_inputs),
            num_samples=predictive_samples,
            device=device,
            scaler=scaler,
            test_dropout_rate=test_dropout_rate,
        )

        interval_inputs = sample_inputs_from_intervals(observed_intervals, num_points=num_points, generator=generator)
        interval_targets = sample_targets_with_interval_noise(
            inputs=interval_inputs,
            target_fn=target_fn,
            default_noise_std=noise_std,
            intervals=observed_intervals,
            interval_noise_stds=observed_interval_noise_stds,
        )
        interval_summary = predict_distribution(
            model=model,
            inputs=scaler.transform_inputs(interval_inputs),
            num_samples=predictive_samples,
            device=device,
            scaler=scaler,
            test_dropout_rate=test_dropout_rate,
        )

    full_inside_95 = (
        (full_targets.squeeze(-1) >= full_summary["observation_q025"])
        & (full_targets.squeeze(-1) <= full_summary["observation_q975"])
    )

    return (
        {
            "full_domain_uniform": compute_interval_coverage(full_summary, full_targets),
            "observed_intervals_only": compute_interval_coverage(interval_summary, interval_targets),
        },
        {
            "inputs": full_inputs.cpu(),
            "targets": full_targets.cpu(),
            "inside_95": full_inside_95.cpu(),
        },
    )


def main() -> None:
    """Train an MC-dropout regressor and plot predictive confidence bands."""

    args = parse_args()
    checkpoint_data = None
    if args.data_from_checkpoint is None:
        args = apply_preset(args)
    else:
        checkpoint_data = load_data_from_checkpoint(args.data_from_checkpoint)

    if args.epochs <= 0:
        raise ValueError("The number of epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("The batch size must be positive.")
    if args.validation_samples <= 0 or args.test_samples <= 0:
        raise ValueError("The number of Monte Carlo dropout samples must be positive.")
    if args.train_points <= 0:
        raise ValueError("The number of training points must be positive.")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("The validation fraction must be in the open interval (0, 1).")
    if args.early_stopping_patience <= 0:
        raise ValueError("The early stopping patience must be positive.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("The early stopping minimum delta must be non-negative.")
    if not 0.0 < args.dropout_rate < 1.0:
        raise ValueError("The dropout rate must be in the open interval (0, 1).")
    if args.test_dropout_rate is not None and not 0.0 < args.test_dropout_rate < 1.0:
        raise ValueError("The test dropout rate must be in the open interval (0, 1).")
    if args.noise_std <= 0.0:
        raise ValueError("The default observation noise standard deviation must be positive.")
    if args.grid_size <= 0:
        raise ValueError("The grid size must be positive.")
    if args.coverage_eval_points < 0:
        raise ValueError("The number of generated coverage-evaluation points must be non-negative.")
    if args.coverage_eval_samples <= 0:
        raise ValueError("The number of predictive samples for coverage evaluation must be positive.")
    if args.guide_points_outside_intervals < 0:
        raise ValueError("The number of guide points outside intervals must be non-negative.")
    if args.guide_points_interior_gaps < 0:
        raise ValueError("The number of guide points in interior gaps must be non-negative.")
    validation_intervals = args.observed_intervals
    if checkpoint_data is not None:
        validation_intervals = checkpoint_data["observed_intervals"]
    if args.observed_interval_noise_stds is not None:
        if len(args.observed_interval_noise_stds) != len(validation_intervals):
            raise ValueError("Provide exactly one observed-interval noise standard deviation per observed interval.")
        if any(noise_std <= 0.0 for noise_std in args.observed_interval_noise_stds):
            raise ValueError("All observed-interval noise standard deviations must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dropout_rate = args.dropout_rate if args.test_dropout_rate is None else args.test_dropout_rate

    if checkpoint_data is None:
        validate_intervals_in_domain(args.observed_intervals, args.domain_min, args.domain_max)
        target_fn, reference_mean_fn = resolve_regression_target(args.target_function)

        raw_dataset = build_regression_dataset(
            intervals=args.observed_intervals,
            num_points=args.train_points,
            noise_std=args.noise_std,
            target_fn=target_fn,
            interval_noise_stds=args.observed_interval_noise_stds,
        )
        observed_inputs, observed_targets = raw_dataset.tensors
        raw_train_dataset, raw_validation_dataset = split_tensor_dataset(
            raw_dataset,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
        train_inputs, train_targets = raw_train_dataset.tensors
        validation_inputs, validation_targets = raw_validation_dataset.tensors

        guide_inputs = None
        guide_targets = None
        guide_dataset = build_outside_interval_guide_dataset(
            observed_intervals=args.observed_intervals,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            num_points=args.guide_points_outside_intervals,
            region_mode=args.guide_region_mode,
            noise_std=args.noise_std,
            target_fn=target_fn,
        )
        interior_guide_dataset = build_interior_gap_guide_dataset(
            observed_intervals=args.observed_intervals,
            num_points=args.guide_points_interior_gaps,
            noise_std=args.noise_std,
            target_fn=target_fn,
        )
        if guide_dataset is not None:
            guide_inputs, guide_targets = guide_dataset.tensors
            train_inputs = torch.cat([train_inputs, guide_inputs], dim=0)
            train_targets = torch.cat([train_targets, guide_targets], dim=0)
        if interior_guide_dataset is not None:
            interior_guide_inputs, interior_guide_targets = interior_guide_dataset.tensors
            if guide_inputs is None or guide_targets is None:
                guide_inputs = interior_guide_inputs
                guide_targets = interior_guide_targets
            else:
                guide_inputs = torch.cat([guide_inputs, interior_guide_inputs], dim=0)
                guide_targets = torch.cat([guide_targets, interior_guide_targets], dim=0)
            train_inputs = torch.cat([train_inputs, interior_guide_inputs], dim=0)
            train_targets = torch.cat([train_targets, interior_guide_targets], dim=0)
        preset = args.preset
        target_function = args.target_function
        domain_min = args.domain_min
        domain_max = args.domain_max
        observed_intervals = args.observed_intervals
        noise_std = args.noise_std
        observed_interval_noise_stds = args.observed_interval_noise_stds
        guide_points_outside_intervals = args.guide_points_outside_intervals
        guide_points_interior_gaps = args.guide_points_interior_gaps
        guide_region_mode = args.guide_region_mode
    else:
        target_function = str(checkpoint_data["target_function"])
        target_fn, reference_mean_fn = resolve_regression_target(target_function)
        domain_min = float(checkpoint_data["domain_min"])
        domain_max = float(checkpoint_data["domain_max"])
        observed_intervals = checkpoint_data["observed_intervals"]
        preset = checkpoint_data["preset"]
        noise_std = float(checkpoint_data["noise_std"])
        observed_interval_noise_stds = (
            args.observed_interval_noise_stds
            if args.observed_interval_noise_stds is not None
            else checkpoint_data["observed_interval_noise_stds"]
        )
        guide_points_outside_intervals = int(checkpoint_data["guide_points_outside_intervals"])
        guide_points_interior_gaps = int(checkpoint_data["guide_points_interior_gaps"])
        guide_region_mode = str(checkpoint_data["guide_region_mode"])
        observed_inputs = checkpoint_data["observed_inputs"]
        observed_targets = checkpoint_data["observed_targets"]
        guide_inputs = checkpoint_data["guide_inputs"]
        guide_targets = checkpoint_data["guide_targets"]
        train_inputs = checkpoint_data["train_inputs"]
        train_targets = checkpoint_data["train_targets"]
        validation_inputs = checkpoint_data["validation_inputs"]
        validation_targets = checkpoint_data["validation_targets"]

    scaler = RegressionStandardizer.fit(train_inputs, train_targets)
    train_dataset = TensorDataset(scaler.transform_inputs(train_inputs), scaler.transform_targets(train_targets))
    validation_dataset = TensorDataset(
        scaler.transform_inputs(validation_inputs),
        scaler.transform_targets(validation_targets),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = DropoutRegressor(
        hidden_dims=args.hidden_dims,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    if args.data_from_checkpoint is not None:
        print(f"Loaded data from checkpoint: {args.data_from_checkpoint}")
    print(f"Observed intervals: {intervals_to_string(observed_intervals)}")
    print(f"Missing intervals: {intervals_to_string(complement_intervals(observed_intervals, domain_min, domain_max))}")
    if preset is not None:
        print(f"Preset: {preset}")
    print(f"Target function: {target_function}")
    print(f"Architecture: 1 -> {args.hidden_dims} -> 1")
    print(f"Training dropout rate: {args.dropout_rate:.3f}")
    print(f"Testing dropout rate: {test_dropout_rate:.3f}")
    if guide_points_outside_intervals > 0 or guide_points_interior_gaps > 0:
        print(
            "Guide observations outside intervals: "
            f"{guide_points_outside_intervals} ({guide_region_mode}), "
            f"interior-gap guide points: {guide_points_interior_gaps}"
        )
    if observed_interval_noise_stds is not None:
        noise_values = ", ".join(f"{interval_noise_std:.4f}" for interval_noise_std in observed_interval_noise_stds)
        print(
            "Observed-interval noise stds: "
            f"{noise_values} (default outside intervals: {noise_std:.4f})"
        )
    print(f"Train / validation points: {len(train_dataset)} / {len(validation_dataset)}")
    print("Training loss: normalized MSE")
    print("Early stopping metric: validation MC-dropout MSE")

    best_validation_mse = float("inf")
    best_epoch = 0
    best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            scaler=scaler,
            optimizer=optimizer,
        )
        validation_metrics = evaluate_regression(
            model=model,
            loader=validation_loader,
            device=device,
            scaler=scaler,
            num_samples=args.validation_samples,
            test_dropout_rate=test_dropout_rate,
        )

        improvement_threshold = best_validation_mse - args.early_stopping_min_delta
        if validation_metrics["mse"] < improvement_threshold:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        should_log = epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0
        if should_log:
            print(
                f"Epoch {epoch:04d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train mse {train_metrics['mse']:.4f} | "
                f"val mc-mse {validation_metrics['mse']:.4f} | "
                f"val pred std {validation_metrics['predictive_std']:.4f}"
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:04d} | "
                f"best epoch {best_epoch:04d} | "
                f"best val mc-mse {best_validation_mse:.4f}"
            )
            break

    if best_epoch == 0:
        best_epoch = args.epochs
        best_validation_mse = validation_metrics["mse"]
        best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    final_validation_metrics = evaluate_regression(
        model=model,
        loader=validation_loader,
        device=device,
        scaler=scaler,
        num_samples=args.validation_samples,
        test_dropout_rate=test_dropout_rate,
    )
    print(
        f"Restored best model from epoch {best_epoch:04d} | "
        f"val mc-mse {final_validation_metrics['mse']:.4f}"
    )

    grid_inputs = torch.linspace(domain_min, domain_max, steps=args.grid_size, dtype=torch.float32).unsqueeze(1)
    scaled_grid_inputs = scaler.transform_inputs(grid_inputs)
    reference_curve = reference_mean_fn(grid_inputs)
    prediction_summary = predict_distribution(
        model=model,
        inputs=scaled_grid_inputs,
        num_samples=args.test_samples,
        device=device,
        scaler=scaler,
        test_dropout_rate=test_dropout_rate,
    )

    print("Uncertainty summary:")
    for line in summarize_region_uncertainty(
        grid_inputs=grid_inputs,
        epistemic_std=prediction_summary["epistemic_std"],
        predictive_std=prediction_summary["predictive_std"],
        observed_intervals=observed_intervals,
        domain_min=domain_min,
        domain_max=domain_max,
    ):
        print(f"  {line}")

    coverage_results = None
    coverage_plot_data = None
    if args.coverage_eval_points > 0:
        coverage_results, coverage_plot_data = evaluate_generated_coverage(
            model=model,
            scaler=scaler,
            target_fn=target_fn,
            observed_intervals=observed_intervals,
            domain_min=domain_min,
            domain_max=domain_max,
            noise_std=noise_std,
            observed_interval_noise_stds=observed_interval_noise_stds,
            num_points=args.coverage_eval_points,
            predictive_samples=args.coverage_eval_samples,
            seed=args.coverage_eval_seed,
            device=device,
            test_dropout_rate=test_dropout_rate,
        )
        print_generated_coverage_results(coverage_results, num_points=args.coverage_eval_points)

    side_panel_lines = [
        f"Train dropout: {args.dropout_rate:.0%}",
        f"Test dropout: {test_dropout_rate:.0%}",
    ]
    if coverage_results is not None:
        side_panel_lines.extend(
            [
                "",
                f"Generated 95% cov, global: {coverage_results['full_domain_uniform']['observation_95']:.1%}",
                f"Generated 95% cov, observed: {coverage_results['observed_intervals_only']['observation_95']:.1%}",
            ]
        )

    if args.plot_path is not None:
        plot_predictions(
            plot_path=args.plot_path,
            observed_inputs=observed_inputs,
            observed_targets=observed_targets,
            guide_inputs=guide_inputs,
            guide_targets=guide_targets,
            coverage_inputs=(
                None
                if (coverage_plot_data is None or not args.show_coverage_points)
                else coverage_plot_data["inputs"]
            ),
            coverage_targets=(
                None
                if (coverage_plot_data is None or not args.show_coverage_points)
                else coverage_plot_data["targets"]
            ),
            coverage_inside_95=(
                None
                if (coverage_plot_data is None or not args.show_coverage_points)
                else coverage_plot_data["inside_95"]
            ),
            grid_inputs=grid_inputs,
            reference_curve=reference_curve,
            summary=prediction_summary,
            observed_intervals=observed_intervals,
            quantile_space=args.plot_quantiles,
            shade_observed_intervals=args.shade_observed_intervals,
            show_summary_box=not args.hide_summary_box,
            title="MC Dropout Regression with Disjoint Observation Intervals",
            side_panel_lines=side_panel_lines,
        )
        print(f"Saved predictive plot to {args.plot_path}")


if __name__ == "__main__":
    main()
