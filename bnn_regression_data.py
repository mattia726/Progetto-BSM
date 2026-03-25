"""Data generation and normalization utilities for Bayesian 1D regression."""

from __future__ import annotations

import argparse
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


Interval = Tuple[float, float]
DEFAULT_OBSERVED_INTERVALS = [(-4.0, -2.0), (-0.5, 0.75), (1.75, 3.5)]
PAPER_FIGURE5_INTERVALS = [(0.0, 0.5)]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_hidden_dims(value: str) -> List[int]:
    """Parse a comma-separated hidden layer specification."""

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Hidden dimensions must not be empty.")

    dims: List[int] = []
    for part in parts:
        try:
            dim = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid hidden dimension '{part}'.") from exc

        if dim <= 0:
            raise argparse.ArgumentTypeError("Hidden dimensions must be positive integers.")
        dims.append(dim)

    return dims


def parse_float_list(value: str) -> List[float]:
    """Parse a comma-separated list of floats."""

    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("The list must not be empty.")

    values: List[float] = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid numeric value '{part}'.") from exc

    return values


def parse_intervals(value: str) -> List[Interval]:
    """Parse intervals written as 'left:right,left:right'."""

    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if len(chunks) < 1:
        raise argparse.ArgumentTypeError("Provide at least one interval.")

    intervals: List[Interval] = []
    for chunk in chunks:
        pieces = [piece.strip() for piece in chunk.split(":")]
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid interval '{chunk}'. Expected format 'left:right'."
            )

        try:
            left = float(pieces[0])
            right = float(pieces[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid interval '{chunk}'. Bounds must be numeric."
            ) from exc

        if left >= right:
            raise argparse.ArgumentTypeError(
                f"Invalid interval '{chunk}'. The left bound must be smaller than the right bound."
            )
        intervals.append((left, right))

    intervals.sort(key=lambda item: item[0])
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise argparse.ArgumentTypeError("Observed intervals must be strictly disjoint.")

    return intervals


def intervals_to_string(intervals: Sequence[Interval]) -> str:
    """Format intervals for logs and checkpoint metadata."""

    return ",".join(f"{left:.3f}:{right:.3f}" for left, right in intervals)


def validate_intervals_in_domain(intervals: Sequence[Interval], domain_min: float, domain_max: float) -> None:
    """Ensure observed intervals lie inside the prediction domain."""

    if domain_min >= domain_max:
        raise ValueError("The domain minimum must be smaller than the domain maximum.")

    for left, right in intervals:
        if left < domain_min or right > domain_max:
            raise ValueError(
                f"Observed interval [{left}, {right}] falls outside the domain "
                f"[{domain_min}, {domain_max}]."
            )


def complement_intervals(intervals: Sequence[Interval], domain_min: float, domain_max: float) -> List[Interval]:
    """Return the gaps inside the domain that contain no training observations."""

    gaps: List[Interval] = []
    cursor = domain_min

    for left, right in intervals:
        if left > cursor:
            gaps.append((cursor, left))
        cursor = right

    if cursor < domain_max:
        gaps.append((cursor, domain_max))

    return gaps


def oscillatory_regression_function(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Oscillatory regression target with several frequencies and no dominant linear trend."""

    epsilon = noise_std * torch.randn_like(x)
    shifted_x = x + epsilon
    return (
        1.10 * torch.sin(1.35 * shifted_x)
        + 0.45 * torch.sin(3.80 * shifted_x + 0.35)
        + 0.20 * torch.cos(6.20 * shifted_x - 0.60)
        + epsilon
    )


def oscillatory_regression_mean(x: torch.Tensor) -> torch.Tensor:
    """Noise-free reference curve for the oscillatory default target."""

    return (
        1.10 * torch.sin(1.35 * x)
        + 0.45 * torch.sin(3.80 * x + 0.35)
        + 0.20 * torch.cos(6.20 * x - 0.60)
    )


def paper_regression_function(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Regression target from Figure 5 of Bayes by Backprop."""

    epsilon = noise_std * torch.randn_like(x)
    shifted_x = x + epsilon
    return (
        shifted_x
        + 0.3 * torch.sin(2.0 * math.pi * shifted_x)
        + 0.3 * torch.sin(4.0 * math.pi * shifted_x)
        + epsilon
    )


def paper_regression_mean(x: torch.Tensor) -> torch.Tensor:
    """Noise-free reference curve for the paper regression target."""

    return x + 0.3 * torch.sin(2.0 * math.pi * x) + 0.3 * torch.sin(4.0 * math.pi * x)


def resolve_regression_target(
    name: str,
) -> Tuple[Callable[[torch.Tensor, float], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Return the noisy sampling function and deterministic mean for the requested target."""

    if name == "oscillatory":
        return oscillatory_regression_function, oscillatory_regression_mean
    if name == "paper":
        return paper_regression_function, paper_regression_mean
    raise ValueError(f"Unsupported regression target '{name}'.")


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Override a subset of arguments with a named experimental preset."""

    if args.preset is None:
        return args

    if args.preset == "paper-figure5":
        args.target_function = "paper"
        args.observed_intervals = list(PAPER_FIGURE5_INTERVALS)
        args.domain_min = -0.4
        args.domain_max = 1.2
        return args

    raise ValueError(f"Unsupported preset '{args.preset}'.")


def sample_inputs_from_intervals(
    intervals: Sequence[Interval],
    num_points: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one-dimensional inputs uniformly from the observed intervals."""

    if num_points < len(intervals):
        raise ValueError("The number of training points must be at least the number of intervals.")

    lengths = torch.tensor([right - left for left, right in intervals], dtype=torch.float32)
    probabilities = lengths / lengths.sum()
    interval_indices = torch.multinomial(probabilities, num_samples=num_points, replacement=True, generator=generator)

    samples = torch.empty(num_points, dtype=torch.float32)
    for index, (left, right) in enumerate(intervals):
        mask = interval_indices == index
        count = int(mask.sum().item())
        if count == 0:
            continue
        samples[mask] = left + (right - left) * torch.rand(count, dtype=torch.float32, generator=generator)

    return samples.unsqueeze(1)


def sample_targets_with_interval_noise(
    inputs: torch.Tensor,
    target_fn: Callable[[torch.Tensor, float], torch.Tensor],
    default_noise_std: float,
    intervals: Sequence[Interval],
    interval_noise_stds: Sequence[float] | None = None,
) -> torch.Tensor:
    """Sample targets with optional interval-specific observation noise."""

    if interval_noise_stds is None:
        return target_fn(inputs, noise_std=default_noise_std)

    if len(interval_noise_stds) != len(intervals):
        raise ValueError("Provide exactly one interval noise standard deviation per observed interval.")

    targets = torch.empty_like(inputs)
    assigned_mask = torch.zeros(inputs.size(0), dtype=torch.bool, device=inputs.device)
    flat_inputs = inputs.squeeze(-1)
    for (left, right), interval_noise_std in zip(intervals, interval_noise_stds):
        mask = (flat_inputs >= left) & (flat_inputs <= right)
        if not mask.any():
            continue
        targets[mask] = target_fn(inputs[mask], noise_std=float(interval_noise_std))
        assigned_mask |= mask

    remaining_mask = ~assigned_mask
    if remaining_mask.any():
        targets[remaining_mask] = target_fn(inputs[remaining_mask], noise_std=default_noise_std)

    return targets


def build_regression_dataset(
    intervals: Sequence[Interval],
    num_points: int,
    noise_std: float,
    target_fn: Callable[[torch.Tensor, float], torch.Tensor],
    interval_noise_stds: Sequence[float] | None = None,
) -> TensorDataset:
    """Create a synthetic regression dataset restricted to observed intervals."""

    inputs = sample_inputs_from_intervals(intervals, num_points=num_points)
    targets = sample_targets_with_interval_noise(
        inputs=inputs,
        target_fn=target_fn,
        default_noise_std=noise_std,
        intervals=intervals,
        interval_noise_stds=interval_noise_stds,
    )
    return TensorDataset(inputs, targets)


def build_outside_interval_guide_dataset(
    observed_intervals: Sequence[Interval],
    domain_min: float,
    domain_max: float,
    num_points: int,
    region_mode: str,
    noise_std: float,
    target_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> TensorDataset | None:
    """Create a small guidance dataset from gaps outside the main observed intervals."""

    if num_points <= 0:
        return None

    sorted_intervals = sorted(observed_intervals)
    if region_mode == "all-gaps":
        guide_intervals = complement_intervals(sorted_intervals, domain_min=domain_min, domain_max=domain_max)
    elif region_mode == "outer-only":
        guide_intervals: List[Interval] = []
        leftmost_left = sorted_intervals[0][0]
        rightmost_right = sorted_intervals[-1][1]
        if domain_min < leftmost_left:
            guide_intervals.append((domain_min, leftmost_left))
        if rightmost_right < domain_max:
            guide_intervals.append((rightmost_right, domain_max))
    else:
        raise ValueError(f"Unsupported guide region mode '{region_mode}'.")

    if not guide_intervals:
        return None

    inputs = sample_inputs_from_intervals(guide_intervals, num_points=num_points)
    targets = target_fn(inputs, noise_std=noise_std)
    return TensorDataset(inputs, targets)


def build_interior_gap_guide_dataset(
    observed_intervals: Sequence[Interval],
    num_points: int,
    noise_std: float,
    target_fn: Callable[[torch.Tensor, float], torch.Tensor],
) -> TensorDataset | None:
    """Create a small guidance dataset from the interior gaps between observed intervals."""

    if num_points <= 0:
        return None

    sorted_intervals = sorted(observed_intervals)
    interior_gaps = [
        (sorted_intervals[index][1], sorted_intervals[index + 1][0])
        for index in range(len(sorted_intervals) - 1)
        if sorted_intervals[index][1] < sorted_intervals[index + 1][0]
    ]
    if not interior_gaps:
        return None

    inputs = sample_inputs_from_intervals(interior_gaps, num_points=num_points)
    targets = target_fn(inputs, noise_std=noise_std)
    return TensorDataset(inputs, targets)


def split_tensor_dataset(
    dataset: TensorDataset,
    validation_fraction: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    """Split a TensorDataset into train and validation subsets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("The validation fraction must be in the open interval (0, 1).")

    inputs, targets = dataset.tensors
    dataset_size = inputs.size(0)
    validation_size = max(1, int(round(dataset_size * validation_fraction)))

    if validation_size >= dataset_size:
        raise ValueError("The validation split leaves no training examples. Reduce the validation fraction.")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(dataset_size, generator=generator)
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    train_dataset = TensorDataset(inputs[train_indices], targets[train_indices])
    validation_dataset = TensorDataset(inputs[validation_indices], targets[validation_indices])
    return train_dataset, validation_dataset


class RegressionStandardizer:
    """Normalize one-dimensional regression inputs and targets."""

    def __init__(self, input_mean: torch.Tensor, input_std: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> None:
        self.input_mean = input_mean
        self.input_std = input_std.clamp_min(1e-6)
        self.target_mean = target_mean
        self.target_std = target_std.clamp_min(1e-6)

    @classmethod
    def fit(cls, inputs: torch.Tensor, targets: torch.Tensor) -> "RegressionStandardizer":
        return cls(
            input_mean=inputs.mean(dim=0, keepdim=True),
            input_std=inputs.std(dim=0, keepdim=True, unbiased=False),
            target_mean=targets.mean(dim=0, keepdim=True),
            target_std=targets.std(dim=0, keepdim=True, unbiased=False),
        )

    @classmethod
    def from_config(cls, config: Dict[str, float | str | torch.Tensor]) -> "RegressionStandardizer":
        return cls(
            input_mean=torch.tensor([[float(config["input_mean"])]]),
            input_std=torch.tensor([[float(config["input_std"])]]),
            target_mean=torch.tensor([[float(config["target_mean"])]]),
            target_std=torch.tensor([[float(config["target_std"])]]),
        )

    def transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.input_mean) / self.input_std

    def transform_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def inverse_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets * self.target_std + self.target_mean

    def scale_target_std(self, std: torch.Tensor) -> torch.Tensor:
        return std * self.target_std

    def config(self) -> Dict[str, float]:
        return {
            "input_mean": float(self.input_mean.squeeze().item()),
            "input_std": float(self.input_std.squeeze().item()),
            "target_mean": float(self.target_mean.squeeze().item()),
            "target_std": float(self.target_std.squeeze().item()),
        }


def to_original_target_std(normalized_std: float, scaler: RegressionStandardizer) -> float:
    """Convert a standard deviation from normalized target space back to original target units."""

    std_tensor = torch.tensor([[normalized_std]], dtype=torch.float32)
    return float(scaler.scale_target_std(std_tensor).squeeze().item())


def estimate_local_observation_std(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Estimate an observation scale from nearby target differences without using the generator noise."""

    if inputs.size(0) < 2:
        fallback = float(targets.std(unbiased=False).item())
        return max(fallback * 0.1, 1e-4)

    sort_indices = torch.argsort(inputs.squeeze(-1))
    sorted_inputs = inputs[sort_indices].squeeze(-1)
    sorted_targets = targets[sort_indices].squeeze(-1)
    input_deltas = torch.diff(sorted_inputs)
    target_deltas = torch.diff(sorted_targets)

    valid = input_deltas > 0
    input_deltas = input_deltas[valid]
    target_deltas = target_deltas[valid]
    if input_deltas.numel() == 0:
        fallback = float(targets.std(unbiased=False).item())
        return max(fallback * 0.1, 1e-4)

    nearest_count = max(1, int(math.ceil(input_deltas.numel() * 0.25)))
    nearest_indices = torch.argsort(input_deltas)[:nearest_count]
    local_abs_deltas = target_deltas[nearest_indices].abs()
    median_abs_delta = float(torch.median(local_abs_deltas).item())
    gaussian_abs_median = 0.6744897501960817
    estimated_std = median_abs_delta / (gaussian_abs_median * math.sqrt(2.0))

    fallback = float(targets.std(unbiased=False).item()) * 0.1
    return max(estimated_std, fallback, 1e-4)


def resolve_global_likelihood_init_std(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    scaler: RegressionStandardizer,
    min_predictive_std: float,
    user_init_std_original_units: float | None,
) -> Tuple[float, float, str]:
    """Choose a global likelihood sigma init without leaking the generator noise to the model."""

    if user_init_std_original_units is None:
        init_std_original = estimate_local_observation_std(train_inputs, train_targets)
        source = "data-driven"
    else:
        if user_init_std_original_units <= 0.0:
            raise ValueError("The global likelihood initial standard deviation must be positive.")
        init_std_original = float(user_init_std_original_units)
        source = "user"

    target_scale = float(scaler.target_std.squeeze().item())
    normalized_floor = max(min_predictive_std / target_scale, 1e-6)
    init_std_normalized = max(init_std_original / target_scale, normalized_floor)
    init_std_original = init_std_normalized * target_scale
    return init_std_normalized, init_std_original, source


def resolve_global_likelihood_prior_mean_std(
    scaler: RegressionStandardizer,
    min_predictive_std: float,
    user_prior_mean_std_original_units: float | None,
    default_prior_mean_std_original_units: float,
) -> Tuple[float, float, str]:
    """Choose the prior mean of the global likelihood sigma in normalized and original units."""

    if user_prior_mean_std_original_units is None:
        prior_mean_std_original = float(default_prior_mean_std_original_units)
        source = "default"
    else:
        if user_prior_mean_std_original_units <= 0.0:
            raise ValueError("The global likelihood prior-mean standard deviation must be positive.")
        prior_mean_std_original = float(user_prior_mean_std_original_units)
        source = "user"

    target_scale = float(scaler.target_std.squeeze().item())
    normalized_floor = max(min_predictive_std / target_scale, 1e-6)
    prior_mean_std_normalized = max(prior_mean_std_original / target_scale, normalized_floor)
    prior_mean_std_original = prior_mean_std_normalized * target_scale
    return prior_mean_std_normalized, prior_mean_std_original, source


def resolve_spline_knots_original(
    domain_min: float,
    domain_max: float,
    user_knots_original_units: Sequence[float] | None,
    num_knots: int,
) -> Tuple[List[float], str]:
    """Resolve spline knots in original x units."""

    tolerance = 1e-6
    if user_knots_original_units is None:
        if num_knots < 2:
            raise ValueError("The spline likelihood model needs at least two knots.")
        knots = np.linspace(domain_min, domain_max, num=num_knots, dtype=np.float64).tolist()
        source = "uniform"
    else:
        if len(user_knots_original_units) < 2:
            raise ValueError("The spline likelihood model needs at least two knots.")
        knots = [float(knot) for knot in user_knots_original_units]
        source = "user"

    for previous, current in zip(knots, knots[1:]):
        if current <= previous:
            raise ValueError("Spline knots must be strictly increasing.")
    if knots[0] < domain_min - tolerance or knots[-1] > domain_max + tolerance:
        raise ValueError("Spline knots must lie inside the configured domain.")
    knots[0] = max(knots[0], domain_min)
    knots[-1] = min(knots[-1], domain_max)

    return knots, source


def resolve_rbf_centers_original(
    domain_min: float,
    domain_max: float,
    num_centers: int,
) -> List[float]:
    """Resolve equally spaced RBF centers in original x units."""

    if num_centers <= 0:
        raise ValueError("The RBF likelihood model needs at least one center.")
    return np.linspace(domain_min, domain_max, num=num_centers, dtype=np.float64).tolist()


def resolve_rbf_lengthscale_original(
    centers_original_units: Sequence[float],
    user_lengthscale_original_units: float | None,
) -> Tuple[float, str]:
    """Resolve the Gaussian RBF lengthscale in original x units."""

    if user_lengthscale_original_units is not None:
        if user_lengthscale_original_units <= 0.0:
            raise ValueError("The RBF lengthscale must be positive.")
        return float(user_lengthscale_original_units), "user"

    if len(centers_original_units) <= 1:
        return 1.0, "default-single-center"

    center_tensor = torch.tensor(centers_original_units, dtype=torch.float32)
    spacing = torch.diff(center_tensor).mean().item()
    if spacing <= 0.0:
        raise ValueError("RBF centers must be strictly increasing.")
    return float(spacing), "spacing"


def normalize_input_locations(locations: Sequence[float], scaler: RegressionStandardizer) -> List[float]:
    """Transform one-dimensional x locations into the model's normalized input space."""

    points = torch.tensor(locations, dtype=torch.float32).unsqueeze(1)
    normalized = scaler.transform_inputs(points).squeeze(1)
    return [float(value.item()) for value in normalized]
