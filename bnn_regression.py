"""Bayesian neural network regression on disjoint intervals using Bayes by Backprop."""

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


Interval = Tuple[float, float]
DEFAULT_OBSERVED_INTERVALS = [(-4.0, -2.0), (-0.5, 0.75), (1.75, 3.5)]


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


def parse_intervals(value: str) -> List[Interval]:
    """Parse intervals written as 'left:right,left:right'."""

    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if len(chunks) < 2:
        raise argparse.ArgumentTypeError("Provide at least two disjoint intervals.")

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


def paper_regression_function(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Oscillatory regression target with several frequencies and no dominant linear trend."""

    epsilon = noise_std * torch.randn_like(x)
    shifted_x = x + epsilon
    return (
        1.10 * torch.sin(1.35 * shifted_x)
        + 0.45 * torch.sin(3.80 * shifted_x + 0.35)
        + 0.20 * torch.cos(6.20 * shifted_x - 0.60)
        + epsilon
    )


def paper_regression_mean(x: torch.Tensor) -> torch.Tensor:
    """Noise-free reference curve for plotting."""

    return (
        1.10 * torch.sin(1.35 * x)
        + 0.45 * torch.sin(3.80 * x + 0.35)
        + 0.20 * torch.cos(6.20 * x - 0.60)
    )


def sample_inputs_from_intervals(intervals: Sequence[Interval], num_points: int) -> torch.Tensor:
    """Sample one-dimensional inputs uniformly from the observed intervals."""

    if num_points < len(intervals):
        raise ValueError("The number of training points must be at least the number of intervals.")

    lengths = torch.tensor([right - left for left, right in intervals], dtype=torch.float32)
    probabilities = lengths / lengths.sum()
    interval_indices = torch.multinomial(probabilities, num_samples=num_points, replacement=True)

    samples = torch.empty(num_points, dtype=torch.float32)
    for index, (left, right) in enumerate(intervals):
        mask = interval_indices == index
        count = int(mask.sum().item())
        if count == 0:
            continue
        samples[mask] = left + (right - left) * torch.rand(count, dtype=torch.float32)

    return samples.unsqueeze(1)


def build_regression_dataset(
    intervals: Sequence[Interval],
    num_points: int,
    noise_std: float,
) -> TensorDataset:
    """Create a synthetic regression dataset restricted to observed intervals."""

    inputs = sample_inputs_from_intervals(intervals, num_points=num_points)
    targets = paper_regression_function(inputs, noise_std=noise_std)
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


def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor | float, sigma: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density."""

    variance = sigma.pow(2)
    return -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - (value - mean).pow(2) / (2.0 * variance)


class PriorDistribution:
    """Interface for priors used by Bayesian layers."""

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def config(self) -> Dict[str, float | str]:
        raise NotImplementedError


class NormalPrior(PriorDistribution):
    """Zero-mean Gaussian prior."""

    def __init__(self, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError("The normal prior sigma must be positive.")
        self.sigma = sigma

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        sigma = value.new_tensor(self.sigma)
        return gaussian_log_prob(value, 0.0, sigma).sum()

    def config(self) -> Dict[str, float | str]:
        return {"prior": "normal", "prior_sigma": self.sigma}


class SpikeAndSlabPrior(PriorDistribution):
    """Two-component zero-mean Gaussian scale-mixture prior."""

    def __init__(self, pi: float, sigma1: float, sigma2: float) -> None:
        if not 0.0 < pi < 1.0:
            raise ValueError("The spike-and-slab mixture weight pi must be in (0, 1).")
        if sigma1 <= 0.0 or sigma2 <= 0.0:
            raise ValueError("Spike-and-slab standard deviations must be positive.")
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        sigma1 = value.new_tensor(self.sigma1)
        sigma2 = value.new_tensor(self.sigma2)

        component1 = math.log(self.pi) + gaussian_log_prob(value, 0.0, sigma1)
        component2 = math.log(1.0 - self.pi) + gaussian_log_prob(value, 0.0, sigma2)
        return torch.logaddexp(component1, component2).sum()

    def config(self) -> Dict[str, float | str]:
        return {
            "prior": "spike-slab",
            "prior_pi": self.pi,
            "prior_sigma1": self.sigma1,
            "prior_sigma2": self.sigma2,
        }


class BayesianLinear(nn.Module):
    """Linear layer with a diagonal Gaussian variational posterior."""

    def __init__(self, in_features: int, out_features: int, prior: PriorDistribution) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior = prior

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def _sample_parameter(self, mu: torch.Tensor, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma = self._sigma(rho)
        epsilon = torch.randn_like(mu)
        return mu + sigma * epsilon, sigma

    @staticmethod
    def _posterior_log_prob(sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return gaussian_log_prob(sample, mu, sigma).sum()

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample:
            weight, weight_sigma = self._sample_parameter(self.weight_mu, self.weight_rho)
            bias, bias_sigma = self._sample_parameter(self.bias_mu, self.bias_rho)

            log_q = self._posterior_log_prob(weight, self.weight_mu, weight_sigma)
            log_q += self._posterior_log_prob(bias, self.bias_mu, bias_sigma)

            log_p = self.prior.log_prob(weight)
            log_p += self.prior.log_prob(bias)

            kl = log_q - log_p
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            kl = x.new_zeros(())

        return F.linear(x, weight, bias), kl


class BayesianRegressor(nn.Module):
    """Small Bayesian MLP that predicts a Gaussian mean and scale."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        prior: PriorDistribution,
        activation: str = "tanh",
        min_predictive_std: float = 1e-3,
    ) -> None:
        super().__init__()

        if min_predictive_std <= 0.0:
            raise ValueError("The minimum predictive standard deviation must be positive.")

        self.min_predictive_std = min_predictive_std
        layer_sizes = [1, *hidden_dims, 2]
        self.layers = nn.ModuleList(
            BayesianLinear(in_features, out_features, prior=prior)
            for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
        )

        if activation == "relu":
            self.activation = F.relu
        elif activation == "tanh":
            self.activation = torch.tanh
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_kl = x.new_zeros(())

        for layer in self.layers[:-1]:
            x, kl = layer(x, sample=sample)
            total_kl = total_kl + kl
            x = self.activation(x)

        output, kl = self.layers[-1](x, sample=sample)
        total_kl = total_kl + kl
        mean = output[:, :1]
        predictive_std = F.softplus(output[:, 1:]) + self.min_predictive_std
        return mean, predictive_std, total_kl


def gaussian_nll(mean: torch.Tensor, predictive_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian observation model."""

    variance = predictive_std.pow(2)
    return 0.5 * (((target - mean) ** 2) / variance + torch.log(2.0 * math.pi * variance))


def compute_kl_weight(epoch: int, warmup_epochs: int) -> float:
    """Linearly anneal the KL term during the first warmup epochs."""

    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, epoch / warmup_epochs)


def run_epoch(
    model: BayesianRegressor,
    loader: DataLoader,
    device: torch.device,
    dataset_size: int,
    scaler: RegressionStandardizer,
    num_samples: int,
    kl_weight: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    """Run one train or evaluation epoch."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    totals = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "mse": 0.0, "predictive_std": 0.0, "examples": 0.0}

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        if is_training:
            optimizer.zero_grad()

        sample_losses = []
        sample_nlls = []
        sample_kls = []
        sample_means = []
        sample_stds = []

        with torch.set_grad_enabled(is_training):
            for _ in range(num_samples):
                mean, predictive_std, kl = model(inputs, sample=True)
                nll = gaussian_nll(mean, predictive_std, targets).mean()
                scaled_kl = kl_weight * kl / dataset_size
                loss = nll + scaled_kl

                sample_losses.append(loss)
                sample_nlls.append(nll)
                sample_kls.append(scaled_kl)
                sample_means.append(mean)
                sample_stds.append(predictive_std)

            mean_loss = torch.stack(sample_losses).mean()
            mean_nll = torch.stack(sample_nlls).mean()
            mean_kl = torch.stack(sample_kls).mean()
            stacked_means = torch.stack(sample_means)
            stacked_stds = torch.stack(sample_stds)
            predictive_mean = stacked_means.mean(dim=0)
            predictive_variance = stacked_stds.pow(2).mean(dim=0) + stacked_means.var(dim=0, unbiased=False)
            original_predictive_mean = scaler.inverse_targets(predictive_mean.detach().cpu()).to(device)
            original_targets = scaler.inverse_targets(targets.detach().cpu()).to(device)
            mse = F.mse_loss(original_predictive_mean, original_targets)
            mean_predictive_std = scaler.scale_target_std(predictive_variance.sqrt().detach().cpu()).mean().to(device)

            if is_training:
                mean_loss.backward()
                optimizer.step()

        totals["loss"] += mean_loss.item() * batch_size
        totals["nll"] += mean_nll.item() * batch_size
        totals["kl"] += mean_kl.item() * batch_size
        totals["mse"] += mse.item() * batch_size
        totals["predictive_std"] += mean_predictive_std.item() * batch_size
        totals["examples"] += batch_size

    total_examples = totals["examples"]
    return {
        "loss": totals["loss"] / total_examples,
        "nll": totals["nll"] / total_examples,
        "kl": totals["kl"] / total_examples,
        "mse": totals["mse"] / total_examples,
        "predictive_std": totals["predictive_std"] / total_examples,
    }


@torch.no_grad()
def evaluate_regression(
    model: BayesianRegressor,
    loader: DataLoader,
    device: torch.device,
    scaler: RegressionStandardizer,
    num_samples: int,
) -> Dict[str, float]:
    """Evaluate Bayesian predictive metrics on a validation split."""

    model.eval()
    totals = {"predictive_nll": 0.0, "mse": 0.0, "predictive_std": 0.0, "examples": 0.0}

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        batch_size = inputs.size(0)

        sample_means = []
        sample_stds = []
        sample_log_probs = []

        for _ in range(num_samples):
            mean, predictive_std, _ = model(inputs, sample=True)
            sample_means.append(mean)
            sample_stds.append(predictive_std)
            sample_log_probs.append(-gaussian_nll(mean, predictive_std, targets))

        stacked_means = torch.stack(sample_means)
        stacked_stds = torch.stack(sample_stds)
        stacked_log_probs = torch.stack(sample_log_probs)

        predictive_log_prob = torch.logsumexp(stacked_log_probs, dim=0) - math.log(num_samples)
        predictive_nll = -predictive_log_prob.mean()

        predictive_mean = stacked_means.mean(dim=0)
        predictive_variance = stacked_stds.pow(2).mean(dim=0) + stacked_means.var(dim=0, unbiased=False)
        original_predictive_mean = scaler.inverse_targets(predictive_mean.detach().cpu()).to(device)
        original_targets = scaler.inverse_targets(targets.detach().cpu()).to(device)
        mse = F.mse_loss(original_predictive_mean, original_targets)
        mean_predictive_std = scaler.scale_target_std(predictive_variance.sqrt().detach().cpu()).mean().to(device)

        totals["predictive_nll"] += predictive_nll.item() * batch_size
        totals["mse"] += mse.item() * batch_size
        totals["predictive_std"] += mean_predictive_std.item() * batch_size
        totals["examples"] += batch_size

    total_examples = totals["examples"]
    return {
        "predictive_nll": totals["predictive_nll"] / total_examples,
        "mse": totals["mse"] / total_examples,
        "predictive_std": totals["predictive_std"] / total_examples,
    }


@torch.no_grad()
def predict_distribution(
    model: BayesianRegressor,
    inputs: torch.Tensor,
    num_samples: int,
    device: torch.device,
    scaler: RegressionStandardizer,
) -> Dict[str, torch.Tensor]:
    """Estimate predictive quantiles and uncertainty with Monte Carlo weight samples."""

    model.eval()
    inputs = inputs.to(device)

    means = []
    predictive_stds = []
    predictive_samples = []
    for _ in range(num_samples):
        sampled_mean, sampled_std, _ = model(inputs, sample=True)
        means.append(sampled_mean.squeeze(-1))
        predictive_stds.append(sampled_std.squeeze(-1))
        predictive_samples.append((sampled_mean + sampled_std * torch.randn_like(sampled_mean)).squeeze(-1))

    stacked_means = torch.stack(means, dim=0)
    stacked_stds = torch.stack(predictive_stds, dim=0)
    stacked_predictive_samples = torch.stack(predictive_samples, dim=0)
    function_samples_original = scaler.inverse_targets(stacked_means.T.cpu()).T
    predictive_samples_original = scaler.inverse_targets(stacked_predictive_samples.T.cpu()).T
    mean = scaler.inverse_targets(stacked_means.mean(dim=0, keepdim=True).T.cpu()).squeeze(-1).squeeze(-1)
    epistemic_std = scaler.scale_target_std(
        stacked_means.std(dim=0, unbiased=False, keepdim=True).T.cpu()
    ).squeeze(-1).squeeze(-1)
    aleatoric_std = scaler.scale_target_std(
        torch.sqrt(stacked_stds.pow(2).mean(dim=0, keepdim=True)).T.cpu()
    ).squeeze(-1).squeeze(-1)
    predictive_std = torch.sqrt(epistemic_std.pow(2) + aleatoric_std.pow(2))

    return {
        "means": stacked_means,
        "predictive_stds": stacked_stds,
        "function_samples": function_samples_original,
        "predictive_samples": predictive_samples_original,
        "mean": mean,
        "function_median": torch.quantile(function_samples_original, q=0.5, dim=0),
        "function_q25": torch.quantile(function_samples_original, q=0.25, dim=0),
        "function_q75": torch.quantile(function_samples_original, q=0.75, dim=0),
        "observation_median": torch.quantile(predictive_samples_original, q=0.5, dim=0),
        "observation_q25": torch.quantile(predictive_samples_original, q=0.25, dim=0),
        "observation_q75": torch.quantile(predictive_samples_original, q=0.75, dim=0),
        "median": torch.quantile(function_samples_original, q=0.5, dim=0),
        "q25": torch.quantile(function_samples_original, q=0.25, dim=0),
        "q75": torch.quantile(function_samples_original, q=0.75, dim=0),
        "epistemic_std": epistemic_std,
        "aleatoric_std": aleatoric_std,
        "predictive_std": predictive_std,
    }


def interval_mask(values: torch.Tensor, left: float, right: float) -> torch.Tensor:
    """Return a mask that includes the right edge only for the final interval."""

    return (values >= left) & (values <= right)


def summarize_region_uncertainty(
    grid_inputs: torch.Tensor,
    epistemic_std: torch.Tensor,
    predictive_std: torch.Tensor,
    observed_intervals: Sequence[Interval],
    domain_min: float,
    domain_max: float,
) -> List[str]:
    """Summarize uncertainty inside observed regions and across missing gaps."""

    grid_inputs = grid_inputs.squeeze(-1).cpu()
    epistemic_std = epistemic_std.cpu()
    predictive_std = predictive_std.cpu()
    missing_intervals = complement_intervals(observed_intervals, domain_min=domain_min, domain_max=domain_max)

    def midpoint_stat(left: float, right: float, values: torch.Tensor) -> float:
        midpoint = 0.5 * (left + right)
        index = int(torch.argmin((grid_inputs - midpoint).abs()).item())
        return float(values[index].item())

    lines = []
    observed_mask = torch.zeros_like(grid_inputs, dtype=torch.bool)
    for left, right in observed_intervals:
        mask = interval_mask(grid_inputs, left, right)
        observed_mask |= mask
        lines.append(
            f"observed [{left:.2f}, {right:.2f}] mean epistemic std = {epistemic_std[mask].mean().item():.4f}, "
            f"mean predictive std = {predictive_std[mask].mean().item():.4f}, "
            f"midpoint predictive std = {midpoint_stat(left, right, predictive_std):.4f}"
        )

    missing_mask = ~observed_mask
    if missing_mask.any():
        lines.append(
            "overall missing-region mean epistemic std = "
            f"{epistemic_std[missing_mask].mean().item():.4f}, "
            f"mean predictive std = {predictive_std[missing_mask].mean().item():.4f}"
        )

    for left, right in missing_intervals:
        mask = interval_mask(grid_inputs, left, right)
        if not mask.any():
            continue
        lines.append(
            f"missing [{left:.2f}, {right:.2f}] mean epistemic std = {epistemic_std[mask].mean().item():.4f}, "
            f"mean predictive std = {predictive_std[mask].mean().item():.4f}, "
            f"midpoint predictive std = {midpoint_stat(left, right, predictive_std):.4f}"
        )

    return lines


def save_checkpoint(
    model: BayesianRegressor,
    save_path: Path,
    hidden_dims: Sequence[int],
    activation: str,
    observed_intervals: Sequence[Interval],
    domain_min: float,
    domain_max: float,
    min_predictive_std: float,
    noise_std: float,
    kl_warmup_epochs: int,
    validation_fraction: float,
    validation_samples: int,
    early_stopping_metric: str,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    early_stopping_start_epoch: int,
    best_epoch: int,
    best_validation_mse: float,
    best_validation_predictive_nll: float,
    scaler: RegressionStandardizer,
    prior_config: Dict[str, float | str],
) -> None:
    """Save the model and experiment configuration."""

    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "observed_intervals": list(observed_intervals),
        "domain_min": domain_min,
        "domain_max": domain_max,
        "min_predictive_std": min_predictive_std,
        "noise_std": noise_std,
        "kl_warmup_epochs": kl_warmup_epochs,
        "validation_fraction": validation_fraction,
        "validation_samples": validation_samples,
        "early_stopping_metric": early_stopping_metric,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_start_epoch": early_stopping_start_epoch,
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_mse,
        "best_validation_predictive_nll": best_validation_predictive_nll,
        **scaler.config(),
        **prior_config,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def plot_predictions(
    plot_path: Path,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    grid_inputs: torch.Tensor,
    summary: Dict[str, torch.Tensor],
    observed_intervals: Sequence[Interval],
    quantile_space: str = "function",
    show_reference: bool = False,
    shade_observed_intervals: bool = False,
) -> None:
    """Save a plot with observed data, predictive median, and interquartile range."""

    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if quantile_space == "function":
        median_key = "function_median"
        q25_key = "function_q25"
        q75_key = "function_q75"
    elif quantile_space == "observation":
        median_key = "observation_median"
        q25_key = "observation_q25"
        q75_key = "observation_q75"
    else:
        raise ValueError(f"Unsupported quantile space '{quantile_space}'.")

    x_train = train_inputs.squeeze(-1).cpu().numpy()
    y_train = train_targets.squeeze(-1).cpu().numpy()
    x_grid = grid_inputs.squeeze(-1).cpu().numpy()
    median = summary[median_key].cpu().numpy()
    q25 = summary[q25_key].cpu().numpy()
    q75 = summary[q75_key].cpu().numpy()
    band_width = q75 - q25

    figure, axis = plt.subplots(figsize=(10, 5))

    if shade_observed_intervals:
        for left, right in observed_intervals:
            axis.axvspan(left, right, color="#d8f0d2", alpha=0.35, linewidth=0)

    axis.scatter(x_train, y_train, color="black", marker="x", alpha=0.7, label="Observations")

    if show_reference:
        reference = paper_regression_mean(grid_inputs).squeeze(-1).cpu().numpy()
        axis.plot(x_grid, reference, color="#888888", linestyle="--", linewidth=1.5, label="Reference function")

    axis.fill_between(x_grid, q25, q75, color="#4f8dd6", alpha=0.32, label="Interquartile range", zorder=1)
    axis.plot(x_grid, q25, color="#4f8dd6", linewidth=1.0, alpha=0.95, zorder=2)
    axis.plot(x_grid, q75, color="#4f8dd6", linewidth=1.0, alpha=0.95, zorder=2)
    axis.plot(x_grid, median, color="#c0392b", linewidth=2.0, label="Median prediction")

    axis.set_title("Bayesian Regression with Disjoint Observation Intervals")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.text(
        0.02,
        0.02,
        f"Mean IQR: {band_width.mean():.3f}",
        transform=axis.transAxes,
        fontsize=10,
        color="#355c8a",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "#b8cce6"},
    )
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)


def build_prior(args: argparse.Namespace) -> PriorDistribution:
    """Instantiate the requested prior distribution."""

    if args.prior == "normal":
        return NormalPrior(sigma=args.prior_sigma)
    if args.prior == "spike-slab":
        return SpikeAndSlabPrior(pi=args.prior_pi, sigma1=args.prior_sigma1, sigma2=args.prior_sigma2)
    raise ValueError(f"Unsupported prior '{args.prior}'.")


def parse_args() -> argparse.Namespace:
    """Define the command line interface."""

    parser = argparse.ArgumentParser(
        description="Bayesian neural network regression with disjoint observed intervals."
    )

    parser.add_argument("--epochs", type=int, default=1500, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[128, 128], help="Hidden layer sizes, e.g. 128,128.")
    parser.add_argument("--activation", choices=["tanh", "relu"], default="tanh", help="Hidden-layer activation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--train-samples", type=int, default=3, help="Weight samples per training batch.")
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=64,
        help="Weight samples used to estimate validation predictive NLL for early stopping.",
    )
    parser.add_argument("--test-samples", type=int, default=200, help="Weight samples for predictive summaries.")
    parser.add_argument("--train-points", type=int, default=192, help="Number of training points sampled from observed intervals.")
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of observed samples held out for validation and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        choices=["mse", "predictive_nll"],
        default="mse",
        help="Validation metric used to choose the best checkpoint and trigger early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop when the chosen validation early-stopping metric has not improved for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation MSE improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--early-stopping-start-epoch",
        type=int,
        default=1,
        help="Start tracking best validation metrics and patience from this epoch onward.",
    )
    parser.add_argument("--noise-std", type=float, default=0.02, help="Noise scale used to generate observations.")
    parser.add_argument(
        "--kl-warmup-epochs",
        type=int,
        default=200,
        help="Linearly increase the KL weight from 0 to 1 over this many epochs. Use 0 to disable warmup.",
    )
    parser.add_argument(
        "--min-predictive-std",
        type=float,
        default=0.02,
        help="Lower bound applied to the learned predictive standard deviation.",
    )
    parser.add_argument("--domain-min", type=float, default=-5.0, help="Minimum x-value of the evaluation domain.")
    parser.add_argument("--domain-max", type=float, default=4.5, help="Maximum x-value of the evaluation domain.")
    parser.add_argument(
        "--observed-intervals",
        type=parse_intervals,
        default=DEFAULT_OBSERVED_INTERVALS,
        help="Comma-separated disjoint intervals such as -4:-2,-0.5:0.75,1.75:3.5.",
    )
    parser.add_argument("--grid-size", type=int, default=500, help="Points used to evaluate the predictive curve.")
    parser.add_argument("--prior", choices=["normal", "spike-slab"], default="normal", help="Prior distribution over weights.")
    parser.add_argument("--prior-sigma", type=float, default=1.0, help="Standard deviation of the normal prior.")
    parser.add_argument("--prior-pi", type=float, default=0.5, help="Large-variance mixture weight for the spike-and-slab prior.")
    parser.add_argument("--prior-sigma1", type=float, default=1.0, help="Large-variance standard deviation for the spike-and-slab prior.")
    parser.add_argument("--prior-sigma2", type=float, default=0.1, help="Small-variance standard deviation for the spike-and-slab prior.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="How often to print training progress.")
    parser.add_argument("--save-path", type=Path, default=None, help="Optional checkpoint path.")
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional path for a predictive uncertainty plot.")
    parser.add_argument(
        "--plot-quantiles",
        choices=["function", "observation"],
        default="observation",
        help="Whether the plotted IQR should come from sampled latent functions or noisy observations.",
    )
    parser.add_argument(
        "--show-reference",
        action="store_true",
        help="Overlay the true synthetic function on the regression plot.",
    )
    parser.add_argument(
        "--shade-observed-intervals",
        action="store_true",
        help="Shade the input intervals that contain training observations.",
    )
    return parser.parse_args()


def main() -> None:
    """Train a Bayesian regressor and report uncertainty in missing intervals."""

    args = parse_args()

    if args.min_predictive_std <= 0.0:
        raise ValueError("The minimum predictive standard deviation must be positive.")
    if args.validation_fraction <= 0.0 or args.validation_fraction >= 1.0:
        raise ValueError("The validation fraction must be in the open interval (0, 1).")
    if args.validation_samples <= 0:
        raise ValueError("The validation samples must be positive.")
    if args.kl_warmup_epochs < 0:
        raise ValueError("The KL warmup epochs must be non-negative.")
    if args.early_stopping_patience <= 0:
        raise ValueError("The early stopping patience must be positive.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("The early stopping minimum delta must be non-negative.")
    if args.early_stopping_start_epoch <= 0:
        raise ValueError("The early stopping start epoch must be positive.")
    if args.epochs <= 0:
        raise ValueError("The number of epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("The batch size must be positive.")
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("The number of Monte Carlo samples must be positive.")

    validate_intervals_in_domain(args.observed_intervals, args.domain_min, args.domain_max)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = build_prior(args)

    raw_dataset = build_regression_dataset(
        intervals=args.observed_intervals,
        num_points=args.train_points,
        noise_std=args.noise_std,
    )
    raw_train_dataset, raw_validation_dataset = split_tensor_dataset(
        raw_dataset,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_inputs, train_targets = raw_train_dataset.tensors
    validation_inputs, validation_targets = raw_validation_dataset.tensors
    scaler = RegressionStandardizer.fit(train_inputs, train_targets)
    train_dataset = TensorDataset(scaler.transform_inputs(train_inputs), scaler.transform_targets(train_targets))
    validation_dataset = TensorDataset(
        scaler.transform_inputs(validation_inputs),
        scaler.transform_targets(validation_targets),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    model = BayesianRegressor(
        hidden_dims=args.hidden_dims,
        prior=prior,
        activation=args.activation,
        min_predictive_std=args.min_predictive_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    print(f"Observed intervals: {intervals_to_string(args.observed_intervals)}")
    print(f"Missing intervals: {intervals_to_string(complement_intervals(args.observed_intervals, args.domain_min, args.domain_max))}")
    print(f"Prior: {prior.config()}")
    print(f"KL warmup epochs: {args.kl_warmup_epochs}")
    print(f"Train / validation points: {len(train_dataset)} / {len(validation_dataset)}")
    if args.early_stopping_metric == "mse":
        print("Early stopping metric: validation MSE of the Monte Carlo mean prediction")
    else:
        print("Early stopping metric: validation predictive NLL")
    print(f"Early stopping starts at epoch: {args.early_stopping_start_epoch}")

    best_validation_score = float("inf")
    best_validation_mse = float("inf")
    best_validation_predictive_nll = float("inf")
    best_epoch = 0
    best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        kl_weight = compute_kl_weight(epoch=epoch, warmup_epochs=args.kl_warmup_epochs)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            dataset_size=len(train_dataset),
            scaler=scaler,
            num_samples=args.train_samples,
            kl_weight=kl_weight,
            optimizer=optimizer,
        )
        validation_metrics = evaluate_regression(
            model=model,
            loader=validation_loader,
            device=device,
            scaler=scaler,
            num_samples=args.validation_samples,
        )

        if epoch >= args.early_stopping_start_epoch:
            current_validation_score = validation_metrics[args.early_stopping_metric]
            improvement_threshold = best_validation_score - args.early_stopping_min_delta
            if current_validation_score < improvement_threshold:
                best_validation_score = current_validation_score
                best_validation_mse = validation_metrics["mse"]
                best_validation_predictive_nll = validation_metrics["predictive_nll"]
                best_epoch = epoch
                best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        should_log = epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0
        if should_log:
            print(
                f"Epoch {epoch:04d} | "
                f"beta {kl_weight:.3f} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train nll {train_metrics['nll']:.4f} | "
                f"train mse {train_metrics['mse']:.4f} | "
                f"val mse {validation_metrics['mse']:.4f} | "
                f"val pred-nll {validation_metrics['predictive_nll']:.4f} | "
                f"val pred std {validation_metrics['predictive_std']:.4f}"
            )

        if epoch >= args.early_stopping_start_epoch and epochs_without_improvement >= args.early_stopping_patience:
            best_metric_label = "val mse" if args.early_stopping_metric == "mse" else "val pred-nll"
            best_metric_value = best_validation_mse if args.early_stopping_metric == "mse" else best_validation_predictive_nll
            print(
                f"Early stopping at epoch {epoch:04d} | "
                f"best epoch {best_epoch:04d} | "
                f"best {best_metric_label} {best_metric_value:.4f}"
            )
            break

    if best_epoch == 0:
        best_epoch = args.epochs
        best_validation_score = validation_metrics[args.early_stopping_metric]
        best_validation_mse = validation_metrics["mse"]
        best_validation_predictive_nll = validation_metrics["predictive_nll"]
        best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    model.load_state_dict(best_state_dict)
    final_validation_metrics = evaluate_regression(
        model=model,
        loader=validation_loader,
        device=device,
        scaler=scaler,
        num_samples=args.validation_samples,
    )
    print(
        f"Restored best model from epoch {best_epoch:04d} | "
        f"val mse {final_validation_metrics['mse']:.4f} | "
        f"val pred-nll {final_validation_metrics['predictive_nll']:.4f}"
    )

    grid_inputs = torch.linspace(args.domain_min, args.domain_max, steps=args.grid_size, dtype=torch.float32).unsqueeze(1)
    scaled_grid_inputs = scaler.transform_inputs(grid_inputs)
    prediction_summary = predict_distribution(
        model=model,
        inputs=scaled_grid_inputs,
        num_samples=args.test_samples,
        device=device,
        scaler=scaler,
    )

    print("Uncertainty summary:")
    for line in summarize_region_uncertainty(
        grid_inputs=grid_inputs,
        epistemic_std=prediction_summary["epistemic_std"],
        predictive_std=prediction_summary["predictive_std"],
        observed_intervals=args.observed_intervals,
        domain_min=args.domain_min,
        domain_max=args.domain_max,
    ):
        print(f"  {line}")

    if args.save_path is not None:
        save_checkpoint(
            model=model,
            save_path=args.save_path,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            observed_intervals=args.observed_intervals,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            min_predictive_std=args.min_predictive_std,
            noise_std=args.noise_std,
            kl_warmup_epochs=args.kl_warmup_epochs,
            validation_fraction=args.validation_fraction,
            validation_samples=args.validation_samples,
            early_stopping_metric=args.early_stopping_metric,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_start_epoch=args.early_stopping_start_epoch,
            best_epoch=best_epoch,
            best_validation_mse=best_validation_mse,
            best_validation_predictive_nll=best_validation_predictive_nll,
            scaler=scaler,
            prior_config=prior.config(),
        )
        print(f"Saved checkpoint to {args.save_path}")

    if args.plot_path is not None:
        plot_predictions(
            plot_path=args.plot_path,
            train_inputs=train_inputs,
            train_targets=train_targets,
            grid_inputs=grid_inputs,
            summary=prediction_summary,
            observed_intervals=args.observed_intervals,
            quantile_space=args.plot_quantiles,
            show_reference=args.show_reference,
            shade_observed_intervals=args.shade_observed_intervals,
        )
        print(f"Saved predictive plot to {args.plot_path}")


if __name__ == "__main__":
    main()
