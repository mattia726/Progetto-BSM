"""Predictive evaluation, checkpointing, and plotting utilities for Bayesian 1D regression."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Sequence

import torch
from torch.utils.data import DataLoader

from .bnn_regression_data import (
    Interval,
    RegressionStandardizer,
    complement_intervals,
    intervals_to_string,
    normalize_input_locations,
    resolve_regression_target,
    sample_inputs_from_intervals,
    sample_targets_with_interval_noise,
    to_original_target_std,
)
from .bnn_regression_model import BayesianRegressor, build_prior_from_config, gaussian_nll


@torch.no_grad()
def evaluate_regression(
    model: BayesianRegressor,
    loader: DataLoader,
    device: torch.device,
    scaler: RegressionStandardizer,
    num_samples: int,
) -> Dict[str, float]:
    """Evaluate Bayesian predictive validation metrics."""

    model.eval()
    totals = {"predictive_nll": 0.0, "predictive_std": 0.0, "examples": 0.0}

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

        predictive_variance = stacked_stds.pow(2).mean(dim=0) + stacked_means.var(dim=0, unbiased=False)
        mean_predictive_std = scaler.scale_target_std(predictive_variance.sqrt().detach().cpu()).mean().to(device)

        totals["predictive_nll"] += predictive_nll.item() * batch_size
        totals["predictive_std"] += mean_predictive_std.item() * batch_size
        totals["examples"] += batch_size

    total_examples = totals["examples"]
    return {
        "predictive_nll": totals["predictive_nll"] / total_examples,
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
        "function_q025": torch.quantile(function_samples_original, q=0.025, dim=0),
        "function_q975": torch.quantile(function_samples_original, q=0.975, dim=0),
        "observation_median": torch.quantile(predictive_samples_original, q=0.5, dim=0),
        "observation_q25": torch.quantile(predictive_samples_original, q=0.25, dim=0),
        "observation_q75": torch.quantile(predictive_samples_original, q=0.75, dim=0),
        "observation_q025": torch.quantile(predictive_samples_original, q=0.025, dim=0),
        "observation_q975": torch.quantile(predictive_samples_original, q=0.975, dim=0),
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
) -> list[str]:
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


def compute_interval_coverage(summary: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
    """Compute empirical coverage for function-space and observation-space quantiles."""

    flattened_targets = targets.squeeze(-1)
    return {
        "observation_iqr": float(
            ((flattened_targets >= summary["observation_q25"]) & (flattened_targets <= summary["observation_q75"]))
            .float()
            .mean()
            .item()
        ),
        "observation_95": float(
            ((flattened_targets >= summary["observation_q025"]) & (flattened_targets <= summary["observation_q975"]))
            .float()
            .mean()
            .item()
        ),
        "function_iqr": float(
            ((flattened_targets >= summary["function_q25"]) & (flattened_targets <= summary["function_q75"]))
            .float()
            .mean()
            .item()
        ),
        "function_95": float(
            ((flattened_targets >= summary["function_q025"]) & (flattened_targets <= summary["function_q975"]))
            .float()
            .mean()
            .item()
        ),
    }


@torch.no_grad()
def evaluate_generated_coverage(
    model: BayesianRegressor,
    scaler: RegressionStandardizer,
    target_fn,
    observed_intervals: Sequence[Interval],
    domain_min: float,
    domain_max: float,
    noise_std: float,
    observed_interval_noise_stds: Sequence[float] | None,
    num_points: int,
    predictive_samples: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
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
        )

    return {
        "full_domain_uniform": compute_interval_coverage(full_summary, full_targets),
        "observed_intervals_only": compute_interval_coverage(interval_summary, interval_targets),
    }


def print_generated_coverage_results(results: Dict[str, Dict[str, float]], num_points: int) -> None:
    """Print generated-point coverage results in a stable human-readable format."""

    print(f"Generated coverage test ({num_points} fresh noisy points):")
    for mode_name, metrics in results.items():
        print(f"  {mode_name}:")
        print(f"    observation IQR coverage = {metrics['observation_iqr'] * 100.0:.1f}%")
        print(f"    observation 95% coverage = {metrics['observation_95'] * 100.0:.1f}%")
        print(f"    function IQR coverage = {metrics['function_iqr'] * 100.0:.1f}%")
        print(f"    function 95% coverage = {metrics['function_95'] * 100.0:.1f}%")


def save_checkpoint(
    model: BayesianRegressor,
    save_path: Path,
    preset: str | None,
    target_function: str,
    likelihood_std_model: str,
    global_likelihood_init_std: float | None,
    global_likelihood_init_std_original_units: float | None,
    global_likelihood_init_source: str | None,
    global_likelihood_prior_mean_std: float | None,
    global_likelihood_prior_mean_std_original_units: float | None,
    global_likelihood_prior_mean_source: str | None,
    global_likelihood_prior_sigma: float,
    spline_knots: Sequence[float] | None,
    spline_knots_original_units: Sequence[float] | None,
    spline_knot_source: str | None,
    spline_coefficient_prior_sigma: float | None,
    rbf_centers: Sequence[float] | None,
    rbf_centers_original_units: Sequence[float] | None,
    rbf_center_source: str | None,
    rbf_lengthscale: float | None,
    rbf_lengthscale_original_units: float | None,
    rbf_lengthscale_source: str | None,
    rbf_lengthscale_prior_sigma: float | None,
    rbf_coefficient_prior_sigma: float | None,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    observed_inputs: torch.Tensor | None,
    observed_targets: torch.Tensor | None,
    guide_inputs: torch.Tensor | None,
    guide_targets: torch.Tensor | None,
    guide_points_outside_intervals: int,
    guide_points_interior_gaps: int,
    guide_region_mode: str,
    observed_interval_noise_stds: Sequence[float] | None,
    hidden_dims: Sequence[int],
    activation: str,
    observed_intervals: Sequence[Interval],
    domain_min: float,
    domain_max: float,
    min_predictive_std: float,
    noise_std: float,
    validation_fraction: float,
    validation_samples: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    best_epoch: int,
    best_validation_predictive_nll: float,
    scaler: RegressionStandardizer,
    prior_config: Dict[str, float | str],
) -> None:
    """Save the model and experiment configuration."""

    checkpoint = {
        "state_dict": model.state_dict(),
        "preset": preset,
        "target_function": target_function,
        "likelihood_std_model": likelihood_std_model,
        "global_likelihood_init_std": global_likelihood_init_std,
        "global_likelihood_init_std_original_units": global_likelihood_init_std_original_units,
        "global_likelihood_init_source": global_likelihood_init_source,
        "global_likelihood_prior_mean_std": global_likelihood_prior_mean_std,
        "global_likelihood_prior_mean_std_original_units": global_likelihood_prior_mean_std_original_units,
        "global_likelihood_prior_mean_source": global_likelihood_prior_mean_source,
        "global_likelihood_prior_sigma": global_likelihood_prior_sigma,
        "global_likelihood_std": model.current_global_likelihood_std(),
        "global_likelihood_std_original_units": None
        if model.current_global_likelihood_std() is None
        else to_original_target_std(model.current_global_likelihood_std(), scaler),
        "spline_knots": None if spline_knots is None else list(spline_knots),
        "spline_knots_original_units": None if spline_knots_original_units is None else list(spline_knots_original_units),
        "spline_knot_source": spline_knot_source,
        "spline_coefficient_prior_sigma": spline_coefficient_prior_sigma,
        "rbf_centers": None if rbf_centers is None else list(rbf_centers),
        "rbf_centers_original_units": None if rbf_centers_original_units is None else list(rbf_centers_original_units),
        "rbf_center_source": rbf_center_source,
        "rbf_lengthscale": rbf_lengthscale,
        "rbf_lengthscale_original_units": rbf_lengthscale_original_units,
        "rbf_lengthscale_source": rbf_lengthscale_source,
        "rbf_lengthscale_prior_sigma": rbf_lengthscale_prior_sigma,
        "rbf_current_lengthscale": model.current_rbf_lengthscale(),
        "rbf_current_lengthscale_original_units": None
        if model.current_rbf_lengthscale() is None
        else model.current_rbf_lengthscale() * float(scaler.input_std.squeeze().item()),
        "rbf_coefficient_prior_sigma": rbf_coefficient_prior_sigma,
        "train_inputs": train_inputs.detach().cpu(),
        "train_targets": train_targets.detach().cpu(),
        "observed_inputs": None if observed_inputs is None else observed_inputs.detach().cpu(),
        "observed_targets": None if observed_targets is None else observed_targets.detach().cpu(),
        "guide_inputs": None if guide_inputs is None else guide_inputs.detach().cpu(),
        "guide_targets": None if guide_targets is None else guide_targets.detach().cpu(),
        "guide_points_outside_intervals": guide_points_outside_intervals,
        "guide_points_interior_gaps": guide_points_interior_gaps,
        "guide_region_mode": guide_region_mode,
        "observed_interval_noise_stds": None if observed_interval_noise_stds is None else list(observed_interval_noise_stds),
        "hidden_dims": list(hidden_dims),
        "activation": activation,
        "observed_intervals": list(observed_intervals),
        "domain_min": domain_min,
        "domain_max": domain_max,
        "min_predictive_std": min_predictive_std,
        "noise_std": noise_std,
        "validation_fraction": validation_fraction,
        "validation_samples": validation_samples,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "best_epoch": best_epoch,
        "best_validation_predictive_nll": best_validation_predictive_nll,
        **scaler.config(),
        **prior_config,
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)


def plot_predictions(
    plot_path: Path,
    observed_inputs: torch.Tensor | None,
    observed_targets: torch.Tensor | None,
    guide_inputs: torch.Tensor | None,
    guide_targets: torch.Tensor | None,
    grid_inputs: torch.Tensor,
    reference_curve: torch.Tensor,
    summary: Dict[str, torch.Tensor],
    observed_intervals: Sequence[Interval],
    quantile_space: str = "function",
    shade_observed_intervals: bool = False,
    show_summary_box: bool = True,
) -> None:
    """Save a plot with observed data, predictive median, IQR, and a 95% interval."""

    import matplotlib.pyplot as plt

    plot_path.parent.mkdir(parents=True, exist_ok=True)

    if quantile_space == "function":
        median_key = "function_median"
        q25_key = "function_q25"
        q75_key = "function_q75"
        q025_key = "function_q025"
        q975_key = "function_q975"
    elif quantile_space == "observation":
        median_key = "observation_median"
        q25_key = "observation_q25"
        q75_key = "observation_q75"
        q025_key = "observation_q025"
        q975_key = "observation_q975"
    else:
        raise ValueError(f"Unsupported quantile space '{quantile_space}'.")

    x_grid = grid_inputs.squeeze(-1).cpu().numpy()
    median = summary[median_key].cpu().numpy()
    q25 = summary[q25_key].cpu().numpy()
    q75 = summary[q75_key].cpu().numpy()
    q025 = summary[q025_key].cpu().numpy()
    q975 = summary[q975_key].cpu().numpy()
    iqr_width = q75 - q25
    interval95_width = q975 - q025

    grid_inputs_cpu = grid_inputs.squeeze(-1).cpu()
    observed_mask = torch.zeros_like(grid_inputs_cpu, dtype=torch.bool)
    for left, right in observed_intervals:
        observed_mask |= interval_mask(grid_inputs_cpu, left, right)
    missing_mask = ~observed_mask

    observed_mask_np = observed_mask.numpy()
    missing_mask_np = missing_mask.numpy()

    observed_iqr_mean = float(iqr_width[observed_mask_np].mean()) if observed_mask.any().item() else float("nan")
    missing_iqr_mean = float(iqr_width[missing_mask_np].mean()) if missing_mask.any().item() else float("nan")
    observed_interval95_mean = (
        float(interval95_width[observed_mask_np].mean()) if observed_mask.any().item() else float("nan")
    )
    missing_interval95_mean = (
        float(interval95_width[missing_mask_np].mean()) if missing_mask.any().item() else float("nan")
    )

    figure, axis = plt.subplots(figsize=(10, 5))

    if shade_observed_intervals:
        for left, right in observed_intervals:
            axis.axvspan(left, right, color="#d8f0d2", alpha=0.35, linewidth=0)

    if observed_inputs is not None and observed_targets is not None:
        x_observed = observed_inputs.squeeze(-1).cpu().numpy()
        y_observed = observed_targets.squeeze(-1).cpu().numpy()
        axis.scatter(x_observed, y_observed, color="black", marker="x", alpha=0.7, label="Interval observations")
    if guide_inputs is not None and guide_targets is not None:
        x_guide = guide_inputs.squeeze(-1).cpu().numpy()
        y_guide = guide_targets.squeeze(-1).cpu().numpy()
        axis.scatter(
            x_guide,
            y_guide,
            color="#d98b2b",
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            s=28,
            alpha=0.9,
            label="Guide observations",
        )
    reference = reference_curve.squeeze(-1).cpu().numpy()
    axis.plot(x_grid, reference, color="#888888", linestyle="--", linewidth=1.5, label="Reference function")

    axis.fill_between(x_grid, q025, q975, color="#7eaee6", alpha=0.18, label="95% predictive interval", zorder=0)
    axis.plot(x_grid, q025, color="#7eaee6", linewidth=0.9, alpha=0.9, zorder=1)
    axis.plot(x_grid, q975, color="#7eaee6", linewidth=0.9, alpha=0.9, zorder=1)
    axis.fill_between(x_grid, q25, q75, color="#4f8dd6", alpha=0.34, label="Interquartile range", zorder=2)
    axis.plot(x_grid, q25, color="#4f8dd6", linewidth=1.0, alpha=0.95, zorder=3)
    axis.plot(x_grid, q75, color="#4f8dd6", linewidth=1.0, alpha=0.95, zorder=3)
    axis.plot(x_grid, median, color="#c0392b", linewidth=2.0, label="Median prediction")

    axis.set_title("Bayesian Regression with Disjoint Observation Intervals")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    if show_summary_box:
        axis.text(
            0.02,
            0.02,
            (
                f"Observed mean IQR: {observed_iqr_mean:.3f}\n"
                f"Missing mean IQR: {missing_iqr_mean:.3f}\n"
                f"Observed mean 95% width: {observed_interval95_mean:.3f}\n"
                f"Missing mean 95% width: {missing_interval95_mean:.3f}"
            ),
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


@torch.no_grad()
def plot_from_checkpoint(args: argparse.Namespace) -> None:
    """Load a saved best checkpoint and generate a predictive plot without training."""

    if args.plot_path is None:
        raise ValueError("Provide --plot-path when using --plot-from-checkpoint.")

    checkpoint = torch.load(args.plot_from_checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = build_prior_from_config(checkpoint)
    scaler = RegressionStandardizer.from_config(checkpoint)
    likelihood_std_model = str(checkpoint.get("likelihood_std_model", "heteroscedastic"))
    global_likelihood_init_std = checkpoint.get("global_likelihood_init_std")
    if global_likelihood_init_std is None:
        global_likelihood_init_std = float(checkpoint.get("noise_std", 0.02))
    global_likelihood_prior_mean_std = checkpoint.get("global_likelihood_prior_mean_std")
    if global_likelihood_prior_mean_std is None:
        global_likelihood_prior_mean_std = float(global_likelihood_init_std)
    spline_knots = checkpoint.get("spline_knots")
    if spline_knots is None:
        spline_knots_original_units = checkpoint.get("spline_knots_original_units")
        if spline_knots_original_units is not None:
            spline_knots = normalize_input_locations(spline_knots_original_units, scaler)
    spline_coefficient_prior_sigma = checkpoint.get("spline_coefficient_prior_sigma")
    if spline_coefficient_prior_sigma is None:
        spline_coefficient_prior_sigma = 1.0
    spline_coefficient_prior_sigma = float(spline_coefficient_prior_sigma)
    rbf_centers = checkpoint.get("rbf_centers")
    if rbf_centers is None:
        rbf_centers_original_units = checkpoint.get("rbf_centers_original_units")
        if rbf_centers_original_units is not None:
            rbf_centers = normalize_input_locations(rbf_centers_original_units, scaler)
    rbf_lengthscale = checkpoint.get("rbf_lengthscale")
    if rbf_lengthscale is None:
        rbf_lengthscale_original_units = checkpoint.get("rbf_lengthscale_original_units")
        if rbf_lengthscale_original_units is not None:
            rbf_lengthscale = float(rbf_lengthscale_original_units) / float(scaler.input_std.squeeze().item())
    rbf_lengthscale_prior_sigma = checkpoint.get("rbf_lengthscale_prior_sigma")
    if rbf_lengthscale_prior_sigma is None:
        rbf_lengthscale_prior_sigma = 1.0
    rbf_lengthscale_prior_sigma = float(rbf_lengthscale_prior_sigma)
    rbf_coefficient_prior_sigma = checkpoint.get("rbf_coefficient_prior_sigma")
    if rbf_coefficient_prior_sigma is None:
        rbf_coefficient_prior_sigma = 1.0
    rbf_coefficient_prior_sigma = float(rbf_coefficient_prior_sigma)
    if likelihood_std_model == "spline" and spline_knots is None:
        raise ValueError("This spline checkpoint is missing saved spline-knot metadata.")
    if likelihood_std_model == "rbf" and (rbf_centers is None or rbf_lengthscale is None):
        raise ValueError("This RBF checkpoint is missing saved RBF metadata.")

    model = BayesianRegressor(
        hidden_dims=checkpoint["hidden_dims"],
        prior=prior,
        activation=str(checkpoint["activation"]),
        likelihood_std_model=likelihood_std_model,
        global_likelihood_init_std=float(global_likelihood_init_std),
        global_likelihood_prior_mean_std=float(global_likelihood_prior_mean_std),
        global_likelihood_prior_sigma=float(checkpoint.get("global_likelihood_prior_sigma", 1.0)),
        spline_knots=None if spline_knots is None else [float(knot) for knot in spline_knots],
        spline_coefficient_prior_sigma=spline_coefficient_prior_sigma,
        rbf_centers=None if rbf_centers is None else [float(center) for center in rbf_centers],
        rbf_lengthscale=1.0 if rbf_lengthscale is None else float(rbf_lengthscale),
        rbf_lengthscale_prior_sigma=rbf_lengthscale_prior_sigma,
        rbf_coefficient_prior_sigma=rbf_coefficient_prior_sigma,
        min_predictive_std=float(checkpoint["min_predictive_std"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    target_function_name = str(checkpoint.get("target_function", "oscillatory"))
    _, reference_mean_fn = resolve_regression_target(target_function_name)
    observed_intervals = [tuple(interval) for interval in checkpoint["observed_intervals"]]
    domain_min = float(checkpoint["domain_min"])
    domain_max = float(checkpoint["domain_max"])

    grid_inputs = torch.linspace(domain_min, domain_max, steps=args.grid_size, dtype=torch.float32).unsqueeze(1)
    scaled_grid_inputs = scaler.transform_inputs(grid_inputs)
    prediction_summary = predict_distribution(
        model=model,
        inputs=scaled_grid_inputs,
        num_samples=args.test_samples,
        device=device,
        scaler=scaler,
    )
    reference_curve = reference_mean_fn(grid_inputs)

    observed_inputs = checkpoint.get("observed_inputs")
    observed_targets = checkpoint.get("observed_targets")
    if observed_inputs is None or observed_targets is None:
        observed_inputs = checkpoint.get("train_inputs")
        observed_targets = checkpoint.get("train_targets")
    guide_inputs = checkpoint.get("guide_inputs")
    guide_targets = checkpoint.get("guide_targets")
    if observed_inputs is None or observed_targets is None:
        print("Checkpoint has no saved training observations; the plot will omit data points.")
        observed_inputs = None
        observed_targets = None

    print(f"Loaded checkpoint: {args.plot_from_checkpoint}")
    print(f"Observed intervals: {intervals_to_string(observed_intervals)}")
    print(f"Missing intervals: {intervals_to_string(complement_intervals(observed_intervals, domain_min, domain_max))}")
    print(f"Target function: {target_function_name}")
    print(f"Likelihood sigma model: {likelihood_std_model}")
    guide_points_outside_intervals = int(checkpoint.get("guide_points_outside_intervals", 0))
    guide_points_interior_gaps = int(checkpoint.get("guide_points_interior_gaps", 0))
    guide_region_mode = str(checkpoint.get("guide_region_mode", "all-gaps"))
    observed_interval_noise_stds = checkpoint.get("observed_interval_noise_stds")
    if guide_points_outside_intervals > 0 or guide_points_interior_gaps > 0:
        print(
            "Guide observations outside intervals: "
            f"{guide_points_outside_intervals} ({guide_region_mode}), "
            f"interior-gap guide points: {guide_points_interior_gaps}"
        )
    if observed_interval_noise_stds is not None:
        noise_values = ", ".join(f"{float(value):.4f}" for value in observed_interval_noise_stds)
        print(f"Observed-interval noise stds: {noise_values}")
    learned_global_std = model.current_global_likelihood_std()
    if learned_global_std is not None:
        learned_global_std_original = float(
            checkpoint.get("global_likelihood_std_original_units", to_original_target_std(learned_global_std, scaler))
        )
        print(
            "Global likelihood std: "
            f"{learned_global_std:.4f} normalized, "
            f"{learned_global_std_original:.4f} target units"
        )
    if likelihood_std_model == "spline":
        spline_knots_original_units = checkpoint.get("spline_knots_original_units")
        if spline_knots_original_units is not None:
            knot_values = ", ".join(f"{float(knot):.3f}" for knot in spline_knots_original_units)
            print(f"Spline likelihood knots: {knot_values}")
        print(f"Spline coefficient prior sigma: {spline_coefficient_prior_sigma:.4f}")
    if likelihood_std_model == "rbf":
        rbf_centers_original_units = checkpoint.get("rbf_centers_original_units")
        if rbf_centers_original_units is not None:
            center_values = ", ".join(f"{float(center):.3f}" for center in rbf_centers_original_units)
            print(f"RBF likelihood centers: {center_values}")
        rbf_lengthscale_original_units = checkpoint.get("rbf_lengthscale_original_units")
        if rbf_lengthscale_original_units is not None:
            print(
                "RBF likelihood lengthscale: "
                f"init/prior-mean {float(rbf_lengthscale_original_units):.4f}, "
                f"prior sigma on log-lengthscale {rbf_lengthscale_prior_sigma:.4f}"
            )
        learned_rbf_lengthscale_original = checkpoint.get("rbf_current_lengthscale_original_units")
        if learned_rbf_lengthscale_original is None:
            learned_rbf_lengthscale = model.current_rbf_lengthscale()
            if learned_rbf_lengthscale is not None:
                learned_rbf_lengthscale_original = learned_rbf_lengthscale * float(scaler.input_std.squeeze().item())
        if learned_rbf_lengthscale_original is not None:
            print(f"Learned RBF lengthscale: {float(learned_rbf_lengthscale_original):.4f}")
        print(f"RBF coefficient prior sigma: {rbf_coefficient_prior_sigma:.4f}")
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

    if args.coverage_eval_points > 0:
        target_fn, _ = resolve_regression_target(target_function_name)
        coverage_results = evaluate_generated_coverage(
            model=model,
            scaler=scaler,
            target_fn=target_fn,
            observed_intervals=observed_intervals,
            domain_min=domain_min,
            domain_max=domain_max,
            noise_std=float(checkpoint.get("noise_std", 0.02)),
            observed_interval_noise_stds=None
            if observed_interval_noise_stds is None
            else [float(value) for value in observed_interval_noise_stds],
            num_points=args.coverage_eval_points,
            predictive_samples=args.coverage_eval_samples,
            seed=args.coverage_eval_seed,
            device=device,
        )
        print_generated_coverage_results(coverage_results, num_points=args.coverage_eval_points)

    plot_predictions(
        plot_path=args.plot_path,
        observed_inputs=observed_inputs,
        observed_targets=observed_targets,
        guide_inputs=guide_inputs,
        guide_targets=guide_targets,
        grid_inputs=grid_inputs,
        reference_curve=reference_curve,
        summary=prediction_summary,
        observed_intervals=observed_intervals,
        quantile_space=args.plot_quantiles,
        shade_observed_intervals=args.shade_observed_intervals,
        show_summary_box=not args.hide_summary_box,
    )
    print(f"Saved predictive plot to {args.plot_path}")
