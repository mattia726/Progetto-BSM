"""CLI entrypoint for Bayesian neural-network regression on disjoint intervals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
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
    resolve_global_likelihood_init_std,
    resolve_global_likelihood_prior_mean_std,
    resolve_regression_target,
    resolve_spline_knots_original,
    normalize_input_locations,
    RegressionStandardizer,
    resolve_rbf_centers_original,
    resolve_rbf_lengthscale_original,
    set_seed,
    split_tensor_dataset,
    to_original_target_std,
    validate_intervals_in_domain,
)
from .bnn_regression_eval import (
    evaluate_generated_coverage,
    evaluate_regression,
    plot_from_checkpoint,
    plot_predictions,
    predict_distribution,
    print_generated_coverage_results,
    save_checkpoint,
    summarize_region_uncertainty,
)
from .bnn_regression_model import BayesianRegressor, build_prior, run_epoch


def parse_args() -> argparse.Namespace:
    """Define the command line interface."""

    parser = argparse.ArgumentParser(
        description="Bayesian neural network regression with configurable observed intervals."
    )

    parser.add_argument(
        "--preset",
        choices=["paper-figure5"],
        default=None,
        help="Optional preset that overrides the target function and interval/domain layout.",
    )
    parser.add_argument(
        "--plot-from-checkpoint",
        type=Path,
        default=None,
        help="Load a saved checkpoint and generate a plot without retraining.",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=[256, 256, 256], help="Hidden layer sizes, e.g. 128,128.")
    parser.add_argument("--activation", choices=["gelu", "relu"], default="relu", help="Hidden-layer activation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--train-samples", type=int, default=3, help="Weight samples per training batch.")
    parser.add_argument(
        "--likelihood-std-model",
        choices=["heteroscedastic", "global", "spline", "rbf"],
        default="global",
        help="Use an x-dependent predicted sigma, a single learned global sigma, a spline model, or an RBF expansion for log(sigma(x)).",
    )
    parser.add_argument(
        "--global-likelihood-init-std",
        type=float,
        default=None,
        help="Initial value of the learned global sigma, or the spline intercept sigma, in original target units. Defaults to a data-driven estimate from training observations.",
    )
    parser.add_argument(
        "--global-likelihood-prior-mean-std",
        type=float,
        default=None,
        help="Prior mean of the learned global sigma, or the spline intercept sigma, in original target units. Defaults to the resolved init value.",
    )
    parser.add_argument(
        "--global-likelihood-prior-sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the Gaussian prior on log(sigma) for the global likelihood option and the spline intercept.",
    )
    parser.add_argument(
        "--spline-knots",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated knot locations in original x units for the spline likelihood sigma model.",
    )
    parser.add_argument(
        "--spline-num-knots",
        type=int,
        default=5,
        help="Number of evenly spaced total knots used by the spline likelihood model when --spline-knots is omitted.",
    )
    parser.add_argument(
        "--spline-coefficient-prior-sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the independent zero-mean Gaussian priors on the spline log-sigma coefficients.",
    )
    parser.add_argument(
        "--rbf-num-centers",
        type=int,
        default=5,
        help="Number of equally spaced RBF centers used by the RBF likelihood sigma model.",
    )
    parser.add_argument(
        "--rbf-lengthscale",
        type=float,
        default=None,
        help="Optional Gaussian RBF lengthscale in original x units. Defaults to the average spacing between centers.",
    )
    parser.add_argument(
        "--rbf-lengthscale-prior-sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the Gaussian prior on log RBF lengthscale.",
    )
    parser.add_argument(
        "--rbf-coefficient-prior-sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the independent zero-mean Gaussian priors on the RBF log-sigma coefficients.",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=64,
        help="Weight samples used to estimate validation predictive NLL for early stopping.",
    )
    parser.add_argument("--test-samples", type=int, default=400, help="Weight samples for predictive summaries.")
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
        help="Stop when validation predictive NLL has not improved for this many epochs.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation predictive NLL improvement required to reset early stopping patience.",
    )
    parser.add_argument("--noise-std", type=float, default=0.02, help="Noise scale used to generate observations.")
    parser.add_argument(
        "--observed-interval-noise-stds",
        type=parse_float_list,
        default=None,
        help="Optional comma-separated noise standard deviations, one per observed interval, used only when generating observations inside those intervals.",
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
        help="Comma-separated disjoint intervals such as -4:-2,-0.5:0.75,1.75:3.5 or a single interval like 0.0:0.5.",
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
    parser.add_argument("--prior", choices=["normal", "spike-slab"], default="normal", help="Prior distribution over weights.")
    parser.add_argument("--prior-sigma", type=float, default=1.0, help="Standard deviation of the normal prior.")
    parser.add_argument("--prior-pi", type=float, default=0.5, help="Large-variance mixture weight for the spike-and-slab prior.")
    parser.add_argument("--prior-sigma1", type=float, default=1.0, help="Large-variance standard deviation for the spike-and-slab prior.")
    parser.add_argument("--prior-sigma2", type=float, default=0.1, help="Small-variance standard deviation for the spike-and-slab prior.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="How often to print training progress.")
    parser.add_argument(
        "--checkpoint-save-every",
        type=int,
        default=20,
        help="Flush the best checkpoint to disk at most once every this many epochs, plus a final save at the end.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("outputs") / "regression" / "weights" / "bnn_regression_best.pt",
        help="Path where the restored best checkpoint will be saved after training.",
    )
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional path for a predictive uncertainty plot.")
    parser.add_argument(
        "--plot-quantiles",
        choices=["function", "observation"],
        default="observation",
        help="Whether the plotted IQR should come from sampled latent functions or noisy observations.",
    )
    parser.add_argument(
        "--hide-summary-box",
        action="store_true",
        help="Omit the plot textbox that reports mean IQR and mean 95% widths.",
    )
    parser.add_argument(
        "--shade-observed-intervals",
        action="store_true",
        help="Shade the input intervals that contain training observations.",
    )
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
        help="Monte Carlo predictive samples used for generated coverage evaluation.",
    )
    parser.add_argument(
        "--coverage-eval-seed",
        type=int,
        default=42,
        help="Random seed used for the generated coverage evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Train a Bayesian regressor and report uncertainty in missing intervals."""

    args = parse_args()
    if args.plot_from_checkpoint is not None:
        if args.test_samples <= 0:
            raise ValueError("The number of Monte Carlo samples must be positive.")
        if args.grid_size <= 0:
            raise ValueError("The grid size must be positive.")
        if args.coverage_eval_points < 0:
            raise ValueError("The number of generated coverage-evaluation points must be non-negative.")
        if args.coverage_eval_samples <= 0:
            raise ValueError("The number of predictive samples for coverage evaluation must be positive.")
        plot_from_checkpoint(args)
        return

    args = apply_preset(args)

    if args.min_predictive_std <= 0.0:
        raise ValueError("The minimum predictive standard deviation must be positive.")
    if args.validation_fraction <= 0.0 or args.validation_fraction >= 1.0:
        raise ValueError("The validation fraction must be in the open interval (0, 1).")
    if args.validation_samples <= 0:
        raise ValueError("The validation samples must be positive.")
    if args.early_stopping_patience <= 0:
        raise ValueError("The early stopping patience must be positive.")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("The early stopping minimum delta must be non-negative.")
    if args.epochs <= 0:
        raise ValueError("The number of epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("The batch size must be positive.")
    if args.train_samples <= 0 or args.test_samples <= 0:
        raise ValueError("The number of Monte Carlo samples must be positive.")
    if args.coverage_eval_points < 0:
        raise ValueError("The number of generated coverage-evaluation points must be non-negative.")
    if args.coverage_eval_samples <= 0:
        raise ValueError("The number of predictive samples for coverage evaluation must be positive.")
    if args.global_likelihood_init_std is not None and args.global_likelihood_init_std <= 0.0:
        raise ValueError("The global likelihood initial standard deviation must be positive.")
    if args.global_likelihood_prior_mean_std is not None and args.global_likelihood_prior_mean_std <= 0.0:
        raise ValueError("The global likelihood prior-mean standard deviation must be positive.")
    if args.global_likelihood_prior_sigma <= 0.0:
        raise ValueError("The global likelihood prior sigma must be positive.")
    if args.spline_num_knots < 2:
        raise ValueError("The spline likelihood model needs at least two knots.")
    if args.spline_coefficient_prior_sigma <= 0.0:
        raise ValueError("The spline coefficient prior sigma must be positive.")
    if args.rbf_num_centers <= 0:
        raise ValueError("The RBF likelihood model needs at least one center.")
    if args.rbf_lengthscale is not None and args.rbf_lengthscale <= 0.0:
        raise ValueError("The RBF lengthscale must be positive.")
    if args.rbf_lengthscale_prior_sigma <= 0.0:
        raise ValueError("The RBF lengthscale prior sigma must be positive.")
    if args.rbf_coefficient_prior_sigma <= 0.0:
        raise ValueError("The RBF coefficient prior sigma must be positive.")
    if args.noise_std <= 0.0:
        raise ValueError("The default observation noise standard deviation must be positive.")
    if args.observed_interval_noise_stds is not None:
        if len(args.observed_interval_noise_stds) != len(args.observed_intervals):
            raise ValueError("Provide exactly one observed-interval noise standard deviation per observed interval.")
        if any(noise_std <= 0.0 for noise_std in args.observed_interval_noise_stds):
            raise ValueError("All observed-interval noise standard deviations must be positive.")
    if args.guide_points_outside_intervals < 0:
        raise ValueError("The number of guide points outside intervals must be non-negative.")
    if args.guide_points_interior_gaps < 0:
        raise ValueError("The number of guide points in interior gaps must be non-negative.")
    if args.checkpoint_save_every <= 0:
        raise ValueError("The checkpoint save interval must be positive.")

    validate_intervals_in_domain(args.observed_intervals, args.domain_min, args.domain_max)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prior = build_prior(args)
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
    scaler = RegressionStandardizer.fit(train_inputs, train_targets)
    global_likelihood_init_std, global_likelihood_init_std_original, global_likelihood_init_source = (
        resolve_global_likelihood_init_std(
            train_inputs=train_inputs,
            train_targets=train_targets,
            scaler=scaler,
            min_predictive_std=args.min_predictive_std,
            user_init_std_original_units=args.global_likelihood_init_std,
        )
    )
    global_likelihood_prior_mean_std, global_likelihood_prior_mean_std_original, global_likelihood_prior_mean_source = (
        resolve_global_likelihood_prior_mean_std(
            scaler=scaler,
            min_predictive_std=args.min_predictive_std,
            user_prior_mean_std_original_units=args.global_likelihood_prior_mean_std,
            default_prior_mean_std_original_units=global_likelihood_init_std_original,
        )
    )
    spline_knots_original = None
    spline_knots_normalized = None
    spline_knot_source = None
    rbf_centers_original = None
    rbf_centers_normalized = None
    rbf_center_source = None
    rbf_lengthscale_original = None
    rbf_lengthscale_normalized = None
    rbf_lengthscale_source = None
    if args.likelihood_std_model == "spline":
        spline_knots_original, spline_knot_source = resolve_spline_knots_original(
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            user_knots_original_units=args.spline_knots,
            num_knots=args.spline_num_knots,
        )
        spline_knots_normalized = normalize_input_locations(spline_knots_original, scaler)
    if args.likelihood_std_model == "rbf":
        rbf_centers_original = resolve_rbf_centers_original(
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            num_centers=args.rbf_num_centers,
        )
        rbf_center_source = "uniform"
        rbf_centers_normalized = normalize_input_locations(rbf_centers_original, scaler)
        rbf_lengthscale_original, rbf_lengthscale_source = resolve_rbf_lengthscale_original(
            centers_original_units=rbf_centers_original,
            user_lengthscale_original_units=args.rbf_lengthscale,
        )
        rbf_lengthscale_normalized = rbf_lengthscale_original / float(scaler.input_std.squeeze().item())
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
        likelihood_std_model=args.likelihood_std_model,
        global_likelihood_init_std=global_likelihood_init_std,
        global_likelihood_prior_mean_std=global_likelihood_prior_mean_std,
        global_likelihood_prior_sigma=args.global_likelihood_prior_sigma,
        spline_knots=spline_knots_normalized,
        spline_coefficient_prior_sigma=args.spline_coefficient_prior_sigma,
        rbf_centers=rbf_centers_normalized,
        rbf_lengthscale=1.0 if rbf_lengthscale_normalized is None else rbf_lengthscale_normalized,
        rbf_lengthscale_prior_sigma=args.rbf_lengthscale_prior_sigma,
        rbf_coefficient_prior_sigma=args.rbf_coefficient_prior_sigma,
        min_predictive_std=args.min_predictive_std,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    print(f"Observed intervals: {intervals_to_string(args.observed_intervals)}")
    print(f"Missing intervals: {intervals_to_string(complement_intervals(args.observed_intervals, args.domain_min, args.domain_max))}")
    if args.preset is not None:
        print(f"Preset: {args.preset}")
    print(f"Target function: {args.target_function}")
    print(f"Prior: {prior.config()}")
    print(f"Likelihood sigma model: {args.likelihood_std_model}")
    if args.guide_points_outside_intervals > 0 or args.guide_points_interior_gaps > 0:
        print(
            "Guide observations outside intervals: "
            f"{args.guide_points_outside_intervals} ({args.guide_region_mode}), "
            f"interior-gap guide points: {args.guide_points_interior_gaps}"
        )
    if args.observed_interval_noise_stds is not None:
        noise_values = ", ".join(f"{noise_std:.4f}" for noise_std in args.observed_interval_noise_stds)
        print(
            "Observed-interval noise stds: "
            f"{noise_values} (default outside intervals: {args.noise_std:.4f})"
        )
    if args.likelihood_std_model in {"global", "spline", "rbf"}:
        print(
            "Likelihood sigma baseline settings: "
            f"init {global_likelihood_init_std:.4f} normalized, "
            f"{global_likelihood_init_std_original:.4f} target units "
            f"({global_likelihood_init_source}); "
            f"prior mean {global_likelihood_prior_mean_std:.4f} normalized, "
            f"{global_likelihood_prior_mean_std_original:.4f} target units "
            f"({global_likelihood_prior_mean_source}); "
            f"log-sigma prior std {args.global_likelihood_prior_sigma:.4f}"
        )
    if args.likelihood_std_model == "spline":
        if spline_knots_original is None:
            raise RuntimeError("Spline likelihood knots were not resolved.")
        knot_values = ", ".join(f"{knot:.3f}" for knot in spline_knots_original)
        print(
            "Spline likelihood sigma settings: "
            f"{len(spline_knots_original)} knots ({spline_knot_source}) at [{knot_values}]; "
            f"coefficient prior std {args.spline_coefficient_prior_sigma:.4f}"
        )
    if args.likelihood_std_model == "rbf":
        if rbf_centers_original is None or rbf_lengthscale_original is None:
            raise RuntimeError("RBF likelihood centers or lengthscale were not resolved.")
        center_values = ", ".join(f"{center:.3f}" for center in rbf_centers_original)
        print(
            "RBF likelihood sigma settings: "
            f"{len(rbf_centers_original)} centers ({rbf_center_source}) at [{center_values}]; "
            f"lengthscale {rbf_lengthscale_original:.4f} ({rbf_lengthscale_source}); "
            f"log-lengthscale prior std {args.rbf_lengthscale_prior_sigma:.4f}; "
            f"coefficient prior std {args.rbf_coefficient_prior_sigma:.4f}"
        )
    print(f"Train / validation points: {len(train_dataset)} / {len(validation_dataset)}")
    print("Early stopping metric: validation predictive NLL")

    best_validation_predictive_nll = float("inf")
    best_epoch = 0
    best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    epochs_without_improvement = 0
    best_checkpoint_dirty = False

    def persist_best_checkpoint(force: bool = False) -> None:
        """Write the current best model to disk so interrupted runs remain usable."""

        nonlocal best_checkpoint_dirty
        if args.save_path is None:
            return
        if not force and (not best_checkpoint_dirty):
            return

        current_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        model.load_state_dict(best_state_dict)

        save_checkpoint(
            model=model,
            save_path=args.save_path,
            preset=args.preset,
            target_function=args.target_function,
            likelihood_std_model=args.likelihood_std_model,
            global_likelihood_init_std=global_likelihood_init_std,
            global_likelihood_init_std_original_units=global_likelihood_init_std_original,
            global_likelihood_init_source=global_likelihood_init_source,
            global_likelihood_prior_mean_std=global_likelihood_prior_mean_std,
            global_likelihood_prior_mean_std_original_units=global_likelihood_prior_mean_std_original,
            global_likelihood_prior_mean_source=global_likelihood_prior_mean_source,
            global_likelihood_prior_sigma=args.global_likelihood_prior_sigma,
            spline_knots=spline_knots_normalized,
            spline_knots_original_units=spline_knots_original,
            spline_knot_source=spline_knot_source,
            spline_coefficient_prior_sigma=args.spline_coefficient_prior_sigma if args.likelihood_std_model == "spline" else None,
            rbf_centers=rbf_centers_normalized,
            rbf_centers_original_units=rbf_centers_original,
            rbf_center_source=rbf_center_source,
            rbf_lengthscale=rbf_lengthscale_normalized,
            rbf_lengthscale_original_units=rbf_lengthscale_original,
            rbf_lengthscale_source=rbf_lengthscale_source,
            rbf_lengthscale_prior_sigma=args.rbf_lengthscale_prior_sigma if args.likelihood_std_model == "rbf" else None,
            rbf_coefficient_prior_sigma=args.rbf_coefficient_prior_sigma if args.likelihood_std_model == "rbf" else None,
            train_inputs=train_inputs,
            train_targets=train_targets,
            observed_inputs=observed_inputs,
            observed_targets=observed_targets,
            guide_inputs=guide_inputs,
            guide_targets=guide_targets,
            guide_points_outside_intervals=args.guide_points_outside_intervals,
            guide_points_interior_gaps=args.guide_points_interior_gaps,
            guide_region_mode=args.guide_region_mode,
            observed_interval_noise_stds=args.observed_interval_noise_stds,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            observed_intervals=args.observed_intervals,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            min_predictive_std=args.min_predictive_std,
            noise_std=args.noise_std,
            validation_fraction=args.validation_fraction,
            validation_samples=args.validation_samples,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            best_epoch=best_epoch,
            best_validation_predictive_nll=best_validation_predictive_nll,
            scaler=scaler,
            prior_config=prior.config(),
        )
        model.load_state_dict(current_state_dict)
        best_checkpoint_dirty = False

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            dataset_size=len(train_dataset),
            scaler=scaler,
            num_samples=args.train_samples,
            optimizer=optimizer,
        )
        validation_metrics = evaluate_regression(
            model=model,
            loader=validation_loader,
            device=device,
            scaler=scaler,
            num_samples=args.validation_samples,
        )

        improvement_threshold = best_validation_predictive_nll - args.early_stopping_min_delta
        if validation_metrics["predictive_nll"] < improvement_threshold:
            best_validation_predictive_nll = validation_metrics["predictive_nll"]
            best_epoch = epoch
            best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
            best_checkpoint_dirty = True
        else:
            epochs_without_improvement += 1

        if epoch % args.checkpoint_save_every == 0:
            persist_best_checkpoint()

        should_log = epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0
        if should_log:
            print(
                f"Epoch {epoch:04d} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"train nll {train_metrics['nll']:.4f} | "
                f"train mse {train_metrics['mse']:.4f} | "
                f"val pred-nll {validation_metrics['predictive_nll']:.4f} | "
                f"val pred std {validation_metrics['predictive_std']:.4f}"
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:04d} | "
                f"best epoch {best_epoch:04d} | "
                f"best val pred-nll {best_validation_predictive_nll:.4f}"
            )
            break

    if best_epoch == 0:
        best_epoch = args.epochs
        best_validation_predictive_nll = validation_metrics["predictive_nll"]
        best_state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        best_checkpoint_dirty = True

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
        f"val pred-nll {final_validation_metrics['predictive_nll']:.4f}"
    )
    learned_global_std = model.current_global_likelihood_std()
    if learned_global_std is not None:
        learned_global_std_original = to_original_target_std(learned_global_std, scaler)
        print(
            "Restored global likelihood std: "
            f"{learned_global_std:.4f} normalized, "
            f"{learned_global_std_original:.4f} target units"
        )

    grid_inputs = torch.linspace(args.domain_min, args.domain_max, steps=args.grid_size, dtype=torch.float32).unsqueeze(1)
    scaled_grid_inputs = scaler.transform_inputs(grid_inputs)
    reference_curve = reference_mean_fn(grid_inputs)
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

    if args.coverage_eval_points > 0:
        coverage_results = evaluate_generated_coverage(
            model=model,
            scaler=scaler,
            target_fn=target_fn,
            observed_intervals=args.observed_intervals,
            domain_min=args.domain_min,
            domain_max=args.domain_max,
            noise_std=args.noise_std,
            observed_interval_noise_stds=args.observed_interval_noise_stds,
            num_points=args.coverage_eval_points,
            predictive_samples=args.coverage_eval_samples,
            seed=args.coverage_eval_seed,
            device=device,
        )
        print_generated_coverage_results(coverage_results, num_points=args.coverage_eval_points)

    if args.save_path is not None:
        persist_best_checkpoint(force=True)
        print(f"Saved checkpoint to {args.save_path}")

    if args.plot_path is not None:
        plot_predictions(
            plot_path=args.plot_path,
            observed_inputs=observed_inputs,
            observed_targets=observed_targets,
            guide_inputs=guide_inputs,
            guide_targets=guide_targets,
            grid_inputs=grid_inputs,
            reference_curve=reference_curve,
            summary=prediction_summary,
            observed_intervals=args.observed_intervals,
            quantile_space=args.plot_quantiles,
            shade_observed_intervals=args.shade_observed_intervals,
            show_summary_box=not args.hide_summary_box,
        )
        print(f"Saved predictive plot to {args.plot_path}")


if __name__ == "__main__":
    main()
