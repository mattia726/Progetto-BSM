"""Model definitions and training utilities for Bayesian 1D regression."""

from __future__ import annotations

import argparse
import math
from typing import Dict, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bnn_regression_data import RegressionStandardizer


def gaussian_log_prob(value: torch.Tensor, mean: torch.Tensor | float, sigma: torch.Tensor) -> torch.Tensor:
    """Elementwise Gaussian log-density."""

    variance = sigma.pow(2)
    return -0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - (value - mean).pow(2) / (2.0 * variance)


def natural_cubic_spline_basis(x: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Return the restricted natural cubic spline basis without the intercept."""

    if knots.ndim != 1:
        raise ValueError("Spline knots must be a one-dimensional tensor.")
    if knots.numel() < 2:
        raise ValueError("At least two spline knots are required.")

    flat_x = x.reshape(-1, 1)
    knots = knots.to(device=flat_x.device, dtype=flat_x.dtype)
    basis_terms = [flat_x]
    if knots.numel() == 2:
        return torch.cat(basis_terms, dim=1)

    last_knot = knots[-1]
    penultimate_knot = knots[-2]
    penultimate_denominator = last_knot - penultimate_knot
    if penultimate_denominator <= 0:
        raise ValueError("Spline knots must be strictly increasing.")

    def truncated_cubic(center: torch.Tensor) -> torch.Tensor:
        return torch.clamp(flat_x - center, min=0.0).pow(3)

    tail_term = (truncated_cubic(penultimate_knot) - truncated_cubic(last_knot)) / penultimate_denominator
    for knot in knots[:-2]:
        denominator = last_knot - knot
        if denominator <= 0:
            raise ValueError("Spline knots must be strictly increasing.")
        raw_term = (truncated_cubic(knot) - truncated_cubic(last_knot)) / denominator
        basis_terms.append(raw_term - tail_term)

    return torch.cat(basis_terms, dim=1)


def gaussian_rbf_basis(x: torch.Tensor, centers: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    """Return Gaussian radial basis functions evaluated at one-dimensional inputs."""

    if centers.ndim != 1:
        raise ValueError("RBF centers must be a one-dimensional tensor.")
    if centers.numel() < 1:
        raise ValueError("At least one RBF center is required.")
    if torch.any(centers[1:] <= centers[:-1]):
        raise ValueError("RBF centers must be strictly increasing.")

    flat_x = x.reshape(-1, 1)
    centers = centers.to(device=flat_x.device, dtype=flat_x.dtype).reshape(1, -1)
    lengthscale = lengthscale.to(device=flat_x.device, dtype=flat_x.dtype)
    if torch.any(lengthscale <= 0):
        raise ValueError("The RBF lengthscale must be positive.")

    scaled_distance = (flat_x - centers) / lengthscale
    return torch.exp(-0.5 * scaled_distance.pow(2))


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


class GlobalLikelihoodSigma(nn.Module):
    """Single learned likelihood standard deviation shared across all inputs."""

    def __init__(self, init_std: float, prior_mean_std: float, prior_sigma: float) -> None:
        super().__init__()

        if init_std <= 0.0:
            raise ValueError("The global likelihood initial standard deviation must be positive.")
        if prior_mean_std <= 0.0:
            raise ValueError("The global likelihood prior-mean standard deviation must be positive.")
        if prior_sigma <= 0.0:
            raise ValueError("The global likelihood prior sigma must be positive.")

        self.log_std = nn.Parameter(torch.tensor(math.log(init_std), dtype=torch.float32))
        self.prior_mean_log_std = math.log(prior_mean_std)
        self.prior_sigma = prior_sigma

    def forward(self, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_std = self.log_std.to(device=reference.device, dtype=reference.dtype)
        sigma = torch.exp(log_std).expand_as(reference)
        prior_mean = reference.new_tensor(self.prior_mean_log_std)
        prior_sigma = reference.new_tensor(self.prior_sigma)
        prior_penalty = -gaussian_log_prob(log_std, prior_mean, prior_sigma)
        return sigma, prior_penalty

    def current_std(self) -> float:
        return float(torch.exp(self.log_std.detach()).item())


class SplineLikelihoodSigma(nn.Module):
    """Natural cubic spline model for log(sigma(x)) with independent coefficient priors."""

    def __init__(
        self,
        init_std: float,
        prior_mean_std: float,
        prior_sigma: float,
        knots: Sequence[float],
        coefficient_prior_sigma: float,
    ) -> None:
        super().__init__()

        if init_std <= 0.0:
            raise ValueError("The spline likelihood initial standard deviation must be positive.")
        if prior_mean_std <= 0.0:
            raise ValueError("The spline likelihood prior-mean standard deviation must be positive.")
        if prior_sigma <= 0.0:
            raise ValueError("The spline likelihood intercept prior sigma must be positive.")
        if coefficient_prior_sigma <= 0.0:
            raise ValueError("The spline likelihood coefficient prior sigma must be positive.")
        if len(knots) < 2:
            raise ValueError("The spline likelihood model needs at least two knots.")

        knot_tensor = torch.tensor(knots, dtype=torch.float32)
        if torch.any(knot_tensor[1:] <= knot_tensor[:-1]):
            raise ValueError("Spline knots must be strictly increasing.")

        basis_size = natural_cubic_spline_basis(
            torch.zeros(1, 1, dtype=torch.float32),
            knot_tensor,
        ).size(1)
        self.register_buffer("knots", knot_tensor)
        self.log_std_intercept = nn.Parameter(torch.tensor(math.log(init_std), dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(basis_size, dtype=torch.float32))
        self.intercept_prior_mean_log_std = math.log(prior_mean_std)
        self.intercept_prior_sigma = prior_sigma
        self.coefficient_prior_sigma = coefficient_prior_sigma

    def forward(self, reference_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_x = reference_x.reshape(-1, 1)
        knots = self.knots.to(device=flat_x.device, dtype=flat_x.dtype)
        basis = natural_cubic_spline_basis(flat_x, knots)
        intercept = self.log_std_intercept.to(device=flat_x.device, dtype=flat_x.dtype)
        coefficients = self.coefficients.to(device=flat_x.device, dtype=flat_x.dtype)
        log_std = intercept + basis.matmul(coefficients.unsqueeze(-1))
        sigma = torch.exp(log_std)

        intercept_prior_mean = flat_x.new_tensor(self.intercept_prior_mean_log_std)
        intercept_prior_sigma = flat_x.new_tensor(self.intercept_prior_sigma)
        coefficient_prior_sigma = flat_x.new_tensor(self.coefficient_prior_sigma)
        prior_penalty = -gaussian_log_prob(intercept, intercept_prior_mean, intercept_prior_sigma)
        prior_penalty = prior_penalty - gaussian_log_prob(coefficients, 0.0, coefficient_prior_sigma).sum()
        return sigma, prior_penalty


class RBFLikelihoodSigma(nn.Module):
    """Gaussian RBF model for log(sigma(x)) with independent coefficient priors."""

    def __init__(
        self,
        init_std: float,
        prior_mean_std: float,
        prior_sigma: float,
        centers: Sequence[float],
        lengthscale: float,
        lengthscale_prior_sigma: float,
        coefficient_prior_sigma: float,
    ) -> None:
        super().__init__()

        if init_std <= 0.0:
            raise ValueError("The RBF likelihood initial standard deviation must be positive.")
        if prior_mean_std <= 0.0:
            raise ValueError("The RBF likelihood prior-mean standard deviation must be positive.")
        if prior_sigma <= 0.0:
            raise ValueError("The RBF likelihood intercept prior sigma must be positive.")
        if coefficient_prior_sigma <= 0.0:
            raise ValueError("The RBF likelihood coefficient prior sigma must be positive.")
        if lengthscale <= 0.0:
            raise ValueError("The RBF lengthscale must be positive.")
        if lengthscale_prior_sigma <= 0.0:
            raise ValueError("The RBF lengthscale prior sigma must be positive.")
        if len(centers) < 1:
            raise ValueError("The RBF likelihood model needs at least one center.")

        center_tensor = torch.tensor(centers, dtype=torch.float32)
        if center_tensor.numel() > 1 and torch.any(center_tensor[1:] <= center_tensor[:-1]):
            raise ValueError("RBF centers must be strictly increasing.")

        self.register_buffer("centers", center_tensor)
        self.log_std_intercept = nn.Parameter(torch.tensor(math.log(init_std), dtype=torch.float32))
        self.log_lengthscale = nn.Parameter(torch.tensor(math.log(lengthscale), dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(center_tensor.numel(), dtype=torch.float32))
        self.intercept_prior_mean_log_std = math.log(prior_mean_std)
        self.intercept_prior_sigma = prior_sigma
        self.lengthscale_prior_mean_log = math.log(lengthscale)
        self.lengthscale_prior_sigma = lengthscale_prior_sigma
        self.coefficient_prior_sigma = coefficient_prior_sigma

    def forward(self, reference_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_x = reference_x.reshape(-1, 1)
        log_lengthscale = self.log_lengthscale.to(device=flat_x.device, dtype=flat_x.dtype)
        lengthscale = torch.exp(log_lengthscale)
        basis = gaussian_rbf_basis(flat_x, self.centers, lengthscale)
        intercept = self.log_std_intercept.to(device=flat_x.device, dtype=flat_x.dtype)
        coefficients = self.coefficients.to(device=flat_x.device, dtype=flat_x.dtype)
        log_std = intercept + basis.matmul(coefficients.unsqueeze(-1))
        sigma = torch.exp(log_std)

        intercept_prior_mean = flat_x.new_tensor(self.intercept_prior_mean_log_std)
        intercept_prior_sigma = flat_x.new_tensor(self.intercept_prior_sigma)
        lengthscale_prior_mean = flat_x.new_tensor(self.lengthscale_prior_mean_log)
        lengthscale_prior_sigma = flat_x.new_tensor(self.lengthscale_prior_sigma)
        coefficient_prior_sigma = flat_x.new_tensor(self.coefficient_prior_sigma)
        prior_penalty = -gaussian_log_prob(intercept, intercept_prior_mean, intercept_prior_sigma)
        prior_penalty = prior_penalty - gaussian_log_prob(log_lengthscale, lengthscale_prior_mean, lengthscale_prior_sigma)
        prior_penalty = prior_penalty - gaussian_log_prob(coefficients, 0.0, coefficient_prior_sigma).sum()
        return sigma, prior_penalty

    def current_lengthscale(self) -> float:
        return float(torch.exp(self.log_lengthscale.detach()).item())


class BayesianRegressor(nn.Module):
    """Small Bayesian MLP that predicts a Gaussian mean and either local or global scale."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        prior: PriorDistribution,
        activation: str = "relu",
        likelihood_std_model: str = "heteroscedastic",
        global_likelihood_init_std: float = 0.02,
        global_likelihood_prior_mean_std: float = 0.02,
        global_likelihood_prior_sigma: float = 1.0,
        spline_knots: Sequence[float] | None = None,
        spline_coefficient_prior_sigma: float = 1.0,
        rbf_centers: Sequence[float] | None = None,
        rbf_lengthscale: float = 1.0,
        rbf_lengthscale_prior_sigma: float = 1.0,
        rbf_coefficient_prior_sigma: float = 1.0,
        min_predictive_std: float = 1e-3,
    ) -> None:
        super().__init__()

        if min_predictive_std <= 0.0:
            raise ValueError("The minimum predictive standard deviation must be positive.")

        self.min_predictive_std = min_predictive_std
        self.likelihood_std_model = likelihood_std_model
        output_dim = 2 if likelihood_std_model == "heteroscedastic" else 1
        layer_sizes = [1, *hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            BayesianLinear(in_features, out_features, prior=prior)
            for in_features, out_features in zip(layer_sizes, layer_sizes[1:])
        )
        self.global_likelihood_sigma: GlobalLikelihoodSigma | None = None
        self.spline_likelihood_sigma: SplineLikelihoodSigma | None = None
        self.rbf_likelihood_sigma: RBFLikelihoodSigma | None = None
        if likelihood_std_model == "global":
            self.global_likelihood_sigma = GlobalLikelihoodSigma(
                init_std=global_likelihood_init_std,
                prior_mean_std=global_likelihood_prior_mean_std,
                prior_sigma=global_likelihood_prior_sigma,
            )
        elif likelihood_std_model == "spline":
            if spline_knots is None:
                raise ValueError("Provide spline knots when likelihood_std_model='spline'.")
            self.spline_likelihood_sigma = SplineLikelihoodSigma(
                init_std=global_likelihood_init_std,
                prior_mean_std=global_likelihood_prior_mean_std,
                prior_sigma=global_likelihood_prior_sigma,
                knots=spline_knots,
                coefficient_prior_sigma=spline_coefficient_prior_sigma,
            )
        elif likelihood_std_model == "rbf":
            if rbf_centers is None:
                raise ValueError("Provide RBF centers when likelihood_std_model='rbf'.")
            self.rbf_likelihood_sigma = RBFLikelihoodSigma(
                init_std=global_likelihood_init_std,
                prior_mean_std=global_likelihood_prior_mean_std,
                prior_sigma=global_likelihood_prior_sigma,
                centers=rbf_centers,
                lengthscale=rbf_lengthscale,
                lengthscale_prior_sigma=rbf_lengthscale_prior_sigma,
                coefficient_prior_sigma=rbf_coefficient_prior_sigma,
            )
        elif likelihood_std_model != "heteroscedastic":
            raise ValueError(f"Unsupported likelihood sigma model '{likelihood_std_model}'.")

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_inputs = x
        total_kl = x.new_zeros(())

        for layer in self.layers[:-1]:
            x, kl = layer(x, sample=sample)
            total_kl = total_kl + kl
            x = self.activation(x)

        output, kl = self.layers[-1](x, sample=sample)
        total_kl = total_kl + kl
        mean = output[:, :1]
        if self.likelihood_std_model == "heteroscedastic":
            predictive_std = F.softplus(output[:, 1:]) + self.min_predictive_std
        elif self.likelihood_std_model == "global":
            if self.global_likelihood_sigma is None:
                raise RuntimeError("The global likelihood sigma module is not initialized.")
            predictive_std, sigma_prior_penalty = self.global_likelihood_sigma(mean)
            total_kl = total_kl + sigma_prior_penalty
        elif self.likelihood_std_model == "spline":
            if self.spline_likelihood_sigma is None:
                raise RuntimeError("The spline likelihood sigma module is not initialized.")
            predictive_std, sigma_prior_penalty = self.spline_likelihood_sigma(raw_inputs)
            total_kl = total_kl + sigma_prior_penalty
        else:
            if self.rbf_likelihood_sigma is None:
                raise RuntimeError("The RBF likelihood sigma module is not initialized.")
            predictive_std, sigma_prior_penalty = self.rbf_likelihood_sigma(raw_inputs)
            total_kl = total_kl + sigma_prior_penalty
        return mean, predictive_std, total_kl

    def current_global_likelihood_std(self) -> float | None:
        if self.global_likelihood_sigma is None:
            return None
        return self.global_likelihood_sigma.current_std()

    def current_rbf_lengthscale(self) -> float | None:
        if self.rbf_likelihood_sigma is None:
            return None
        return self.rbf_likelihood_sigma.current_lengthscale()


def gaussian_nll(mean: torch.Tensor, predictive_std: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under a Gaussian observation model."""

    safe_std = predictive_std.clamp_min(1e-8)
    variance = safe_std.pow(2)
    return 0.5 * (((target - mean) ** 2) / variance + torch.log(2.0 * math.pi * variance))


def run_epoch(
    model: BayesianRegressor,
    loader: DataLoader,
    device: torch.device,
    dataset_size: int,
    scaler: RegressionStandardizer,
    num_samples: int,
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
                scaled_kl = kl / dataset_size
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


def build_prior(args: argparse.Namespace) -> PriorDistribution:
    """Instantiate the requested prior distribution."""

    if args.prior == "normal":
        return NormalPrior(sigma=args.prior_sigma)
    if args.prior == "spike-slab":
        return SpikeAndSlabPrior(pi=args.prior_pi, sigma1=args.prior_sigma1, sigma2=args.prior_sigma2)
    raise ValueError(f"Unsupported prior '{args.prior}'.")


def build_prior_from_config(config: Dict[str, float | str | torch.Tensor]) -> PriorDistribution:
    """Instantiate a prior from checkpoint metadata."""

    prior_name = str(config["prior"])
    if prior_name == "normal":
        return NormalPrior(sigma=float(config["prior_sigma"]))
    if prior_name == "spike-slab":
        return SpikeAndSlabPrior(
            pi=float(config["prior_pi"]),
            sigma1=float(config["prior_sigma1"]),
            sigma2=float(config["prior_sigma2"]),
        )
    raise ValueError(f"Unsupported prior '{prior_name}'.")
