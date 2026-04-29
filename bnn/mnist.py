"""Bayes-by-Backprop MNIST model and training primitives.

Importing this module does not load MNIST data or create a model. Dataset access is
limited to build_dataloaders(); downloads require an explicit request.
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int) -> None:
    """Seed all random generators used by this script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BayesianLinear(nn.Module):
    """Linear layer with a diagonal Gaussian posterior over weights and bias."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the posterior parameters."""
        nn.init.xavier_uniform_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5.0)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        """Convert rho to a strictly positive standard deviation."""
        return F.softplus(rho)

    def _sample_parameter(self, mu: torch.Tensor, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from a Gaussian posterior using the reparameterization trick."""
        sigma = self._sigma(rho)
        epsilon = torch.randn_like(mu)
        sample = mu + sigma * epsilon
        return sample, sigma

    def _kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL divergence between q(w)=N(mu,sigma^2) and p(w)=N(0,prior_sigma^2)."""
        prior_sigma = sigma.new_tensor(self.prior_sigma)
        prior_variance = prior_sigma.pow(2)
        posterior_variance = sigma.pow(2)
        kl = torch.log(prior_sigma / sigma)
        kl += (posterior_variance + mu.pow(2)) / (2.0 * prior_variance)
        kl -= 0.5
        return kl.sum()

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Bayesian linear layer and return both output and KL contribution."""
        if sample:
            weight, weight_sigma = self._sample_parameter(self.weight_mu, self.weight_rho)
            bias, bias_sigma = self._sample_parameter(self.bias_mu, self.bias_rho)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            weight_sigma = self._sigma(self.weight_rho)
            bias_sigma = self._sigma(self.bias_rho)
        output = F.linear(x, weight, bias)
        kl = self._kl_divergence(self.weight_mu, weight_sigma)
        kl += self._kl_divergence(self.bias_mu, bias_sigma)
        return output, kl


class BayesianMLP(nn.Module):
    """Simple Bayesian multilayer perceptron for MNIST."""

    def __init__(self, hidden_dim: int = 400, prior_sigma: float = 1.0):
        super().__init__()
        self.layer1 = BayesianLinear(28 * 28, hidden_dim, prior_sigma)
        self.layer2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma)
        self.output_layer = BayesianLinear(hidden_dim, 10, prior_sigma)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full network and accumulate KL from every Bayesian layer."""
        x = x.view(x.size(0), -1)
        x, kl1 = self.layer1(x, sample=sample)
        x = F.relu(x)
        x, kl2 = self.layer2(x, sample=sample)
        x = F.relu(x)
        logits, kl3 = self.output_layer(x, sample=sample)
        total_kl = kl1 + kl2 + kl3
        return logits, total_kl


def predict_probabilities(model: nn.Module, images: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average predictions over several sampled networks."""
    probabilities = []
    kl_values = []
    for _ in range(num_samples):
        logits, kl = model(images, sample=True)
        probabilities.append(F.softmax(logits, dim=1))
        kl_values.append(kl)
    mean_probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
    mean_kl = torch.stack(kl_values, dim=0).mean()
    return mean_probabilities, mean_kl


def save_checkpoint(model: nn.Module, save_path: Path, hidden_dim: int, prior_sigma: float) -> None:
    """Save model weights together with the architecture settings."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "prior_sigma": prior_sigma,
    }
    torch.save(checkpoint, save_path)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_size: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    num_samples: int = 1,
) -> Dict[str, float]:
    """Run one full pass over a dataloader for either training or evaluation."""
    is_training = optimizer is not None
    model.train(mode=is_training)
    totals = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "correct": 0.0, "examples": 0.0}
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)
        if is_training:
            optimizer.zero_grad()
            sample_losses = []
            sample_nlls = []
            sample_kls = []
            sample_probabilities = []
            for _ in range(num_samples):
                logits, kl = model(images, sample=True)
                nll = F.cross_entropy(logits, targets, reduction="mean")
                # Scale KL by all training examples, not by the current batch.
                scaled_kl = kl / dataset_size
                loss = nll + scaled_kl
                sample_losses.append(loss)
                sample_nlls.append(nll)
                sample_kls.append(scaled_kl)
                sample_probabilities.append(F.softmax(logits, dim=1))
            mean_loss = torch.stack(sample_losses).mean()
            mean_nll = torch.stack(sample_nlls).mean()
            mean_kl = torch.stack(sample_kls).mean()
            mean_loss.backward()
            optimizer.step()
            mean_probabilities = torch.stack(sample_probabilities).mean(dim=0)
            predictions = mean_probabilities.argmax(dim=1)
        else:
            with torch.no_grad():
                probabilities, mean_kl_unscaled = predict_probabilities(model, images, num_samples=num_samples)
                predictions = probabilities.argmax(dim=1)
                # Score the averaged predictive distribution at evaluation time.
                mean_nll = F.nll_loss(probabilities.clamp_min(1e-8).log(), targets, reduction="mean")
                mean_kl = mean_kl_unscaled / dataset_size
                mean_loss = mean_nll + mean_kl
        totals["loss"] += mean_loss.item() * batch_size
        totals["nll"] += mean_nll.item() * batch_size
        totals["kl"] += mean_kl.item() * batch_size
        totals["correct"] += predictions.eq(targets).sum().item()
        totals["examples"] += batch_size
    total_examples = totals["examples"]
    return {
        "loss": totals["loss"] / total_examples,
        "nll": totals["nll"] / total_examples,
        "kl": totals["kl"] / total_examples,
        "accuracy": totals["correct"] / total_examples,
    }


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create MNIST loaders from local data, downloading only when requested."""
    from torchvision import datasets, transforms
    # Match the standard MNIST tensor range and normalization used by the GUI.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda}
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
