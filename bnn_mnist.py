"""Bayesian neural network for MNIST using Bayes by Backprop."""

# argparse is used to expose a small command line interface so the script can be
# run with different hyperparameters without editing the code.
import argparse

# random is used to seed Python's built-in random number generator.
import random

# Path makes filesystem paths clearer and more robust than raw strings.
from pathlib import Path

# These typing helpers make function signatures easier to understand.
from typing import Dict, Optional, Tuple

# NumPy is seeded for reproducibility because some environments mix NumPy and PyTorch.
import numpy as np

# torch is the main deep learning library used to build and train the model.
import torch

# nn contains PyTorch module classes such as nn.Module and nn.Parameter.
from torch import nn

# functional exposes stateless operations such as relu, linear, and cross_entropy.
from torch.nn import functional as F

# DataLoader handles batching and shuffling for the dataset.
from torch.utils.data import DataLoader

# torchvision provides the MNIST dataset and common image transforms.
from torchvision import datasets, transforms

# These constants are reused by both training code and the drawing app.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seed(seed: int) -> None:
    """Seed all random generators used by this script."""

    # Seed Python's own RNG.
    random.seed(seed)

    # Seed NumPy's RNG.
    np.random.seed(seed)

    # Seed PyTorch on CPU.
    torch.manual_seed(seed)

    # Seed PyTorch on every available GPU.
    torch.cuda.manual_seed_all(seed)


class BayesianLinear(nn.Module):
    """Linear layer with a diagonal Gaussian posterior over weights and bias."""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        # Initialize the parent nn.Module class.
        super().__init__()

        # Store the number of inputs seen by this layer.
        self.in_features = in_features

        # Store the number of outputs produced by this layer.
        self.out_features = out_features

        # Store the standard deviation of the Gaussian prior p(w).
        self.prior_sigma = prior_sigma

        # weight_mu is the learnable posterior mean for each weight.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))

        # weight_rho is transformed into a positive standard deviation via softplus.
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))

        # bias_mu is the learnable posterior mean for each bias term.
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # bias_rho plays the same role as weight_rho but for biases.
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        # Fill the parameters with reasonable initial values.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the posterior parameters."""

        # Xavier initialization gives the means a sensible starting scale.
        nn.init.xavier_uniform_(self.weight_mu)

        # A strongly negative rho makes the initial sigma very small.
        nn.init.constant_(self.weight_rho, -5.0)

        # Start biases near zero.
        nn.init.zeros_(self.bias_mu)

        # Use the same small initial uncertainty for bias parameters.
        nn.init.constant_(self.bias_rho, -5.0)

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        """Convert rho to a strictly positive standard deviation."""

        # softplus(x) = log(1 + exp(x)), which is always positive.
        return F.softplus(rho)

    def _sample_parameter(self, mu: torch.Tensor, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from a Gaussian posterior using the reparameterization trick."""

        # Turn rho into the posterior standard deviation sigma.
        sigma = self._sigma(rho)

        # Draw epsilon from a standard normal with the same shape as mu.
        epsilon = torch.randn_like(mu)

        # Sample w = mu + sigma * epsilon so gradients can flow through mu and sigma.
        sample = mu + sigma * epsilon

        # Return both the sampled parameter and sigma because sigma is needed for KL.
        return sample, sigma

    def _kl_divergence(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """KL divergence between q(w)=N(mu,sigma^2) and p(w)=N(0,prior_sigma^2)."""

        # Move the scalar prior standard deviation onto the same device and dtype as sigma.
        prior_sigma = sigma.new_tensor(self.prior_sigma)

        # Compute the prior variance once because it is used multiple times.
        prior_variance = prior_sigma.pow(2)

        # Compute the posterior variance for the current parameter tensor.
        posterior_variance = sigma.pow(2)

        # First term of the closed-form Gaussian KL.
        kl = torch.log(prior_sigma / sigma)

        # Second term combines the variance mismatch and the posterior mean magnitude.
        kl += (posterior_variance + mu.pow(2)) / (2.0 * prior_variance)

        # Final constant term in the KL expression.
        kl -= 0.5

        # Sum over every scalar parameter in this tensor.
        return kl.sum()

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the Bayesian linear layer and return both output and KL contribution."""

        # In Bayesian mode we draw a fresh set of weights and biases.
        if sample:
            # Sample a weight matrix from the learned posterior q(W).
            weight, weight_sigma = self._sample_parameter(self.weight_mu, self.weight_rho)

            # Sample a bias vector from the learned posterior q(b).
            bias, bias_sigma = self._sample_parameter(self.bias_mu, self.bias_rho)
        else:
            # Deterministic mode uses the posterior means directly.
            weight = self.weight_mu

            # Deterministic mode also uses the posterior mean biases.
            bias = self.bias_mu

            # Even without sampling, sigma is still needed to compute the KL term.
            weight_sigma = self._sigma(self.weight_rho)

            # Same reasoning for the bias posterior standard deviation.
            bias_sigma = self._sigma(self.bias_rho)

        # Apply the usual affine transformation xW^T + b.
        output = F.linear(x, weight, bias)

        # Add the KL cost from the weight posterior.
        kl = self._kl_divergence(self.weight_mu, weight_sigma)

        # Add the KL cost from the bias posterior.
        kl += self._kl_divergence(self.bias_mu, bias_sigma)

        # Return the layer output and how much this layer contributes to the ELBO penalty.
        return output, kl


class BayesianMLP(nn.Module):
    """Simple Bayesian multilayer perceptron for MNIST."""

    def __init__(self, hidden_dim: int = 400, prior_sigma: float = 1.0):
        # Initialize the parent nn.Module class.
        super().__init__()

        # First Bayesian layer maps the 784-pixel image vector to hidden_dim units.
        self.layer1 = BayesianLinear(28 * 28, hidden_dim, prior_sigma)

        # Second Bayesian hidden layer keeps the same hidden width.
        self.layer2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma)

        # Final Bayesian layer outputs 10 logits, one for each digit class.
        self.output_layer = BayesianLinear(hidden_dim, 10, prior_sigma)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the full network and accumulate KL from every Bayesian layer."""

        # Flatten each 28x28 image into a vector of length 784.
        x = x.view(x.size(0), -1)

        # Pass through the first Bayesian linear layer.
        x, kl1 = self.layer1(x, sample=sample)

        # Apply a nonlinearity after the first affine transformation.
        x = F.relu(x)

        # Pass through the second Bayesian linear layer.
        x, kl2 = self.layer2(x, sample=sample)

        # Apply another nonlinearity.
        x = F.relu(x)

        # Produce class logits from the final Bayesian layer.
        logits, kl3 = self.output_layer(x, sample=sample)

        # Sum the KL terms from all layers to build the full model KL.
        total_kl = kl1 + kl2 + kl3

        # Return logits for classification and total KL for the ELBO objective.
        return logits, total_kl


def predict_probabilities(model: nn.Module, images: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average predictions over several sampled networks."""

    # This list will store a probability vector for each Monte Carlo sample.
    probabilities = []

    # This list tracks KL values associated with each sampled forward pass.
    kl_values = []

    # Repeat the forward pass with newly sampled weights each time.
    for _ in range(num_samples):
        # Get logits and the KL term for one sampled model.
        logits, kl = model(images, sample=True)

        # Convert logits to class probabilities and save them.
        probabilities.append(F.softmax(logits, dim=1))

        # Save the KL term from this sample.
        kl_values.append(kl)

    # Stack the probability tensors and average across the sample dimension.
    mean_probabilities = torch.stack(probabilities, dim=0).mean(dim=0)

    # Average the KL terms for consistency with the averaged predictive distribution.
    mean_kl = torch.stack(kl_values, dim=0).mean()

    # Return the Bayesian predictive probabilities and mean KL.
    return mean_probabilities, mean_kl


def save_checkpoint(model: nn.Module, save_path: Path, hidden_dim: int, prior_sigma: float) -> None:
    """Save model weights together with the architecture settings."""

    # Pack everything needed to rebuild the model later.
    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_dim": hidden_dim,
        "prior_sigma": prior_sigma,
    }

    # Write the checkpoint to disk.
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

    # If an optimizer is provided, this is a training epoch; otherwise it is evaluation.
    is_training = optimizer is not None

    # Switch the module into train or eval mode.
    model.train(mode=is_training)

    # Accumulate epoch-level metrics in plain Python numbers.
    totals = {"loss": 0.0, "nll": 0.0, "kl": 0.0, "correct": 0.0, "examples": 0.0}

    # Loop over every minibatch from the dataloader.
    for images, targets in loader:
        # Move the images onto CPU or GPU.
        images = images.to(device)

        # Move the labels onto the same device.
        targets = targets.to(device)

        # Keep the batch size to scale summed metrics correctly.
        batch_size = images.size(0)

        # Training mode performs gradient updates.
        if is_training:
            # Clear old gradients before the new backward pass.
            optimizer.zero_grad()

            # Store the total ELBO loss from each Monte Carlo sample.
            sample_losses = []

            # Store only the data-fit term from each sample.
            sample_nlls = []

            # Store only the complexity penalty from each sample.
            sample_kls = []

            # Store predicted class probabilities from each sampled network.
            sample_probabilities = []

            # Draw multiple sampled networks if requested.
            for _ in range(num_samples):
                # Forward pass with freshly sampled Bayesian weights.
                logits, kl = model(images, sample=True)

                # Cross-entropy is the negative log-likelihood term for classification.
                nll = F.cross_entropy(logits, targets, reduction="mean")

                # Divide KL by dataset size so it acts like an average per-example penalty.
                scaled_kl = kl / dataset_size

                # Negative ELBO = expected negative log-likelihood + scaled KL.
                loss = nll + scaled_kl

                # Save the total loss for later averaging.
                sample_losses.append(loss)

                # Save the likelihood term for reporting.
                sample_nlls.append(nll)

                # Save the KL term for reporting.
                sample_kls.append(scaled_kl)

                # Save probabilities so accuracy uses the mean prediction over samples.
                sample_probabilities.append(F.softmax(logits, dim=1))

            # Average the sampled losses before backpropagation.
            mean_loss = torch.stack(sample_losses).mean()

            # Average the sampled data-fit terms for logging.
            mean_nll = torch.stack(sample_nlls).mean()

            # Average the sampled KL penalties for logging.
            mean_kl = torch.stack(sample_kls).mean()

            # Differentiate the average loss with respect to all model parameters.
            mean_loss.backward()

            # Apply one optimizer step.
            optimizer.step()

            # Average sampled probabilities to get a Bayesian predictive distribution.
            mean_probabilities = torch.stack(sample_probabilities).mean(dim=0)

            # Convert probabilities to hard class predictions for accuracy.
            predictions = mean_probabilities.argmax(dim=1)
        else:
            # Evaluation does not need gradients.
            with torch.no_grad():
                # Average predictions from multiple weight samples.
                probabilities, mean_kl_unscaled = predict_probabilities(model, images, num_samples=num_samples)

                # Use the most likely class as the prediction.
                predictions = probabilities.argmax(dim=1)

                # Compute NLL from the averaged predictive probabilities.
                mean_nll = F.nll_loss(probabilities.clamp_min(1e-8).log(), targets, reduction="mean")

                # Scale the KL term in the same way as during training.
                mean_kl = mean_kl_unscaled / dataset_size

                # Reconstruct the evaluation loss for reporting.
                mean_loss = mean_nll + mean_kl

        # Add the batch loss contribution to the epoch total.
        totals["loss"] += mean_loss.item() * batch_size

        # Add the batch NLL contribution to the epoch total.
        totals["nll"] += mean_nll.item() * batch_size

        # Add the batch KL contribution to the epoch total.
        totals["kl"] += mean_kl.item() * batch_size

        # Count how many predictions were correct in this batch.
        totals["correct"] += predictions.eq(targets).sum().item()

        # Track how many examples have been processed.
        totals["examples"] += batch_size

    # Read the total number of examples once for normalization.
    total_examples = totals["examples"]

    # Return average metrics for the whole epoch.
    return {
        "loss": totals["loss"] / total_examples,
        "nll": totals["nll"] / total_examples,
        "kl": totals["kl"] / total_examples,
        "accuracy": totals["correct"] / total_examples,
    }


def build_dataloaders(data_dir: Path, batch_size: int, test_batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders for MNIST."""

    # Compose the preprocessing pipeline applied to each image.
    transform = transforms.Compose(
        [
            # Convert PIL images to tensors in the range [0, 1].
            transforms.ToTensor(),

            # Normalize using the standard MNIST mean and standard deviation.
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )

    # Download or load the training split of MNIST.
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)

    # Download or load the test split of MNIST.
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # Check whether CUDA is available so pinned memory can be enabled when useful.
    use_cuda = torch.cuda.is_available()

    # Collect DataLoader keyword arguments in one place.
    loader_kwargs = {"num_workers": num_workers, "pin_memory": use_cuda}

    # Build the shuffled training loader.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)

    # Build the non-shuffled test loader.
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **loader_kwargs)

    # Return both dataloaders.
    return train_loader, test_loader


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments."""

    # Create the top-level CLI parser.
    parser = argparse.ArgumentParser(description="Bayesian neural network on MNIST with PyTorch.")

    # Choose where the MNIST files are stored.
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Where MNIST will be stored.")

    # Set the minibatch size used during training.
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")

    # Set the minibatch size used during evaluation.
    parser.add_argument("--test-batch-size", type=int, default=512, help="Evaluation batch size.")

    # Control how many epochs the model trains for.
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")

    # Control the width of the hidden layers.
    parser.add_argument("--hidden-dim", type=int, default=400, help="Hidden units in each Bayesian layer.")

    # Control the spread of the Gaussian prior p(w).
    parser.add_argument("--prior-sigma", type=float, default=1.0, help="Standard deviation of the Gaussian prior.")

    # Set the Adam learning rate.
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")

    # Set how many weight samples are used for each training batch.
    parser.add_argument("--train-samples", type=int, default=1, help="Weight samples per training batch.")

    # Set how many weight samples are averaged at test time.
    parser.add_argument("--test-samples", type=int, default=10, help="Weight samples for Bayesian prediction.")

    # Control dataloader worker processes; 0 is often simplest on Windows.
    parser.add_argument("--num-workers", type=int, default=0, help="Data loader workers. Keep 0 on Windows if needed.")

    # Set the random seed for reproducibility.
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Optionally save the trained weights at the end.
    parser.add_argument("--save-path", type=Path, default=None, help="Optional path to save model weights.")

    # Parse the provided CLI arguments and return them.
    return parser.parse_args()


def main() -> None:
    """Entry point for training and evaluating the Bayesian MNIST model."""

    # Read user-specified hyperparameters from the command line.
    args = parse_args()

    # Make the run reproducible across Python, NumPy, and PyTorch.
    set_seed(args.seed)

    # Prefer GPU if available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print the selected device so the user knows where training is running.
    print(f"Using device: {device}")

    # Create dataloaders for the MNIST training and test sets.
    train_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
    )

    # Instantiate the Bayesian neural network and move it to the target device.
    model = BayesianMLP(hidden_dim=args.hidden_dim, prior_sigma=args.prior_sigma).to(device)

    # Use Adam to optimize the variational parameters mu and rho.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Store the full number of training examples for KL scaling.
    train_size = len(train_loader.dataset)

    # Store the full number of test examples for consistent metric reporting.
    test_size = len(test_loader.dataset)

    # Repeat training and evaluation for the requested number of epochs.
    for epoch in range(1, args.epochs + 1):
        # Run one training epoch with gradient updates.
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            dataset_size=train_size,
            optimizer=optimizer,
            num_samples=args.train_samples,
        )

        # Run one evaluation epoch without parameter updates.
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            device=device,
            dataset_size=test_size,
            optimizer=None,
            num_samples=args.test_samples,
        )

        # Print a compact summary of the epoch results.
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} | train acc {train_metrics['accuracy']:.4%} | "
            f"test loss {test_metrics['loss']:.4f} | test acc {test_metrics['accuracy']:.4%}"
        )

    # Save the learned parameters if the user requested an output path.
    if args.save_path is not None:
        # Ensure the parent directory exists before saving.
        args.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the model weights and architecture metadata to disk.
        save_checkpoint(
            model=model,
            save_path=args.save_path,
            hidden_dim=args.hidden_dim,
            prior_sigma=args.prior_sigma,
        )

        # Confirm where the file was saved.
        print(f"Saved weights to {args.save_path}")


# Run main() only when this file is executed as a script.
if __name__ == "__main__":
    main()
