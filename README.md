# Bayesian Neural Network for MNIST

This project contains a small PyTorch implementation of a Bayesian neural network for the MNIST digit dataset.

The model uses variational inference with Bayes by Backprop:

- each weight and bias has a Gaussian posterior with learnable `mu` and `rho`
- `sigma = softplus(rho)` keeps the standard deviation positive
- weights are sampled with the reparameterization trick
- training minimizes the negative ELBO:
  `cross_entropy + KL(q(w) || p(w)) / N`

## Files

- `bnn_mnist.py`: training and evaluation script
- `bnn_regression.py`: compatibility launcher for the regression CLI
- `regression/bnn_regression.py`: CLI entrypoint for Bayes-by-Backprop regression on disjoint observation intervals
- `regression/bnn_regression_data.py`: interval handling, synthetic targets, dataset builders, and normalization helpers
- `regression/bnn_regression_model.py`: priors, Bayesian layers, spline/global sigma models, and training utilities
- `regression/bnn_regression_eval.py`: predictive evaluation, checkpointing, plotting, and checkpoint-only replotting
- `draw_digit_app.py`: Tkinter app to draw a digit and classify it with the trained BNN
- `requirements.txt`: minimal dependencies

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python bnn_mnist.py --epochs 5 --test-samples 10
```

Useful options:

- `--hidden-dim 400`
- `--prior-sigma 1.0`
- `--train-samples 1`
- `--test-samples 10`
- `--save-path checkpoints/bnn_mnist.pt`

## Drawing App

First train and save a checkpoint:

```bash
python bnn_mnist.py --epochs 5 --save-path checkpoints/bnn_mnist.pt
```

Then launch the desktop app:

```bash
python draw_digit_app.py --checkpoint checkpoints/bnn_mnist.pt --test-samples 25
```

In the app:

- draw a digit with the mouse on the black canvas
- click `Predict`
- the app converts the drawing to a centered 28x28 MNIST-style image
- the BNN averages multiple weight samples and shows the probability for each digit

## Notes

- `test-samples` controls how many weight samples are averaged at prediction time.
- the drawing app needs a saved checkpoint before it can make predictions.
- `num-workers` defaults to `0`, which is usually the safest choice on Windows.
- This is a clear teaching implementation, not an optimized research codebase.

## Regression Example

`bnn_regression.py` trains a Bayesian regressor on a synthetic target where observations are only available inside one or more observed intervals. It then evaluates the predictive distribution across the full domain and reports uncertainty inside both observed and missing regions.

The regression model:

- samples training inputs only from configurable observed intervals
- uses Bayes by Backprop with a diagonal Gaussian posterior
- predicts a Gaussian mean and scale for each input
- reports Monte Carlo uncertainty summaries over the full domain
- supports a `normal` prior by default and an optional spike-and-slab prior
- supports both the default oscillatory target and the Figure 5 paper regression target

Example with the default normal prior:

```bash
python bnn_regression.py --epochs 500 --plot-path outputs/regression/plots/bnn_regression.png
```

Example with custom intervals:

```bash
python bnn_regression.py --epochs 500 --observed-intervals=-4:-2.5,-0.4:0.5,2.0:3.2 --plot-path outputs/regression/plots/bnn_regression_custom.png
```

Example with the paper regression target and a single observed interval:

```bash
python bnn_regression.py --target-function paper --observed-intervals=0.0:0.5 --domain-min -0.2 --domain-max 1.2 --plot-path outputs/regression/plots/bnn_regression_paper.png
```

Example with the non-default paper preset:

```bash
python bnn_regression.py --preset paper-figure5 --plot-path outputs/regression/plots/bnn_regression_paper_preset.png
```

Example with the default single learned likelihood sigma shared across all inputs:

```bash
python bnn_regression.py --preset paper-figure5 --plot-path outputs/regression/plots/bnn_regression_paper_global_sigma.png
```

Example with a natural cubic spline model for `log sigma(x)`:

```bash
python bnn_regression.py --preset paper-figure5 --likelihood-std-model spline --spline-num-knots 5 --plot-path outputs/regression/plots/bnn_regression_paper_spline_sigma.png
```

Example with a Gaussian radial-basis expansion for `log sigma(x)`:

```bash
python bnn_regression.py --preset paper-figure5 --likelihood-std-model rbf --rbf-num-centers 5 --plot-path outputs/regression/plots/bnn_regression_paper_rbf_sigma.png
```

Example with a few guide observations outside the main intervals:

```bash
python bnn_regression.py --target-function paper --domain-min -0.4 --domain-max 1.2 --observed-intervals=0.0:0.2,0.6:0.8 --guide-points-outside-intervals 6 --plot-path outputs/regression/plots/bnn_regression_paper_guided.png
```

Example with interval-specific observation noise inside the two observed intervals:

```bash
python bnn_regression.py --likelihood-std-model spline --target-function paper --domain-min -0.4 --domain-max 1.2 --observed-intervals=0.0:0.2,0.6:0.8 --observed-interval-noise-stds 0.01,0.06 --guide-points-outside-intervals 10 --guide-region-mode outer-only --guide-points-interior-gaps 4 --plot-path outputs/regression/plots/bnn_regression_paper_interval_noise.png
```

Plot again from a saved checkpoint without retraining:

```bash
python bnn_regression.py --plot-from-checkpoint outputs/regression/weights/bnn_regression_best.pt --plot-path outputs/regression/plots/bnn_regression_replot.png
```

Example with the spike-and-slab prior:

```bash
python bnn_regression.py --epochs 500 --prior spike-slab --prior-pi 0.5 --prior-sigma1 1.0 --prior-sigma2 0.1 --plot-path outputs/regression/plots/bnn_regression_spike_slab.png
```

Useful regression options:

- `--epochs 1000`
- `--hidden-dims 256,256,256`
- `--activation relu`
- `--train-points 192`
- `--train-samples 3`
- `--validation-samples 64`
- `--test-samples 400`
- `--validation-fraction 0.2`
- `--early-stopping-patience 80`
- `--early-stopping-min-delta 0.0`
- `--domain-min -5.0 --domain-max 4.5`
- `--preset paper-figure5`
- `--target-function oscillatory`
- `--target-function paper`
- `--observed-intervals=-4:-2,-0.5:0.75,1.75:3.5`
- `--observed-interval-noise-stds 0.01,0.06` overrides the generated observation noise inside each observed interval while leaving the default noise for guide points and uncovered regions
- `--guide-points-outside-intervals 6`
- `--guide-region-mode outer-only`
- `--guide-points-interior-gaps 2`
- `--checkpoint-save-every 20`
- `--likelihood-std-model global` is the default
- `--likelihood-std-model heteroscedastic`
- `--likelihood-std-model spline`
- `--likelihood-std-model rbf`
- `--global-likelihood-init-std 0.02` is interpreted in original target units; if omitted, the global sigma init is estimated from nearby training-target differences
- `--global-likelihood-prior-mean-std 0.02` sets the prior mean of the global sigma in original target units; if omitted, it defaults to the resolved init value
- `--global-likelihood-prior-sigma 1.0`
- `--spline-num-knots 5` uses evenly spaced total knots across the domain when the spline sigma model is selected
- `--spline-knots=-0.4,0.0,0.4,0.8,1.2` sets explicit knot locations in original x units for the spline sigma model
- `--spline-coefficient-prior-sigma 1.0` controls the independent zero-mean Gaussian priors on the spline log-sigma coefficients
- `--rbf-num-centers 5` uses equally spaced Gaussian RBF centers across the domain when the RBF sigma model is selected
- `--rbf-lengthscale 0.2` sets the initial and prior-mean Gaussian RBF lengthscale in original x units; if omitted, it defaults to the average center spacing
- `--rbf-lengthscale-prior-sigma 1.0` controls the Gaussian prior on log RBF lengthscale
- `--rbf-coefficient-prior-sigma 1.0` controls the independent zero-mean Gaussian priors on the RBF log-sigma coefficients
- `--coverage-eval-points 500`
- `--coverage-eval-samples 1000`
- `--prior normal`
- `--prior spike-slab`
- `--save-path outputs/regression/weights/bnn_regression_best.pt`
- `--plot-from-checkpoint outputs/regression/weights/bnn_regression_best.pt`

When `--plot-path` is provided, the saved figure contains:

- black `x` markers for the observed training points
- a dashed gray reference curve for the synthetic target function
- a red median predictive curve
- a light-blue 95% interval
- a darker blue interquartile band
- a textbox with mean IQR and mean 95% width, split between observed regions and missing regions

Optional plotting flags:

- `--plot-quantiles observation` uses quantiles of noisy sampled observations and is the default
- `--plot-quantiles function` uses quantiles of sampled latent functions
- `--hide-summary-box` removes the textbox with mean IQR and mean 95% widths
- `--shade-observed-intervals` highlights the intervals that contain observations

Training note:

- early stopping uses validation predictive NLL
- the KL term is always included as `kl / dataset_size` during training
- for `--likelihood-std-model spline`, `log sigma(x)` is modeled as an intercept plus a natural cubic spline basis on normalized inputs, and each spline coefficient has its own independent Gaussian prior penalty
- for `--likelihood-std-model rbf`, `log sigma(x)` is modeled as an intercept plus Gaussian radial basis functions centered at equally spaced points; the RBF lengthscale is learned with a Gaussian prior on log-lengthscale, and each RBF coefficient has its own independent Gaussian prior penalty
- after training, the script restores the best validation checkpoint and saves it to `outputs/regression/weights/bnn_regression_best.pt` by default so it can be reused for inference or plotting
