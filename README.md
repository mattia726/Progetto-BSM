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
- `bnn_regression.py`: Bayes-by-Backprop regression on disjoint observation intervals
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

`bnn_regression.py` trains a Bayesian regressor on a synthetic oscillatory target where observations are only available inside several disjoint intervals. It then evaluates the predictive distribution across the full domain and reports uncertainty inside both observed and missing regions.

The regression model:

- samples training inputs only from configurable disjoint intervals
- uses Bayes by Backprop with a diagonal Gaussian posterior
- predicts a Gaussian mean and scale for each input
- reports Monte Carlo uncertainty summaries over the full domain
- supports a `normal` prior by default and an optional spike-and-slab prior

Example with the default normal prior:

```bash
python bnn_regression.py --epochs 500 --plot-path checkpoints/bnn_regression.png
```

Example with custom intervals:

```bash
python bnn_regression.py --epochs 500 --observed-intervals=-4:-2.5,-0.4:0.5,2.0:3.2 --plot-path checkpoints/bnn_regression_custom.png
```

Example with the spike-and-slab prior:

```bash
python bnn_regression.py --epochs 500 --prior spike-slab --prior-pi 0.5 --prior-sigma1 1.0 --prior-sigma2 0.1 --plot-path checkpoints/bnn_regression_spike_slab.png
```

Useful regression options:

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
- `--observed-intervals=-4:-2,-0.5:0.75,1.75:3.5`
- `--prior normal`
- `--prior spike-slab`
- `--save-path checkpoints/bnn_regression_best.pt`

When `--plot-path` is provided, the saved figure contains:

- black `x` markers for the observed training points
- a dashed gray reference curve for the synthetic target function
- a red median predictive curve
- a light-blue 95% interval
- a darker blue interquartile band

Optional plotting flags:

- `--plot-quantiles observation` uses quantiles of noisy sampled observations and is the default
- `--plot-quantiles function` uses quantiles of sampled latent functions
- `--shade-observed-intervals` highlights the intervals that contain observations

Training note:

- early stopping uses validation predictive NLL
- the KL term is always included as `kl / dataset_size` during training
- after training, the script restores the best validation checkpoint and saves it to `checkpoints/bnn_regression_best.pt` by default so it can be reused for inference or plotting
