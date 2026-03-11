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
