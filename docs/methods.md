# Methods and available material

The coauthored report *Metodi Bayesiani per le Reti Neurali* (Brunetto Daniele and Mattia Faini, 12 May 2026) develops Bayes by Backprop (BbB) and Monte Carlo Dropout (MCD), then discusses cubic regression and MNIST/Fashion-MNIST comparisons. The supplied Python implements only some of that scope. The [English translation](../bayesian_methods_en.pdf) reconstructs the complete report from its PDF; the original editable source was not supplied. Its ten figure panels are extracted from the Italian PDF, not recreated by model execution.

| Report component | Available source |
|---|---|
| BbB variational weights and MNIST classification | `bnn/mnist.py` and `bnn_mnist.py`; the report's accuracy/uncertainty experiment logs and plots are absent. |
| BbB synthetic regression | `regression/` implements oscillatory and `paper` targets, separate from the report's noisy cubic example. |
| MC Dropout regression/classification | No implementation supplied. |
| Cubic example, 20 inputs on `[-4,4]`, noise `N(0,9)` | No corresponding Python experiment supplied. |
| Fashion-MNIST evaluation, probability-boxplot and entropy figures | No evaluator or figure-generation pipeline supplied. |
| Desktop digit drawing | `draw_digit_app.py`; a demonstration, not an OOD study. |

The `paper-figure5` regression preset refers to a target attributed in the source to Figure 5 of *Bayes by Backprop*; it is not a reconstruction of the course report. The 23 supplied `outputs/plots/` PNGs are historical regression illustrations. Their names do not establish complete run settings, a successful test, or a mapping to the report's four figures. Two inspected copies are published under `assets/`; all originals and 19 checkpoints remain local.

## Weight distributions and objectives

Both implemented Bayesian linear layers use diagonal Gaussian variational weights with `softplus(rho)` standard deviations, Xavier-initialized weight means, zero bias means, and `rho` initialized at `-5`. In both, `sample=True` still samples weights under `model.eval()`. The classes differ in their penalty estimator and must not be interchanged:

| Path | Parameter penalty |
|---|---|
| MNIST | Closed-form KL from the variational Gaussian to a zero-mean Gaussian prior. Even `sample=False` returns this KL. |
| Regression | Sampled `log q(w) - log p(w)` for weights and biases. `sample=False` returns zero for that field. The optional `spike-slab` prior is a continuous two-Gaussian scale mixture evaluated with `logaddexp`, not a point mass or sparsity selector. |

Each training path uses a mean per-example likelihood loss plus its summed parameter penalty divided by the training-dataset size. MNIST training averages cross-entropy across weight draws. Testing averages **softmax probabilities first**, then computes NLL; it does not average logits or individual-network NLLs. The displayed test KL is scaled by test-set size, unlike the training penalty's denominator. The model is a 784 -> hidden -> hidden -> 10 ReLU MLP, with hidden width 400 by default and source normalization constants 0.1307 and 0.3081. The standard MNIST test split is evaluated each epoch without a separate validation-selection split. The filename of the supplied MNIST checkpoint supplies no accuracy evidence.

Regression averages Gaussian negative log-likelihoods during training. Validation estimates predictive-mixture NLL using `logsumexp` across weight draws, and early stopping saves/restores the best state by that criterion with the configured patience and minimum change. Predictive NLL is computed in standardized target coordinates; error and standard-deviation summaries converted back to target units are separate quantities. No Jacobian correction or new loss definition was introduced. In the global, spline, and RBF scale modes, the quantity logged as `total_kl` also contains negative log-prior penalties for point-estimated noise parameters; the name does not mean it is entirely a variational KL.

## Synthetic data and likelihood scale

The two supplied generators use the **same** sampled epsilon both inside the nonlinear function as `x + epsilon` and as an additive output term. The `*_mean` functions provide zero-noise reference curves, not established conditional expectations of these generators. Main inputs are assigned to configured intervals with probabilities proportional to interval lengths; this does not guarantee a point in every interval. The main observed sample is split into training and validation first. Exterior or interior-gap guide observations are appended only to training, after which standardization is fitted on the resulting training sample, including guides, using population standard deviation. Broad regions outside main sampling intervals can therefore contain guide observations. `interval_mask()` includes both endpoints.

| Likelihood-scale mode | Implementation |
|---|---|
| `global` | One optimized log-scale parameter with a Gaussian log-scale prior penalty; the regression CLI default. |
| `heteroscedastic` | A second Bayesian output passed through softplus plus its configured floor; the regressor constructor default. |
| `spline` | An optimized intercept and natural-cubic-spline coefficients for log scale with coefficient penalties. |
| `rbf` | Optimized radial-basis coefficients and log lengthscale with prior penalties; centers and lengthscale use the source definitions. |

The global scale can be initialized from nearby training-target differences; when no prior location is supplied, it reuses the resolved initialization. That is a data-informed prior location, not an observation-independent prior. The Gaussian prior location is on **log sigma**; it is not the arithmetic mean of sigma after transformation. Source CLI arguments distinguish original target units from standardized target units. `min_predictive_std` does not act as the same output floor in every mode, so these modes are not reduced to a single noise parameterization.

## Prediction, coverage, and checkpoints

Regression predictive variance is summarized as variance across sampled function means (epistemic) plus the mean of squared likelihood scales (aleatoric). The aleatoric standard deviation is a root-mean-square scale, not the mean of the individual sigmas. Population variance (`unbiased=False`) is used where specified, and standard deviations are converted using the fitted target scale. Function quantiles use sampled means; observation quantiles additionally draw Gaussian observation noise. The historical `median`, `q25`, and `q75` aliases refer to **function** samples, while plotted quantiles default to **observation** samples.

The generated-coverage routine compares both function and observation intervals to newly generated **noisy** targets. In particular, `function_95` is not coverage of the latent noiseless curve. These are diagnostics for one fitted model on generated observations, not repeated-dataset coverage or proof of calibration. Plotting or replotting from a checkpoint performs Monte Carlo inference.

The MNIST checkpoint writer stores `state_dict`, `hidden_dim`, and `prior_sigma`, with layer keys including `layer1`, `layer2`, `output_layer`, and `weight_mu/rho`, `bias_mu/rho`. The drawing app accepts that dictionary or bare state weights, infers width from `layer1.weight_mu`, and defaults missing `prior_sigma` to 1.0. The regression checkpoint includes model state, target/prior/mode fields, basis settings, input and target standardization, interval and guide metadata, and scale initialization. Its layer and likelihood-module key structure remains in place. Runtime loaders now request `torch.load(weights_only=True)` and do not fall back to unrestricted pickle loading. The existing `.pt` files were not loaded, so compatibility remains a static expectation rather than a test result.

## Translation notes and limits

The Italian report sometimes describes `F(D, theta)` as a cost to minimize and elsewhere as an ELBO to maximize. Its scale-mixture prior grouping, variance-versus-`rho` wording, and MCD precision formula retain their printed forms in the English translation; those ambiguities were not turned into new equations. The first panel of report Figure 3 has an internal "MC Dropout" title while its printed subcaption says "BbB". The English caption records both labels and preserves the original panel order. Italian artwork lettering remains, with translated caption keys rather than painted-over plots.
