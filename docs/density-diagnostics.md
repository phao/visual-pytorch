# Conditional density and predictive-mixture diagnostics

GaussianFilter computes a normalized finite-state recursion for supplied
parameters. That alone does not prove a normalized density in the observations
if its mean/transition network sees those same current/future observations.
The graph now supplies `density_conditioning` metadata for evidence outputs,
also returned by `ports`. TaskSpec2 checks it before allocating a worker.

The supported proof follows explicit causal lag features, pointwise operations,
time-preserving reshapes/permutations and single-input/single-output Module bodies.
Normalizing/mixing across time, feeding the current observation into its own
parameters, an observation-dependent initial law, or unknown temporal composition
produces a located refusal for likelihood training, not a failure to draw/compile
the ordinary tensor graph. An adaptive observation transform similarly requires
an explicitly specified change-of-variables/density objective; no Jacobian is
silently inferred. Frozen/adaptive transforms and more complex temporal bodies
may be meaningful, but are not automatically certified by this bounded proof.

Other root inputs are explicitly supplied conditioning, not inferred statistical
independence. Their availability and construction remain part of the scientific
data contract. Binding conditioning and scored observations to the same field
currently refuses as ambiguous; an intentionally conditional reconstruction
objective would require a separate explicit declaration. Derived prefix lengths
and masks remain allowed metadata. This gate does not establish identifiability,
calibration, absence of external data leakage or correct target density units.

Cross-entropy refuses known probabilities, variances or evidence as class logits.
Normalized log probabilities are still valid class scores: another LogSoftmax is
mathematically idempotent. Unknown raw tensors may be logits; captions do not
prove that they are. Gaussian NLL uses actual positive variance, not Normal scale.
See [PyTorch distributions](https://docs.pytorch.org/docs/2.14/distributions.html)
for Normal scale and mixture distributions. The
[statsmodels switching-autoregression reference](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_autoregression.MarkovAutoregression.html)
distinguishes lagged/exogenous transition covariates and filtering/smoothing;
its full regime-dependent lag formulation is not claimed equivalent to this
report's simpler state-wise affine mean. These references inform semantics, not
numerical acceptance of this implementation.

## Inspect actual predictive densities

`app.sequence_diagnostics.inspect_model(model, inputs, filter_id)` reads checked
compiled/generated graph intermediates and returns ephemeral JSON. `inputs`
may use stable Input IDs or the model's native names. It does not fit, store a
simulation, change weights/graphs, or manufacture state labels. Runtime temporal
model inference retains its normal guarded contract. Direct `summarize(...)`
accepts explicitly expanded `[B,T,...]` filter roles.

For each executed step, it shows the state distribution **before** observing the
current value and the filtered state distribution **after** it. Scored steps
include mixture mean, total covariance and mixture log density. Total covariance
includes both within-state diagonal variance and between-state mean variation;
it is not merely the diagonal of the average component covariance. Mixture-mean
error alone does not measure density quality: two mixtures can have the same
mean and different covariance/log density.

Skipped/padded steps preserve belief, have no predictive emission score, and
ignore resets. An executed unscored step still advances the state law; its
density/moments are absent rather than fictitious zeros. Applied resets happen
before prediction. Empty trajectories have zero evidence, not a fabricated fit.
The report is not smoothing or latent-label prediction. Means, covariance and
independent density diagnostics use float64 arithmetic on the model's actual
values; the filter score retains the original dtype. Small roundoff differences
are not hidden or replaced with a fabricated exact match.

The default diagnostic output budget is one million numerical elements, checked
before filter/covariance work. This includes full covariance's D² cost; the
kernel's separate input/work admission remains active. Large diagrams can still
be specified, without implying their full diagnostic covariance is inexpensive.

Tests compare densities with PyTorch's independent `MixtureSameFamily`, moments
with scalar formulas, masked/reset sequences, high dynamic range, immutable
inputs/weights, empty B/T and actual controlled neural MS-VAR forward evidence.
