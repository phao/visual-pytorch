# Exact causal neural MS-VAR composition

The requested K=5, p=3, D=3 architecture is now a real editable 23-block graph,
not the earlier supervised regime-classifier approximation. Open
`examples/neural-msvar.vtw` to inspect the diagram, family, unlabelled trajectory
dataset, target-free task, grouped split and bounded sanity-fit checkpoint.
The example does not establish identifiability, calibration or stationarity.

## Computation and ownership

At observed time $t$, the causal block returns
$h_{t-1}=\operatorname{vec}(Y_{t-3},Y_{t-2},Y_{t-1})$, oldest to newest.
The explicit initial history supplies missing negative-index observations.
$Y_t$ is never part of its own transition or mean predictor.

| Visible branch | Shape including batch | Learned elements |
|---|---|---:|
| CausalLagWindow, three bound inputs | `[B,T,3]` + `[B,3,3]` + `[B,1]` → `[B,T,9]` | 0 |
| Transition Linear | `[B,T,9]` → `[B,T,32]`, bias on | 320 |
| GELU | `[B,T,32]` → `[B,T,32]` | 0 |
| Transition Linear | `[B,T,32]` → `[B,T,25]`, bias on | 825 |
| Reshape, LogSoftmax on next-state axis | `[B,T,25]` → `[B,T,5,5]` | 0 |
| State-wise affine VAR Linear | `[B,T,9]` → `[B,T,15]`, bias on | 150 |
| Means Reshape | `[B,T,15]` → `[B,T,5,3]` | 0 |
| Initial logits, state LogSoftmax | `[B,5]`, initial law before first transition | 5 |
| Raw state variances, PositiveVariance | `[B,5,3]`, shared across time | 15 |
| Exact GaussianFilter | posterior, final belief, step/sequence evidence | 0 |

Total: 1,315 elements in eight unique parameter tensors. Earlier axes share
Linear weights; changing T does not create new owners. Transition rows represent
previous states, columns next states. The exact recursion sums over latent
states, with full diagonal-Gaussian normalization constants. This is filtering,
not smoothing and not supervised latent-state classification.

The positive-variance branch currently uses `softplus(raw)+0.0001`. It is a
declared floored variance parameterization, not a standard deviation and not a
literal `exp(log_variance)` block. That distinction remains visible. Variance
does not depend on the lag window. All controls are explicitly bound.

## Training and data

The task minimizes negative marginal sequence log likelihood conditional on
initial observed histories. The remaining six outputs are diagnostics, without
fake regime targets. The example dataset contains independently generated
synthetic trajectories, not shuffled overlapping windows. Split membership is
grouped by the explicit trajectory field. Padding is bounded by prefix lengths;
sequence evidence sums only active steps. Per-step objectives instead require
their own explicit loss masks, as described in the likelihood contract.

The bundled twelve-update run is a sanity check, not a recommended training
budget or proof of statistical recovery. Teacher and fitted transition network
are distinct. The builder never overwrites existing files; execution is opt-in:

`python examples/build_neural_msvar.py --output NEW.vtw --run`

Without `--run`, the builder creates inspectable inputs and experiment only.

## Executable boundaries

CausalLagWindow1 has three exact role bindings: observations, history, lengths.
Diagram controls are float32 tensors `[B,1]` containing exact integer lengths;
the checked adapter converts them to int64 `[B]`. Shapes and conservative work
are checked before conversion/masks. The numerical kernel then sanitizes inactive
padding before windowing. Version 1 has no missing-history imputation or resets.
Source export verifies actual helper definitions and isolates constructor namespaces.

Causal grammar uses ModelRevision9, Family13, Specialization12 and container31;
older headers/models/families refuse it. Likelihood uses TaskSpec2, concrete
Bundle5 or temporal-v2 Bundle6, Result2, Experiment7 and Run7. The original
supervised versions keep meaning. Report1 recipes consume those exact references.

## Explicit controlled version

`python examples/build_neural_msvar.py --output NEW.vtw --controlled` creates
the 27-block controlled variant without training; add `--run` for the bounded
sanity fit. The existing bundled example and version-1 factory retain their
original softplus-plus-floor interpretation. The controlled variant uses
`PositiveVariance2(transform="exp", positive_floor=0)` on learned log variances:
the emitted variance is exactly `exp(raw)`, not a standard deviation. A declared
positive floor is also supported, but is never inserted implicitly. Nonfinite
inputs, overflow and nonpositive underflow refuse rather than clamp. Choosing
softplus instead requires an explicit version-2 transform declaration.

The controlled graph has the same 1,315 learned elements; four extra inputs
declare execution, reset, emission and reset-history controls. Role bindings
are ordered and explicit, with one physical edge per unique source:

| Role | Diagram shape including B | Meaning |
|---|---|---|
| lengths | `[B,1]` | Exact integer prefix end; converted to int64 `[B]` |
| execution_mask | `[B,T]` | Exact 0/1; whether transition, emission and observed-history update execute |
| reset_mask | `[B,T]` | Exact 0/1; reset before an executed step, ignored on skipped/padded slots |
| emission_mask | `[B,T]` | Exact 0/1; whether that executed observation contributes density evidence |
| reset_history | `[B,T,p,D]` | Explicit oldest-to-newest observed context at executed resets |
| TaskSpec2 loss mask | Declared dataset field | Objective inclusion, not filter execution or resetting |

The lag block resets to supplied history **before** constructing features and
appends the current observation **after** producing its features. Skipped steps
produce zero features and leave observed history unchanged. The filter resets
to its supplied initial law **before** transition; skipped steps preserve belief
and have zero evidence. Evidence sums over the executed reset segments. An
emission-masked executed step still advances the transition and observed history.
Reset histories must be paired with a reset mask; no reset context is inferred.
Controls are per-call; there is no hidden cross-call state.

These are distinct statistical operations. Skipping execution changes the
sequence clock to executed observations; it is not a missing-data integration.
An emission-masked observation may be excluded from scoring while remaining a
known, finite covariate for future autoregressive features. If it is genuinely
unobserved, exact marginalization of that continuous history through nonlinear
transition networks is **not** implemented: the lag block refuses nonfinite
executed observations. Supplying observed context is an explicit conditioning
assumption, not a claim to have integrated unknown histories. Unused padded or
skipped numerical roles are sanitized before evaluation at these primitives;
upstream ordinary NN operations are not automatically mask-aware.

The controlled synthetic fixture is an execution/replay witness, not a claim
that its teacher obeys the fitted reset law. Its loss mask includes trajectories
with scored executed observations; an all-invalid population is unavailable,
has zero optimizer updates and no fabricated objective. Existing TaskSpec2
numerator/denominator and multi-head coefficient semantics are unchanged.

These operators require ModelRevision10, Family14, Specialization13 and
container32, including retained/archived versions. Runtime-temporal-v2 families
may vary only T across observations and time-varying control/history arrays;
K, D and p stay constructor dimensions. There are no new SQL tables, task,
result, Experiment or Run versions for these controls.

## Verification and remaining scope

Tests independently construct lag features and affine/neural branches, enumerate
all 625 initial/latent paths for K=5,T=3, compare density and final posterior,
check no current-observation leak and exact native/exported parameter gradients.
A real twelve-update fit checks grouped populations, checkpoint admission and
standalone predictors. Family inference exercises T=2/7 with the same owners.
Real saved/rerun Jupyter kernels reproduce full-model checkpoints and metrics.
Sandboxed Qt checks sockets, Inspector, native compact drag, history and reload;
its host imports no Torch/NumPy.

The original example has a [pinned manifest and explicit replay](example-packs.md).
[Predictive-mixture diagnostics](density-diagnostics.md) explain actual state
beliefs, mean/covariance and density, with separate causal-likelihood admission.
[Shared recipes/edit previews/doctor](edit-planning.md) provide inspectable agent
and desktop authoring without new workbenches. Final integrated regression
and local main acceptance remain. There is no generic production Loss/Optimizer/Dataset canvas
or deferred optimizer-resume/HPC feature in this milestone.

Implementation references: [PyTorch Tensor.unfold](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)
documents sliding-window views; the kernel explicitly permutes lag/coordinate
axes before flattening. [PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)
distinguishes Normal scale from variance. These are implementation references,
not numerical acceptance evidence; tests exercise the installed runtime.
