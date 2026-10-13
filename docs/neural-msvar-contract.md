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
padding before windowing. No automatic missing-history imputation or resets exist.
Source export verifies actual helper definitions and isolates constructor namespaces.

Causal grammar uses ModelRevision9, Family13, Specialization12 and container31;
older headers/models/families refuse it. Likelihood uses TaskSpec2, concrete
Bundle5 or temporal-v2 Bundle6, Result2, Experiment7 and Run7. The original
supervised versions keep meaning. Report1 recipes consume those exact references.

## Verification and remaining scope

Tests independently construct lag features and affine/neural branches, enumerate
all 625 initial/latent paths for K=5,T=3, compare density and final posterior,
check no current-observation leak and exact native/exported parameter gradients.
A real twelve-update fit checks grouped populations, checkpoint admission and
standalone predictors. Family inference exercises T=2/7 with the same owners.
Real saved/rerun Jupyter kernels reproduce full-model checkpoints and metrics.
Sandboxed Qt checks sockets, Inspector, native compact drag, history and reload;
its host imports no Torch/NumPy.

The complete requested scope is not yet finished. Reset/execution semantics,
autoregressive missing-history conditioning, broader trajectory diagnostics,
CLI construction/planning/example-pack/doctor contracts and final integrated
regression remain. There is no generic production Loss/Optimizer/Dataset canvas
or deferred optimizer-resume/HPC feature in this milestone.

Implementation references: [PyTorch Tensor.unfold](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html)
documents sliding-window views; the kernel explicitly permutes lag/coordinate
axes before flattening. [PyTorch distributions](https://docs.pytorch.org/docs/stable/distributions.html)
distinguishes Normal scale from variance. These are implementation references,
not numerical acceptance evidence; tests exercise the installed runtime.
