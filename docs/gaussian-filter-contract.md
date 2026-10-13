# Gaussian filter and probability operators: prototype contract

Container28, ModelRevision8, Family12 and Specialization11 introduce these
closed graph semantics. Old supported containers/revisions retain their meanings;
earlier headers cannot hide probability/filter drawings or archived new revisions.
No SQLite tables, TaskSpec loss grammar or optimizer-resume semantics change.

## Probability transforms

`Activation(activation="LogSoftmax",dim=...)` uses the same sample-axis indexing
as Softmax, excluding B. Native PyTorch dim is the resolved sample axis plus1.
Its outputs are log probabilities; exp(output) sums to1 along that axis.
`Activation(activation="Softplus")` fixes beta1 and threshold20 and rejects
extra constructor interpretation. It has no positive floor and can underflow to0.

`PositiveVariance` requires op_version1 and an explicit finite positive
`positive_floor` representable in float32. It computes precisely
`softplus(raw,beta=1,threshold=20)+positive_floor`, preserving shape/axis names.
There is no clipping, arbitrary code, trainable parameter or buffer. The floor
is visible model content/statistical constraint, not a hidden numerical repair.
Production is dense CPU float32; its helper also supports float64 for independent
gradient/reference checks. The primitive forward guard bounds one million
elements; it is not a whole-run RAM estimate.

## GaussianFilter fields and ports

Required closed fields: `inputs`, `initial_representation`,
`transition_representation`, `transition_mode`, `mean_mode`, `variance_mode`.
Missing op_version defaults to1. The filter owns no parameters or share_with.

`inputs` maps the six required roles to explicit stable source keys:
observations, initial, transition, means, variances, lengths. A key is a node ID
or `node@port`. Optional emission_mask has a separate role. Each unique source
has one incoming wire; one tensor may bind several shape-compatible roles.

| Role | Exact tensor shape, including B |
| --- | --- |
| observations | [B,T,D] |
| initial | [B,K] |
| transition | shared: [B,K,K]; sequence: [B,T,K,K] |
| means | shared: [B,K,D]; sequence: [B,T,K,D] |
| variances | shared: [B,K,D]; sequence: [B,T,K,D] |
| lengths | [B,1], exact float32-encoded integers in [0,T] |
| emission_mask | optional [B,T], exact0/1 |

Initial/transition representations explicitly select `probabilities` or
`log-probabilities`; no implicit logits interpretation. Probability rows must
be finite/nonnegative and sum to1 within2e-6. Structural zeros become -inf logs
with safe zero derivatives. Log rows use the audited kernel's normalization
contract. Transition rows are previous state and columns next state. The initial
belief is before the first transition, not before the first emission.

Shared modes expand views across T; they do not allocate or learn T independent
parameter copies. Every raw role first receives exact rank/shape/B validation:
no implicit broadcasting across records or missing batch axes. Numerical work
is admitted before probability/control conversion and adapter allocations:
T≤4096 and max(B,1)×(max(T,1)×(8K²+16KD+12K+4D+8)+4K)≤1,000,000.
Whole-model/run RAM admission remains a separate heuristic.

Unnamed axes are accepted. When named, observation axes mean time/coordinate
(T/D accepted), transitions time/from_state/to_state or from_state/to_state,
means/variance time/state/coordinate or state/coordinate. Known upstream
normalization metadata must select the next-state columns, not from_state or B.
The positive variance is not a standard deviation, raw variance or log variance.

| Stable output socket | Returned shape | Meaning |
| --- | --- | --- |
| filtered | [B,T,K] | Filtered log state probabilities after each active emission |
| final | [B,K] | Last filtered log belief, or initial belief for length0 |
| step_evidence | [B,T] | Conditional per-step log evidence |
| sequence_evidence | [B,1] | Sum of per-step evidence |

The full Gaussian normalization constant is included. Missing whole emissions
still transition and contribute exactly zero evidence. Padding carries the last
belief and has zero evidence; inactive NaN density inputs are sanitized by the
kernel before arithmetic. Emission masks are not execution masks or loss masks.
Reset/execution controls, smoothing and path decoding are not supported by this
operator version. Those omissions remain requirements of the full MS-VAR work.

## UI, families, export and scope

The palette exposes LogSoftmax/Softplus and PositiveVariance. Gaussian filter adds
an inspectable14-block two-state p=0 HMM component alongside existing work: real
initial/transition/mean/raw-variance owners, normalization/floor boxes, observed
sequence and prefix-length inputs, four posterior/evidence outputs. It creates
no dataset, fit or invented state labels. Shape/parameter controls remain in the
existing Inspector; the filter's six-field JSON editor checks exact graph context
and all candidate shape/binding semantics before writing. Failed edits preserve
SQLite bytes. Normal sockets, folds, compact drag, undo and .vtw persistence apply.

This is an exact small HMM/filter component, not the full neural p3 MS-VAR.
LogSoftmax, floor and filter semantics use the same native/app-free helpers;
private captured namespaces prevent source/variance constructor-name collisions.
Family12/Spec11 can retain fixed or explicit temporal-v2 execution; real tests
exercise different T with the same10 HMM parameter elements. Captured family/
notebook closures include positive_variance, gaussian_filter and gaussian_nodes;
current bundles require imported dependencies before kernel startup.

Known operator representations appear in CLI port contracts when proven and are
never inferred from labels. Sequence evidence is signed: narrow Gaussian densities
can exceed1. Concrete target-free training now has the explicit
[TaskSpec2 / Bundle5 / Result2 contract](likelihood-task-contract.md); supervised
contracts retain their original meaning. Legacy
synthetic/packed-MSE controls refuse filter graphs rather than fabricate valid
integer controls or pretend MSE fits hidden states. Runtime-family/Experiment/
notebook likelihood contracts and causal p3 windows still need implementation.

## Checked evidence and limits

Focused native/plain-export/family/runtime/storage/adversarial cases pass,
including all10 source gradients, shared time reuse, finite-state normalization,
wrong batches and pre-allocation guard refusal. An independent review found
broadcasting and early-adapter-allocation gaps; both were corrected with tests.
Latest30 focused cases28.935s and375 JavaScript cases pass; earlier66 cases and40
route cases pass (three native-only skips). Actual sandboxed Qt tests cover native
palette click, visible internals, six inputs/four outputs, Inspector, failed-edit
no-write, undo/redo/save-reload; the host imports no numerical runtime. Existing
policy saved/rerun Jupyter kernels still replay exactly. The separate likelihood
contract records real concrete worker/checkpoint/export acceptance, not Gaussian
notebook-kernel or full source-wide acceptance.
