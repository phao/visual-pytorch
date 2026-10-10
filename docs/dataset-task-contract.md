# Typed datasets and tasks — Stage C work in progress

This is the executable first inline adapter, not completion of Stage C. It is
available through Python/CLI and the branch-only [typed dataset/task GUI](experiment-gui.md).
Experiment visual authoring is still unfinished. Independent
dataset/task/model revisions persist in `.vtw` v6; fitted task training/export is
now executable. Architecture-owned legacy `dataset.x/y` continues unchanged.
Experiment/run/report persistence is specified in [experiment-objects.md](experiment-objects.md).
The full Stage C end-to-end exit gates remain pending; notebook APIs and actual
kernel evidence now have their own [contract](notebook-reports.md).

## Independent objects and identities

`app.datasets` validates DatasetSpec version 1. Required keys: `version`, `id`,
`name`, `storage`, `fields`, `records`. Storage is exactly
`{"kind":"inline-records","version":1}`. Optional provenance/usage are finite
JSON. `freeze_dataset` returns private bytes represented as JSON values plus a
SHA-256 revision of the canonical JSON object excluding `revision` itself.
Changing metadata, field order, vocabulary order or record bytes produces a new
revision. This content identity is not proof of provenance or independence.

Every field has stable `id`, `dtype`, sample `shape`; optionally `name`, `axes`,
`nullable`, `vocabulary`, `modality`, `unit`, `coordinates`. Dtypes: float32,
int64 (exact JSON integers within ±(2⁵³−1)), bool, category, string. Numeric
float32 values are finite and representable; bool is not silently an integer.
The first sample dimension may be −1 for a ragged sequence; other dimensions are
positive. Scalars have `shape: []`. Optional axes match rank and are distinct.
Categorical fields require version-1 vocabulary with an ordered list of at least
two distinct string/integer values. `1`, `"1"` are different labels. The order
defines class indices and is never inferred from observed batches. Missing
whole fields or tensor leaves require `nullable: true`.

Records have unique stable `id` and exactly the schema's field IDs in `values`.
This adapter is finite, random-access and repeatable with known cardinality; it
does not claim resumable streaming positions. Preview uses sorted observation
IDs, 1–100 records, and labels its ranges/counts as a bounded preview, not a full
audit. It neither fits transforms nor consumes a global/training RNG.

## Splits

`make_split(dataset, protocol)` returns a SplitManifest containing immutable
dataset revision, method, actual observation IDs by train/validation/test role,
explicit exclusions, and the protocol. `check_manifest` rejects revision
mismatch, unknown/duplicate/overlapping IDs, undeclared omission, or empty
train/test roles. Validation may be empty only if requested fraction is zero.

Row/group protocols declare independent integer seed, fractions `[train,val,test]`,
and, for group, a scalar grouping field ID. Sorted independent units are shuffled
with a private Python MT19937/Fisher–Yates procedure. Integer-floor allocation
applies to units, not record counts: group sizes can yield unequal row fractions.
A group cannot occur in two roles, including exclusions. Temporal allocation
uses distinct numeric timestamp values in increasing order; tied timestamps
remain in one unit and role boundaries must be strictly chronological. This does
not prove that feature construction is causal. External protocols declare all
three ID arrays and exclusions explicitly. Replay uses saved IDs, not a seed.

## Stable task bindings and preparation

TaskSpec version 1 has `id`, `name`, `inputs`, `heads`; optional `description`
and verified content `revision`. Each input binds stable `port` to `field` with
an ordered `transforms` array. Every model input is bound exactly once. Shape and
declared axes must match the validated graph contract, without row-order/name
guessing. Scalar numeric fields map to `[1]`. Numeric data maps to float32 input
tensors; string and category inputs need an explicit applicable transform.

Transforms: scalar categorical `onehot` using fixed vocabulary; first-axis `pad`
with explicit positive `length` and `value`, never silent truncation; fixed-shape
`impute-mean` per feature; `standardize` per feature (population variance/ddof0,
constant-feature scale 1); invertible `affine` with explicit nonzero scale and
offset. Transform order is semantic. Missing values require explicit imputation.
`prepare` fits each learned step solely on train IDs after previous steps and
applies that exact state to validation/test. Artifact `fitted` retains train IDs,
dataset/task revisions, input and target pipeline states. Source data is not
modified. Derived sequence `length` or `mask` input bindings declare the same
source field and padded `length`, with no transforms. They yield `[1]` or
`[length]` float32 inputs and reject overlong/missing source sequences. Each derived
binding must match a pad input binding for the same field and length. Padding
positions participate in deterministic per-feature transforms unless the task
explicitly provides another adapter; no automatic padding-aware fitting is claimed.

Each named head declares stable output `port`, target `field`, positive objective
`weight`, `loss` and target `transforms`. Every output has one head. MSE/MAE
targets match output sample shape; cross-entropy uses categorical target indices
and logits with sample shape `[vocabulary_size, *target_shape]`. Category targets
have no transforms. The task worker supports these classification/regression heads.
Optional masks/weights refer to nonmissing numeric scalar or target-shaped fields.
Masks are bool/exact 0 or 1; sample weights are finite and nonnegative. Missing
categorical targets currently report unsupported masked-target adaptation rather
than fabricating a label. The trainer reports empty weighted populations as
unavailable and skips minibatches with no supervised head; preparation itself
does not invent metrics.

`prepare` returns populations keyed by split role, each with observation IDs,
inputs keyed by stable port IDs, targets/masks/sample_weights keyed by head IDs,
plus fitted state and captured graph contract. This representation can be
tensorized by the numerical worker. Captured human port names are only an adapter
to the existing model's named forward, not the identity of a binding.

`app.task_runtime.tensor_batch` selects population indices without mutating source
arrays, tensorizes with explicit CPU float32 inputs/regression targets or int64
class indices, and preserves actual IDs (including duplicate sampler entries).
`forward_ports` maps stable IDs through captured names to the native graph's
valid forward interface. These adapters feed the task-aware CPU trainer;
[task-training-contract.md](task-training-contract.md) specifies budgets, terminal
evidence, masks/multi-head objectives, fitted checkpoints and standalone inference.

`head_terms` implements MSE, MAE and indexed cross-entropy with strict shapes and
no regression broadcasting. A per-record scalar mask/weight is broadcast only
within that record; otherwise it must be exactly loss-shaped. It returns additive
weighted numerator/denominator, float64 differentiable reduction, and included
element counts. Zero denominator has status unavailable and `mean: None` (the
forthcoming trainer must explicitly skip such a batch). Nonfinite predictions,
arithmetic or weights are errors even when masked. Classification accuracy uses
the same mask/weight denominator. MetricAccumulator combines only the same loss,
checkpoint and population, retaining actual ID sequence, unique and evaluated
counts. It adds numerators/denominators, never averages batch means. Head objective
weights are not applied inside a head's population metric; multi-head combination
belongs to the forthcoming trainer.

## Read-only CLI

These commands use the same current `.vtw` architecture contract without storing
new objects or modifying its disk bytes:

```sh
python vpw_cli.py lab.vtw preview-dataset observations.json --limit 20
python vpw_cli.py lab.vtw prepare-task MODEL_ID --dataset observations.json \
    --task task.json --split split-protocol.json
python vpw_cli.py lab.vtw prepare-task MODEL_ID --dataset observations.json \
    --task task.json --manifest saved-manifest.json
```

`prepare-task` returns preparation artifact in the normal JSON envelope, including
fitted state and actual split IDs. It does not train, update a checkpoint or save
a workspace. The manifest is the `value.manifest` object from a previous result.
Constrained mode comes from the workspace; percentage is read without writes from
user configuration or explicitly overridden with `--memory-limit-percent`. An
invalid user config requires repair or an explicit override. Preview diagnostics
are bounded; preparation is execution of the declared transforms on the selected
finite population and is therefore subject to RAM admission.

Constrained preparation checks expanded shapes before padding/copy allocation.
Its version-1 estimate is 8 MiB + 128 bytes × selected rows × (input elements +
3 × output elements), against the user-level available-RAM percentage. It is a
conservative heuristic, not an OOM guarantee or total training estimate. The
existing model/training memory admission still applies independently. No fixed
64-feature limit is introduced. Tests inject observed RAM; users may explicitly
disable constrained mode. Excluded records are not transformed into populations.

## Independent revision persistence

Current `.vtw` version is 6 on the Stage C branch. DatasetSpec/TaskSpec and frozen
ModelRevision documents are immutable independent revisions with mutable browsing
heads. The migration runs only in the in-memory working copy before explicit
save, preserving v1–5 legacy meanings. Hashes, relational identity, schema and
immutability guards are checked on load/save. CLI writes use the existing atomic
revision-checked ordered edit workflow. Undo restores browsing heads but retains
immutable revision bytes. Exact semantics are in [revision-store.md](revision-store.md).
ExperimentSpec/RunRecord/ReportRecipe persistence uses the same guarded registry.
Direct task training returns a self-contained artifact; explicit candidate
execution captures its validated evidence in an independent RunRecord, retaining
actual manifests and exact dependencies instead of duplicating raw datasets.
Unknown placeholder kinds are not silently accepted.

Next steps: integrate experiment selection/history/notebook GUI using the
implemented result-report/explicit-rerun bundle APIs and actual-kernel tests.
Stage C exits only after the complete end-to-end gates in the roadmap.

## References and decisions

[scikit-learn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
motivates splitting before fitted preprocessing and retaining unchanged evaluation
state. This implementation uses independent scalar tests and pure Python, not a
scikit-learn dependency. [PyTorch's data API](https://docs.pytorch.org/docs/2.8/data.html)
informs explicit adapter capability claims; iterable/distributed storage is not
implemented here. [CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html)
informs indexed logits/target shapes; custom masks and sample weights use explicit
unreduced elements and independently tested denominators rather than implicit
library mean semantics. These are design references, not proof of app correctness.
