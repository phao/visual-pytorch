# Architecture family experiment axes — experiment/run version 2

This is an implemented local Stage D checkpoint, not tied weights, runtime
recurrence, dynamic spatial shapes or durable jobs. Reference implementations
are app/family_axes.py, app/experiment_objects.py and app/family_records.py.
Version-2 experiment/run documents require a .vtw version-8 container. The
SQLite tables and immutable hash rules are unchanged. Version-1 experiments and
runs keep their existing grammar and content hashes.

## Explicit model choices

ExperimentSpec version 2 has the same top-level fields as version 1. Its ordered
models array may mix exact model references and family choices. A family choice
has exactly id, family, bindings, axes:

```json
{
  "id": "width-depth",
  "family": {"kind": "family", "id": "width-depth-mlp", "revision": "<64 lowercase hex digits>"},
  "bindings": {},
  "axes": [
    {"symbol": "hidden-width", "values": [2, 3]},
    {"symbol": "hidden-depth", "values": [1, 2]}
  ]
}
```

The placeholder hash above must be replaced by an actual retained exact family
reference. Choice IDs are unique among family choices. bindings maps stable
symbol IDs to exact integers, not names or expressions. axes is an ordered list
of at most 128 distinct symbol IDs, each with 1–1000 distinct explicit integer
values. Fixed/axis symbols are disjoint, must exist in that pinned family and
use its exact editor integer range, −(2⁵³−1) through 2⁵³−1; booleans, floats,
strings, inferred values and extra fields are rejected. The family validates
its own narrower symbol bounds and operator constraints when a point resolves.
Empty axes means one explicit fixed/default selection. Missing required defaults
remain conditional rather than making up a binding.

The model axis concatenates each choice's product in stored order. Within a
family, the last binding axis changes fastest. Outside it, the original order
is model × data/task × split × initialization × training configuration, last
axis fastest. Preview uses mixed-radix arithmetic to skip prefixes, not an
eager list of compiled architectures. Counts are exact Python integers even
when the product exceeds browser Number or machine-size sequence length.
Existing public page offsets remain bounded exact editor integers; this is not
a new arbitrary-size browser cursor protocol. Projection totals use decimal
strings and family candidate parameter counts use parameters_exact strings.

## Exact candidate coordinates and visible failures

The outer candidate coordinates remain model, data_task, split, initialization,
protocol. A concrete model coordinate remains its exact reference. A family
model coordinate has exactly id and bindings: the choice ID and every selected
fixed/axis binding, with no omitted, extra, changed-fixed or out-of-axis values.
Defaults are not copied into the coordinate; the exact family revision supplies
them. Every resolved binding, including defaults, is retained in the canonical
specialization receipt. Different declared choices can select the same concrete
instance; candidate identities distinguish choices, while their canonical model
and specialization identities correctly agree.

Pair whitelists and exclusions name these exact model coordinates, not a
wildcard family or guessed parameter matching. Exclusions precede resolution.
Conditional, incompatible and excluded points remain visible without manifest/
training claims. A compatible family point has model, family, specialization,
dataset and task exact references. The derived model/specialization need not
exist in SQLite during preview: they are predicted canonically from the pinned
source and bindings, without allocation, fitting, SQL writes or editable drawings.
Version-2 full candidates additionally contain parameters_exact, null until
compatibility establishes a trainable model. Version-1 candidate fields stay
unchanged. Reasons are complete in full native/CLI reads and explicitly partial
in existing bounded UI projections.

## Terminal provenance and atomic delivery

RunRecord version matches its selected experiment's version. Version-2 run
coordinates can contain a family point; resolved version-1 execution envelopes
keep their existing manifest/protocol fields but references can additionally
contain family and specialization, always together. The run's dependency closure
includes those exact derived records, so result reports, save/load, history and
portable exports do not depend on mutable heads or drawings.

Only explicit execution runs the existing task worker. Terminal admission
reconstructs the expected family model/receipt from the exact experiment, checks
the references and retains the missing derived records plus validated run under
one SQLite savepoint. A late protocol/checkpoint/admission failure rolls back
new provenance and the dirty flag. Qt completion uses the same service in its
captured working copy; no arbitrary worker-provided dependency rows are trusted.
Missing workers/RAM failures keep their actual terminal stage/status and captured
provenance. Unbound/incompatible points have compatibility-stage unavailable
attempts without a fabricated model/checkpoint. Retry retains original selection,
manifest, complete resolved references and append-only attempt history.

## Qt and agent workflows

In the existing experiment editor, Add family selection explicitly upgrades the
draft to grammar version 2 and inserts a choice atomically. Exact family choices
use the existing historical revision selector. Choice ID, fixed binding JSON and
ordered axis JSON have owned typed controls. JSON integer tokens are sent as
text without Number conversion; invalid text remains visible and blocks Check/
Save/Preview. Large values use existing staged editing. No new session token
is persisted and no new CLI draft service exists. Downgrading an owned version-2
draft is denied, avoiding an accidental semantic version change.

Family coordinates in typed pair/exclusion cards have explicit JSON controls;
existing local model pickers continue to browse only selected concrete model
references, not enumerate a potentially enormous family product. To create a
family pairing, explicitly edit the entry JSON first; the typed family-coordinate
control is then available. Rich individual symbol/axis and family-point picker
affordances remain future work, not a silently rounded or truncated selection.

Agents use ordinary ordered CLI put-spec edits (family before experiment),
preview-experiment / preview-experiment-page, run-candidate, report-data and
notebook export/execution. These call the same grammar/resolver/admission as Qt.
Archive/head selection never substitutes a newer family revision.
An ordered admit-run CLI edit accepts document and optional boolean select
(default true). It uses the same atomic terminal service as Qt completion to
recover a captured version-2 run from its already retained exact experiment/
family, without fitting or trusting additional dependency rows. select:false
retains the run bytes without adding its browsing head. Ordinary put-spec keeps
its stricter requirement that all dependencies are already retained.

Portable report runtime now includes family_axes.py in its 24-file closure.
Generation remains train-free. Actual rerun kernels retain derived provenance,
export the generated concrete architecture sources, checkpoint predictors and
per-run curves. Later display cells reconstruct exact derived instances from
the saved run summaries; they do not train again. Older complete 23-file runtimes
remain accepted only when their source does not import this dependency; genuine
earlier 20/21-file fixtures retain their original sources and checksums. A new
closure missing family_axes.py is rejected before kernel launch.

## Evidence and limits

tests/test_family_axes.py checks strict membership/order/counts, exact historical
sources, no pre-retention writes, conditional/invalid points, atomic terminal
delivery, real owned sessions, CLI parity, v7 migration/v8 minimum and a fresh
standard-library process without Torch/NumPy/Qt. Independent forward/input/
parameter-gradient and standalone export checks cover all four model points;
eight actual worker trials, deterministic retry and a fresh standalone predictor
are tested. A real Jupyter kernel executes all four points with source/predictor/
curve/provenance assertions. The actual sandboxed Qt smoke edits ordered axes,
preserves invalid JSON, adds a family choice, previews 20 candidates, runs one
through the real worker/delivery and saves/reloads its exact specialization.

These are local bounded CPU-float32 fixtures, not universal training quality,
hostile-file/native-cost/platform/accessibility certification, shared weights,
runtime recurrence or full Stage D/E acceptance.
