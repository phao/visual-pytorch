# Retained temporal architecture families — Stage D checkpoint

Family6 and specialization5 require a `.vtw`15 container. They retain explicitly
chosen recurrent constructor/runtime semantics, not hidden state or inferred data
bindings. No SQL tables are added. Existing family1–5 and their revision hashes
keep their grammar; family1–5 still reject Recurrent. Existing spatial axes do not
become time axes. This capability is local on `task/stage-d-families`, not full D
acceptance or a public release.

## Closed family6 document

Required root fields: `version`, `id`, `name`, `root`, `models`, `symbols`,
`assignments`, `repeats`, `runtime_policy`. Optional: `constructors` (and `revision`
only on a retained document). Version must be exact integer6, not boolean.
All prior scoped symbol, typed scalar, assignment, sharing, bounded finite-repeat
and closure rules remain. `runtime_inputs` is **not** a family6 field.

`runtime_policy` is required even for fixed shapes. Its value is either JSON null
or exactly:

```json
{
  "version": 1,
  "kind": "runtime-temporal-v1",
  "inputs": {"frames": [0], "mask": [0], "reset": [0]}
}
```

Keys in `inputs` are stable root Input node IDs, not labels, child IDs or
constructor symbol IDs. `[0]` denotes the leading **sample** axis, excluding B.
Every declared axis must be explicitly named `T`; ranks and other dimensions stay
fixed. Only actual declared Inputs belong in this map; optional mask/reset Inputs
are not automatically created. A null policy means concrete fixed shapes, including
fixed T if the graph contains recurrence. Nonnull policy selects the pure
[temporal dependency proof](runtime-temporal-contracts.md); specialization must
prove frame/state and learned dimensions independent of T. Actual lengths, masks,
reset values, equal-T roles, RAM admission and shape constraints remain per-call
guards. A verified baseline is not proof that every supplied case can execute.

Family6 templates may contain the closed versioned
[Recurrent reference](recurrent-graph-contract.md): explicit `ref`,
`sequence_inputs`, `states`, `sequence_outputs`, optional `controls` and ordinary
presentation/sharing fields. The complete reachable Module/Recurrent closure is
required; unrelated templates are rejected. Sequence/state/control/returned port
IDs and their order retain their ordinary graph interpretation.

Scoped constructor annotations may target either a Module or Recurrent direct
child scope. Required child choices are supplied explicitly through parent call
bindings or valid fallbacks. Resolution uses the existing bounded context grammar;
different incompatible contexts cannot claim one shared parameter owner.
Recurrent is not a unary finite-repeat body. A unary wrapper containing recurrence
can still repeat where ordinary constraints allow it. Repeating an upstream unary
chain rewrites matching downstream recurrent argument source IDs, not its roles.

## Resolution and immutable specialization5

`app.families.specialize` returns an allocation-free resolution receipt version4
for family6. It includes exact fallback bindings, scoped `instances`, frozen model
closure, baseline contract, `runtime` and `runtime_specification`. The latter is
null for fixed policy, or the exact pure temporal specification.

Retained specialization5 has exactly `version`, `id`, `name`, `family`, `bindings`,
`model`, `instances`, `runtime` (plus its computed `revision`). `family` and `model`
are exact kind/ID/revision references. `instances` is the existing ordered complete
constructor-context record. A nonnull runtime witness is exactly:

```json
{
  "version": 2,
  "kind": "runtime-temporal-v1",
  "inputs": {"frames": [0]},
  "specification_sha256": "<64 lowercase hexadecimal characters>"
}
```

Null runtime corresponds to null policy. Admission/load/save re-resolve the exact
family and all bindings, compare canonical contexts and frozen model bytes, and
recompute the witness digest. Shape/parameter-count agreement alone is insufficient.
Forged kinds, axes, hashes, null witnesses or context changes are rejected.
Reachable Recurrent requires frozen ModelRevision3; family6 without recurrence
retains the applicable ordinary model version. Family6 always maps to
specialization5, even with null policy.

Retention is one atomic registry/drawing operation. Optional materialization
inserts Module **and** Recurrent children before parents, preserving explicit
stable IDs and refusing drawing collisions rather than replacing existing work.
A failure after registry or drawing writes restores both SQL and dirty status.
Materialized drawings are ordinary concrete baseline instances: opening them does
not implicitly select a dynamic runtime domain. Historical records remain exact
even after their mutable browsing heads advance or are archived.

## Native and checked exported construction

```python
from app.family_records import compile_specialization

model = compile_specialization(workspace, {
    "kind": "specialization", "id": specialization_id,
    "revision": specialization_revision,
}, memory_limit_percent=10.)
result = model(named_cpu_float32_tensors)
```

This explicitly constructs from exact retained dependencies; it does not use the
current drawing/head, edit SQL or train. Omitted `constrained` uses the workspace
flag. `initial_shapes` can supply the first actual case for admission. A temporal
witness selects `compile_model(..., runtime_temporal=...)`; spatial witness4 selects
the spatial API; null selects concrete execution. Current-B/T admission precedes
packing, control conversion and recurrent work. Owners are built once and reused
across valid T values, with no hidden between-call state.

The existing checked family export captures the exact retained policy. Its
`ArchitectureFamily(..., runtime_temporal=None, initial_shapes=None)` defaults to
a nonnull captured family6 declaration. An explicit temporal map is a separately
proved instance override; family1–5 reject that override. Both temporal and spatial
maps together are rejected. Exporting/importing does not instantiate/train;
construction is explicit. Class import requires Torch, while source generation
and native preview/retention do not. Exact state-dict paths, output/gradient
behavior and current RAM checks use the captured ordinary compiler runtime.

Learned semantic IDs traverse stable family/template/Recurrent/node identities,
not generated context IDs or baseline T. Strict warm starts can transfer compatible
weights across baseline-T constructor choices; incompatible shapes/identities or
forged provenance fail. This is not optimizer continuation or hidden state carry.

## Owned Qt and agent workflow

Drawing capture containing reachable Recurrent seeds family6 with
`runtime_policy:null`; dynamic T is never guessed. The owned family form exposes
a JSON policy card, stages large content through the existing editor, and preserves
invalid raw text. Nonblank edits explicitly upgrade to6 before setting policy;
blank text writes JSON null, not property removal. Older spatial fields must be
removed explicitly; they are never silently converted. Session owner, sequence,
head/origin/workspace and active-job fences remain in force.

Preview is read-only and allocation-free. Retain, workspace Save, exact checked
export and Create concrete instance are separate actions. The CLI's existing
`preview-family`, `preview-stored-family`, atomic `specialize-family` apply edit
and `export-family` use the same grammar/provenance. These actions do not select a
temporal experiment, infer task heads or data/state/control/loss roles.

## Verified boundary and next integration

Tests cover closed grammar, scoped choices, exact historical retention, container
gates, forgery refusal, late atomic rollback, complete materialization, actual CLI,
pure imports, owner/gradient parity for row/CNN cases, warm starts, mixed-T fitting,
current RAM and fresh app-free inference. A real sandboxed Qt/WebChannel workflow
edits/previews/retains/exports/materializes and reloads a version15 workspace.

Variable-T **experiment/data/task/worker/notebook execution selection is still
pending**. Capturing temporal helpers or creating a concrete instance does not
prove that integration. Existing fixed recurrent experiments and explicit spatial
runtime experiments retain their meanings. Existing-node recurrent interface
reconfiguration remains explicit CLI edits; the Qt recurrent modal creates nodes.

Run `python examples/temporal_family.py` for a disposable retained .vtw15/native/
checked-export example at T=3/7/11. Run
`PYTHONPATH=.:tests python tests/probe_family6_legacy.py` for the pinned pure
family1–5 receipt/record comparison. Neither modifies a user's files/preferences.
