# Architecture families: initial specialization checkpoint

Implemented locally on `task/stage-d-families`, not a completed Stage D or public
release. `app/families.py` resolves an external, versioned JSON family into the
existing ordinary executable architecture closure. Neither specialization nor
preview imports Torch/Qt, allocates tensors, trains, changes global preferences
or infers bindings from data. Compilation/training still use ordinary constrained
run admission. Immutable family/specialization records now live in `.vtw` v7;
the existing Qt scientific browser can inspect/import them. This checkpoint now
adds [typed owned family authoring](typed-family-sessions.md),
[explicit experiment axes](family-experiment-axes.md) and version-2
[shared finite uses](shared-parameters.md). Version3 additionally supports
[explicit boolean/enum constructor choices](constructor-choices.md), their
typed retained bindings and experiment axes. Version4 supports per-Module
[scoped constructor bindings](scoped-constructors.md) and required child arguments
without global defaults. Version5 explicitly retains runtime H/W; version6 permits
Recurrent transitions and explicitly selects a nullable temporal policy. See
[temporal families](temporal-families.md); earlier grammars do not acquire recurrence.
Version7 adds [explicit fixed-T LSTM constructors](lstm-families.md), including
typed integer shape selection for projection/direction-dependent supplied states.
All seven family grammars support [checked standalone class export](checked-family-export.md),
with exact constructor guards/provenance and semantic shape-aware warm starts.

## Family document version 1

Required, closed top-level fields:

| Field | Meaning |
| --- | --- |
| `version` | Integer 1, not boolean |
| `id`, `name` | Nonempty trimmed strings, at most 100 characters |
| `root` | Root template model ID |
| `models` | Complete reachable closure, keyed by matching model ID; no unrelated models |
| `symbols` | Ordered symbol definitions |
| `assignments` | Integer-field replacements on existing template nodes |
| `repeats` | Explicit independent finite unary-chain expansions |

Each template model has exactly `id`, `name`, `graph`. The graph has `nodes`,
`edges`, optional `input_order`, and must already be a valid concrete graph.
Existing concrete values are its checked seed, not implicit defaults for symbols.
Template nodes use the current closed operator/presentation fields; an unknown
sharing annotation cannot silently masquerade as independent weights. Wires
contain only `from`, `to`, optional `from_port`. Ordinary axes, ports, Module
references and ordered Concat sources keep their existing semantics.

A symbol has required `id`, `scope`, `name` and optional `default`, `minimum`,
`maximum`. Its stable ID is unique across the family; its human-readable name is
unique only in its model-ID scope. Two modules can both name a symbol `width`.
Renaming does not change the ID used in a binding. Defaults for bounds are 1 and
9007199254740991. Bounds/defaults must be exact integers; lower ≤ upper and any
default must satisfy the bounds. Every symbol needs an assignment or repeat-count
dependent. Cross-scope references are rejected in this initial version; explicit
cross-scope parameter bindings remain future work.

An assignment has exactly `model`, `node`, `field`, `value`. Duplicate targets
are rejected. The target field must exist in the template, preventing accidental
addition of an ignored setting. Supported fields:

Input `features` cannot be assigned when explicit `shape` takes precedence;
assign its shape instead. Activation `dim` is meaningful only for Softmax.

| Node type | Fields |
| --- | --- |
| Input | `features`, `shape` |
| Output, Reshape | `shape` |
| Linear | `features` |
| Conv2d | `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups` |
| AvgPool2d | `kernel_size`, `stride`, `padding` |
| MaxPool2d | `kernel_size`, `stride`, `padding`, `dilation` |
| AdaptiveAvgPool2d | `output_size` |
| Flatten | `start_dim`, `end_dim` |
| Permute | `dims` |
| Activation | `dim` (Softmax sample axis, excluding B) |

Values are exact integer constants, typed expressions, or arrays of expressions
for array-valued fields. No Python strings, calls, attributes, eval or algebraic
solver are accepted. Expression forms:

```json
{"symbol": "hidden-width"}
{"op": "mul", "args": [2, {"symbol": "hidden-width"}]}
{"op": "div", "args": [6, {"symbol": "channels"}]}
```

`add`/`mul` take 2–16 arguments; `sub`, exact `div`, explicit `floordiv` take two.
Division requires a positive denominator; `div` rejects a remainder. Every
intermediate, not only the final value, must remain within ±9007199254740991.
An expression is limited to depth 24 and 256 nodes. Arrays have 1–8 entries;
operator-owned rules check rank, positivity, axis permutations, channel
divisibility and reshape element preservation after binding.

## Repetition is not parameter sharing

A repeat has exactly `id`, `model`, `nodes`, `count`, `mode`. `nodes` is an ordered
nonempty unary-operation chain, with one ingress and one egress and no branch
crossings. Input, Output, Concat, Splitter and Replicator are not repeat bodies.
Recurrent is not a unary repeat body, even in family6. An ingress from a router
retains its explicit `from_port`. Downstream Concat sources and Recurrent
sequence/state/control bindings are rewritten in place, preserving explicit
wire identities and feature packing. Bodies cannot overlap
or nest in version 1. Consecutive repeat bodies are supported.

Version1 supports only `mode: "independent"`. Version2 and3 additionally accept
`tied` and explicit local template `share_with` ownership; see
[shared-parameters.md](shared-parameters.md). `recurrent` and unknown modes fail
explicitly, never becoming independent copies. `count` is a positive
integer expression; zero is not silently treated as a valid identity operation.
Occurrence node IDs are `repeat-id/index/template-node-id`. Prefix collisions
with any existing node are rejected. Retained indices have stable local paths
when depth changes; changing the binding still creates a distinct specialization.
An independent Module occurrence constructs a fresh module unless explicit
local ownership ties its complete subtree to another occurrence.
Copies of equal values or reuse of a template reference do not tie parameters.

Expansion counts are checked before copies are materialized. The current concrete
prototype's 64 blocks/128 wires per model still apply. Families allow at most
64 template models, 128 symbols, 512 assignments and 32 repeats. These are explicit
current implementation bounds, not new hardware-safety rules or final large-model
limits. New seed positions use simple left-to-right layers for expanded models
or templates lacking coordinates; existing fully positioned integer-only
templates preserve geometry. This is not the configurable browser layout engine.

## Results and identities

`specialize(family, bindings)` either returns a verified concrete instance or
raises GraphError. Bindings map stable symbol IDs to declared scalar values,
not symbol labels: integers for v1/v2, optional booleans/enum strings for v3.
Unspecified symbols use explicit defaults; otherwise execution is refused.
Templates are never modified. All concrete models are shape-checked before return.
Errors name the model and assigned fields; arithmetic errors name their exact
assignment/repeat origin. No incompatible shape is guessed or repaired.

The receipt includes version, family ID/hash, resolved bindings, specialization
hash, original→resolved model-ID mapping, concrete models/root, repeat occurrence
paths, root contract and the existing computational architecture hash.
`family.revision` is the existing canonical `digest(family)` SHA-256 convention.
`specialization` hashes `{version:1,family:<hash>,bindings:<resolved-bindings>}`.
Each model ID is `family-` plus the digest of
`{specialization:<hash>,model:<template-model-id>}`. Module references are remapped;
boundary block/port IDs and declared input order are preserved. Deterministic
IDs never authorize checkpoint warm starts or implicit replacement. The family
hash includes its complete presentation/name bytes; the ordinary architecture
hash follows existing presentation-independent rules.

`preview` returns `verified` plus the resolved receipt, `conditional` plus required
symbol IDs/scopes/names, or `incompatible` plus a reason. Bad grammar, unknown IDs
and supplied wrong scalar types raise; valid integers outside declared bounds or
v3 strings outside declared enum domains are
reported incompatible even if other bindings remain unresolved. The current
preview is allocation-free for tensors, not a universal memory/cost guarantee.

## CLI and .vtw boundary

`preview-family FAMILY_JSON [--bindings BINDINGS_JSON]` reads an existing .vtw
workspace for the ordinary file/revision checks but does not write it. Omitted
bindings use defaults. As elsewhere in the CLI, `-` denotes stdin.

An ordinary atomic `apply` batch now accepts:

```json
[{"op":"specialize-family", "family": {}, "bindings":{"hidden-width":4}}]
```

Replace `{}` with the full family document. This resolves, checks collisions
against existing architecture IDs, inserts child architectures before parents,
then uses the existing final-batch validation and atomic workspace save. There is
no implicit replacement or new SQLite table. Family/specialization semantics
require container version 7. Result `edits[].resolved` is the complete receipt;
`family`, `model`, `specialization` are exact retained references and
`materialized` records whether editable drawings were inserted. A failing batch or stale
`--expect-revision` leaves disk unchanged. The imported concrete models are
ordinary editable architectures, can be frozen with `freeze-model`, selected in
current experiments, rendered/inspected and exported without family-runtime code.

The family, resolved bindings and frozen model are retained automatically inside
`.vtw`. Editing the concrete instance does not update any of those immutable
records. Typed owned GUI editing and [family sweep axes](family-experiment-axes.md)
are now implemented, as are [explicit finite tied uses](shared-parameters.md).
The [runtime spatial backend/API](runtime-spatial-contracts.md) now accepts an
explicit H/W declaration in native/checked exported constructors with statically
fixed parameter shapes and actual-case guards. Family v5, specialization v4 and
the typed editor now retain this declaration/witness in `.vtw`12. Ordinary frozen
drawings and worker/task datasets still have fixed shapes; retaining provenance
does not upgrade those paths to variable-resolution execution. Testing two concrete
specializations alone does not prove dynamic safety. Recurrent state/length/mask
contracts remain subsequent Stage D work.

### Retained records and exact stored preview

`family` uses version1, the explicit sharing extension in version2 or typed
constructor version3, scoped Module constructor version4 or retained runtime
contract version5, with only the optional verified registry
`revision` added. Its revision is the external definition's original digest.
Unbound families may be retained as drafts; retaining them does not compile or
materialize a model. A `specialization` has exactly `version:1`, `id`, `name`,
`family`, `bindings`, `model`, plus optional verified `revision`. Both references
are exact `{kind,id,revision}` objects of their declared kind. Bindings must
include every stable symbol ID, even defaults; booleans are not integers here.
Specialization v2 retains family v3's exact typed values; v1 retains family v1/v2.
Specialization v3 pairs with family v4 and retains canonical available fallbacks
and all exact resolved local contexts; child-only arguments need not exist as
global bindings. See [scoped-constructors.md](scoped-constructors.md).
Specialization v4 pairs with family v5 and adds mandatory `runtime`, either null
or the closed canonical inputs/specification hash witness reproduced on every
admission/load/save. See [runtime-spatial-contracts.md](runtime-spatial-contracts.md).

Admission/load/save re-resolve the exact family with those bindings and compare
the expected frozen `model_document` identity under the reference JSON hash.
A forged binding or same-shaped model with another activation is rejected even
if its own document hash was recomputed correctly. This is integrity, not a
hostile-file sandbox or proof of experiment validity.

`preview-stored-family ID REVISION [--bindings BINDINGS_JSON]` previews one exact
retained family read-only. It never substitutes the current browsing head.
The `specialize-family` edit accepts exactly one of `family` (full definition) or
`family_ref` (exact retained reference); optional `bindings`, `id`, `name`,
`materialize` configure the specialization. `materialize` defaults to true;
false stores provenance/frozen model without adding or replacing drawings.
Selecting a historical family for resolution does not move its browsing head.

The in-memory admission uses one SQLite savepoint for all revisions and drawings.
Existing per-operation connection-context commits are suppressed only inside
this synchronous internal operation, with no event-loop yield. A late failed
write rolls back all logical rows and restores the prior dirty flag/connection;
an existing caller transaction remains open. SQLite `total_changes` is diagnostic
and is not rolled back. The CLI still applies its final disk CAS/atomic save.
Undo removes the new drawing but retains immutable provenance/dependencies.
Archive removes a head, not family/model/specialization bytes. Old containers
migrate only in memory until explicit Save; old reader compatibility is not
promised for new v7 files.

The Qt **Browse** selector includes alphabetically ordered Architecture families
and Specializations. Metadata pages and summaries remain bounded; full intrinsic
validation is still native and may be expensive. Raw import/ordinal inspection
are supported. The [typed family editor](typed-family-sessions.md) also creates
families from drawings, edits symbols/assignments/repeats, previews bindings and
creates exact retained concrete instances. This is not full D acceptance.

## Verification and references

`examples/families/width-depth.json` produces 37 parameters at width 3/depth 2,
77 at width 4/depth 3. `tests/test_families.py` covers independent forward/input/
parameter gradients, concrete export in a fresh process, multiple batch sizes,
actual fitting/checkpoint admission, scoped names, exact arithmetic, rejected
sharing requests, collisions, bounds, branches/router/Concat repeat boundaries,
derived reshape and atomic save/reload/freeze. The grouped Conv2d/adaptive-head
fixture has exactly 179 parameters; incompatible group bindings explain why.
`tests/qt_family_smoke.py` tests the CLI-imported instance after .vtw reload in
the real sandboxed Qt editor, without importing host Torch or claiming a family UI.
`tests/test_family_records.py` covers exact dependencies, forged bindings,
historical revisions, failed-write rollback, undo/archive, v6 migration, v7
roundtrip, bounded browser reads and CLI CAS/failed-batch preservation.

The pinned nepr symbolic specification motivates this construction. PyTorch's
[Module documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
documents actual module/parameter identity and duplicate removal: independent
copies and tied modules must not be treated as the same operation. That distinction
is preserved here by explicit owner identity, never implying it from
equal widths or template reuse. Numerical claims above are tested locally, not
inferred from documentation.

[SQLite savepoints](https://www.sqlite.org/lang_savepoint.html) and
[Python SQLite connection context managers](https://docs.python.org/3/library/sqlite3.html#how-to-use-the-connection-context-manager)
informed the atomic admission: a connection-context exit commits a transaction,
while an inner savepoint release must not commit its outer caller transaction.
