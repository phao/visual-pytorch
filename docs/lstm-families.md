# Explicit LSTM architecture families

This local Stage D checkpoint introduces Family7, Specialization6 and `.vtw18`.
It does not complete D, add runtime-T LSTM, change ModelRevision4, or implement
LoRA/MS-VAR. Family1–6 and runtime-temporal-v1 keep their original meanings and
reject LSTM; old spatial runtime policies reject it too.

This document specifies the preserved fixed-T grammar. The later
[Family8 runtime-T extension](runtime-temporal-v2-families.md) is separate;
it does not reinterpret Family7 or complete D/E.

## Closed grammar and constructor-dependent shapes

Family7 keeps Family6's required nullable `runtime_policy` and optional scoped
`constructors`. Any closure containing LSTM requires `runtime_policy: null`.
An LSTM-free Recurrent family may still select the existing temporal-v1 policy.
Templates remain checked concrete seeds, not executable expressions or partially
valid layers. Their ordinary LSTM fields, source-role bindings, supplied states,
length adapter and three named returned ports follow [the operator contract](lstm-graph-contract.md).

Integer assignments admit existing `hidden_size`, `num_layers`, `proj_size`
fields. Boolean constructor assignments admit existing `bidirectional` and
`bias`. Node/source/port IDs are not inferred from labels or diagram geometry.
The family's ordinary scoped symbol/constructor-context rules still apply.

Family7 introduces this integer expression form, also preserved in Family8:

```json
{"select": {"symbol": "directions"}, "then": 2, "else": 1}
```

The predicate is an exact boolean literal, a same-scope boolean symbol reference,
or an exact two-integer comparison:

```json
{"select": {"cmp": "gt", "args": [{"symbol": "projection"}, 0]},
 "then": {"symbol": "projection"}, "else": {"symbol": "cell-width"}}
```

Comparison names are exactly `eq`, `ne`, `gt`, `ge`, `lt`, `le`; their two
arguments use the same integer expression grammar. `select/then/else` and
`cmp/args` have no extra keys. Both branches are grammar/scope checked; only
the selected branch is evaluated. No Python evaluation, tensor control flow,
float coercion, integer/enum truthiness or implicit symbol binding is allowed.
All symbols in an instantiated scope still need defaults or explicit bindings,
including those referenced only in an inactive branch. Expressions share the
existing depth24/node256 budget, exact JavaScript-round-trippable integer range,
positive-denominator and exact-division rules. Older grammars reject `select`.

Let $H$ be cell width, $P$ projection width, $L$ layers and $D$ directions.
The example explicitly derives $D=2$ for bidirectional, otherwise1, and
$Q=P$ for positive projection, otherwise$H$. Sample shapes are sequence
`[T,D*Q]`, hidden `[L*D,Q]`, cell `[L*D,H]`; supplied sequence is `[T,I]`.
These expressions are real assignments to Input/Output shapes, not captions.
Projection must remain zero or strictly below cell width. All graph checks and
current-RAM admission apply after resolution and before numerical allocation.

For one layer/direction, independent counting uses
$4H I_{layer}+4H Q+8H\mathbf{1}_{bias}+HP\mathbf{1}_{P>0}$,
where $I_{layer}=I$ for the first layer and $DQ$ thereafter. Sum over layers and
directions, counting canonical shared owners once. LSTM is not a unary finite
repeat body. Repeating an upstream unary chain explicitly rewrites downstream
sequence/hidden/cell/length source bindings; it does not approximate recurrence.

## Retention, compatibility and authoring

Specialization6 requires exact family/model references, full bindings, constructor
contexts and nullable `runtime`. Fixed LSTM has null runtime. Its allocation-free
resolved receipt is version5; the frozen concrete model remains ModelRevision4.
Load/save re-specialize and compare exact retained model meaning, not only shapes
or counts. A `.vtw17` or earlier header carrying Family7/Specialization6 refuses,
including an unmaterialized family. Opening a genuine older workspace upgrades
only its in-memory working version; disk changes only on explicit Save.
No SQL table, task/experiment/protocol grammar or sidecar is introduced.

New architecture family from a concrete LSTM drawing seeds Family7/null policy.
Owned cards and staged JSON preserve its version7 acknowledgement; editing an
existing older family may explicitly upgrade but not downgrade or forge a version.
The v2 runtime-policy edit explicitly queues Family8 before retaining that policy;
it still requires an admitted v2 closure and explicit removal of fixed Output
annotations if those are incompatible with the desired actual T.
Preview does not allocate/train/write. Retention and concrete materialization use
the same guarded services as CLI, with collision refusal and atomic rollback.
Undo may remove a browsing head/drawing but preserves immutable retained rows;
historical evidence must be addressed by exact revision, not the current head.

```sh
python examples/lstm_family.py NEW_WORKSPACE.vtw
python vpw_cli.py NEW_WORKSPACE.vtw preview-stored-family FAMILY_ID REVISION
python vpw_cli.py NEW_WORKSPACE.vtw export-family FAMILY_ID REVISION --output NEW_FAMILY.py
```

The example accepts an optional `--bindings JSON_FILE`, never overwrites a target,
and creates no dataset, fit, checkpoint or implicit initial state. Defaults have
1,400 parameters; the forward-only, unprojected default-width two-layer binding
has784. Concrete results may use existing typed tasks; changing time is a new
concrete specialization, not a runtime-T execution policy.

## Standalone class and warm-start identity

The checked export captures and initializes `lstm_nodes` alongside the complete
resolver/compiler closure. Constructing `ArchitectureFamily` uses those captured
bytes, not an installed mutable application. Input/state/parameter gradients and
all returned ports agree with native and ordinary concrete source exports.

Stable semantic parameter IDs extend the existing JSON component paths with
the canonical LSTM owner and real gate/layer/direction/projection parameter
suffix. They do not depend on time binding, resolved model hash or layer-list
ordinal. Shared aliases have one owner. Strict warm start validates exact source
family/runtime/bindings/schema/tensors and the entire load plan before writing;
time-only changes can load compatible weights. Constructor changes may alter
shapes/parameter sets and require explicit partial loading, with reported
missing/skipped/unused entries. This is weight initialization, not optimizer
resume or parameter freezing.

## Verification boundary and references

Dedicated pure tests cover16 constructor combinations,100 additional bounded
points, independent counts, conditional AST adversaries, source rewrites, shared
owners, old grammar refusals, retention rollback, `.vtw18` fences and real CLI.
Numerical tests compare16 combinations through native/checked-family/concrete
export against an independent explicit gate scan, including all gradients,
NaN padding, zero lengths, semantic warm starts, a35-update fit and typed-worker
checkpoint parity. Actual sandboxed Qt checks exercise drawing seeding, typed
selection/default cards, invalid text preservation, preview, exact source export,
materialization, history and reload with no host Torch import. Complete regression
results and any outstanding acceptance are recorded in development.md.

Remaining work includes explicit runtime-T LSTM and its experiment/kernel routes,
then the original nepr/nars D acceptance audit. A successful fixed-T family is
not a substitute for those requirements or for Stage E.

[PyTorch LSTM documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html)
checked2026-10-11 defines the projected hidden/cell and direction/layer contracts
used here; numerical verification uses installed PyTorch2.13.0, not a claim of
testing every2.14 backend. Formula and gate checks are independent numerical
evidence, not inferred correctness from matching API names.
