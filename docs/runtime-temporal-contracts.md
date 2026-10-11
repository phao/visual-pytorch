# Runtime temporal contracts — Stage D execution foundation

`app/temporal_shapes.py` defines the separate, explicit `runtime-temporal-v1`
policy. It is not an upgrade of `runtime-spatial-v1`, a constructor repeat,
hidden between-call state, or finite copying. The ordinary graph remains concrete
unless a caller selects this policy. [Family6](temporal-families.md) now retains
this policy and its exact specialization5 witness, with owned Qt editing and
CLI preview/retention/export. Temporal experiment execution selection remains
unimplemented. Family1–5 continue to reject Recurrent, and spatial runtime declarations
continue to reject recurrence anywhere in the reachable closure.

## Declaration and pure proof

Root Input node IDs map to exactly `[0]`, indexing the leading **sample** axis,
not batch B. Every declared root Input must explicitly name that axis `T`.
All other sample dimensions and rank are fixed. Sequence Inputs can have shapes
`[T,D]` or `[T,C,H,W]`; the transition receives one `[D]` or `[C,H,W]` frame.
Mask/reset Inputs have shape `[T]` and need the same explicit axis name when
declared dynamic. Lengths `[1]` and initial states remain fixed-shaped inputs.
The concrete recurrent authoring form does not automatically annotate controls
with semantic axes or select a runtime policy. Those are separate explicit edits.

```python
from app.temporal_shapes import specification, shape_contract

declaration = {"sequence": [0], "mask": [0], "reset": [0]}
spec = specification(models, root_id, declaration)
actual = shape_contract(models, root_id, spec, {
    "sequence": [17, 2], "initial": [2],
    "lengths": [1], "mask": [17], "reset": [17],
})
```

Planning allocates no tensors, imports neither Torch nor Qt/NumPy, changes no
random state, and writes no workspace. It validates the complete reachable
Module/Recurrent closure. The version1 specification has scope
`explicit-eager-runtime-temporal-contract`, status `conditional`, normalized
`runtime_inputs`, ordered `baseline_input_groups`, parameter-independence
statement, guards and output dimension dependencies. For example, the packed
trajectory-plus-final-state output depends on `sequence.T`; final-state-only
output dimensions do not. This does not say the *values* are independent of T.

The proof shares ordinary operator dependency propagation with the spatial
engine, without changing the latter's public declaration or returned bytes.
Linear's learned last-axis width and Conv2d channels cannot depend on T.
Flatten followed by a T-dependent Linear allocation is rejected before layer
construction. A last-axis Linear applied to `[T,D]` is legal: T is preserved
and one fixed matrix applies at every time. Reshape's inferred dimension
conservatively retains incoming dependencies; literal shape/product and output
shape restrictions remain actual-case guards, not inferred universal guarantees.

At each Recurrent call, frame dimensions after the leading time axis and all
state dimensions must be statically independent of runtime T. Its transition
is planned with fixed shapes. Returned sequence dimensions prepend T; final
state dimensions stay fixed. Guards retain the exact stable body Input/Output
IDs, returned port IDs, ordered distinct wire sources and control adapters.
All sequences of **one** recurrent call must have equal positive T, and its
mask/reset shapes must match that T. A declaration with fixed mask/reset inputs
therefore remains conditional; it cannot execute a different T without changing
those shapes under an explicit declaration. Lengths remain constrained by the
existing `exact-int64-v1` T ≤ 2²⁴ rule. No padding/reshaping is silently inferred.

Module calls preserve their existing single-tensor runtime behavior; variable
packed multiple-input Modules are refused because they cannot recover changing
port sizes. Recurrent has its own explicit multiple-input bindings. Reference,
per-model and runtime-context limits remain in force. A concrete recurrent
transition may itself contain recurrence; its inner time is fixed, and the
outer time remains the declared runtime axis.

## Construct once; execute and admit each actual B/T

```python
from app.core import compile_model

model = compile_model(models, root_id, runtime_temporal=declaration)
view = model.preview_shapes(actual_input_shapes, batch_size=3)
result = model(named_cpu_float32_tensors)
```

`runtime_inputs` still selects the spatial API. Supplying it together with
`runtime_temporal` is rejected. `initial_shapes` can supply an admitted first
case before construction; it never changes learned dimensions or bindings.

The model captures a private reachable source/parameter-owner snapshot and
constructs those owners once. Changing T does not rebuild, duplicate, rebind or
replace parameters or optimizer state. The original concrete `model.contract`
remains unchanged; immutable per-call plans contain actual steps and returned
port shapes. Specification/preview returns are defensive copies. Shared
Recurrent aliases use the same cell parameters with distinct explicit state
arguments. Calls never retain hidden state for the next call.

Inputs must be dense CPU float32 tensors with one common positive B, exact named
ports, fixed frame/state dimensions and declared T variations. A packed tensor
is rejected for a multiple-input runtime graph; its changing partition cannot
be guessed. Float64, integer, sparse, meta-device and mismatched batch/shape
inputs fail before packing or transition execution. Existing explicit graph
adapters alone convert float32 control inputs to bool/int64; probabilities,
NaNs and fractional/out-of-range lengths are not silently accepted.

`forward`, `forward_tensor` and `intermediate_values` all use the same actual
plan. Returned names/order and individual port tensor shapes remain intact;
`forward_tensor` packs the actual sequence-plus-state dimensions, not baseline T.
Existing active-only execution, zero padded outputs, state holds, resets to
call-initial state before active steps, differentiable all-masked zero anchors
and full BPTT without implicit detach remain unchanged. Execution controls are
not task/loss masks; this API introduces no implicit loss or optimizer policy.

Before packing/control conversion, admission charges the whole current-B/T
graph plus conservative recurrent sequence/state/output copies and controls.
Inner recurrent charges are multiplied by the outer unroll, including through
Module wrappers; memoization avoids expanding shared reference paths. This can
double-charge some graph buffers intentionally. Learned parameters are counted
in the root graph estimate, not reallocated per step. Current available RAM,
the caller's percentage (default10%) and constrained flag govern admission;
equality is admitted and unknown RAM fails closed in constrained mode. The
policy also reaches each owned recurrent runner for its immediate pre-unroll
check. Preview charges gradients or module training mode; inference estimates
require eval mode and `gradients=False`.

This is a conservative CPU heuristic, not an allocator limit, an OOM guarantee,
a GPU policy, or accounting for arbitrary external caches/concurrent jobs.
Caller-created input tensors already exist; pure shape/admission previews can
be used before allocating large inputs. Unconstrained mode still validates shapes,
types, exact controls and parameter independence.

## Checked standalone source and report closure compatibility

`app/temporal_export.py:temporal_source` generates source without constructing a
model. The artifact contains a frozen concrete ModelRevision reachable closure,
the explicit temporal declaration/proof digest, execution policy and exact
eleven-module runtime hash. It includes no dataset observations or unrelated
drafts. Core/evidence are the same selected declarations used by checked family
exports; other runtime modules are captured unchanged. No second approximate
forward interpreter is maintained.

Importing the generated file and calling `runtime_specification()` require
neither an app installation nor Torch. `build_model()` imports Torch and creates
parameters explicitly. Its returned graph has the same state-dict paths and
parameter shapes as the native temporal graph, so exact weights can be loaded
without a key translation. Runtime bytes, model bytes and reproduced declaration
proof are checked before construction. Policy overrides are explicit keyword
arguments and are validated by the captured compiler. No optimizer-continuation
or family-constructor claim is made.

New report bundles capture30 runtime files including `temporal_shapes.py`;
checked family source captures it too because the selected core compiler now
has an explicit optional temporal route. Missing imported dependencies are
rejected before a Jupyter kernel. Genuine older capsules remain accepted only
when their original captured source does not import the missing modules. This
closure addition alone does not enable temporal experiment execution. No SQL
table, container version, family grammar or experiment/run grammar changes here.

## Verification and remaining integration

`tests/test_temporal_shapes.py` covers declarations, allocation-free proof,
fixed frame/state/parameter requirements, actual-case guards, multiple streams,
final-state-only outputs, nested charges and checked source tampering.
`tests/test_temporal_shapes_execution.py` uses independent row/CNN/two-level
recurrences for outputs and all input/parameter gradients; it checks variable-T
owner/optimizer identity, shared cells, NaN padding, all-masked zeros, no state
carry, early memory refusal and exact captured/fresh-process predictions.

Next retain an explicitly versioned temporal family/specialization witness and
select it through owned GUI/CLI and experiment/data/task execution. Stable input,
state/control/output bindings and model/parameter IDs must survive save/reload,
workers and notebooks. Existing family1–5 and spatial meanings must not change.
Stage D, Stage E and the overall A–E goal remain incomplete.
