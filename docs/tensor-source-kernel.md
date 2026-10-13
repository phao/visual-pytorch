# Tensor sources and explicit batch context

`app/tensor_sources.py` provides pure closed-v1 normalization and lazy CPU float32
source construction. The Neural MS-VAR task branch registers `Constant` and
`TensorParameter` in diagrams, owner schemas, native/generated/runtime compilers,
CLI discovery and captured notebooks. This is prototype groundwork, not an exact
Neural MS-VAR likelihood implementation. New contracts require container27,
ModelRevision7, Family11 and Specialization10; SQLite tables are unchanged.

`Constant` has a nonempty positive sample shape, optional unique axis names and
exactly one of compact `fill` or rectangular nested `values`. Numeric literals
round to finite float32 with signed zero preserved. It has zero parameters and
one nonpersistent buffer named `value`: fixed data comes from immutable graph
content, not checkpoint weights. `TensorParameter` has one real zero-default
`nn.Parameter` named `value`. Initializers and freezing belong to the existing
external parameter-policy contract, not arbitrary new node fields.

`forward(batch_size)` returns `[B,*shape]` by expanding a view; it does not receive
a dummy data tensor. B can be zero. Consumers must not mutate the expanded view
in place. Construction checks a numerical element budget before allocations.
An explicit `admitted_elements` can override the prototype's one-million-element
default, but is not authority to allocate or an RSS bound. Production construction
must first pass complete model/run RAM admission. Shape inspection permits large
compact declarations without allocating their data or imposing64-feature limits.

Module import and normalization do not import Torch/NumPy/Qt. `runtime_class(nn)`
is shared with the inspected app-free source emitter. `constructor_args` refuses
unresolved sharing; graph integration must reuse the actual owner module and
stable component identity, not guess from labels or equal values. Nonparametric
literal buffers still consume memory even though optimizer counts are zero.

Main reviewed all193 source/410 initial test lines and reran21 cases0.858s.
After integration,80 source/filter/reference/layout cases1.313s pass. Fresh
app-free Python agrees on tensor bytes, signed zeros, owner state and gradients;
allocation guards, external initialization/freezing and explicit alias/hash-once
behavior are checked. A process-wide test interop-pool reset was removed from
test-module import and explicitly checked after prior Torch work. No source
fit, actual new-source GUI interaction or new-schema conformance is claimed.

## Graph, ownership and persistence

Sources have zero incoming sockets and one tensor output with ordinary fan-out.
They enter the DAG as roots, but never appear in dataset input packing. A graph
with real inputs takes B from those inputs. A graph with no Inputs takes
`model(batch_size=B)` or `model.forward_tensor(batch_size=B)` explicitly; no
dummy tensor, data column or hidden batch default is introduced. B may be zero
on the concrete nullary path. Existing runtime input policies require a real
Input; they do not turn sources into variable constructor dimensions.

A nullary referenced architecture is invoked by an explicit
`{"type":"Module","ref":"...","batch_only":true}` with no incoming wire.
Ordinary Module calls still take exactly one tensor input. The flag and child
interface must agree. Sources participate in operation-layer ordering: checkpoint
paths use `layer_N.value`. Semantic owners use canonical JSON component arrays,
for example `["architecture-id","source-id","value"]`. A `share_with` occurrence
must name a compatible local TensorParameter shape/axes and reuse its exact owner
module. It does not create a second checkpoint key or optimizer parameter.

Constants own no parameter-policy selector. Their literal buffers are rebuilt
from immutable graph data and counted separately from parameters. Shared Module
buffers count once; recurrent transition buffers do not multiply with T.
Construction and source-bearing calls admit RAM before source allocation/batch
packing. Batch estimates distinguish gradient-enabled execution and no_grad
inference. These remain heuristics, not OOM guarantees.

Family11 permits source shape assignments using the existing checked integer
expression grammar and optional unique axis names. Sources and nullary Module
calls are not unary repeat-chain bodies. Retained specialization reproduces the
exact family, bindings, frozen Model7, owner policy and nullable temporal-v2
witness. Older headers cannot hide new drawings or archived revisions.

## UI and CLI

Building blocks adds independent Constant or Tensor parameter branches, each
with an initial Output. Shape and optional axis names live in Inspector; learned
sources reuse existing initialization/freezing/sharing controls. Constant has
either a finite uniform fill or a rectangular JSON literal. Changing editor mode
alone writes nothing. The small literal editor accepts up to2000 characters;
larger literals and atomic shape/value changes use the existing CLI JSON edits.
Source roles have distinct icons, theme colors and outgoing-only sockets; normal
compact-icon drag/folding and undo/redo apply.

Workspace-free `operator-schema Constant`, `operator-schema TensorParameter`,
`capabilities` and `format-schema`, plus workspace `ports` and parameter-policy
queries, use actual registered grammar. Do not infer source semantics from labels.
Existing supervised task training can learn source values in graphs with real
inputs. A new target-free/nullary training-task grammar is not implied.

## Validation and remaining work

The initial integrated125 focused Python cases (three Qt-only skips separately
verified natively) and372 JavaScript cases pass. New source tests cover actual
optimizer updates, shared gradients, buffer/signed-zero/owner counts, nullary
native/app-free/family parity, three runtime policies and version/archive refusal.
Real native task worker checkpoints and predictions match; actual saved/rerun
Jupyter kernels replay source checkpoints and metrics. A sandboxed Qt smoke
checks palette insertion, source sockets/Inspector, signed-zero/literal editing,
undo/redo, compact-icon dragging and save/reload without numerical host imports.
Review follow-up fixes explicit runtime batch keywords and inference RAM parity;
their final regression must be read from current test evidence, not this initial
gate. Full source-wide suite and additional binding/adversarial coverage remain.

Next: representation transforms, signed target-free Gaussian sequence likelihood,
causal lag windows and full K5,p3,D3 neural MS-VAR. The internal filter primitive
and these real source boxes do not complete that model or its CLI report scope.
