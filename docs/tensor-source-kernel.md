# Internal source constructors: not graph admission

`app/tensor_sources.py` provides pure closed-v1 normalization and lazy CPU float32
source construction for the next MS-VAR stage. The app cannot yet draw, store or
compile these operator types. No container/model/family version or discovery
claim changes at this checkpoint.

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

Required next: source roots and packing/batch context, owner schema/sharing,
Constant buffer RAM, native/app-free/spatial/temporal/temporal-v2 factories,
family symbols/specializations, retained initialization binding, checkpoints,
worker/notebook captured closures, version/archive/missing-code refusal, and
GUI/CLI schema/socket/policy/undo/persistence parity. Fully nullary architecture
semantics need an explicit batch convention; do not replace them with a dummy
Input box. Then signed target-free likelihood and complete causal Neural MS-VAR.
