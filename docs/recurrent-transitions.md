# Explicit recurrent transition execution v1

## Status and boundary

`app/recurrent.py` is an implemented eager execution primitive over an ordinary
compiled graph. The [recurrent graph contract](recurrent-graph-contract.md) now
exposes it through a closed multiport graph node, container14/ModelRevision3,
existing atomic CLI graph edits, concrete task workers, saved predictors and
captured notebook helpers, with guarded Qt palette/Inspector authoring. A separate
[runtime-temporal-v1 backend API](runtime-temporal-contracts.md) now permits
declared root T variations; retained temporal family/experiment selection remains
Stage D work. Old ModelRevision/family hashes and
task interpretations are preserved; finite tied repeats do not become recurrent.

The existing graph defines one transition; it stays acyclic. Recurrence lives
in a shared transition object plus explicit state bindings and a runtime loop.
This is not an independent finite graph repeat, nor implicit mutable hidden
state. CNN operations, nested architecture references, multiple inputs/outputs
and explicit parameter sharing inside that graph retain their existing meaning.

## Closed specification

The exact v1 keys are `version`, `sequence_inputs`, `states`, `sequence_outputs`.
Version is integer `1` (not boolean/float). All IDs below are nonempty strings.

```json
{
  "version": 1,
  "sequence_inputs": ["observation"],
  "states": [{"input": "previous", "output": "next", "port": "hidden"}],
  "sequence_outputs": [{"output": "next", "port": "trajectory"}]
}
```

`sequence_inputs` is a nonempty ordered list of unique transition Input IDs.
`states` is a nonempty ordered list of objects with exactly `input`, `output`,
`port`. Inputs, next-state Output IDs and returned state ports are each unique
within this list. Every transition Input must be bound exactly once, either as
sequence or state. Labels, node order and packed dataset offsets are not bindings.

`sequence_outputs` has objects with exactly `output`, `port`. It may be empty
for final-state-only use. Returned sequence ports are unique. Several explicit
ports may reference the same transition Output; that Output may also be a next
state. Sequence and state results live in separate dictionaries and need not
have disjoint names. Not every transition Output must be emitted.

Every referenced Output must exist. Each next-state Output has exactly the
sample shape and semantic axes of its initial-state Input. No inferred state
broadcast, reshape, zero initialization or axis adapter is supplied. Such
operations must be explicit in the transition graph or caller's input creation.

`recurrent_contract(models, root_id, specification)` performs allocation-free
validation. It returns a private interface snapshot, graph architecture identity,
semantic recurrent identity, exact parameter schema and statistics. The semantic
identity includes transition meaning and complete interface, not labels/geometry.
Parameter state paths add `transition.` to existing native state-dict paths;
stable parameter IDs remain those of the transition architecture. Parameters are
counted once, including explicit sharing, never multiplied by runtime T.

## Compile and call

```python
from app.recurrent import compile_recurrent_transition

network = compile_recurrent_transition(models, root_id, specification)
result = network(
    {"observation": observations}, {"hidden": initial_hidden},
    lengths=lengths, mask=mask, reset=reset,
)
trajectory = result["sequences"]["trajectory"]
final_hidden = result["state"]["hidden"]
```

The native object is an `nn.Module` with exactly one registered `transition`.
Compilation takes a private graph/interface snapshot of only the reachable
runnable closure, excluding datasets, results, training settings and unrelated
drafts. Runtime calls never
instantiate another cell or parameter set. Every call explicitly supplies initial
state; the object does not retain final state for the next call.

All sequence tensors use CPU float32 `[B, T, *transition_input_sample_shape]`.
All initial-state tensors use CPU float32 `[B, *state_sample_shape]`. B and T
are positive and common to all streams; T is runtime-variable without changing
parameters. State shapes, feature shapes and axes are fixed. Empty temporal
containers, ragged tensors, implicit padding, coercion, sorting and broadcasting
are rejected. A zero-length observation is represented by length 0 inside a
nonempty padded temporal container. Parameters must remain CPU float32 too.

Optional controls are checked before any cell call:

* `lengths`: CPU int64 `[B]`, exact values from 0 through T;
* `mask`: CPU bool `[B,T]`, including arbitrary holes;
* `reset`: CPU bool `[B,T]`.

Controls omitted mean full validity and no resets. At time t, row b is active
iff `(lengths is absent or t < lengths[b])` and `(mask is absent or mask[b,t])`.
Only active rows are gathered into the transition. Their reset flag replaces
previous state with the initial state supplied to **this call**, before the step.
Inactive resets do nothing. Inactive rows retain previous state and emit zeros
at that time. No transition runs if a whole step is inactive; an entirely masked
call never invokes it. This also prevents NaN padding from entering cell math.

Returned sequences are `[B,T,*transition_output_sample_shape]`; returned states
are `[B,*state_sample_shape]`. Inputs/controls are not changed in place. Out-of-place
state replacement preserves full backpropagation through time. There is no hidden
detach. To truncate BPTT, the caller explicitly detaches carried state. For a reset
in a later chunk, the reset baseline is that chunk's supplied initial state,
not a remembered beginning-of-stream state.

Differentiable zero anchors use empty slices, never padding values. Entirely masked
sequence losses can be backpropagated and give zero parameter/input gradients.
This executor does not define a loss, normalization denominator or optimizer
policy. A trainer must mask its objective and skip optimizer updates when there
are no valid objective elements: a zero gradient alone does not prevent Adam's
existing momentum or weight decay from moving weights. Final state remains
the initial state numerically in an entirely inactive call.

## Safety

`recurrent_memory_report(contract, B, T, constrained=True,
memory_limit_percent=10., training=True)` is pure and Torch-free. It observes
current free RAM, including observable cgroup-v2 headroom, and charges:
runtime headroom, one parameter/optimizer/evidence set, sequence/output/state
copies, retained per-step activations, conservative per-step convolution workspace
and controls. Constrained execution requires the total to fit the selected fraction
of current available RAM; unknown RAM fails closed. It is an admission heuristic,
not a hard allocator limit or an OOM guarantee. It does not discount masked steps.

Construction checks a B=1/T=1 inference estimate before importing/allocating Torch
layers. Each call checks actual B/T before constructing the unroll, including
inference under `eval()`/`no_grad()` versus potential training/autograd execution.
Disabling constraints is explicit and does not disable type/shape/control checks.

## Portable source

`generate_recurrent_code(models, root_id, specification, constrained=True,
memory_limit_percent=10.)` returns app-free Python source. It contains the exact
compiled transition, interface, policies, RAM observation/estimate and shared
forward helpers. Exported `model` is the recurrent module. Its state-dict paths
match native paths; `recurrent_identity` matches native contract identity. There
are no raw dataset records, learned weights or dependencies on Qt or `app`.
Callers explicitly load compatible weights when needed. Source capture rejects
helpers whose on-disk implementation differs from the loaded implementation.
No `torch.compile`, tracing, ONNX, fusion or universal symbolic shape proof is claimed.

## Research basis and validation

PyTorch's [RNN contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RNN.html)
separates sequence input, initial hidden state and final hidden state, and describes
reuse of recurrent weights. This informed the explicit-state boundary, not a claim
that this generic executor is an optimized `nn.RNN` kernel. Our mask-hole and
reset-to-call-initial semantics are explicit project choices.
PyTorch's [out-of-place index_copy](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_copy.html)
provides the selective state replacement operation; gradient behavior is verified
independently rather than assumed from the UI.

Tests include independent per-row functional outputs/input/state/parameter
gradients, variable T, zero lengths, hole masks, inactive resets, NaN padding,
multiple state streams, tied parameters, spatial CHW state, nested references,
manual chunk carry/full BPTT, bounded MLP/CNN fits, RAM-before-allocation refusal,
native/export equality and real fresh-process app-free inference/training.
