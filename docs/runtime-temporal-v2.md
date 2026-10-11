# Runtime-variable T with explicit LSTM states: version 2

Status: bounded native/backend and concrete standalone-export implementation on
retained D. **Not yet admitted by architecture families, typed TaskBundles,
ExperimentSpecs, GUI runtime-policy forms or the CLI.** Those routes require
separate explicit version fences and remain the next implementation work.
This is not full Stage D or E completion.

## Selection and backward compatibility

The new opt-in is app.core.compile_model(models, root,
runtime_temporal_v2={"sequence": [0]}, constrained=...), or
app.temporal_shapes_v2.compile_temporal_model(...).
It is mutually exclusive with runtime_inputs (spatial) and runtime_temporal
(temporal-v1). Version 1 still rejects any LSTM in the transitive closure;
neither its proof grammar nor its interpretation is widened.

Pure specification returns version 2 and scope
explicit-eager-runtime-temporal-v2-contract. Root Input IDs explicitly declare
the exact list [0], with leading semantic axis T. All undeclared dimensions,
all ranks and every supplied state dimension stay fixed. The existing LSTM
operator remains op_version: 1; no operator or SQLite field is reinterpreted.
Container18, ModelRevision4, Family7 and Specialization6 are unchanged.
Their current policies still refuse runtime-T LSTM.

## Computation and proof

Inputs are named dense CPU float32 tensors. Sequence samples have [T,F];
batch size B is common and positive. Write D=2 for a bidirectional block
and D=1 otherwise, L=num_layers, H=hidden_size, and
Q=proj_size when positive, otherwise Q=H.

| Role | Input/output shape including batch |
| --- | --- |
| sequence input | [B,T,F] |
| hidden input and return | [B,L*D,Q] |
| cell input and return | [B,L*D,H] |
| sequence return | [B,T,D*Q] |

The learned gate/projection constructors do not depend on T. A static
dependency proof propagates declared axes through routes and refuses runtime
dependencies in learned widths, supplied hidden/cell dimensions or lengths
widths. In particular Flatten followed by learned Linear cannot absorb runtime
T into its weight shape; a last-axis Linear head may preserve T.

Per-call pure graph planning checks actual shapes, optional declared Output
shapes, reshape divisibility, source/port roles and nested reference contexts.
The existing 64-context guard remains. A proof is conditional, not a promise
that every T passes fixed reshape/output declarations. Multi-input Module
boundaries retain their explicit packed, fixed-shape contract: v2 does not
invent named runtime sockets for them.

There is no hidden initialization, state carry, detach, loss mask, dropout,
state persistence or dataset inference. Every call uses the supplied hidden and
cell. Existing gate order, layer/direction order and projection semantics remain.
Lengths are optional fixed [B,1] float32 inputs under the explicit
exact-int64-v1 adapter. Values must be exact finite integers in [0,T], with
T <= 2**24. Without lengths every frame is active. Zero-length rows return
zero sequences and unchanged supplied states; inactive padding may contain NaN.
Input and parameter gradients must be exact zero for an all-empty sequence
loss. Active frames and initial states must be finite.

## Current-RAM admission

Initial actual shapes are checked before constructing learned owners. Without
initial shapes the declared baseline is checked. For every call, shape and RAM
admission precede graph packing, control conversion or LSTM execution.

Version-2 reports retain the existing whole-graph and recurrent-tree estimate
and add lstm_unrolls for gates, states/backward storage and boundary copies.
For each LSTM call with plan activation elements A, batch B, input sample
elements I and total returned sample elements O, the additional charge is
32*B*A + 16*B*(I+O) bytes. LSTM calls inside an outer Recurrent body are charged
outer-T times; Module descendants are charged too. Aliases count parameter
owners once but charge each execution's activations. The parameter allowance is
at least 128*unique_parameter_elements, including under eval/no_grad, matching
the conservative existing per-LSTM allowance.

Backward/gate storage is conservatively retained for inference. Current available
RAM and the explicitly supplied user percentage determine admission; unknown
available RAM refuses constrained execution. Exact equality with the allowance
passes. Unconstrained mode disables the RAM refusal, not shape/version/type
guards. This is a heuristic, not an allocator limit or an OOM guarantee.
Per-LSTM and generic Recurrent wrappers retain the selected constrained policy.

## Standalone source

app.temporal_export_v2.temporal_source(...) generates version-2 checked
concrete source with policy runtime-temporal-v2. Generation/import/proof
construct no parameters and need neither Torch nor the installed application.
build_model(...) then constructs the fixed owners once. The captured closure
includes and initializes temporal_shapes_v2, LSTM and recurrent helpers.
Model bytes, runtime source bytes and shape-proof identity are checked.
Calling with different T reuses the same state-dict and optimizer owners.
Generated notebook runtime closures also capture the new helper, so existing
spatial/temporal-v1 family exporters keep working in fresh kernels. Omitting a
required helper is refused before a kernel starts; genuine old bundles retain
their own captured code. This does not admit a v2 TaskBundle.

This is not a checked-family source, typed scientific experiment export or a
notebook rerun. Those routes must gain supported versioned contracts before
claiming parity. Existing fixed-T families and temporal-v1 exports remain usable.

## Verification and research

Dedicated tests use explicit row/time gate equations, not nn.LSTM or
LSTMCell, as the independent numerical oracle. They cover all 16 combinations
of projection, directions, bias and layer count, multiple T values, all
input/state/parameter gradients, NaN padding, empty rows, ownership, nested
full-BPTT, 35 variable-time Adam updates and fresh app-free checkpoint inference.
Pure tests exercise 64 bounded constructor points at three T values, old-version
refusals, actual-shape guards, nested/alias charges, exact RAM boundaries and
pre-allocation refusals. Check campaign evidence for actual test results and
limits; this paragraph describes coverage, not a full-platform PASS claim.
The separate probe_temporal_v2_branches.py also checks independently varying
time lengths across two shared-weight branches, with all input/state/owner
gradients through native and captured export. Manual probes are not counted as
extra unittest cases.

Primary references checked 2026-10-11:

- [PyTorch LSTM](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html):
  distinct projected hidden versus cell widths and bidirectional state order.
- [PyTorch packed sequences](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html)
  and [padding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.rnn.pad_packed_sequence.html):
  explicit batch-first packing and restoration to the supplied total length.

Production tests use the locally installed Torch version recorded with evidence.
The application adapter, not an assumed PyTorch packed-sequence convention,
defines the zero-length-row behavior.
