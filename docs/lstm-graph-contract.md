# LSTM graph operator v1

This local Stage D operator implements a concrete CPU float32 LSTM, including
stacked layers, bidirectionality and hidden-state projection. It is not the
completion of Stage D or a public release. The numerical reference is
`app/lstm_nodes.py`; the pure shape proof does not import Torch.

## Closed constructor and bindings

Common node metadata is unchanged. Additional fields are `inputs`, `hidden_size`,
`num_layers`, `bidirectional`, `proj_size`, `bias` and optional `lengths`.
`op_version` is integer 1, defaulting to 1. Unknown fields fail closed.

```json
{
  "id": "lstm", "type": "LSTM", "op_version": 1,
  "hidden_size": 7, "proj_size": 4, "num_layers": 2,
  "bidirectional": true, "bias": true,
  "inputs": {"sequence": "frames", "hidden": "initial_h", "cell": "initial_c"},
  "lengths": {"source": "valid_lengths", "adapter": "exact-int64-v1"}
}
```

Positive exact integers specify cell width and layer count; the current explicit
prototype limit is 64 stacked layers. Projection is an exact integer from zero
through `hidden_size - 1`; zero means no projection. Defaults are one layer,
forward direction, no projection and bias present. Boolean fields do not accept
numeric substitutes. Dropout is fixed at zero; batch-first, device and dtype
are runtime policies, not user constructor fields in this operator version.

`inputs` has exactly three nonempty stable source keys: `sequence`, `hidden`,
`cell`. A source key is a node ID or `node_id@port_id`, never a displayed label.
`lengths` is null/absent or exactly the source/adapter object above. Incoming
wires must exactly cover distinct bound keys, without duplicates. One source
may explicitly serve multiple compatible roles. Evaluation order is sequence,
hidden, cell, then lengths, deduplicated in that order, not JSON/edge order.

## Tensor and state interpretation

Let L be layer count, D be direction count (1 or 2), H be cell width, Q be
projection width when positive and H otherwise, and I be incoming feature width.
The graph's batch B is always implicit in sample shapes.

| Role / returned port | Graph tensor | Sample axes when declared |
| --- | --- | --- |
| sequence input | `[B,T,I]` | leading `T` required |
| initial / final hidden | `[B,L*D,Q]` | `LD,Hout` |
| initial / final cell | `[B,L*D,H]` | `LD,Hcell` |
| sequence output | `[B,T,D*Q]` | `T,F` |
| optional supplied lengths | `[B,1]` | no inferred loss semantics |

There are three fixed named output sockets: `sequence`, `hidden`, `cell`.
Outgoing wires must select one explicitly. State rows are layer-major, then
forward/reverse direction. The sole native adapter transposes graph states to
`[L*D,B,width]` and transposes returned states back. Sequence stays batch-first.
The cell width does not shrink when projection is enabled. Reverse final hidden
state must not be recovered from the last reverse sequence position.

Initial states are supplied tensors. There is no inferred zero initialization,
hidden carry between calls, automatic reset, inferred mask or inferred loss head.
This primitive does not add internal feedback edges to the surrounding DAG.

## Lengths and padding

The explicit adapter accepts finite CPU float32 values exactly equal to integers
in `[0,T]`, requiring `T <= 2**24`, then converts them to CPU int64. Positive
rows execute packed prefixes with correct row-specific reverse order. Padding
returns zeros and is not evaluated; nonfinite padding is allowed. Active frames
and all initial states must be finite.

Zero-length rows return zero sequence and unchanged supplied hidden/cell states.
Sequence loss gives exact zero gradients, including for all-zero batches; losses
on returned states retain the identity gradient to supplied states. Empty-slice
autograd anchors do not inspect NaN padding. Lengths control execution only.
Task loss masks remain independently specified; length fields never create them.

## Parameters, safety and serialization

Native parameter names/shapes are recorded without allocation. Every layer and
direction owns input/recurrent gate weights and, if requested, two bias vectors.
Projected layers additionally own projection weights. Layer zero uses I input
features; subsequent layers use D*Q. `share_with` reuses a compatible local LSTM
constructor, counts weights once, stores one canonical state-dict path and sums
all call gradients. Sequence length is not a weight constructor dimension.

Compilation uses the existing whole-graph construction preflight. Each native
constrained call additionally estimates actual batch/state/gate/backward and
parameter/optimizer copies before packing or running LSTM. It compares the
estimate to the configured fraction of currently available RAM. This is a
conservative admission estimate, not an allocator guarantee. Generated standalone
forward code has explicit unconstrained inference policy, matching existing export
behavior; it is not permission to run unbounded training.

`.vtw` container version 17 and ModelRevision version 4 fence this new grammar.
SQL tables are unchanged. Older headers containing LSTM drawings or ModelRevision
4 refuse without modifying the file. Ordinary models retain their earlier
revision versions/hashes. Opening an older valid workspace is read-only until Save.
Notebook/runtime closures include `lstm_nodes.py`; portable generated Python
embeds its forward helpers and needs no application import.

## Current authoring and verification boundary

`python examples/projected_lstm.py example.vtw` creates a small explicit drawing
and frozen revision without Torch, refusing overwrite. Open it in the desktop;
the Inspector explains all states/adapters/shapes and named sockets. Existing
wire editing, output insertion, folding, compact-icon drag, resizing and history
apply. CLI `validate`, `export` and atomic explicit graph edit batches support it.
Building blocks now has a typed LSTM branch-creation form, and Inspector has a
guarded constructor/source/owner editor. CLI `prepare-lstm` / `prepare-lstm-edit`
and atomic `add-lstm` / `edit-lstm` call the same allocation-free builders.
See [authoring contract](lstm-authoring.md) for exact options, token/instance
guards, affected-path proof, preservation and no-op/history semantics.

Family versions 1–6 and runtime-spatial/temporal-v1 deliberately do not admit this
new operator. Extending symbolic/runtime LSTM authoring requires a new explicit
grammar, not reinterpreting retained families. Generic Recurrent remains available
for already-implemented reset/mask/runtime-T contracts. Concrete LSTM tasks have
been exercised through the actual worker, saved predictor and result/rerun
notebook kernels. This does not enable runtime-T or family LSTM experiments.

`tests/test_lstm_contract.py` covers pure grammar, proof, identities, file fences
and real stdlib CLI. `tests/test_lstm_execution.py` uses an independent explicit
gate scan for all input/state/parameter gradients, packed padding, zero-length
rows, projection/direction ordering, tied updates, nested/mixed graphs and a
70-update fit with fresh-process exact exported predictions.
`tests/lstm-editing.test.js` and `tests/qt_lstm_smoke.py` exercise socket editing
and actual sandboxed desktop gestures; the desktop fixture is not a creation form.
`tests/test_lstm_task_execution.py` checks a 60-update masked task fit, exact real
worker history/checkpoints, explicit unavailable supervision, app-free trained
prediction and real saved/rerun kernels with the captured LSTM runtime closure.

Dedicated `test_lstm_authoring`, `test_lstm_authoring_execution` and
`lstm-authoring.test.js` add pure/serialized CLI, independent authored forward/
gradient/fitting and mounted-modal lifecycle checks. `qt_lstm_authoring_native_smoke`
and `qt_lstm_authoring_smoke` exercise real dispatch and actual sandboxed forms.

Primary reference: [PyTorch LSTM documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html),
reviewed 2026-10-11. Reference shape/gate semantics were checked against installed
Torch 2.13.0 documentation; numerical checks run on that installed version.
