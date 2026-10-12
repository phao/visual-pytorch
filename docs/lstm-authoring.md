# Concrete LSTM authoring: shared Qt and CLI contract

Status: local Stage D implementation, not completion of D/E or a public release.
The [LSTM graph contract](lstm-graph-contract.md) specifies the executable forward.
Authoring adds no SQL table, container/revision version, hidden state, loss or
execution policy. It retains `.vtw`17 / ModelRevision4.

## Drawing an independent branch

Building blocks → LSTM opens a typed modal: label, sequence length T, input
width I, cell width H, stacked layers L, bidirectionality, projection P, bias and
optional supplied lengths. Defaults are T4/I3/H8/L1, unidirectional, projection0,
bias on, no lengths. Positive exact integers follow the existing safe-integer
limit; L≤64; P=0 or P<H. Labels are nonblank and at most60 characters. Supplied
float32 lengths require T≤2²⁴. Preparation/preview/shape proof allocate no model.

Let D=2 for bidirectional, else1; Q=P when P>0, else H. Preview shows hidden
`[B,L*D,Q]`, cell `[B,L*D,H]`, sequence output `[B,T,D*Q]`. Creation appends three
Inputs, one LSTM and three Outputs; optional lengths add an Input `[B,1]`.
Sequence input is `[B,T,I]`. Initial states are supplied tensors, not assumed
zero. Returned ports are `sequence`, `hidden`, `cell`. No task heads or loss masks
are inferred from these outputs or lengths.

Existing nodes/wires remain unchanged. New physical IDs and boundary labels are
unique; original input order is pinned and new IDs appended. Placement is below
existing work. Only the independent new branch must pass its shape proof, so
unrelated unfinished work remains visible. Whole-graph validation is still
required before execution. Creation refuses the prototype's64-block capacity.

## Stable in-place editing

Inspector opens the same modal in edit mode: prefilled constructor/label,
sequence/initial-hidden/initial-cell source keys, optional lengths and nullable
local `share_with`. Keys are stable `node_id` or `node_id@port_id`, not labels.
Browsing is alphabetic; scientific role/tensor order is not. Unproven source
dimensions are labelled; unavailable current sources are not replaced by guesses.

Apply preserves target ID, geometry/folding, non-target nodes, outgoing wires
and input order. Incoming wires follow explicit bindings, deduplicating a physical
source reused in multiple roles. Affected upstream/downstream paths and sharing
peers must validate. H/L/D/P changes do not resize state Inputs or downstream
blocks: choose compatible sources or edit those blocks explicitly. Disused Inputs
remain and may need explicit cleanup before whole-graph execution. Compatible
sharing counts one canonical owner; it does not mean parameter freezing.

Successful real GUI edits are one undo/redo step. Proved no-ops preserve stored
bytes/history/dirty state even when the form made implicit defaults explicit.
Real edits clear legacy attached dataset/result, not independently retained
immutable revisions/runs.

## Tokens, transport and UI delivery

`app.lstm_authoring.prepare` returns defaults and exactly the creation token
`{id,root_graph,root_model}`. `prepare_edit` adds `node`, current options and a
source catalog. Digests are SHA-256. `root_graph` captures the raw graph including
geometry; `root_model` captures reachable IDs/names/graphs and missing-reference
markers, not attached data/results. Changes to graph/name/geometry/reference
invalidate preparation. Unrelated models do not. No mutable latest scientific
revision is silently substituted for these current-editor preconditions.

Desktop reads: `lstmAuthoring`, `lstmEditing`; writes: `addLSTM`, `editLSTM`.
All require exact `id`, `workspace_id`, `panel_scope`; edits add `node`. Writes
also require `root_graph`, `root_model`, strict `options_json` text≤65536
characters. Extra/missing fields, duplicate JSON keys, unsupported versions,
stale/reloaded instances and incompatible paths refuse before mutation. Busy
jobs refuse writes. Acknowledgement carries `id,node,workspace_id,panel_scope`;
the page rechecks ownership before refreshing/selecting a block.

One preparation/apply is pending, controls disable while pending, and generation/
context guards fence late results. Native errors retain editable field values.
Cancel sends no new write and fences late UI delivery; it cannot roll back an
already authorized Apply. Stale graph errors require reopening for a new token,
not automatic replay. Existing typing/modal/floating-panel safeguards remain.

## CLI parity and closed documents

```sh
python vpw_cli.py lab.vtw prepare-lstm PARENT_ID
python vpw_cli.py lab.vtw prepare-lstm-edit PARENT_ID LSTM_BLOCK_ID
python vpw_cli.py lab.vtw --expect-revision FILE_SHA256 apply edits.json
```

Preparation is read-only, returning token fields plus `options` (editing also
`sources`). Each operation has exactly `op,token,options`; do not use the entire
response as the token. `op` is `add-lstm` or `edit-lstm`. Creation options:

```json
{"version":1,"label":"LSTM","steps":4,"input_size":3,
 "hidden_size":8,"num_layers":1,"bidirectional":false,
 "proj_size":0,"bias":true,"lengths":false}
```

Edit options omit T/I, use explicit sources and have exactly:

```json
{"version":1,"label":"LSTM","hidden_size":8,"num_layers":1,
 "bidirectional":false,"proj_size":0,"bias":true,
 "inputs":{"sequence":"sequence_input","hidden":"hidden_input","cell":"cell_input"},
 "lengths":null,"share_with":null}
```

Supplied edit lengths use `{"source":"length_input","adapter":"exact-int64-v1"}`;
sharing names a compatible local owner ID. Example IDs must be replaced by actual
socket IDs. Boolean versions/integers, duplicate keys, NaN/Infinity, overflowing
floats and unknown fields are not alternative grammars. All CLI JSON documents
use the same strict `read_json` file/stdin seam. Atomic ordered `apply` checks
each token against that operation's current batch state. Late failure leaves file
bytes unchanged. `--allow-invalid` does not bypass builder proofs or stale guards.
CLI edits are not a GUI undo archive; desktop external-file conflict/reload rules
still apply.

## Verification and remaining limits

Dedicated pure/serialized CLI tests cover dimensions/counts, stable packing/IDs,
drafts, no-op/defaults, stale references/geometry, rebinding, sharing and file
atomicity. Numerical authored/native/generated forwards and all gradients are
checked against an independent gate scan, including projected bidirectional
lengths/zero rows/NaN padding. A35-update teacher fit verifies exact loaded export.
Native dispatcher and actual sandboxed WebEngine forms exercise creation/editing,
no-op/refusal/cancel/stale/history/save-reload with private configuration and no
Torch in the Qt host. See development history for actual suite results and limits.

Family1–6 and runtime-spatial/temporal-v1 still reject LSTM. This is concrete
fixed-T authoring, not new symbolic/runtime-T grammar, state-data generation,
freezing/LoRA, MS-VAR, optimizer resume or E's durable/distributed jobs. Existing
concrete task/worker/notebook execution remains a separate runtime capability.
