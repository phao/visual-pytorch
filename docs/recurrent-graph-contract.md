# Recurrent graph reference v1

## Delivered boundary

`Recurrent` is now an admitted executable graph node. It references an ordinary
acyclic transition architecture by stable ID. The outer graph also remains
acyclic: no feedback edge is added. One registered transition object is invoked
at every active step; its weights are shared over time. There is no hidden state
between calls and no inferred zero initial state. The algorithm and independent
execution API are specified in [recurrent-transitions.md](recurrent-transitions.md).

This checkpoint implements concrete graph shapes, SQLite/CLI admission, native
and app-free forward, exact checkpoints, existing task workers and portable
notebook execution, plus guarded palette branch creation, a didactic Inspector,
distinct input/output sockets and selected-port editing in the existing Qt UI.
It does not implement a temporal architecture-family/runtime grammar. Existing family versions1–5 and
`runtime-spatial-v1` explicitly reject this new node rather than interpreting
time as a spatial axis or silently opting in. Direct recurrent execution permits
variable T; a concrete root graph fixes T in its Input shape, just like any other
concrete tensor dimension. No new task/experiment/report grammar is implied.

## Closed node document

The common node metadata fields (ID, label, layout, size and optional local
`share_with`) are unchanged. The only additional fields are `ref`,
`sequence_inputs`, `states`, `sequence_outputs`, `controls` and integer
`op_version=1` (missing version is normalized to1). Unknown fields fail.

```json
{
  "id": "scan", "type": "Recurrent", "op_version": 1,
  "ref": "transition",
  "sequence_inputs": [{"input": "observation", "source": "sequence"}],
  "states": [{"input": "previous", "output": "next",
              "port": "hidden_final", "source": "initial"}],
  "sequence_outputs": [{"output": "next", "port": "trajectory"}],
  "controls": {
    "lengths": {"source": "lengths", "adapter": "exact-int64-v1"},
    "mask": {"source": "mask", "adapter": "exact-bool-v1"},
    "reset": {"source": "reset", "adapter": "exact-bool-v1"}
  }
}
```

All identifiers are nonempty stable strings, not displayed names. `input` and
`output` refer to child Input/Output node IDs. `source` is an incoming wire key:
the source node ID, or `node_id@port_id` for a multiport edge. The actual graph
edges must cover exactly the distinct declared keys, with no duplicate/ambiguous
keys. Reusing one wire for several explicit bindings is allowed. Sources are
ordered by sequence bindings, state bindings, then lengths/mask/reset; persisted
edge array order cannot change meaning.

At least one sequence and state are required. Every child Input must be bound
exactly once as sequence or initial state. State next-output IDs are unique;
multiple states are supported. Sequence outputs may be empty or explicitly
alias one child Output. Returned state and sequence port IDs are collectively
unique. An outgoing edge must specify `from_port`, for example:

```json
{"from": "scan", "from_port": "trajectory", "to": "prediction"}
```

The child next-state Output must have exactly the same shape and axes as its
corresponding state Input. Sequence sources have sample shape `[T,*step_shape]`;
all sequences must agree on positive T. Declared leading axis must be `T`.
If the child declares axes, the sequence axes must be exactly `T` followed by
those axes. Initial state preserves the child Input shape/axes. Sequence outputs
have `[T,*output_shape]`, with `T` followed by the child axes, or generated
`D0,D1,...` for otherwise unnamed dimensions. Final state outputs retain the
state Input shape/axes. These per-port axes are used by subsequent nodes, not a
packed aggregate's unspecified axes; Conv2d cannot accidentally interpret T as C.

## Explicit control adapters

Controls are optional and never inferred from data. Omitted controls mean every
step is active and no reset is requested. The existing graph/task packing uses
CPU float32 tensors, so each declared control explicitly names its validator and
conversion. A categorical or float probability is not silently treated as a mask.

* `exact-int64-v1`: graph sample shape `[1]`; actual `[B,1]` float32 values must
  be finite exact integers in `[0,T]`. T must be at most2²⁴ for this adapter.
  The result is CPU int64 `[B]`. The direct primitive's int64 length API does not
  use this float32 bound.
* `exact-bool-v1`: graph sample shape `[T]`; actual `[B,T]` float32 values must
  be exactly0/1, never NaN/Inf/fractional. The result is CPU bool `[B,T]`.

Lengths and masks intersect to determine active rows. Holes are allowed. An
active reset replaces state with that call's supplied initial state before the
transition; inactive reset is ignored. Inactive sequence outputs are zero and
state is held. Active-row execution never evaluates padded frames. A data loss
mask is a separate explicit TaskSpec head binding: execution masks do not
automatically remove loss terms. Every exposed root Output still requires a
positive-weight task head under TaskSpec1; hide a non-supervised state behind an
internal node, or provide an explicitly masked state head. All-empty supervision
skips optimizer updates (Adam momentum/decay must not update an absent objective).

## Ownership, resource admission and evidence

Parameters count once per recurrent owner. Step operations, activations and
convolution workspace contribute T times their body estimate. Explicit compatible
`share_with` reuses a whole recurrent layer; transition reference and state/output
specification must match. Wire keys and controls can differ per call. CNN bodies,
nested Modules and sharing inside the body remain ordinary graph operations.

Before adapter conversion and every unroll, constrained native execution checks
current free RAM against the actual B/T estimate, including state/history,
outputs, activation/backward storage and convolution workspace. This is a
conservative heuristic, not an allocator/OOM guarantee. Plain generated graph
code retains the existing explicitly unconstrained export convention; saved task
predictor export takes its ordinary constrained-policy argument.
Ordinary Module wrappers containing a recurrent descendant propagate both the
caller’s constrained flag and RAM percentage through the complete reference path.
Legacy Module-only closures retain their previous construction-only nested policy.

Native state paths include `layer_i.transition.layer_j...`. Stable parameter IDs
follow root/recurrent-node/body-node paths, not labels or registration indices.
Model identity includes reachable child semantics, bindings and adapters, while
excluding drawing geometry/labels. Checkpoints retain exact paths and IDs.

Container14 adds no SQL tables. `architecture_refs` indexes both Module and
Recurrent nodes, preventing target deletion and propagating current-result
invalidation. ModelRevision3 captures the entire reachable Module/Recurrent
closure; versions1/2 reject recurrence. Earlier container headers reject both
mutable recurrent nodes and model v3 records. Ordinary model v1/v2 hashes remain
unchanged. Opening an older container never rewrites disk bytes.

No special CLI verb is needed: existing ordered atomic `apply` create/replace
operations admit these documents, `freeze-model` captures v3, validation/counts/
export use the same graph interpreter. TaskBundle1 can carry exact ModelRevision3;
the existing concrete worker, saved predictor and result/rerun notebook paths
share the captured execution helpers. Notebook bundles carry `recurrent.py` and
`recurrent_nodes.py`; omitting an imported helper fails before kernel execution.

Run the disposable constrained graph/storage example from the product checkout:

```sh
PYTHONPATH=. python examples/recurrent_graph.py
```

It builds a two-input recurrent graph, freezes ModelRevision3, saves/reopens a
private temporary `.vtw`, and runs an ordinary PyTorch forward. It writes no user
workspace or configuration. The primitive-only example remains
`examples/recurrent_transition.py`.

## Qt authoring and agent parity

Building Blocks → Reusable architecture → **Recurrent transition** opens the
selected transition's concrete interface. Every child Input requires an explicit
sequence/state choice; a state requires an explicitly selected shape/axis-matching
next-state Output. The form selects concrete positive T, a block name, returned
sequences (possibly none), and optional supplied lengths/mask/reset inputs. It
adds a separate branch: all existing blocks/wires remain, new boundary names and
IDs are unique, and existing input order stays first. Legacy attached data and
results are cleared because new supplied Inputs change that boundary; independent
scientific records remain intact. There is one graph-history undo step.

Initial states are ordinary supplied Input boxes, never automatically zeroed or
learned. All generated outputs are ordinary root Output boxes: TaskSpec1 requires
positive-weight heads for all of them. This is not an automatic loss proposal.
Change supplied tensor shapes/T in Input inspectors; editing a transition body
updates its references. Existing invalid parent drafts remain editable: the new
branch is independently validated without deleting/repairing unrelated work.
The prototype's 64-block limit and eight-dimensional sample limit remain; adding
T must fit within the latter. No Torch is imported for authoring.

The Recurrent Inspector identifies every child role, incoming source, shape,
next-state mapping, exact control adapter and returned sequence/state port. It
explains mask/reset/full-BPTT/no-hidden-carry semantics, opens the transition in
the current/new tab, and provides existing explicit compatible parameter sharing.
The top-right loop icon follows the existing fold/compact-drag/expand behavior
and semantic theme tokens. Physical inputs are distinct source keys in binding
order; deliberately shared argument sources use one socket with all roles in
its tooltip. Outputs are sequence ports then final-state ports, each explicitly
identified. Tooltips include individual tensor shapes, not a packed aggregate.

Socket wiring replaces only the selected recurrent source binding, including all
roles deliberately sharing it; other bindings and wires remain. Selecting an
already-used source for another argument explicitly aliases those arguments into
one physical wire. Deleting a wire retains its argument for repair. Dragging a
destination within the same Recurrent box to a different empty argument gives
the previous argument a `disconnected/<unique-id>` source key so it stays visibly
unbound rather than silently receiving the moved wire too. Such drafts cannot
execute until repaired. Deleting a recurrent box never guesses a bypass. Unary
insertion/removal and router insertion update exact downstream source keys.
Output creation, Splitter/Replicator insertion and later layer defaults use the
selected returned tensor's shape/last axis, not all returned tensors' packed size.

Native read action `recurrentAuthoring` has exactly `action`, `workspace_id`,
`panel_scope`, `id`, `ref`. It returns those identities plus `root_graph`,
`transition`, `name`, `inputs`, `outputs`; the latter two are ordered concrete
contract groups. `root_graph` is the digest of the full current parent graph
(including layout); `transition` digests its reachable ModelRevision document.
The existing workspace-instance `panel_scope` rejects even same-UUID reloads.
Native `addRecurrent` adds exactly `root_graph`, `transition`, `options` to that
request shape and returns `id`, `node`, `workspace_id`, `panel_scope`. It checks
identity/instance, both current digests, reference cycles, explicit roles and
compatible shapes before writing. Busy graph writes fail. Failed/stale requests
leave SQL/history unchanged; repeated submissions fail after the first edit.
No full parent-document replacement is sent by the modal. The graph append uses
fresh current metadata and broadcasts through the normal UI/history path.

Closed options version1:

```json
{
  "version": 1, "steps": 4, "label": "Recurrent transition",
  "inputs": [
    {"input": "x", "role": "sequence", "output": null},
    {"input": "h", "role": "state", "output": "next"}
  ],
  "sequence_outputs": ["next"],
  "controls": ["lengths", "mask", "reset"]
}
```

Unknown fields fail. IDs refer to the chosen child's declared nodes, not names.
Input rows cover every child exactly once; sequence rows require `output:null`.
State rows name distinct matching next outputs. Sequence outputs/control roles
are distinct. Controls are generated in lengths/mask/reset order. Boundary IDs
are fresh; output ports are `sequence/<child-output-id>` and
`state/<child-input-id>`. Label length is1–60, T is an exact positive integer;
lengths controls additionally require T≤2²⁴. Expanded sample products must fit
the graph's exact integer domain. Reconfigure existing node bindings/output
declarations through explicit CLI node edits for now; the creation modal is not
a general existing-node interface editor.

CLI `prepare-recurrent PARENT_ID TRANSITION_ID` returns the same read-only
interface/digests (without ephemeral GUI ownership). Atomic `apply` operation
`{"op":"add-recurrent","token":{"id":...,"ref":...,"root_graph":...,"transition":...},"options":...}`
uses the same pure `app/recurrent_authoring.py` builder and returns the new block
ID. `--expect-revision` additionally protects the file write. Existing complete
create/replace/add-node/set-node operations remain supported. No schema or grammar
version is added by this UI/convenience command.
