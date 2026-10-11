# Existing recurrent interface editing

Inspector → **Edit recurrent interface…** opens an advanced explicit editor for
an existing `Recurrent` block. This uses the established node grammar, not a new
SQLite schema or another execution interpreter. Container16 remains current;
ordinary recurrent nodes require14. Branch creation is a separate operation.

## Qt behavior

The modal prefills the actual bindings, return ports, label and parameter owner.
It shows the transition's stable input/output IDs and shapes, and an expandable
list of existing source sockets. Transition choices are alphabetical; execution
and ports retain their declared order. At this checkpoint, bindings are edited
as JSON, not a graphical role table. Selecting another transition reads its
interface and new guards but retains the JSON: it never guesses replacement IDs.
Invalid JSON, duplicate keys, invalid bindings, stale state and unsupported fields
stay visible as errors, with the draft retained. Cancel writes nothing.

Apply changes only the existing block and explicitly changed incoming wires.
Non-target nodes, IDs, geometry, folding, size, root input order and downstream
wires are preserved. No Input/Output boxes or bypasses are invented. Change
supplied dimensions/T on their Input boxes. A real edit forms one undo/redo step;
an exact no-op changes neither SQL, history nor dirty status. Real changes
conservatively clear legacy attached data/results, not independent scientific
revisions, retained runs or their exact dataset/task bindings.

## Closed options version1

Every field below is required; `share_with:null` means independent parameters.
Identifiers are stable IDs, not labels. Example bindings:

```json
{
  "version": 1, "label": "Recurrent transition",
  "sequence_inputs": [{"input": "x", "source": "frames"}],
  "states": [{"input": "h", "output": "next", "port": "final", "source": "initial"}],
  "sequence_outputs": [{"output": "next", "port": "trajectory"}],
  "controls": {"mask": {"source": "mask", "adapter": "exact-bool-v1"}},
  "share_with": null
}
```

Tensor/control rules are those of [Recurrent v1](recurrent-graph-contract.md).
Source keys are `node_id` or `node_id@port_id`; owners must be compatible local
recurrent blocks. Labels have1–60 characters; version is integer1, not boolean.

## Preservation and validation

Physical incoming wires are deduplicated from explicit sequence/state/lengths/
mask/reset bindings. Unchanged wires retain their bytes and edge-array position;
new sources are appended in declared order. Multiple roles may explicitly share
one physical source. Removing/renaming a connected return port is refused:
disconnect its wires explicitly first. Adding an unused return port does not add
a root Output or a task head. Keeping a port ID while changing its child mapping
is an explicit semantic edit.

The pure builder proves affected paths, upstream dependencies, parameter owners
and affected sharing peers, including actual shape/axis/control/cycle/downstream
constraints. Private synthetic sinks for unconsumed proof leaves never enter the
workspace. Unrelated invalid drafts remain editable. Removing a control or
rebinding an initial state may leave its old Input unused; the editor keeps it.
The ordinary whole-drawing compiler still rejects unused/invalid branches until
explicit cleanup. Interface acceptance does not claim the entire drawing trains.

## Native transport

Read `recurrentEditing` requires exactly `action`, `workspace_id`, `panel_scope`,
`id`, `node`, `ref`. It returns those identities plus `root_graph`, `root_model`,
`transition`, `name`, ordered `inputs`/`outputs` and prefilled `options`.
The pure token is exactly `id`, `ref`, `node`, `root_graph`, `root_model`,
`transition`. `root_graph` digests the complete parent graph including layout.
`root_model` digests parent ID/name and all current referenced frozen closures,
without requiring the editable parent to execute. `transition` separately digests
the chosen replacement closure. Canonical project digest rules apply.

Write `editRecurrent` adds exactly `root_graph`, `root_model`, `transition`,
`options_json` to that request shape. JSON text is at most65536 characters; strict
native parsing rejects duplicate keys and nonfinite constants. The result is
`id`, `node`, `workspace_id`, `panel_scope`. Instance scope rejects same-UUID
reloads. Busy edits fail before history capture; all refusals leave SQL/history
unchanged. Changed-graph token replay fails; exact no-op replay is harmless.
GUI response guards also check block/reference/workspace and modal generation.

## Agent CLI

```sh
python vpw_cli.py lab.vtw prepare-recurrent-edit PARENT_ID BLOCK_ID
python vpw_cli.py lab.vtw prepare-recurrent-edit PARENT_ID BLOCK_ID --ref TRANSITION_ID
python vpw_cli.py lab.vtw --expect-revision FILE_SHA256 apply edits.json
```

Preparation is read-only, returning the same information without ephemeral GUI
ownership. Atomic batches accept exactly
`{"op":"edit-recurrent","token":TOKEN,"options":OPTIONS}`. Copy the six token
fields and edit the returned options explicitly. Qt/CLI share
`app/recurrent_reconfiguration.py`; CLI imports no Torch/Qt. Existing low-level
node/wire edits remain supported. Failure later in a batch leaves original disk
bytes unchanged. Offline batches do not persist GUI undo history.

## Verification and remaining work

Pure tests cover no-op preservation, roles, sources/router sockets, multiple
states/sequences, controls, connected/added ports, current/replacement closures,
sharing peers, downstream constraints, invalid drafts and CLI save/replay/rollback.
Numerical tests independently verify edited native/app-free outputs and all
data/parameter gradients, plus a bounded fit and exact loaded-weight predictions.
Native dispatcher tests check SQL/history, JSON, busy/replay/reload guards and
no host Torch/config writes. Real sandboxed Qt tests use native box/Apply/Cancel
clicks for drafts, sockets, undo/redo, stale refusal and save/reload with MLP/CNN
bodies. JavaScript tests independently check response identity and closure tokens.
Tests are `test_recurrent_reconfiguration*.py`, `qt_recurrent_edit*_smoke.py`
and `recurrent-editing.test.js`.

This delivered advanced JSON editor does not establish full D acceptance or E
durable/heavyweight execution. No live user workspace/config or public/main
integration is implied.
