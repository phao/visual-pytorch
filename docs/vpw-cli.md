# Agent and user CLI contract

vpw_cli.py follows the requested filename; workspaces remain .vtw. It needs Python
standard libraries, not Qt/PyTorch. Each command returns one JSON object:
`{"ok":true,"value":...}` with exit 0 or `{"ok":false,"error":"..."}` with exit 1.
Argument usage errors use argparse exit 2. Architecture names never substitute
for explicit IDs.

```sh
python vpw_cli.py lab.vtw init
python vpw_cli.py lab.vtw inspect
python vpw_cli.py lab.vtw show ARCHITECTURE_ID
python vpw_cli.py lab.vtw validate ARCHITECTURE_ID
python vpw_cli.py lab.vtw export ARCHITECTURE_ID
python vpw_cli.py lab.vtw export-trained ARCHITECTURE_ID
python vpw_cli.py lab.vtw preview-dataset observations.json --limit 20
python vpw_cli.py lab.vtw prepare-task ARCHITECTURE_ID --dataset observations.json --task task.json --split split.json
python vpw_cli.py lab.vtw spec-list --kind dataset --all-revisions
python vpw_cli.py lab.vtw spec-show dataset DATASET_ID --revision DATASET_SHA256
python vpw_cli.py lab.vtw prepare-stored --model MODEL_ID MODEL_SHA256 --dataset DATASET_ID DATASET_SHA256 --task TASK_ID TASK_SHA256 --split split.json
python vpw_cli.py lab.vtw train-stored --model MODEL_ID MODEL_SHA256 --dataset DATASET_ID DATASET_SHA256 --task TASK_ID TASK_SHA256 --manifest manifest.json --protocol protocol.json --torch-python /path/to/numerical/python
python vpw_cli.py lab.vtw export-task-trained task-result-value.json
python vpw_cli.py lab.vtw --expect-revision SHA256 apply edits.json
python vpw_cli.py lab.vtw apply -
```

init creates an empty version-5 collection, never overwriting a file. inspect
returns working format version, exact SQLite DDL, architecture IDs/names/counts,
policy, independent specification browsing selections and on-disk SHA-256 revision. Architecture summaries are alphabetical by
case-folded name, with ID as tie-breaker. show returns a materialized document with
dataset. validate derives the graph contract without allocation; export returns
generated Python in the value.code field, with fresh parameter initialization.
export-trained requires a matching current checkpoint identity and embeds learned
weights, strict state loading and evaluation mode for standalone inference.
It also checks version, preprocessing, stable parameter IDs, exact weight keys/
shapes and finite float32 values without importing Torch. Corrupt/unsupported
checkpoints fail instead of producing a script that only fails when executed.

Stage C task-branch commands `preview-dataset` and `prepare-task` are read-only:
they do not save or train. Typed dataset/task schemas, explicit split protocols,
manifest replay, fitted transforms and RAM-policy behavior are specified in
[dataset-task-contract.md](dataset-task-contract.md). `prepare-task` takes exactly
one of `--split` (protocol) or `--manifest` (saved actual IDs); percentage defaults
to user config and can be explicitly overridden with `--memory-limit-percent`.
Independent dataset/task/model persistence is implemented in `.vtw` v5;
experiment/run/report kinds, selection GUI and notebooks remain pending.
[revision-store.md](revision-store.md) specifies validators/migration.
`prepare-stored` requires exact model/dataset/task ID+revision pairs and the same
explicit split/manifest/RAM settings; it never reads mutable heads as execution
references. `spec-show` may read a browsing head when --revision is omitted;
`spec-list --all-revisions` includes archived/unselected immutable documents.

`train-stored` executes the same exact-reference selection in an isolated worker,
requiring explicit TrainingProtocol JSON and one split protocol or saved manifest.
The CLI/GUI host still imports no Torch; `--torch-python` selects the numerical
interpreter. RAM percentage defaults to the read-only user preference and supports
an explicit override. It returns a self-contained task-training-result in `value`
without modifying the workspace. The command envelope `ok:true` means a valid
artifact was returned: inspect `value.status`/`failure`, not `ok`, to distinguish
completed, failed, cancelled or unavailable training. Admission/process/protocol
errors use the usual `ok:false` envelope; they cannot yet be stored as RunRecords.
No background/detached execution or independent run-store UI is claimed.

`export-task-trained` reads the saved **value object**, validates its frozen bundle
and fitted inference checkpoint, and emits standalone Python in `value.code`.
It never trains or writes `.vtw`. The source requires only Torch and Python stdlib,
does not contain raw observations, and exposes `predict_records(list_of_field_maps)`
returning tensors keyed by stable Output IDs. Classification outputs remain logits;
regression stays in declared transformed target space. Task/target/vocabulary
metadata is exported alongside preprocessing and strict weights. See
[task-training-contract.md](task-training-contract.md) for the exact artifact
protocol, terminal behavior, RAM and verification limits. Export uses the same
read-only user RAM preference/override as preparation/training.

apply reads an ordered JSON array from a file or stdin. It edits an in-memory
copy and persists only if the whole batch succeeds. Normal apply validates every
graph, including affected references. --allow-invalid explicitly permits semantic
drafts, never structural/reference/lock violations. A failed batch leaves file
bytes unchanged; successful edits are not automatically archived as undo history.

| op | Fields besides op | Effect |
| --- | --- | --- |
| create | document, or optional name/inputs/outputs | New architecture; existing ID refused; defaults are a Linear graph, not a preset MLP. |
| replace | id, document | Full replacement preserving ID |
| rename | id, name | Display name |
| move | id, node ID, x, y | Layout only; latest learned result retained |
| resize | id, node ID, width, height | Preferred world-unit diagram size; finite positive numbers ≤100000; data/results/forward unchanged |
| set-node | id, node ID, values; optional remove_fields | Property edits; changing/removing ID/type refused |
| add-node | id, node object | Explicit new block |
| remove-node | id, node ID | Remove block/incident wires; prune Concat keys; no guessed rewiring |
| connect | id, edge object | Append explicit wire; update ordered Concat sources separately |
| disconnect | id, exact edge object | Remove exactly that wire or fail |
| delete | id | Delete unreferenced graph; last architecture protected |
| mode | enabled boolean | Persist workspace safety policy only |
| put-spec | kind, document, optional select boolean (default true) | Validate/store immutable dataset/task/model revision, optionally select for browsing |
| select-spec | reference={kind,id,revision} | Select a retained exact revision, never edit its bytes |
| archive-spec | kind, id | Hide browsing head; revision bytes are retained and recoverable |
| freeze-model | id | Capture current valid architecture's complete reachable closure as an independent model revision |
| rebind-dataset | id | Explicitly bind valid existing x/y columns to the current input/output port contract; no numerical reordering/transformation; clears current result, preserves historical runs |

Create referenced targets before their owners. Custom documents follow
[vtw-format.md](vtw-format.md). Computational edits invalidate current results in
the graph and its users. When changing input/CSV packing, explicitly clear or
replace the dataset rather than assuming old columns retain their meaning.
For batch-local references, assign unique IDs in create documents: automatically
generated UUIDs are reported only after the complete batch commits.

Conv2d, pooling, Flatten and Permute use the same add-node/set-node/create/replace
grammar as existing blocks. Writes normalize spatial scalars to explicit pairs
and op_version=1. Normal apply checks grouped channel divisibility, layout and
positive output sizes without importing Torch. Invalid batches leave disk bytes
unchanged. A structured dataset adapter mismatch is not repaired by rebind-dataset.
See [the CNN example](cnn-workflow.md) for a runnable workspace generator.

Example layout repair (substitute actual IDs):

```json
[
  {"op":"move","id":"MODEL_ID","node":"linear1","x":384,"y":192},
  {"op":"resize","id":"MODEL_ID","node":"linear1","width":240,"height":96},
  {"op":"rename","id":"MODEL_ID","name":"Reviewed model"}
]
```

Restore automatic sizing with set-node, values={} and
remove_fields=["width","height"]. Rendered dimensions respect typography and port
minimums even if an agent stores smaller preferences. This is not a network shape
edit or memory-policy limit. Batches and graph/port arrays retain explicit order.

The response includes affected/created IDs and the new revision. Recommended
collaboration: user saves; agent inspects and captures revision; agent explains
authorized edits in chat and applies them with --expect-revision. A clean idle GUI
reloads in about one polling interval. If it became dirty, it warns and retains
user work. Re-inspect conflicts; do not blindly repeat stale edit batches.
External reload starts fresh GUI undo history. CLI batches have atomic failure,
not a persistent undo log. User-level theme/spacing/RAM percentage settings are
not part of a .vtw edit batch; workspace mode is. Editing values or validating a
graph does not promise that the local execution memory policy will admit it.

The CLI does not control an unsaved GUI buffer/modal dialogs/active jobs, train,
execute arbitrary Python or generate notebooks. Notebook/prompt generation is
future design work, not a hidden feature of this interface.

## Destination rewiring and GUI alignment

The successor's destination-drag gesture changes existing edge records; it does
not introduce a new workspace format or UI-private connection state. Agents can
perform the equivalent in one atomic apply batch: disconnect the exact old
edge, connect a new edge with the same from/from_port and a new to. For Concat,
also set-node its sources explicitly: remove the old source/port key from the
old destination and append it at the new destination without reordering peers.
The GUI prunes these keys automatically; CLI connect/disconnect deliberately
keep the explicit edit contract above. Deletion's GUI unary-bypass behavior also
differs from remove-node's explicit removal; agents must supply any desired
replacement edges. Graph/type validation and .vtw persistence are shared.
Default batches reject invalid final graphs; --allow-invalid is needed for an
intentional unfinished draft. Inspect/revision-check, save/reload, validate and
export remain the collaboration workflow, covered by CLI regression tests.
