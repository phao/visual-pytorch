# Agent and user CLI contract

vpw_cli.py follows the requested filename; workspaces remain .vtw. It needs Python
standard libraries, not Qt/PyTorch. Each command returns one JSON object:
`{"ok":true,"value":...}` with exit 0 or `{"ok":false,"error":"..."}` with exit 1.
Argument usage errors use argparse exit 2. Architecture names never substitute
for explicit IDs.

The local prototype also accepts `Activation` nodes with `activation:Softmax`
and `dim` (default −1). This axis indexes sample dimensions, excluding batch B;
CLI validate/export and ordinary `set-node` edits use exactly the GUI convention.
See [softmax.md](softmax.md) for valid axes, PyTorch mapping and loss semantics.

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
python vpw_cli.py lab.vtw export-run-trained RUN_ID RUN_SHA256 --attempt 1
python vpw_cli.py lab.vtw preview-experiment EXPERIMENT_ID EXPERIMENT_SHA256 --offset 0 --limit 100
python vpw_cli.py lab.vtw preview-experiment-page EXPERIMENT_ID EXPERIMENT_SHA256 --offset 0 --limit 25
python vpw_cli.py lab.vtw run-candidate --experiment EXPERIMENT_ID EXPERIMENT_SHA256 --coordinates coordinates.json --run-id RUN_ID --name "Trial" --torch-python /path/to/numerical/python --timeout 60
python vpw_cli.py lab.vtw report-data REPORT_ID REPORT_SHA256
python vpw_cli.py lab.vtw run-evidence RUN_ID RUN_SHA256 --attempt 1 --offset 0 --limit 25
python vpw_cli.py lab.vtw report-evidence REPORT_ID REPORT_SHA256 --offset 0 --limit 5
python vpw_cli.py lab.vtw run-curve RUN_ID RUN_SHA256 --attempt 1 --head HEAD_ID --offset 0 --limit 100
python vpw_cli.py lab.vtw --expect-revision SHA256 apply edits.json
python vpw_cli.py lab.vtw apply -
```

init creates an empty version-6 collection, never overwriting a file. inspect
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
Independent dataset/experiment/model/report/run/task persistence is implemented
in `.vtw` v6; the branch-only selection/history GUI is described in
[experiment-gui.md](experiment-gui.md). Portable
notebook generation/explicit real-kernel execution are described below.
[revision-store.md](revision-store.md) specifies validators/migration.
`prepare-stored` requires exact model/dataset/task ID+revision pairs and the same
explicit split/manifest/RAM settings; it never reads mutable heads as execution
references. `spec-show` may read a browsing head when --revision is omitted;
`spec-list --all-revisions` includes archived/unselected immutable documents.

For bounded reads, use `spec-page KIND [--query TEXT] [--offset N] [--limit 1..25]
[--heads-only]`, `spec-summary KIND ID REVISION`, or `spec-inspect KIND ID
REVISION [--path JSON_ORDINAL_ARRAY] [--part value|key] [--offset N]
[--limit 1..25]`. Summary/inspection require an exact retained revision. Follow
returned `next_offset`; pages can be shortened by their compact 65536-byte
projection limit. Scalar chunks use at most 2000 Unicode code points. Complete
legacy reads remain available, not silently truncated. See
[scientific-browser.md](scientific-browser.md) for scope/identity/order, byte
encoding, CLI-envelope and backend-cost limits. All three commands are read-only.

The Qt [chunked draft value editor](chunked-draft-values.md) uses transient owned
session/buffer tokens, not another `.vtw` grammar or persistent CLI session.
Agents still use complete specification import/ordered apply with normal atomic
admission and disk conflict checks. Large integer tokens remain literal JSON;
the raw Qt path does not require rounding them through JavaScript Number.
The [typed dataset form](typed-dataset-sessions.md) now also preserves literal
tokens and uses the same owned session for a child value editor. Dataset int64
storage accepts the full signed 64-bit range through GUI/CLI/backend validation;
explicit float32 numerical preparation does not promise exact integer arithmetic.
There are no persistent CLI draft tokens or new SQLite tables.
Native owned task card/transform pages and literal edits are described in
[typed-task-sessions.md](typed-task-sessions.md). They do not add CLI commands or
change the agent-facing TaskSpec grammar. The actual task form uses them now,
with named-property targets and a linked value editor. Native origin-proof tokens
remain transient editor state, not commands for durable agent sessions.
Native experiment projections/edits have the same boundary; see
[typed-experiment-sessions.md](typed-experiment-sessions.md). They do not change
ExperimentSpec/CLI import/apply or replace deliberate full CLI preview/export.
The actual experiment form now uses those commands, including same-session
staged raw values and exact local choices. Agent import/apply still operates on
complete scientific specifications, not browser projection DTOs or session tokens.

Recovery browsing likewise uses transient window scope/sequence, not CLI tokens
or a persisted job table; see [recovery-browser.md](recovery-browser.md).
Agents can explicitly export complete recovery JSON, inspect its origin/contents,
then use validated dependency-first existing specification edits. A small
notification/tree page or the outer recovery envelope is not a put-spec document.

`task-context-page fields|input_ports|output_ports --model ID REVISION --dataset
ID REVISION [--query TEXT] [--offset N] [--limit 1..25] [--pin STABLE_ID]` shares
the actual task GUI's exact bounded field/port suggestion backend. It preserves
scientific ordinals, reports missing inspected IDs explicitly and labels partial
shape previews. It does not transmit records/complete vocabularies/metadata or
validate a task's compatibility. It is read-only, with unchanged file bytes and
no new persistent grammar. See [task-context-pages.md](task-context-pages.md)
for the complete DTO, byte limit and unbounded native registry/sort costs.

`spec-page --pin ID REVISION` resolves a selected exact reference independently
of search/page/head filtering. A nonexistent pin fails by default; adding
`--allow-missing-pin` returns that exact unavailable reference as `missing_pin`
with `pinned: null`, never a replacement head or invented metadata. The authoring
GUI uses this explicit variant. Identities/page flags still undergo validation.

`train-stored` executes the same exact-reference selection in an isolated worker,
requiring explicit TrainingProtocol JSON and one split protocol or saved manifest.
The CLI/GUI host still imports no Torch; `--torch-python` selects the numerical
interpreter. RAM percentage defaults to the read-only user preference and supports
an explicit override. It returns a self-contained task-training-result in `value`
without modifying the workspace. The command envelope `ok:true` means a valid
artifact was returned: inspect `value.status`/`failure`, not `ok`, to distinguish
completed, failed, cancelled or unavailable training. Admission/process/protocol
errors use the usual `ok:false` envelope for this read-only direct command.
The separate run-candidate wrapper below stores terminal attempts.
No background/detached execution is claimed. Saved runs are browsable in the
branch-only GUI; this direct command does not store one.

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

`export-run-trained RUN_ID RUN_SHA256 [--attempt N] [--memory-limit-percent P]`
generates the same source artifact as the branch-only Qt checkpoint-predictor
preview. It returns code/code_sha256 plus exact attempt/status/checkpoint/update
provenance, creating no source file and preserving `.vtw` bytes. Omission selects
the latest retained attempt of that exact revision, NOT a newer head. A failed
latest attempt without a checkpoint fails; explicitly select an earlier retained
attempt instead. Valid cancelled/zero-update checkpoints are honestly labelled.
Generation validates under workspace RAM/preparation policy, without importing
Torch or executing/training. Outputs remain stable-ID logits/declared target space;
no optimizer resume is available. See [saved-predictor-export.md](saved-predictor-export.md)
for exact artifacts, units, privacy and exclusive Qt file creation.

`preview-experiment ID REVISION` reads an exact ExperimentSpec, never its mutable
head. `--offset` defaults to 0; `--limit` defaults to 100 and is in 1..1000.
The bounded page lists crossed/paired combinations, including excluded/invalid
ones with reasons. It neither fits nor trains nor writes. The exact coordinate
object from a candidate row is the entire contents of coordinates.json.

`preview-experiment-page ID REVISION` is the byte-bounded alternative used by
the actual Qt candidate browser: offset0/limit25 by default, limit1..25, exact
decimal total, original candidate order/status/references/counts and visibly
partial reason descriptors. Full manifests and resolved protocols stay native;
the complete `preview-experiment` command above is unchanged. Successful compact
projection JSON is at most65536 bytes, excluding CLI formatting/envelope; follow
its actual next_offset, not a25-row stride. Neither command fits, trains or saves.
See [candidate-browser.md](candidate-browser.md) for precise scope and cost limits.

`run-candidate` requires `--experiment ID REVISION`, `--coordinates FILE`,
`--run-id ID`, `--name DISPLAY_NAME`. Optional `--retry` explicitly appends a
fresh attempt to that same selection; without it an existing run ID is refused.
`--torch-python` defaults to the CLI interpreter. `--memory-limit-percent`
defaults to the read-only user setting; `--timeout` is an optional transport
deadline in seconds. Actual split IDs and resolved protocol are frozen before
admission. Retries use the captured manifest and longest retained attempt history,
not a regenerated split or older browsing head.

This command runs synchronously and saves a terminal RunRecord in one atomic
revision-checked Save. It returns `value.reference` and the complete
`value.record`. A successful Save adds `persisted:true,workspace_revision`.
If concurrent edits or another Save error prevent persistence, it instead returns
`persisted:false,save_error`, retaining the complete recoverable evidence in
the response and leaving external file bytes untouched. The `ok:true` envelope
means a valid record was produced, NOT a completed fit or successful Save.
Check `record.attempts[-1].status` and `persisted` separately. Compatibility
exclusion, admission, missing executable and worker failure can be persisted
without inventing metrics/checkpoints. Invalid command/selection grammar remains
an `ok:false` error. No automatic retry or optimizer continuation occurs.

On Save conflict, keep the returned record, inspect the current file/dependencies,
and recover it with an explicit `put-spec` run operation plus `--expect-revision`.
Do not repeat the numerical fit just to recover its already produced evidence.
If its logical run ID now has conflicting history, resolve the identity conflict
explicitly; never rewrite earlier attempts.

`report-data ID REVISION` reads an exact ReportRecipe and saved run evidence.
It is read-only and never starts a worker, including for a rerun recipe. Failed
attempts remain visible; recipe references do not follow newer run heads.
Result-report and rerun modes are deliberately separate. This is JSON
materialization, not notebook generation/execution (`notebook_status:not-generated`).
See [experiment-objects.md](experiment-objects.md) for exact closed schemas,
dependency validation, descriptive analysis and remaining capabilities.

`run-evidence ID REVISION [--attempt N] [--offset 0 --limit 25]` and
`report-evidence ID REVISION [--offset 0 --limit 5]` return the same structured
read-only projections as Qt. Exact hashes are mandatory. Attempt defaults to the
latest retained attempt of that exact revision, not a newer revision or earlier
success. Metric limits are 1..25; report run limits 1..10. Invalid requests or
missing/corrupt dependencies fail without changing file bytes. No workers,
notebook execution, weights or raw observations are returned. Saved scalar
metrics/denominators/counters retain head/population/checkpoint identity, including
cancellation/unavailable cases. Report order follows the recipe; rerun recipes
return labelled zero saved rows, not retraining. Paging bounds response rows,
not full validation/inline data costs. See [saved-evidence-views.md](saved-evidence-views.md).

`run-curve ID REVISION [--attempt N --head ID --offset 0 --limit 100
--head-offset 0 --head-limit 25]` reads the same exact saved-attempt head curve as
Qt. Attempts default to the latest in that exact revision, heads to the first
declared head; neither follows a newer browsing head. Point limits are 1..100,
head-choice limits 1..25, and their cursors are independent. The response preserves
saved checkpoint metrics/counters and missingness, excludes weights/raw rows/test
curves, and uses fixed axes across retained pages. Reading never fits, evaluates
or trains and changes no file bytes. These are row bounds, not complete validation
or metadata byte bounds. See [saved-run-curves.md](saved-run-curves.md).

`generate-report ID REVISION --output NEW_DIRECTORY` creates a portable notebook
bundle from an exact ReportRecipe; it never executes cells or trains. Optional
`--offset 0 --limit 100 --allow-partial` selects an explicitly bounded rerun page
(limit 1..1000); without `--allow-partial`, incomplete coverage is rejected.
Result-report recipes select exact runs and reject candidate paging flags.
`--memory-limit-percent` overrides the read-only user preference; optional
`--worker-timeout` bounds each numerical worker in the rerun notebook.
The directory must not exist; no files are overwritten. Bundles include raw
inline observations and weights: review privacy before sharing.

`execute-report BUNDLE --kernel-python PYTHON --cell-timeout SECONDS` explicitly
executes trusted notebook code with a fresh real ipykernel. Defaults are the CLI
interpreter and 600 seconds per cell (positive integer). It verifies requested
bytes/input closure, retains requested/executed notebook copies, terminal receipt
and partial outputs under a new attempt directory. `ok:true` means a receipt was
produced: inspect `value.execution.status` for kernel success and each retained
run's status independently. Source `.vtw` bytes remain unchanged. The positional
workspace is still required as CLI context, but the bundle's exact inputs, not its
current heads/drafts, determine execution. Manual Jupyter execution works from
the bundle directory without the product checkout; install its requirements.
See [notebook-reports.md](notebook-reports.md), including trusted-code and replay limits.
The GUI now offers explicit asynchronous bundle inspection/execution with the
same notebook/receipt grammar, separate from generation and RunRecord admission.
Its runner uses the chosen interpreter for the optional Jupyter stack; the CLI
adapter still requires those optional execution packages in its invoking
environment. See [notebook-gui-execution.md](notebook-gui-execution.md). No new
SQLite objects, implicit fits during reads or optimizer-resume command is added.

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
| layout | value | Closed workspace layout profile version 1 in metadata; read-only inspect returns layout_preferences. Never changes global preferences. |
| set-node | id, node ID, values; optional remove_fields | Property edits; changing/removing ID/type refused |
| add-node | id, node object | Explicit new block |
| remove-node | id, node ID | Remove block/incident wires; prune Concat keys; no guessed rewiring |
| connect | id, edge object | Append explicit wire; update ordered Concat sources separately |
| disconnect | id, exact edge object | Remove exactly that wire or fail |
| delete | id | Delete unreferenced graph; last architecture protected |
| mode | enabled boolean | Persist workspace safety policy only |
| put-spec | kind, document, optional select boolean (default true) | Validate/store immutable dataset/experiment/model/report/run/task revision and exact dependencies, optionally select for browsing |
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

The CLI does not control an unsaved GUI buffer/modal dialogs/active jobs or
execute arbitrary Python through edit operations. Notebook generation does not
execute code; explicit execute-report runs trusted notebook cells in an isolated
kernel, not a hostile-code sandbox. Training is only through explicit commands
or the clearly labelled rerun resolution cell. Agent-prompt generation is not
implemented.

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
