# Concrete experiment GUI — branch-only Stage C checkpoint

The accepted floating Experiment dock now has an **Experiments & history** tab.
This is not in the initial public prerelease. It uses the immutable v6 registry,
not a parallel UI database, and preserves legacy training/code/weights tabs.

## Interaction

Browse datasets, experiments, frozen architectures, report recipes, run history
and tasks/bindings. Metadata is alphabetical; execution axes, tensor ports and
splits are not reordered. Every option contains an exact hash and marks current
head versus historical revision. Initial browse visibly defaults to a current
head. Refresh retains the exact selection, even after retry creates a newer one;
select the new head explicitly. Creating/importing selects the returned exact
revision. Changing workspace clears stale selections and candidate cards.

Read-only browsing now uses searchable 25-row metadata pages, retains historical
selections outside a page as a marked pin, and displays a typed summary rather
than eagerly transferring the complete object. Specification/evidence opens an
explicit paged tree, including long tensor values and dictionary keys. Pending
and stale replies are guarded, including a notified disk reload with unchanged
UUID. These are projection/DOM bounds, not lazy storage or registry-cost bounds.
See [scientific-browser.md](scientific-browser.md) for exact schemas and CLI parity.

For non-root dataset/task/experiment values, **Edit this value…** opens a separate
owned draft and raw chunk editor from the selected exact revision. Apply fragment
preserves unfinished JSON; Use staged value parses/replaces the whole value only
on success; Validate and Retain remain explicit. UTF-8 byte cursors and literal
tokens preserve long values and integer precision. Constrained RAM admission
estimates precede large-value parsing and undo capture. This actual Qt path is
also used as a linked child by the typed dataset form; its full protocol and cost limits are in
[chunked-draft-values.md](chunked-draft-values.md).

Task context and experiment stored-revision selectors now use independent search/
page controls instead of complete metadata indexes. They preserve historical and
explicitly unavailable pins without rebuilding/discarding other text controls.
Task context defaults once to a returned head; experiment selections remain
explicit. Field/input-port/output-port suggestions now use independent search/
pages and exact-ID inspection, preserving manual binding text. Shape/axes previews
and category counts omit full vocabularies/metadata; browsing retains scientific
ordinals. Local draft axes/pairing choices and experiment full drafts remain
unbounded. See [authoring-revision-choices.md](authoring-revision-choices.md) and
[task-context-pages.md](task-context-pages.md).

Owned task card/transform pages, literal edits and compatibility commands are
tested through the actual native bridge; [typed-task-sessions.md](typed-task-sessions.md)
specifies that transport. The actual task form below now uses it, including
literal queued edits and linked child staging. Experiment authoring remains on
the complete-draft path; this does not certify all native resource costs.

Freeze architecture captures the normalized drawing closure, not trained weights.
Import specification reads strict finite JSON from a native chooser into the
chosen registry kind; dependencies must already exist. Invalid imports fail.
The dataset, task and experiment authoring forms below create typed observations,
reusable bindings and exact experiment selections. They share the existing
strict registry/CLI contracts; no raw whole-specification JSON editor is required.
The scientific tab header says Exact revisions · explicit runs, rather than
applying the legacy drawing's full-batch MSE contract to every independent task.

### Typed dataset authoring

New dataset opens an incomplete private draft; it does not add fake observations
or write SQLite. Edit dataset revision starts from the selected exact revision,
including a historical one. The immutable base and current browsing head are
displayed/recorded separately. Field and observation IDs are stable and not
renumbered. Adding a field adds explicit null cells to existing records, requiring
either nullable semantics or actual user-entered values before valid admission.
Removing fields/observations affects only the private draft; Cancel discards it.

Field controls specify name, dtype, sample shape, optional ordered axis names,
nullable flag and category vocabulary. Blank shape means scalar; only the first
ragged axis may be -1. Vocabularies are ordered, type-sensitive fixed labels, not
inferred from observations. Modality, unit, coordinates, provenance and usage are
optional finite JSON metadata; their text parser rejects duplicate keys, including
escaped aliases. These are metadata editors, not an arbitrary DatasetSpec textarea.
Cells accept JSON scalars, nested tensor arrays and null. Values are never silently
reshaped/coerced when the schema changes. Cells/vocabularies carry literal JSON
tokens; int64 values cover the full signed 64-bit range without JavaScript Number
rounding. Category labels retain their type-sensitive vocabulary contract.
Float32 magnitude and shape/type compatibility are
checked by the existing server validator. Invalid local text remains visible and
blocks paging/save rather than being discarded on a rerender.

Field definitions/columns page by at most six; observations by at most twenty,
using actual byte-shortened cursors. The native owned session retains hidden
data; complete drafts no longer cross the channel for dataset authoring. Small
descriptors contain bounded literal JSON; large tensors/metadata/vocabularies
show explicit type/size markers and offer staged editing. Pending typed edits
flush by stable target before opening a value child in the same session. Child
Finish/Discard return to the parent without retaining or discarding its draft.
Retry page recovers failed reads; missing initial pages disable page-dependent
actions. Failed batches preserve invalid text; atomicity is per accepted batch,
not a multi-batch form transaction. See [typed-dataset-sessions.md](typed-dataset-sessions.md)
for closed commands, DTOs, edits, preservation and native evidence. Transport,
small-preview work and DOM bounds are not lazy storage or universal host RAM/
validation/history bounds. Experiment forms still transfer complete drafts;
field/port suggestions now have bounded pages, though native registry decoding
still has complete costs. External/lazy storage remains open.
Validate draft is read-only, without fitting/transforms/RNG/execution. Retain
revision invokes normal strict registry admission and selects the exact result;
explicit Save workspace is still required to persist the working copy to disk.
An unchanged selected revision produces no SQL/dirty/history change.

The shared host draft version 1 has exactly version, kind, id, workspace_id, base, expected_head
and document. The document has no revision hash: one is computed on admission.
Begin verifies any exact base before removing its hash for editing. Check/Save
require unchanged stable identity, the same workspace UUID and the captured head
(which can be null for an archived object). New drafts cannot shadow ANY retained
ID, even one with no head. Historical editing is explicit, never substituted with
current bytes. A changed head fails before insertion, preserving draft text.
Changing workspace disables the retained form; Cancel/reopen is explicit. These
are optimistic content/head checks, not a complete event/ABA or file-save conflict
detector. Existing external file Save conflict checks remain independent.

Drafts are transient UI state, not additional .vtw tables. Undo restores heads and
retains immutable evidence through the normal history wrapper. Pending terminal
delivery blocks draft admission. In-flight commands disable duplicate submission;
the existing keyboard typing safeguards apply to all inputs/textarea/selects.
The pure backend supports dataset/task/experiment drafts under their existing
closed grammars. Dataset/task use owned page/edit commands; experiment
authoring remains on the complete-draft AuthorBegin/Check/Save path.

### Typed task authoring

New task opens an incomplete private TaskSpec draft with empty input/head lists.
It does not guess bindings, targets, losses or observations from list position.
Edit selected revision works for all three authorable kinds, including exact historical
revisions. Names are editable labels, not port/field identities. Task ID remains
fixed throughout editing. The task is reusable: the optional model/dataset chosen
for suggestions and compatibility are **not stored dependencies in TaskSpec**.
Creating a task does not require a model/dataset context; executing it does.

Input cards explicitly specify stable input port ID, source field ID and an
ordered transform chain. Head cards specify stable head/output-port/target-field
IDs, loss, positive objective multiplier and optional mask/sample-weight field
IDs. Add/remove and up/down controls change only the draft. Boundary arrows stay
disabled. Reordering preserves identities; transform execution is never sorted.
The add-transform list is alphabetical: affine, impute-mean, onehot, pad,
standardize. Numeric controls accept finite explicit values; padded lengths must
be positive JavaScript-safe integers. Backend validation also rejects zero affine
scale and all unsupported transform combinations. There is no arbitrary code box.

Input-derived mode is None, Sequence length or Padding mask. Each derived binding
has explicit padded length and must match a pad binding for the same source field
and length. Its transforms must be empty. Changing derived mode does not silently
delete an existing chain: validation explains the conflict until the user removes
it. Likewise cross-entropy requires categorical index targets, a fixed vocabulary,
raw logits with the class dimension first, and no target transforms. Selecting CE
does not remove incompatible target transforms. MSE/MAE target transformations
remain explicit. Objective multipliers and observation/element sample weights are
different controls; a mask controls participation. Every output needs one head.

The actual form pages input/head cards independently (at most six each), with
separate six-step pages for visible transform chains. Small fields queue complete
literal JSON tokens without Number serialization. Invalid text remains visible
and blocks mutation; batches are atomic individually. Structural edits flush
queued named properties, reload current paths, then mutate. Initial/return-page
failures offer Retry without another Open; admission stays disabled when a required
reload failed. Large descriptions/properties offer a same-session raw child;
Finish/Discard returns to the parent without retaining/discarding the whole task.
Exact context refresh does not flush or rebuild binding text. Native origin proof
is required before resuming a retained draft after returning to its workspace;
notified reload requires cancel/reopen. See [typed-task-sessions.md](typed-task-sessions.md).
These are concrete tensor bindings, not recurrent-state or symbolic-family support.

The context dropdowns show exact frozen model/dataset hashes, current-head versus
historical status, input/output sample shapes and source field schema. Datalists
suggest IDs but never select a binding automatically. Ports and fields retain
contract order. Refresh preserves an existing exact selection even if new heads
exist. If refreshed metadata omits it, an unavailable-exact option retains its
identity instead of silently substituting another head. Missing exact bytes fail
visibly. Context refresh changes suggestions only; unsaved/invalid form text is
not rerendered away. Responses from a different workspace UUID/draft are ignored;
these checks do not distinguish replacing a database with the same UUID.
Draft Check/Save replies have the same UUID fence. If native admission completed
in the originating working copy before the workspace changed, its reply must not
select that origin's reference in the replacement workspace or clear its retained
draft. This is not detached persistence or recovery of a discarded working copy.

Validate draft checks reusable intrinsic TaskSpec grammar and prospective content
hash. Check exact interface compatibility also checks the selected graph ports,
field dtypes/shapes/axes, fixed class ontology, masks/weights and derived relations.
It returns compatible/incompatible with the reason and exact context hashes; this
is explicitly **interface-only**. It does not fit preprocessing, allocate tensors,
check RAM admission, audit mask/weight values or guarantee training success.
An intrinsically valid reusable task may be retained even if one example context
is incompatible. Preparation/execution retains its independent stronger checks.
Typing, changing bindings or changing context clears an older compatibility
result and old prospective-validation text immediately. A different workspace
UUID clears suggestions and disables the
retained draft, including compatibility actions, until explicit Cancel/reopen.

All bindings/transforms are currently rendered; their size is not bounded or
virtualized. Context returns field/interface metadata, not observation values,
but its backend still validates/loads the full inline dataset. Full data transport
and metadata/rendering bounds remain unfinished, not proven by the dataset grid's
paging. Task revisions use the existing v6 registry, history and explicit file
Save, without new tables or an alternative task store.

### Typed experiment authoring

New experiment opens an incomplete private ExperimentSpec with empty architecture,
dataset/task, split, seed and training-configuration axes. No model, dataset or
task is guessed from list position. Each exact reference dropdown displays name,
hash and historical/current-head status. Refresh changes available choices only:
selected hashes and invalid local text remain. A missing selected choice is
retained as an unavailable identity until explicitly replaced. Editing a historical
experiment does not substitute the current head. Names/question are labels; the
experiment's stable ID stays fixed.

Dataset/task pairing, split and training-configuration IDs are explicitly editable.
Up/down controls preserve ordered semantic axes; remove affects the draft only.
Renaming/removing an ID does not silently repair comparison references. Intrinsic
validation explains resulting conflicts. Crossed mode requires an empty pair
whitelist; paired mode requires explicit model–dataset/task entries. Changing mode
keeps conflicting entries until explicitly removed. Exclusions require a reason
and remain visible in preview rather than being dropped.

The alphabetically browsed split methods are external, group, row and temporal.
Row/group controls specify a separate split seed; group/time specify a literal
field ID. Train/validation/test fractions stay explicit. External splits specify
each role and excluded observations with one literal stable ID per line. Spaces
and commas inside IDs are preserved, and neither sorting nor deduplication occurs.
Each existing entry has a fixed method: changing it requires explicit remove/add
and the desired ID, not a coercion that discards method-specific settings.

Initialization seeds form an explicit ordered axis. The CPU float32/Adam version-1
configuration exposes batch size, epochs/updates/examples/seconds bounds, evaluation
interval, retained history points, learning rate, both betas, epsilon, weight decay,
base initialization seed and independent shuffle seed/flag. Blank budget fields
mean null/disabled; at least one positive finite bound is required. Execution stops
at the first observed bound under the existing trainer contract. The experiment's
initialization coordinate overrides the base protocol initialization seed; the
shuffle seed remains independent. The UI labels fixed no-augmentation,
last-successful-checkpoint and final-completed-only test policies. Adding a
configuration explicitly inserts the same defaults as the authoritative Python
factory; it does not train.

Preview draft validates the intrinsic/dependency/head/UUID contract and computes
the prospective hash, then returns at most 25 coordinate results using existing
interface/split compatibility. It never admits a revision/run, fits preprocessing,
allocates tensors, consumes global RNG state or checks RAM admission. Incompatible
interfaces/splits and exclusions remain visible with reasons. An interface-compatible
row does not guarantee training success. Total Cartesian size is also returned as
an exact decimal string for counts beyond JavaScript's safe integer range; paging
offsets remain within the existing exact-integer contract. The preview has no Run
buttons: retain a valid revision first, then explicitly run its stored candidates.
Typing/reordering/changing choices clears an old preview immediately. Late preview
or metadata replies cannot repopulate another workspace UUID/draft. These optimistic
checks do not prove same-UUID database replacement, ABA/event or crash safety.

Candidate pages bound returned/rendered rows, not full inline dependency transport,
manifest bytes, metadata lists or the number of authoring controls. All experiment
axis cards currently render. Preview is not a lazy-data implementation or execution
resource admission. Retain/check/no-op/undo/Save-reload use the same v6 rules as the
other draft forms; no new SQL tables or parallel experiment store are introduced.

### Stored candidates, run evidence and reports

Experiment browsing displays its question and pages of 25 coordinates. All
compatible, excluded and incompatible rows remain visible with reasons. Only
compatible rows enable Run this candidate; it explicitly launches one fit, not
a silent sweep. Active work disables competing actions and scientific edits;
reads remain available. Stop invokes the existing worker cancellation event.

Independent run history is not the drawing's latest-result pointer. It shows the
selected immutable RunRecord, attempts/status/errors and resolved evidence.
Retry is explicit fresh training of the same selection/captured split, not
optimizer resume; longest retained history wins after old-head selection.
Create saved-result report points at the selected exact run revision and has an
editable question. Reading saved or rerun recipes never launches fits. Rerun mode
is labelled as such. The existing one-run report action remains available.

Saved-run cards show actual counters, population objectives, per-head metrics and
expandable exact additive ledgers/checkpoint identities/captured context. Retained
attempt selection is explicit; a failed latest retry never substitutes earlier
metrics. Metric rows page by 25; raw JSON does not open automatically.

Add exact run to comparison opens a transient ordered basket. An identical
reference is idempotent; another revision of the same logical run requires
explicit removal first, not counting a retry as another run. Name/question and
Create saved comparison recipe retain an immutable result-report without training.
Save workspace persists it. Cards page by five runs, including unsuccessful cases;
Open exact run pins the recipe's reference, never the current head. No metrics
are pooled or independence asserted. See [saved-evidence-views.md](saved-evidence-views.md)
for response fields, attempt/units/guards and paging-versus-validation boundaries.

View/Export checkpoint predictor use the displayed retained attempt, not an unread
number or mutable latest head. Generated source contains saved preprocessing and
strict weights; generation never executes it. A native chooser exclusively creates
a new Python file after rechecking origin/busy state. Missing result disables
export; cancelled or zero-update checkpoints keep their actual status/counts.
Preview is safe literal source in a themed base-font scroll pane. See
[saved-predictor-export.md](saved-predictor-export.md) for exact source/receipt/CLI,
prediction units, privacy and validation-resource boundaries.

Generate notebook bundle creates a new complete directory via a native chooser,
without executing cells or overwriting files. Install optional
`requirements-report.txt` in the GUI interpreter for authoring; the numerical
worker may use another Python. Missing optional dependencies fail clearly before
output creation. The Notebook execution disclosure also asynchronously inspects
existing/generated bundles and explicitly runs trusted notebooks through a real
kernel in the chosen report/numerical environment, not the Qt process. Its
mode-aware native confirmation defaults to Cancel; Stop, timeout/failure and
verified external receipts stay distinct. Rerun outputs are never auto-imported
into the source workspace. Review inline observations/weights for privacy before
sharing bundles. See [notebook-gui-execution.md](notebook-gui-execution.md) for
exact selection, progress/terminal events, dependency/origin/queued-busy guards,
receipt presentation and remaining lifecycle/transport limits.

## Bridge and state

All commands require an open workspace. Read references contain kind/id/revision;
a null hash is rejected, not interpreted as a mutable head.

| Action | Inputs | Effect |
| --- | --- | --- |
| scienceIndex | none | All exact revision metadata, current-head flags |
| scienceShow | reference | Exact stored document, no execution |
| sciencePreview | experiment reference, offset/limit | Existing bounded compatibility preview |
| scienceReport | report reference | Read saved evidence or labelled rerun recipe |
| scienceRunEvidence | exact run reference, optional attempt, offset/limit (default 0/25, limit 1..25) | Read-only saved-attempt projection; never execute |
| scienceReportEvidence | exact report reference, offset/limit (default 0/5, limit 1..10) | Read-only exact saved-run projections; never execute |
| scienceRunPredictor | workspace_id, exact run reference, optional attempt | Read-only retained inference source generation; never execute |
| scienceExportRunPredictor | workspace_id, exact run reference, optional attempt/output | New source file; modal origin/busy recheck, no overwrite/execution |
| scienceAuthorBegin | kind, optional exact reference or new id | Read-only draft with base/head/workspace capture |
| scienceAuthorCheck | draft | Strict read-only intrinsic/dependency validation, prospective hash |
| scienceAuthorPreview | experiment draft, offset/limit (default 0/25, limit 1..25) | Read-only prospective compatibility page; never retain or fit |
| scienceAuthorSave | draft | Optimistic checks then undoable immutable admission; exact no-op is read-only |
| scienceTaskContext | exact model, dataset references | Legacy complete port/field suggestions; no records/fitted state |
| scienceTaskContextPage | workspace_id, exact model/dataset, collection; optional query/offset/limit/pin | Actual bounded field/port suggestions; task-context-pages.md specifies DTO |
| scienceTaskCompatibility | task draft, exact model/dataset references | Intrinsic draft check plus interface-only compatible/incompatible result |
| scienceFreeze | drawing id | Freeze complete normalized model closure |
| sciencePut | kind, document | Strict registry admission |
| scienceSelect | reference | Explicit browsing-head change |
| scienceImport | kind | Native JSON chooser, strict admission |
| scienceResultReport | runs, question, optional id/name/workspace_id | Reading recipe over exact runs; supplied workspace UUID checked before admission |
| scienceTrain | experiment reference, coordinates, optional run_id/name/retry | Explicit isolated fit/retry |
| scienceGenerateNotebook | report reference, optional output/offset/limit/allow_partial | Generate, never execute |
| scienceNotebookInspect | workspace_id, optional directory/python/request_id | Async input inspection, no kernel; selection via terminal event |
| scienceExecuteNotebook | workspace_id, exact inspected selection, optional python/timeout | Explicit confirmation, async actual kernel, new external artifacts |
| scienceNotebookHistory | none | Read copied session notebook execution rows |
| scienceNotebookEvidence | id | Read/revalidate retained receipt/artifact bytes; never execute |

Scientific mutations join the undo wrapper; immutable revisions survive head
undo. Modal choosers recheck workspace identity and active work after returning.
Reads do not change workspace bytes. Save workspace explicitly to persist edits
and terminal evidence. State has a separate experiment_job ID: it does not imply
that the legacy drawing/latest-result is the executing scientific revision.
Notebook inspection/execution has its own notebook_job state, not an invented
experiment RunRecord. Its durable receipts are external; session history is not
stored in the workspace or domain undo stack.

`scienceTaskContext` returns version 1, model/dataset references, input_ports,
output_ports, fields and record_count. It is a supported legacy read, not the
actual GUI suggestion path. `scienceTaskContextPage` uses independent exact
collection pages with IDs/ordinals/labelled shape previews, not whole vocabularies/
metadata. See [task-context-pages.md](task-context-pages.md) for closed grammar
and CLI parity. `scienceTaskCompatibility` returns version
1, prospective task reference, model/dataset references, status, reason (string
or null), and scope `interface-only-no-fitting-or-execution`. Invalid intrinsic
drafts, stale heads/workspaces and invalid/missing exact references raise errors;
valid intrinsic drafts with incompatible interfaces return a labelled result.
Both actions are reads, including while a numerical job is active. GUI authoring
controls are disabled while busy; native draft admission still uses the central
pending-terminal guard before attempting to parse an edit.

scienceAuthorPreview returns version 1, preview (the existing candidate-page
schema with prospective experiment reference) and total_combinations_exact (decimal
string). Invalid intrinsic drafts/dependencies/head/UUID, offsets or limits raise
before writes. It is also a read action; GUI draft controls stay disabled while busy.

`experiment_ui.capture` copies the exact experiment dependency closure, candidate
manifest and longest prior history for retry, not unrelated registry datasets.
ExperimentJob builds its own Workspace on the Qt thread and runs the existing
validated numerical executor; it never mutates the GUI SQLite connection there.
Terminal records are admitted on the GUI thread. Progress is run_id/point;
scienceComplete is exact reference/record; scienceChanged refreshes browsing.
scienceRecovery exposes the full result plus admission error. Window-owned
recovery retains it even after deleting a completed job. This is not durable
crash recovery or a detached job receipt.

### Terminal delivery and recovery

Execution stays busy until both the thread and queued terminal admission finish.
An explicit GUI-thread QObject receiver handles progress, terminal and finish
slots; `isRunning() == false` alone does not allow replacement, edits or another
fit. Superseded progress is ignored. A terminal is delivered at most once and
its reference/hash is checked before SQL insertion. Changed workspace/job or
failed admission retains the complete captured closure and result in a
window-owned recovery collection, never in the new drawing.

Read `scienceRecoveries` for summaries and `scienceRecoveryShow {id}` for a copied
envelope. Export explicitly using `scienceExportRecovery {id}` (optional `output`
for agents/tests). The GUI has Show recovery and Export controls. Export refuses
existing paths and never trains, admits a run or saves a workspace. Version-1 JSON
has format `visual-pytorch-experiment-recovery`, ID, snapshot, result, error and
exported_to. Snapshot includes origin workspace ID/path/disk revision, exact
dependency documents, selection, run ID/name, prior history and split manifest.
Result is the existing record/reference, or null on pre-terminal failure. Origin
is provenance, not permission to overwrite a file. The exported file records its
pre-export state; the session marks exported_to only after a successful flushed
export. A failed write may leave a partial new file; the evidence remains in the
session and can be exported to a different new path. No execution receipt or
numerical result is fabricated from a thread error.

Captured observations/parameters may be sensitive. To recover, re-inspect the
intended workspace and explicitly admit dependency-first documents followed by
the retained record through validated CLI edits, without repeating the fit. Do
not pass the outer envelope directly to put-spec. No automatic cross-workspace
recovery import occurs. Memory survives workspace changes, not process death.
Close during pending delivery requests stop and leaves the event loop alive;
close again after evidence arrives. Unexported recovery requires explicit discard
confirmation before exit. Stop stays accessible when browsing run history.

Qt states that queued slots run when the receiver event loop resumes, not when
the worker merely stops. This motivates the separate admission busy condition:
[Threads and QObjects](https://doc.qt.io/qt-6/threads-qobject.html),
[QThread](https://doc.qt.io/qt-6/qthread.html).

## Verification and remaining work

Seven pure boundary tests cover exact read-only indexing/capture, isolated
dependencies, changed heads, invalid references/name/retry, longest histories,
captured manifests, saved-report failures and exclusions. A real sandboxed
WebEngine test browses eight rows, performs a three-update fit and retry,
preserves the historical dropdown revision, explicitly selects the new head,
creates a saved-result recipe, generates without training and saves/reloads v6.
It checks zero console errors and no GUI-host Torch import. Notebook generation
uses a private Qt+nbformat environment; user environments/config/files are untouched.

At the initial GUI checkpoint, 353 Python/57 JS tests passed before the separate palette UI followup, plus inherited WebEngine gestures/themes and native
layout checks. New rendered CSS regression checks 240 block/palette pairs.
Subsequent checkpoints implement structured metric/predictor views, saved multi-run
recipes, explicit GUI notebook execution and bounded rendered-notebook/figure
acceptance. Their contracts are linked above and current checks are recorded in
[development.md](development.md). This is still not full C completion: full
metadata/raw-data/manifest transport and authoring/report presentation bounds,
richer in-app figures and requirement-by-requirement C convergence remain.
D/E stay open.

Eleven native Qt lifecycle tests include an actual thread finished while GUI
delivery is deliberately withheld, central edit/run/replace/close guards, exact
head identity, duplicate terminal, superseded progress, invalid-reference
atomicity, failed admission, independent recovery/export/refusal to overwrite,
missing terminal, exit warning and cooperative real-worker cancellation. The
cancelled checkpoint has actual partial counters and no test-population metrics;
the host imports no Torch. WebEngine also forces admission failure after a real
fit, destroys the completed QThread, reads/exports retained evidence and verifies
no new run was admitted. These are local lifecycle gates, not crash recovery,
detached execution, or exhaustive reentrancy/model checking.

Task authoring has fourteen pure tests and seven helper tests for closed grammar,
exact context/history, explicit port-order changes without rebinding, independent
drafts, field/axis/ontology and mask/weight compatibility, derived conflicts,
finite number parsing and ordered moves. The real `qt_task_authoring_smoke.py`
starts without any stored tasks, creates weighted classification/regression and
ragged pad/length/mask tasks through typed controls, checks stored bindings/chains,
then runs both exact saved revisions for three actual worker updates each. It
checks intrinsic conflict retention, invalid local text/context refresh, immediate
compatibility invalidation, missing choice/no head fallback, workspace replacement
during a context response, undo/redo/no-op/stale/busy guards and v6 Save/reload.
It also switches workspace during real Check/Save callback delivery, preserving
the draft and preventing origin references from leaking into replacement state.
Dark/GitHub/Math forms pass at fonts 14/18/28 and spacings 1/4/10; sandbox remains
enabled and GUI-host Torch absent. These are bounded real examples, not proof of
all task/operator combinations or complete Stage C.

Experiment authoring adds thirteen pure tests and six helper tests: exact
prospective hashes, read-only/no-fit/RNG isolation, paging, paired/excluded rows,
all four actual split manifests, order/seed override, changed/archived heads,
invalid grammar/dependencies/UUID and lossless Cartesian count formatting. The
real sandboxed qt_experiment_authoring_smoke.py starts without stored experiments,
authors every configuration field, checks the retained bytes and executes all
eight two-model × two-dataset × two-initialization combinations through explicit
GUI clicks. Group/time/external authored splits each execute another actual fit.
A 96-row draft pages 25/25/25/21 with 48 compatible, 32 incompatible and 16
reasoned exclusions; paired/crossed conflicts stay until explicitly corrected.
Invalid text survives metadata refresh; exact historical/missing selections,
no-op/undo/redo, stale heads, late preview workspace switch, pending delivery and
v6 Save/reload are checked. Dark/GitHub/Math fonts 14/18/28 and spacing 1/4/10 pass;
the form capture was inspected. No console errors or GUI-host Torch import.
These are bounded correctness/authoring checks, not training-quality evidence,
an implicit sweep or the remaining C/D/E gates.

Current checkpoint: 394 Python tests (134.602 s), 83 JS tests, all three real
authoring forms, the existing fit/retry/report/notebook-generation/recovery flow,
11 native Qt lifecycle tests and inherited sandboxed editor pass. Compile/syntax/
diff checks pass. Partial Stage C remains branch-only and unpublished.
