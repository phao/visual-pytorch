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

Freeze architecture captures the normalized drawing closure, not trained weights.
Import specification reads strict finite JSON from a native chooser into the
chosen registry kind; dependencies must already exist. Invalid imports fail.
The dataset authoring form below creates and edits typed observations. Task and
experiment authoring forms remain unfinished; use the CLI or specification import
for those objects. Do not treat an imported experiment as completed GUI authoring.

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
reshaped/coerced when the schema changes. Integer/category numerals must remain
JavaScript-safe exact integers; float32 magnitude and shape/type compatibility are
checked by the existing server validator. Invalid local text remains visible and
blocks paging/save rather than being discarded on a rerender.

Field definitions/columns page by six; observations page by twenty. Hidden values
remain in the draft, not truncated. This bounds rendered cell count, NOT transport
or draft size: scienceShow still loads the complete inline document. External/lazy
storage, bounded metadata transport and richer diagnostics are still open work.
Validate draft is read-only, without fitting/transforms/RNG/execution. Retain
revision invokes normal strict registry admission and selects the exact result;
explicit Save workspace is still required to persist the working copy to disk.
An unchanged selected revision produces no SQL/dirty/history change.

Draft version 1 has exactly version, kind, id, workspace_id, base, expected_head
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
closed grammars, but only the dataset form is currently connected to the UI.

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
is labelled as such. Multi-run reports can be imported/created through the CLI;
the current GUI creation action selects one run.

Generate notebook bundle creates a new complete directory via a native chooser,
without executing cells or overwriting files. Install optional
`requirements-report.txt` in the GUI interpreter for authoring; the numerical
worker may use another Python. Missing optional dependencies fail clearly before
output creation. Notebook execution remains explicit CLI/Jupyter work, not a GUI
action. Review inline observations/weights for privacy before sharing bundles.

## Bridge and state

All commands require an open workspace. Read references contain kind/id/revision;
a null hash is rejected, not interpreted as a mutable head.

| Action | Inputs | Effect |
| --- | --- | --- |
| scienceIndex | none | All exact revision metadata, current-head flags |
| scienceShow | reference | Exact stored document, no execution |
| sciencePreview | experiment reference, offset/limit | Existing bounded compatibility preview |
| scienceReport | report reference | Read saved evidence or labelled rerun recipe |
| scienceAuthorBegin | kind, optional exact reference or new id | Read-only draft with base/head/workspace capture |
| scienceAuthorCheck | draft | Strict read-only intrinsic/dependency validation, prospective hash |
| scienceAuthorSave | draft | Optimistic checks then undoable immutable admission; exact no-op is read-only |
| scienceFreeze | drawing id | Freeze complete normalized model closure |
| sciencePut | kind, document | Strict registry admission |
| scienceSelect | reference | Explicit browsing-head change |
| scienceImport | kind | Native JSON chooser, strict admission |
| scienceResultReport | runs, question, optional id/name | Reading recipe over exact runs |
| scienceTrain | experiment reference, coordinates, optional run_id/name/retry | Explicit isolated fit/retry |
| scienceGenerateNotebook | report reference, optional output/offset/limit/allow_partial | Generate, never execute |

Scientific mutations join the undo wrapper; immutable revisions survive head
undo. Modal choosers recheck workspace identity and active work after returning.
Reads do not change workspace bytes. Save workspace explicitly to persist edits
and terminal evidence. State has a separate experiment_job ID: it does not imply
that the legacy drawing/latest-result is the executing scientific revision.

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

353 Python/57 JS tests passed before the separate palette UI followup, plus inherited WebEngine gestures/themes and native
layout checks. New rendered CSS regression checks 240 block/palette pairs.
This is not full C completion. Rich authoring/metric/predictor views, multi-run
report selection, GUI notebook execution, bounded metadata/raw-data presentation, rendered report
figure review and requirement-by-requirement C convergence remain. D/E stay open.

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
