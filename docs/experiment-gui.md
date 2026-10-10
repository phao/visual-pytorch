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
Use the existing CLI to author/edit/select/archive scientific objects. Full
schema-specific visual dataset/task/experiment authoring forms are not provided.

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
scienceRecovery exposes the full result plus admission error; the job also retains
it in memory. This is not durable crash recovery or a detached job receipt.

## Verification and remaining work

Seven pure boundary tests cover exact read-only indexing/capture, isolated
dependencies, changed heads, invalid references/name/retry, longest histories,
captured manifests, saved-report failures and exclusions. A real sandboxed
WebEngine test browses eight rows, performs a three-update fit and retry,
preserves the historical dropdown revision, explicitly selects the new head,
creates a saved-result recipe, generates without training and saves/reloads v6.
It checks zero console errors and no GUI-host Torch import. Notebook generation
uses a private Qt+nbformat environment; user environments/config/files are untouched.

349 Python/57 JS tests passed, plus inherited WebEngine gestures/themes and native
layout checks. New rendered CSS regression checks 240 block/palette pairs.
This is not full C completion. Rich authoring/metric/predictor views, multi-run
report selection, GUI notebook execution, cancellation/late-admission/workspace
recovery stress tests, bounded metadata/raw-data presentation, rendered report
figure review and requirement-by-requirement C convergence remain. D/E stay open.
