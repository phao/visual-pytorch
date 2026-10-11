# Stage C requirement audit

This is the requirement evidence map for locally integrated Stage C. It follows the
external nepr review's “datasets, selections and notebook reports” programme
(review commit `777dd1c25e11583877e607064ffe7e61dfdb54a4`). The original requirement
paragraph and exit gate are distinguished below from remaining product work.
Do not promote an import, test count or one browser fixture into full platform
acceptance. [Final acceptance](stage-c-acceptance.md) records the current local
integration and actual desktop/kernel checks; no new publication is implied.

## Original implementation requirements

| Requirement | Authoritative implementation and tested evidence | Current scope |
| --- | --- | --- |
| DatasetSpec, TaskSpec, ExperimentSpec, RunRecord and ReportRecipe are independent objects | `app/revisions.py`, `app/experiment_objects.py`; `test_revisions.py` independent dataset/task revisions, guarded SQL, immutable hashes/heads/history; `test_experiment_objects.py` exact dependencies, terminal attempts and v6 save/reload | Implemented, closed version-1 contracts in v6 SQLite. A browsing head is not an execution identity. No detached running job is implied. |
| Preserve the legacy inline regression adapter | Existing MLP/legacy evidence, worker and migration suites; `test_revisions.py::test_actual_legacy_table_migration_pins_two_input_meaning_before_new_revisions` and `test_actual_v1_to_v4_schema_migration_is_readonly_and_preserves_legacy_checkpoint` | Legacy adapter remains alongside, not rewritten as a scientific object on opening. Tests cover supported inherited formats, not arbitrary SQLite. |
| Group/time splits | `app/datasets.py`; `test_datasets.py::test_group_split_is_subject_disjoint_and_counts_units_not_rows`, `test_temporal_split_keeps_ties_together_and_rejects_future_mixing`, external complete/disjoint-selection checks | Row/group/temporal/external protocols are explicit. Subject and equal-time blocks stay together; manifests retain actual stable observation IDs. No general forecasting or RNN stream policy yet. |
| Training-only fitted transforms | `app/datasets.py`, `app/task_execution.py`; preparation tests for held-out extreme values, imputation → units → standardization, independent target fitting; standalone fitted-predictor tests | Supported ordered transforms fit on train IDs only. Inference replays retained fitted values. Source units and transformed-loss space remain separate. |
| Structured losses | `app/task_runtime.py`; independent scalar MSE/MAE/CE gradients, 24 masked dense cases/partitions, unequal-batch additive reductions; multihead scalar Adam and spatial CNN CE tests in `test_task_training.py` | Named heads, masks and observation weights; unavailable denominators are not zero-valued successes. No mean-of-batch-means aggregation or silent head-weight renormalization. CPU float32 training with specified validation tolerances. |
| Minibatching | `app/task_training.py`; `test_task_training.py::test_exact_epochs_updates_examples_and_partial_batches`, skipped/failed/cancelled/time-budget cases and initial/checkpoint timing tests | Last short batches are retained. Examples, attempts and successful updates are different counters. Cooperative budget checks are not hard real-time deadlines. |
| Preview isolation | `preview_experiment`, authoring preview, typed dataset preview; `test_experiment_objects.py::test_preview_is_paged_without_fitting_training_or_global_rng`, no-fit dataset/authoring tests and native authoring fixtures | Prospective read/validation does not fit or execute. Candidate response rows are paged, but complete inline dependencies and registry/form transport are still not cost-bounded. |
| Generate notebooks through an appropriate environment | `app/notebooks.py`, explicit CLI/GUI generation; format/stable-cell-ID checks, no-worker authoring checks, native generation-only inspection fixture | Optional nbformat authoring environment, closed copied runtime, manifest published last; generation does not execute. |
| Execute notebooks through an appropriate environment | `app/notebooks.py`, `app/notebook_execution.py`, `app/desktop_notebooks.py`; actual fresh-ipykernel tests and private sandboxed native GUI saved/rerun execution | Selected report/kernel interpreter; GUI/stdlib host need not load Torch or Jupyter. Trusted Python execution, not a hostile-code sandbox. |
| Preserve failures and artifact closure | Real failing cells, timeout, independent kernel death and actual Stop tests; requested/executed notebook + receipt/artifact hash checks; missing/symlink/changed source and artifact rejection | Actual terminal evidence and prior outputs are retained, later cells remain unexecuted. Ordinary owned Linux kernel/worker cleanup is tested. Crash/detached and arbitrary descendants belong to E, not this claim. |

The exact storage/CLI contract is in [revision-store.md](revision-store.md),
[experiment-objects.md](experiment-objects.md),
[task-training-contract.md](task-training-contract.md) and
[notebook-reports.md](notebook-reports.md). Native authoring/evidence/predictor/
notebook behavior is specified in [experiment-gui.md](experiment-gui.md).

## Original exit-gate evidence

1. **Two models × two datasets × declared replicates expand reproducibly.**
   `test_experiment_objects.py::test_two_models_two_datasets_two_replicates_have_exact_reproducible_coordinates`
   checks eight unique ordered coordinates, initialization/shuffle seeds and
   paired actual manifests. `test_all_eight_candidates_execute_and_independent_retries_reproduce_checkpoints`
   executes all eight through the real numerical worker. Actual experiment GUI
   authoring and portable rerun notebook fixtures also execute this matrix.
   This is fixture/environment-specific reproducibility, not cross-hardware
   bitwise equality or a statistical independence assertion.
2. **Incompatible combinations are visible.** Shape/split incompatibilities,
   paired exclusions with reasons and incomplete selections are tested and
   displayed in prospective GUI/notebook pages. They are not silently filtered
   or assigned fabricated metrics. The GUI fixture additionally checks a
   96-coordinate preview with compatible/incompatible/excluded cases.
3. **Selected runs retain hashes and splits.** Exact dependency revisions,
   manifest/protocol hashes and actual IDs travel with terminal run revisions.
   `test_record_reads_retries_and_reload_use_saved_manifest_not_split_regeneration`
   prohibits split regeneration through reads/retry/reload. Old attempt bytes
   survive head changes/history and later retries. Hashes are integrity checks,
   not signatures or atomic filesystem snapshots.
4. **Saved-result reports do not retrain.** Pure materialization tests prohibit
   host worker dispatch and retain an earlier exact successful/failed revision
   after later retries. Real notebook tests additionally install a training
   fence **inside the fresh kernel** before evidence resolution and all report
   sections. It guards both candidate entry points, task training, numerical
   worker transport and Torch imports. Saved evidence completes with zero
   violations; the same guarded rerun fails at the first candidate dispatch,
   retaining an actual failure receipt. A parent-process monkeypatch alone
   cannot establish this subprocess property. This is a test probe of declared
   application entry points, not an OS-level security boundary.
5. **Rerun notebooks state their execution behavior.** Metadata, visible prose
   and mandatory captured selection preview explicitly distinguish generation
   from execution and saved-result from fresh rerun. Real kernel runs produce
   eight fresh terminal artifacts; generation produces no execution directory.
   Partial coverage requires explicit permission and remains visible.
6. **Actual notebook kernel execution is verified.** Tests use nbclient and an
   actual fresh ipykernel, not `exec` of notebook source strings. Completed,
   failing, timed-out, dead-kernel and stopped outcomes have distinct verified
   output/receipt behavior. The optional rendered fixture shows both complete
   notebook modes offline in sandboxed Qt WebEngine, including all nine sections.
7. **Portability is verified outside the earlier review's sandbox limitation.**
   `test_actual_rerun_kernel_portable_eight_candidates_and_repeated_attempts`
   copies the complete bundle to an unrelated directory and performs two real
   fresh-kernel attempts. Its runtime is the captured copied closure, not the
   editor checkout; the original bundle and source workspace stay unchanged.
   Standalone exported fitted predictors execute separately without the app
   package/refitting. A fresh stdlib-only host inspects/executes via the selected
   numerical environment. These are local Linux/interpreter checks, not remote,
   cross-platform, GPU, every-Jupyter-frontend or optimizer-resume guarantees.

## Historical convergence and current cost boundaries

The computational exit examples above and the practical desktop/CLI convergence
are now tested and locally integrated. The following records the convergence
work and retained native cost boundaries; it must not be read as a declaration
that final GUI acceptance is still pending. The original C programme did not add
universal lazy-storage/RAM/platform proof gates. See the final acceptance note
for window tests, source-tree equality and next D/E work.

- The main read browser now uses bounded `scienceBrowse` metadata pages,
  `scienceSummary` typed summaries and explicit `scienceInspect` container/scalar
  pages, shared with CLI. Historical pins, unchanged-UUID notified reload and
  stale delivery guards are tested. Complete legacy `scienceIndex`/`scienceShow`/
  `scienceReport` reads remain available; authoring stored-revision selectors now
  use independent bounded choices without full metadata indexes. Task field/port
  suggestions now have independent bounded exact collection pages, inspected IDs
  and explicitly labelled shape previews; actual experiment editing now uses
  owned card/value/local-choice/edit/preview projections. Actual recovery choices/
  inspection and terminal notices now have separate bounded projections, specified
  in [recovery-browser.md](recovery-browser.md). Thread-internal full terminal
  payloads/native retention/admission costs remain. Main browser bounds are not
  universal resource bounds.
- Legacy `scienceAuthorBegin/Check/Save/Preview` deliberately remain complete-draft
  compatibility APIs. The actual dataset/task/experiment forms use owned-session
  projected reads, sequence-fenced edits and Check/Commit without whole draft
  transport. Dataset has migrated;
  [typed-dataset-sessions.md](typed-dataset-sessions.md) specifies literal bounded
  descriptors, stable-target edits, failed-read retry and linked staging in the
  same session. The actual task form now also uses owned literal cards/edits and
  linked child staging; [typed-task-sessions.md](typed-task-sessions.md) specifies
  independent bounded binding/pipeline pages, current named-property targets,
  exact acknowledgements and retry/origin fences. The actual experiment form
  now also uses independent six-card axes, exact local choices, named literal
  queues, linked raw children and projected previews;
  [typed-experiment-sessions.md](typed-experiment-sessions.md) distinguishes its
  real-controller evidence from earlier native-only tests. Current native full
  validation/manifest costs remain.
  [owned-authoring-sessions.md](owned-authoring-sessions.md) records the
  tested transport scope. The separate actual chunk editor now stages raw large
  values with explicit acceptance and constrained admission estimates; see
  [chunked-draft-values.md](chunked-draft-values.md). Dataset now avoids fully
  encoding a huge tensor/metadata/vocabulary merely to preview it, preserves
  hidden values and full int64 tokens, and limits actual result bytes/control
  count. This does not bound all host/storage/dependency/history costs.
  Task contexts now avoid full field/vocabulary/metadata transport; see
  [task-context-pages.md](task-context-pages.md). Host registry decoding still has
  complete costs. Experiment rendering now bounds visible axes/selection cards
  and independently searched local choices; task rendering bounds visible
  bindings and each visible pipeline, without bounding all native costs.
- Saved metric/run pages and notebook figure tables bound returned/rendered
  rows, not arbitrary metadata size, full registry validation/inline costs,
  whole-file CSV reading or notebook source/payload size. Do not conflate these
  with lazy storage or universal RAM/OOM guarantees.
- Actual stored-candidate browsing now uses `sciencePreviewPage`, a separate
  byte-bounded projection preserving exact candidate identity/order/status/
  references/counts and decimal totals; deliberate complete CLI preview remains.
  [candidate-browser.md](candidate-browser.md) describes actual cursor/failure/
  stale-selection/run guards and a real96-coordinate Qt fixture with large
  manifests. Native complete dependency validation/manifest generation costs
  remain; this is not universal cost or execution convergence.
- Request-local exact dependency/contract/compatibility/split preparation now
  removes repeated successful native preview work without changing candidate
  results or direct training/retry validation. Eleven focused equivalence/isolation
  cases and a matched identical-wire measurement support this scoped optimization;
  [preview-resolution.md](preview-resolution.md) records limits. Initial complete
  registry/data/manifest/copy costs remain, not a new universal resource gate.
- In-app saved-attempt training curves now read exact retained head metrics in
  run views and independent report cards, with CLI parity, missing-value gaps,
  stable page axes and exact tables; [saved-run-curves.md](saved-run-curves.md)
  specifies the projection and its cost boundaries. Final native window checks
  cover60 one/two-tab font/zoom/theme/spacing configurations with actual focused
  controls, mouse clicks and native Tab reveals. The notebook-rendering
  checkpoint fixes its reproduced clipping/description/table defects; it does
  not certify every editor view, font or assistive technology.
- The task branch and tests/specs must be reviewed and integrated together.
  Final regression scope includes inherited A/B/editor/history/CLI behavior,
  actual authoring and fit/export/report/kernel UI flows. An unmerged checkpoint
  is not an integrated release. The sole development checkout now holds the
  C branch, so the existing desktop entry launches it on the next normal start.

The bounded **read-only exact-revision scientific browser** is now implemented
on the retained branch; [scientific-browser.md](scientific-browser.md) specifies
the byte encoding, ordinal/cursor/selection contracts and their cost limits.
It does not silently truncate identities, masquerade partial previews as complete
JSON, change record/port ordering or regenerate evidence. Existing complete CLI
reads/explicit exports remain available. Actual experiment typed-authoring
migration is now tested; recovery/completion webview transport now uses small
notifications and paged inspection. Remaining native inline/retention costs,
whole-window geometry and integrated acceptance are next. Hiding
full JSON in a collapsed panel is not a cost bound.
Native task binding/transform page, literal-edit and owned compatibility primitives
now exist with actual bridge checks; [typed-task-sessions.md](typed-task-sessions.md)
records their exact scope. The actual task controller now uses them with bounded
visible cards and same-session child staging. Stage C is accepted locally;
symbolic/recurrent and detached execution remain future D/E work.

After these concrete C convergence items and the integration audit, proceed to
the unchanged D family/recurrent and E durable batch/distributed requirements.
Do not count optimizer resume, distributed reduction, detached reconnection or
remote jobs as C features merely because notebook execution is available.

## Reproduction and evidence status

The production figure checkpoint passed 445 Python and 94 JavaScript tests.
With the two subsequent kernel-fence tests, that checkpoint passed
447 Python tests (169.871 s) and 94 JavaScript tests (547 ms). The focused
kernel-fence pair passes in 4.599 s. Details and the exact limits are in
[development.md](development.md); do not substitute these totals for the
requirement-level evidence above or the remaining convergence items.

The bounded scientific browsing checkpoint passes 457 Python tests (172.967 s)
and 103 JavaScript tests (535 ms), including ten new backend/CLI and nine new
asynchronous browser tests. The actual Qt 1250 × 512 inline-data fixture verifies
historical selection, page/key/value traversal, same-UUID reload, scoped font/
theme geometry, no complete-document main-browser reads and unchanged SQL bytes.
These checks do not certify authoring/resource costs or full C integration.

The subsequent dataset migration passes 490 Python tests (172.052 s) and 148
JavaScript tests. Eleven new backend tests and fifteen typed-controller/four
linked-editor cases cover bounded exact DTOs, atomic ordered edits, full int64
range, focus/invalid text, page retry and parent/child ownership. Actual Qt tests
200 × 512 tensor observations, hidden historical metadata, same-session staging,
two injected page failures, font14/18/28, one history entry/undo/redo/save-reload;
21 successful projections peak 16385 bytes, no host Torch/console errors.
Rendered ID/header column squeezing is repaired and checked against actual font
geometry. Native dataset/task/experiment/session/browser/raw-editor regressions
also pass, including real multihead/ragged/eight-candidate/all-split fits.
These are dataset/shared-hook evidence, not completed task-form/resource/C acceptance.

The subsequent field/port context checkpoint passes 498 Python tests (172.967 s)
and 156 JavaScript tests (714 ms). Eight new backend/CLI and eight asynchronous
controller cases cover exact historical refs, independent searches/pins/actual
cursors, explicit unavailable IDs, literal/partial descriptors, Unicode byte
budgets, arbitrary JSON field-name compatibility and preserved manual controls.
Actual sandboxed Qt tests 600 fields, 32 input/output ports and 20000 category
labels: 17 projections peak 8147 bytes, unchanged SQL, no complete context/index
requests, host Torch or console errors. Historical selection, invalid weight
text, rank-8 previews, supplementary-character IDs, failed-page retry and actual
font14/18/28 geometry pass. Native dataset/task/experiment regressions pass,
including real fits. These are context projection/editor preservation checks,
not task/experiment draft migration, universal host costs or integrated C.

The native task-card/transform/literal-edit/owned-compatibility checkpoint passes
508 Python tests (171.320 s)/156 JavaScript tests (689 ms). Ten new stdlib task
cases and actual sandboxed QWebChannel exercise 80 binding/1000 transform reads,
literal integer precision, ordinal/sequence/owner/origin/staging/admission,
atomic rollback/strict retention and unchanged SQL/history. This is native
transport only: at that historical checkpoint the visible task form still used
the complete legacy draft path; it migrated in the later checkpoint below.
The 80-binding fixture does not certify compilation of an 80-port model.

The subsequent actual task controller adds named current-property resolution,
native origin proof, 20 asynchronous controller tests and three backend cases.
The actual sandboxed 80-binding/1000-transform form checks bounded controls,
historical description editing, exact literal integers, injected initial/pipeline/
child-return read failures, shared staging and one admission/history entry with
undo/redo/save-reload. Twenty-two projections peak 4124 bytes; font14/18/28 and
compact typed-field capture are reviewed. Existing native multihead/ragged fits
remain separately checked; an 80-binding TaskSpec editor is not an 80-port fit.
Current full-suite and native regression results are in [development.md](development.md).
This is task-form convergence, not completion of experiment forms, recovery/
terminal payload costs, universal native memory bounds, C integration or D/E.

The subsequent native experiment transport covers all seven ordered collections,
closed named nested targets, native literal edits, current value lookup, independent
selected-model/pairing pages and manifest-free preview projections. Fifteen focused
stdlib tests include exact legacy candidate/order/count equality, no-fit/SQL/RNG
purity, byte-shortened card/choice/preview cursor coverage, Unicode maximum-length
references, hidden large values, historical/missing/duplicate pins, atomic failure
and current sequence/origin/staging/admission guards. Extended real sandboxed
QWebChannel checks a 160000-coordinate preview and all five new commands, exact
literal integers, closed requests, pending-terminal fences, no pre-Commit SQL/
history change and actual owner cleanup. This is not actual experiment-form
migration or a 160000-fit claim; that historical checkpoint preceded the actual
controller below.

The actual experiment controller adds 28 asynchronous form/choice tests. They
cover literal focus/precision/invalid text, 25-operation batch splitting,
current structural paths, settled parallel reads, failed Back/return retry,
linked child adoption, exact decimal previews, owner/sequence/origin/job guards
and independent exact local pins. Native Qt exercises 80-entry axes/1000 seeds,
historical 20000-code-point question staging, local outside pins, injected
failures, exact Commit, one undo/redo/history entry and save/reload. Its 163
projections peak39205 bytes; fonts14/18/28 and the compact font18 capture are
checked. The 1024000000-coordinate preview is a declared product space, not
billion-run execution; the fixture forbids fit dispatch. Separately, the
existing actual GUI regression passes eight candidate fits and all split
methods on the new controller. Full suites pass 526 Python tests (171.889 s)
and 204 JavaScript tests (1000.658 ms). These establish actual form migration,
not lazy storage, full C integration or D/E.

The later actual recovery browser replaces full automatic state/recovery/result
transfers with a window scope/sequence header and small terminal notifications.
Independent choice pages and read-only literal tree preserve complete native
evidence for explicit export, with exact outside pins and failed-read retry.
Nine backend and fourteen new JS cases cover byte/cursor/identity/precision and
late-navigation guards. Fourteen native lifecycle cases include real queued
delivery/cancellation and chooser sequence races/no overwrite. Real sandboxed
Qt80-recovery fixture checks large hidden tensors/diagnostics, exact integers,
outside pins, failed list/tree Back retry, unchanged SQL/history and exact export;
24 projections peak33406 bytes. Full535 Python/218 JS pass. These are webview
projection/lifecycle bounds, not bounded native retention/admission/storage,
durable execution, whole-window geometry, complete C or D/E.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests
node --test tests/*.test.js
PYTHONPATH=tests:. python -m unittest \
  test_notebooks.NotebookTests.test_actual_saved_report_kernel_forbids_training_worker_and_torch_import \
  test_notebooks.NotebookTests.test_actual_rerun_kernel_is_rejected_by_same_training_fence_negative_control
```

Use the report/numerical test environment from `requirements-report.txt` and
`requirements.txt`. Actual native GUI fixtures additionally require Qt WebEngine
and a display; keep Chromium sandboxing enabled and use private workspaces,
profiles and conservative sequential CPU fits. Rendered-notebook reproduction
is in [report-figure-presentation.md](report-figure-presentation.md), with optional
test-only nbconvert requirements. No user app, file or configuration is modified.
