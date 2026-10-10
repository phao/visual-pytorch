# Development state and roadmap

The canonical source repository is https://github.com/phao/visual-pytorch,
with integration branch `main`. Local research campaigns are separate from this
public source tree. Do not maintain a second live application fork in a campaign.
Campaigns can record product commits, research notes and frozen experimental
snapshots. Concurrent product writers use separate worktrees and serialize
integration as described in AGENTS.md; campaign locks alone are insufficient.

## Initial source import

### UI refinements after the initial prerelease (local implementation)

The Inspector no longer exposes removal, diagram dimensions or Automatic size.
The diagram-block context menu has Inspect block, Delete block and Automatic size
before the general separator. Automatic sizing captures model/block IDs, rechecks
training and removes only width/height preferences. It is undoable, preserves
scientific identity and produces no history entry when already automatic.

Building-block palette buttons now have a distinct About <button> native menu
entry. The themed, readable dialog covers all 15 buttons, selected activation
and reference behavior, concrete shapes, learned parameters and supported limits.
It uses safe text and local MathJax, not execution or graph insertion. The portable
[UI handoff](UI-AGENT-HANDOFF.md) specifies both context descriptors and actions.

199 Python/62 JavaScript tests pass. The real sandboxed WebEngine harness checks
all 15 palette targets, unchanged SQLite/graph, selected Sigmoid, modal Escape,
training-time help, removed Inspector controls, automatic-size identity/no-op/
undo/redo/stale-target guards and nine theme/font/spacing configurations. Native
Qt menus confirm sections, targeted actions and disabled resize during training.
No user workspace/config/app restart and no new GitHub publication were performed.

These are UI refinements, not completion of Stage C or new model-family support.

The first curated import preserves the GitHub starter commit
`4c249d9387173938cf1fec95ba41dfe345e5affd`. Application, assets, tests, licenses and
portable specifications came from the local reviewed development checkpoint
`1f43953cfbc836df7eb9345ce03c4bbb075b7e72`. Private campaign Git histories,
research report closures, user workspaces, preferences, logs and receipts were
not imported. Source files are copied once; future implementation happens here.

The imported checkpoint passed 116 Python tests, 46 JavaScript tests, syntax/diff
checks, real sandboxed WebEngine interaction tests and a native Qt/PyTorch worker
fit. These are specific checks, not proof of full platform correctness.

## A — Trustworthy computation and evidence (bounded exit gates verified)

Implemented repairs include optional affine-summary isolation, post-update
evaluation metrics, actual cancellation counts, stable input column identities,
dataset/current-result invalidation and identity-bound trained predictor export.
`.vtw` version 3 records stable input packing and explicit dataset bindings.
Worker stdout/stderr are drained concurrently with bounded protocol handling.

The worker transport gate now has 26 focused tests with real private child
processes (plus admission mocks), including simultaneous stdin/stdout/stderr
backpressure, protocol corruption, bounded diagnostics, callback/startup failure,
deadline cleanup and cooperative/forced cancellation. These reproduced gaps in
the inherited transport and pass after the repairs. See [worker contract](worker-protocol.md).

Checkpoint admission now validates version, preprocessing, stable parameter IDs,
exact weight keys/shapes and finite float32 values without importing Torch.
Independent predictions cover bias on/off, MLP, tensor inputs, tanh/ReLU branches
joined into sigmoid, fresh references, Splitter/Replicator outputs and SQLite/CLI
roundtrips. Native and exported Linear constructors explicitly use CPU float32.
The GUI checks dataset column bindings before launching, shows recorded/current
columns, offers confirmed undoable rebind and labels legacy/stale/invalid results.
Late completions are validated against the worker snapshot; incompatible drafts
keep the run in history without presenting it as current.

The bounded A exit gates and their evidence are mapped in
[Stage A verification](stage-a-verification.md). Full platform correctness,
training-resume checkpoints and the C–E features below are not claimed. The
subsequent bounded CNN slice is described below; dataset/experiment contracts
are the next implementation milestone, not additional cosmetic work.

## B — Coherent tensor/operator composition (first CNN slice verified)

Version-4 workspaces add version-1 Conv2d, spatial pooling, Flatten and Permute,
explicit CHW/axis semantics, recorded structured data adapters and convolution
workspace estimates. The same UI/CLI/worker can construct, validate, train,
save/reload and export a small CNN. Native and exported constructors share pure
normalized specs; independent scalar values/gradients prevent shared mistakes.
The old v3 checkpoint meaning survives opening and explicit-save migration.

The bounded first slice is mapped in [CNN verification](cnn-workflow.md).
195 Python/57 JS tests, the real sandboxed editor-built CNN fit and the inherited
WebEngine/UI contracts passed during development. This is not completion of the
rich dataset/task, symbolic/recurrent or detached-execution work below.

## C — Datasets and experiments

Separate DatasetSpec/TaskSpec/ExperimentSpec/RunRecord/ReportRecipe. Strengthen
explicit dataset bindings, immutable run evidence, group/time splits,
training-only fitted transforms, minibatching and an experiment/history interface.
Notebook reports must distinguish reading existing results from explicit
retraining. Avoid silent changes to column meaning when ports change.

### Typed dataset GUI checkpoint — local task branch only

The existing floating Experiment/history tab now creates and edits typed inline
datasets without importing hand-written DatasetSpec JSON. All five dtypes,
scalar/tensor/first-axis ragged shapes, nullable cells, fixed ordered vocabularies,
axis names and provenance/usage/unit/modality/coordinate metadata remain explicit.
Stable field/observation IDs and unconverted values prevent silent rebinding.
The six-field/twenty-observation grid bounds rendered controls, not full transport.

The common dataset/task/experiment draft API records workspace UUID, exact base
and expected browsing head, validates through the existing grammar/dependencies
without fitting, and rejects stale/invalid admission before writes. Historical
editing is explicit; unchanged head admission is a no-op. Undo retains evidence.
Only the dataset form is wired; task/experiment forms, richer metric/predictor/
multi-run reporting, GUI notebook execution, bounded transport and full C gates
are unfinished. D/E remain pending. This checkpoint is not merged/published.

Fourteen new pure Python tests and eight JS tests cover types/identities, nullable
ragged/category data, prospective hashes, exact historical/archived bases,
collisions, malformed/stale/foreign-workspace drafts, no-op, immutable history,
Save/reload, finite/duplicate-key parsing, prototype-like field IDs and two-axis
paging. The real sandboxed WebEngine harness starts with an empty scientific
registry, authors rich data, checks stored bytes, invalid text retention, atomic
shape failures, undo/redo/no-op, stale heads/workspace switches and pending-job
guards. Dark/GitHub/Math at fonts 14/18/28 and spacings 1/4/10 preserve readable
themed form controls and bounded grid width; host Torch remains absent. The form
capture was inspected. Whole-window header overflow at larger fonts still needs
a separate convergence audit; these checks do not prove all application layout.

Current checkpoint validation: all 367 Python tests pass (130.105 s), all 70 JS
tests pass, and the new actual WebEngine form test passes with no console errors
or host Torch. The existing actual fit/retry/history/report/notebook-generation/
forced-admission-recovery flow and inherited editor smoke also pass. Eleven
native Qt terminal/cancel/recovery tests pass. No new remote publication or user
app/workspace/configuration change was performed.

### First executable checkpoint — not Stage C completion

On `task/stage-c-experiments` (base main
`283dd9028a349a802e97743598221f7479ad1480`), independent version-1 typed inline
DatasetSpec/TaskSpec, fixed vocabularies, nullable/multimodal/sequence fields,
stable field→port bindings, saved row/group/temporal/external split IDs and
train-only fitted transform chains are executable. Explicit ragged padding has
matching length/mask inputs and rejects silent truncation. Preparation admits
expanded shapes against the existing user RAM policy before allocation.

Read-only CLI `preview-dataset`/`prepare-task` works in fresh Python without
Torch/Qt/NumPy and leaves workspace/config bytes untouched. Numerical-worker
adapters tensorize stable port batches and execute named graph forward. MSE/MAE/
indexed cross-entropy have independently tested scalar values/gradients, additive
weighted numerators/denominators, explicit unavailable empty populations and
checkpoint/population-bound accumulation. Unequal batches are not averaged as
equally weighted means. A real 160-update minibatch fit uses these adapters;
this is a test of the components, not a completed task trainer or experiment UI.

Checkpoint verification: 234 Python tests (39 added), including 24 dense masked
configurations under three batch partitions, and 57 JavaScript tests pass.
Compile/diff checks pass. No GUI source changed; no fresh native Qt check is
claimed for this checkpoint. The previous A/B native checks remain prior evidence.
See [the executable dataset/task contract](dataset-task-contract.md) for exact
schema, CLI, semantics, memory estimate and boundaries.

At that first checkpoint the container was v4; the following checkpoint advances
persistence. C's full exit gates and D/E are not achieved.

### Independent revision checkpoint — still branch-only

Container v5 adds independent immutable dataset/task/model documents and mutable
browsing heads. Frozen models retain only the complete reachable normalized
graph closure, preserving the Stage A scientific hash without node presentation
or architecture-owned data/results. Deleting an authoring model cannot delete
its frozen revision. Task syntax validates independently; selected field/port
compatibility still gates preparation. Other experiment/run/report kinds are
not accepted as unvalidated placeholder JSON.

Working-copy migration accepts v1–4, leaves source bytes untouched until Save,
retains legacy checkpoint identity and packing, and validates schema/guards and
all content hashes. SQL guards block UPDATE, DELETE, REPLACE and duplicate-key
UPSERT writes even with recursive triggers disabled. Head archival/reselection
is recoverable. Undo/redo restores mutable heads/drafts while unioning retained
revision bytes, not silently removing newly recorded scientific objects.

CLI writes participate in the existing ordered atomic revision-checked save.
`spec-list`/`spec-show` browse retained objects; `prepare-stored` requires EXACT
model/dataset/task ID+revision pairs and explicit split/manifest. Changing a
browsing head cannot substitute a different dataset into that preparation.
Fresh CLI executions import no Torch/Qt/NumPy and preserve read-only file bytes.
Specs: [revision storage](revision-store.md), [current format](vtw-format.md),
[CLI](vpw-cli.md). The exact SQL code blocks are executed in a regression test.

Verification: 255 Python/57 JS tests, compile/diff checks, native Qt history and
binding checks, and the real sandboxed WebEngine GUI suite pass.
Native disk-update polling also passes: clean workspaces reload agent writes,
dirty workspaces retain unsaved edits, and display preferences persist.
New tests include
all four actual old SQL container versions, old two-input packing, current legacy
checkpoint identity, malicious/missing guards and JSON/row/hash mismatches,
frozen closure retention, undo/redo and fresh exact-revision CLI replay. User
workspaces/configuration/live app were not changed; no new GUI menus claimed.

### Task-aware training checkpoint — branch-only, not Stage C completion

The actual worker now dispatches frozen task bundles with exact model/data/task
revisions and split IDs. Closed CPU/Adam protocols declare independent init/shuffle
seeds, minibatches, budgets, evaluation schedule and bounded curve retention.
Named MSE/MAE/classification heads use actual masks/weights with additive population
denominators; wholly unsupervised batches skip updates, and entirely unsupervised
epochs terminate explicitly. Test is evaluated only after completed execution,
not exposed in intermediate train/validation curves. Counter/timing semantics,
last-finite failure/cancel checkpoints and combined RAM admission are explicit.

Nonidentity fitted preprocessing now belongs to the separate task-predictor
checkpoint. Pure admission checks frozen identities, expected train-only state,
stable parameter IDs, exact tensor shapes/keys and finite weights. New field-valued
records use captured imputation/standardization, units, one-hot, ragged padding and
derived masks without refitting. Standalone generated inference requires Torch
and stdlib only, carries task/target/vocabulary metadata, excludes raw observations,
and returns logits or declared target-space tensors keyed by stable Output IDs.

CLI `train-stored` runs the real worker without Torch/Qt in the host and leaves
workspace bytes unchanged. `export-task-trained` emits validated fitted source
without training. A native `TaskTrainingJob` adapter has passed real Qt
thread/signal tests; no selection menus or run-history GUI are claimed yet.
See [task training contract](task-training-contract.md) for exact versions/fields,
budgets, metrics, artifacts, export semantics, limits and primary design sources.

Verification includes an independent scalar two-head gradient/Adam oracle, actual
classification and spatial CNN classification, finite/error/cancel recovery,
observed RAM guards, RNG isolation, exact budgets, bounded curves, and real worker
and CLI transport. Standalone source ran from an unrelated directory with no app
imports and bitwise native/export predictions. Branching two-input Tanh/ReLU/Concat,
ragged masks and one-hot meanings are covered. Final checks: 289 Python/57 JS tests;
real Qt task worker classification (240 updates, held-out accuracy 1.0 and CE
0.02411829524); old native linear/MLP workers (~1.19e-12/~7.74e-5 held-out MSE);
native Qt history/binding/disk sync; sandboxed real WebEngine layout, compact icon
drag, undo/rewire/delete/typing/binding/theme checks with no console errors;
compile/diff checks. User workspaces/config/live app remain untouched; no push.
These are bounded fixture checks, not generalization or full platform guarantees.

At that checkpoint, next work was independent experiment/run/report persistence, durable
admission/initialization/process-failure attempts, reproducible model × dataset
selection expansion with invalid combinations visible, GUI/run history, and
result-report versus explicit-rerun notebooks with real kernel/portable-bundle
gates. A task-training-result artifact is NOT a stored RunRecord. Full C gates
and D/E remain open; this partial branch must not be merged as completed Stage C.

### Concrete experiment/run/report checkpoint — branch-only

Container v6 adds closed ExperimentSpec/RunRecord/ReportRecipe version-1
semantics to the immutable registry without changing SQL tables. Exact
dependencies are checked on admission/load/save, even after archival/undo.
Selections cross explicit model/data-task/split/initialization/configuration
axes or declare pairings/exclusions. Bounded mixed-radix preview keeps invalid
and excluded combinations visible without fitting/training or global RNG use.

Explicit candidate execution captures actual split IDs and resolved protocol
before RAM admission/worker launch. Stored attempts include compatibility,
admission, missing executable/transport and execution failures without invented
metrics. Completed evidence is admitted against its frozen closure: counters,
budget boundaries, fitted checkpoint, population IDs, weighted metric arithmetic,
curve timing/retention, final-only test, environment/sampler/RAM components.
This is internal validation, not authenticated proof of execution or independent
recomputation of neural predictions. No queued/running/detached receipt is claimed.

Run revisions preserve all earlier attempts and frozen resolution. Explicit retry
uses the longest retained history, not an older head, and performs fresh training,
not optimizer resume. Historical reads/retries do not regenerate the captured
manifest. Report recipes select exact run revisions or explicitly request rerun;
read-only materialization never launches and keeps unsuccessful attempts visible.
Two revisions/retries of the same logical run cannot be counted as distinct
descriptive run-level units. Notebook generation is still absent.

CLI preview-experiment/run-candidate/report-data remains Torch/Qt-free in the
host. Save conflicts return recoverable evidence with persisted:false while
preserving concurrent user bytes; explicit put-spec recovery does not retrain.
The executable [example](../examples/experiment_workspace.py) saves two models ×
two datasets × two initializations by default, only previews, refuses overwrites,
and optionally runs one bounded candidate with a saved result recipe.

Verification: 322 Python/57 JS tests, including 33 new experiment/run/report/example
tests. All eight combinations of a separate 2×2×2 fixture execute and explicitly
retry with identical checkpoints/metrics in the same local environment.
Tests cover immutable dependency/history corruption, captured-manifest reads
with split generation disabled, fractional masked weights and classification,
real RAM/missing-executable failures, concurrent Save evidence recovery, report
read isolation and fresh stdlib-only readers. Native Qt task classification still
completes 240 updates with accuracy 1.0 and CE 0.02411829524; native binding/history/
disk-sync and the real sandboxed WebEngine gesture/layout/bias suite pass with
no console errors. Compile/diff checks pass. Exact format/revision/data/task/CLI
contracts are synchronized; [experiment contract](experiment-objects.md) defines
every new field and evidence limitation. No user state, private source snapshots
or remote publication changed.

Next: integrate selection and independent run history into the accepted floating
Experiment UI, generate both result-report and explicit-rerun notebooks, execute
them through an actual Jupyter kernel and verify portable artifact closure.
Retain failed-cell evidence and distinguish requested from executed notebooks.
These are required C exit gates, not optional polish. Then D/E remain required.
This unfinished branch stays unmerged into main.

### Portable notebook and workspace-layout checkpoint — branch-only

Both notebook modes now generate self-contained, integrity-checked bundles from
exact report revisions. Generation never executes; result-report reads evidence
without fits, while executing rerun explicitly trains the captured selection.
Real fresh ipykernels retain requested/executed documents, errors, timeouts and
receipts. Portability and fitted predictor tests run outside the producer checkout.
Captured manifests survive replay; retry substitution is refused before launch.
Metric/curve artifacts retain checkpoint/population identity and unavailable
values. See [notebook contract](notebook-reports.md) for schemas and limitations.

342 Python/57 JS tests passed, including 16 notebook tests and six workspace-layout
tests; native Qt layout/history/config-isolation checks and real sandboxed
WebEngine gestures/save/reload/undo controls passed. SVG coordinate tests are
numerical evidence; final rendered-figure review remains open before publication.
These additions are a useful partial C checkpoint, not all C exit gates.

Collapsed/expanded layout profiles now belong to `.vtw` metadata and participate
in undo/redo, atomic CLI edits and cross-tab state, not global config. Floating
panel geometry/visibility remains session-only and is still a portability item.
Selection/history/notebook GUI integration and full C convergence are next;
then D/E remain required. This task branch stays separate from the initial public
preview. Do not advertise its unfinished features as released.

### Current Stage C GUI checkpoint — still branch-only

The floating dock now browses exact scientific revisions, compatibility pages
and independent run/attempt history. Explicit fits use a Qt thread with a separate
captured Workspace and the validated task worker. Saved splits/retry meaning stay
pinned. Reading recipes never trains; notebook generation is an explicit separate
action. See [experiment GUI](experiment-gui.md) for bridge/thread/dependency contracts.

349 Python/57 JS passed. The real sandboxed GUI fit/retry/history/report/generation/
save/reload test passed in a private Qt+nbformat environment, with no host Torch.
Inherited WebEngine gestures/themes and native layout checks pass. Seven pure
boundary tests cover exact closure/head/retry/manifest and failed/excluded data.
Richer authoring/metric/multi-run report controls, GUI notebook execution,
rendered report figures and full C convergence
remain open; this is not C completion/publication. D/E remain required.

The subsequent lifecycle checkpoint keeps the shell busy through queued terminal
admission, not only QThread execution. Explicit GUI receivers discard stale
progress and deliver a terminal once. Hash/reference mismatches cannot insert a
record. Recovery belongs to the window and can be explicitly exported with its
exact closure; closing cannot silently discard unexported evidence. It is not an
automatic durable job receipt. Qt’s queued-connection docs motivated this split;
see [experiment GUI](experiment-gui.md) for exact actions and privacy/exit limits.

353 Python/57 JS plus 11 native Qt lifecycle tests pass at this checkpoint.
Actual WebEngine worker fit/retry/report/notebook generation/save/reload and
forced-admission failure/recovery export pass without host Torch or console
errors. Cancellation preserves partial counters/checkpoint and excludes test
metrics. Missing/delayed/duplicate/superseded terminal and replacement/close
guards have explicit tests. Metadata/raw presentation bounds, richer authoring,
report views and all remaining C gates stay open before integration/publication.

After merging the reviewed Inspector/palette followups, 353 Python/62 JS pass
again. The current sandboxed WebEngine editor and experiment/recovery checks,
all 11 native lifecycle tests, and native history/disk-sync checks also pass.
The shared C task branch is intentionally not integrated as a completed stage.

## D — Symbolic families and mixed CNN/RNN composition

Develop concrete-first architecture families with optional scoped symbols and
specialization, explicit repetition versus tied weights, recurrent state,
lengths/masks and independently tested mixed CNN/RNN compositions. Teaching
links specification, mathematics, execution and measured evidence. Symbolic
families and recurrent operators are not implemented yet. Concrete experiment
notebook generation/execution belongs to the partial C work described above;
it is not evidence of symbolic or recurrent family support.

## E — Heavyweight execution

Separate editing from detached jobs, durable receipts, checkpoint/attempt
lineage, distributed reduction semantics, resource-aware execution, cancellation,
progress and failure handling. External machines, paid services and remote jobs
need appropriate operational authorization; the roadmap alone does not grant it.

## UI contracts

The Layout/compact-summary change passed 119 Python tests and 54 JavaScript tests,
native menu checks and real sandboxed WebEngine gestures. A real trained
checkpoint retains bitwise-identical inference and identical trained export
across presentation-only bulk folding/position edits. The new layered layout
tests include 100 uneven 64-node DAG configurations, cyclic/disconnected drafts,
ports, geometry bounds and deterministic repeat layout. Those UI checks remain
independent of the subsequently verified Stage A backend gates.

Keep icon folding visual-only. A compact icon is a drag surface, not solely an
expansion button; single click selects, double-click restores, and Enter/Space
toggles folding. Begin pointer capture after actual movement for compact blocks,
so stationary double-click events remain targeted at the icon. Ports, snapping,
preferred expanded dimensions and undo/redo remain functional.

Layout adds independent collapsed/expanded profiles with one implemented Layered
style, LR/TB direction, model-unit gaps and optional rearrange/fit. Controls apply
to the next action; Save layout preferences updates the workspace draft, and
Save workspace persists the profiles in `.vtw` metadata. Global configuration
does not own these profiles; legacy global layout settings are ignored.
Each bulk action is a single undoable visual edit. Small summaries remain visible
on compact blocks. The algorithm is deterministic and size-aware, handles draft
cycles without changing their graph, and reduces crossings heuristically rather
than promising optimal routing. See [layout contract](layout.md).

Preserve persistent selection on empty canvas and contextual Escape behavior,
including typing/dialog safeguards. Group selection gestures remain deferred.

## Initial public preview — 0.1.0-alpha.1

### Theme correction after the preview

Concat had a higher-specificity hard-coded dark fill than the generic token rule.
It now explicitly uses `node-input`/`blue` tokens, preserving the selected border.
The real WebEngine cascade test checks all 15 block types across 16 concrete
palettes (240 pairs), title/detail colors and selected Concat fill/border. System
resolves through the existing light/dark palettes. This is rendered CSS evidence,
not a claim of complete accessibility certification. No numerical graph changes.

The public preview includes the verified A/B slices above and workspace-owned
layout profiles. It does not include the incomplete Stage C development branch.
Six focused workspace-layout tests cover independent files, validation, defaults,
save/reload, presentation-only identity, undo/redo and atomic CLI edits. Native Qt
checks also verify busy/empty guards and that global preferences remain untouched.
The real WebEngine editor test checks that undo/redo restores the layout controls.

Release validation: 199 Python tests and 57 JavaScript tests passed, along with
native Qt workspace-layout and real sandboxed WebEngine UI/CNN checks. The CNN
harness builds and edits its graph through real controls, trains an actual worker,
checks a current checkpoint, save/reload and trained export, and captures the
experiment, learned tensors and fresh-weight source views after painted frames.
These are bounded local checks, not a clean-install/platform certification.

Floating-panel geometry and visibility remain session-only. Moving remaining
workspace-specific layout state into `.vtw` is an open portability item. Rich
experiments/notebook reporting, symbolic/recurrent composition and detached jobs
remain roadmap work; none are implied by this release's screenshots.
