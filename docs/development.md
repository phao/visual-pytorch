# Development state and roadmap

The canonical source repository is https://github.com/phao/visual-pytorch,
with integration branch `main`. Local research campaigns are separate from this
public source tree. Do not maintain a second live application fork in a campaign.
Campaigns can record product commits, research notes and frozen experimental
snapshots. Concurrent product writers use separate worktrees and serialize
integration as described in AGENTS.md; campaign locks alone are insufficient.

## Initial source import

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
Specs: [revision storage](revision-store.md), [v5 format](vtw-format.md),
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

Next: task-aware minibatch trainer/worker and preprocessing checkpoint/export;
independent experiment/run/report validators and frozen execution protocol;
model × dataset selections with invalid combinations visible; GUI/run history;
result-report versus explicit-rerun notebooks with real kernel/portable-bundle
gates. Persisting three object types does NOT satisfy those remaining gates.

## D — Symbolic families and mixed CNN/RNN composition

Develop concrete-first architecture families with optional scoped symbols and
specialization, explicit repetition versus tied weights, recurrent state,
lengths/masks and independently tested mixed CNN/RNN compositions. Teaching
links specification, mathematics, execution and measured evidence. Symbolic
families, recurrent operators and notebook generation are not implemented yet.

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
to the next action; an explicit save persists profiles in user configuration.
Each bulk action is a single undoable visual edit. Small summaries remain visible
on compact blocks. The algorithm is deterministic and size-aware, handles draft
cycles without changing their graph, and reduces crossings heuristically rather
than promising optimal routing. See [layout contract](layout.md).

Preserve persistent selection on empty canvas and contextual Escape behavior,
including typing/dialog safeguards. Group selection gestures remain deferred.
