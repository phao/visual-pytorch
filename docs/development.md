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

## A — Trustworthy computation and evidence (in progress)

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

Remaining gates include broader checkpoint roundtrips and independent predictions,
binding/migration/CLI coverage, GUI
binding admission and honest evidence-status presentation. Do not mark A complete
merely because its five initial regression counterexamples pass.

## B — Coherent tensor/operator composition

Define and implement a bounded CNN workflow with explicit shape/layout, ports,
padding and reference semantics. Check computation and gradients independently.
Do not build a palette of operators without a usable end-to-end model.

## C — Datasets and experiments

Separate DatasetSpec/TaskSpec/ExperimentSpec/RunRecord/ReportRecipe. Strengthen
explicit dataset bindings, immutable run evidence, group/time splits,
training-only fitted transforms, minibatching and an experiment/history interface.
Notebook reports must distinguish reading existing results from explicit
retraining. Avoid silent changes to column meaning when ports change.

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
ports, geometry bounds and deterministic repeat layout. This does not complete
the outstanding Stage A backend gates listed above.

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
