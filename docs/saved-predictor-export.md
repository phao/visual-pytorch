# Exact saved-attempt predictor export — branch-only Stage C

View/Export checkpoint predictor use the **displayed retained attempt** of an
exact RunRecord revision. An unread edit to the number control does not change
the target: first successfully Read this attempt. Selection/metric refresh clears
older source. This complements the legacy fresh drawing export and direct
task-result exporter, without changing `.vtw` tables/grammars. It is not in the
initial public prerelease.

## Source selection and contract

`app.run_export.predictor_source(workspace, ref, attempt=None,
memory_limit_percent=10.)` requires an exact run kind/id/revision. Omission chooses
the latest retained attempt of THAT revision, never a newer head. It reconstructs
the frozen bundle with original dependencies, ordered coordinates, resolved
protocol and captured manifest, without substituting current drawings/heads or
regenerating splits. The existing saved-run/checkpoint validators check fitted
preprocessing identity, stable parameters, keys/shapes and finite float32 weights.

Failed latest retries without results cannot borrow earlier successes. Explicit
earlier attempts remain exportable. Completed/cancelled/failed/unavailable attempts
may retain valid last-finite inference checkpoints. A zero-update initial
checkpoint is visibly labelled status/zero updates, not successful learning.
Missing result/checkpoint disables controls and fails natively/through CLI.
Malformed references, null hashes and invalid/boolean attempts fail atomically.

The artifact has exactly version 1, kind `visual-pytorch-saved-predictor-source`,
scope `generate-source-only; no-training-or-inference`, provenance, code and
code_sha256. Provenance has version 1, exact reference, attempt/status, checkpoint
hash, completed_updates, prediction_semantics and optimizer_resume false. It is
also embedded as Python `saved_run_provenance`. SHA-256 covers complete UTF-8
source including provenance; it detects mutation, not authorship/honest training.

Generation reconstructs/checks saved fitted preprocessing in pure Python under
the existing workspace constrained/RAM preparation policy. It never imports
Torch, runs generated code, performs inference/training, consumes global RNG,
changes SQL/heads/dirty/history or writes `.vtw`. Full dependency/inline validation
cost remains unbounded; this is not lazy validation or a runtime allocator limit.

`write_new_predictor(source, output)` exclusively creates a new UTF-8/LF file.
It first verifies that the actual source bytes still match code_sha256; mismatch
fails before file creation. This is integrity checking, not source authentication.
Existing files/directories and final symlinks, including dangling ones, are refused
without overwriting/following them. Parents must exist. Its receipt contains
version 1, absolute path, provenance, code_sha256 and status
`predictor-source-exported; not-executed`. Failure may leave an inspectable partial
new file: no crash-safe/atomic receipt or automatic retry is promised. The helper
expects a generated artifact, not arbitrary caller-supplied source dictionaries.

## Standalone inference semantics and privacy

The module needs Torch plus stdlib, not the app, Qt, Jupyter, workspace or producer
checkout. Import initializes the graph, immediately strictly restores saved CPU
float32 weights and sets eval mode; it never trains. `predict_records` takes field
maps, applies saved imputation/standardization/affine/one-hot/padding/derived
length-mask states without refitting and returns tensors keyed by stable Output
IDs. Classification stays logits; regression stays in declared transformed target
space, not automatically inverse-transformed physical units. Missing/invalid
inputs, overlong sequences and nonfinite arithmetic fail. No optimizer resume.
Executing outside the editor has no editor RAM admission/execution sandbox.

Source contains weights, fitted statistics and task/input/target schema/vocabulary
metadata, but no DatasetSpec.records payload. Metadata may itself be sensitive;
review before sharing. Hashes provide neither privacy nor safe execution of
untrusted Python. Source previews render literal text, not HTML/untrusted math.

## Qt and CLI

Both native actions require workspace_id, exact run reference and optional attempt.
scienceRunPredictor is read-only, including during active/pending jobs.
scienceExportRunPredictor is an external file action, centrally blocked before
parsing while jobs/terminal admission are pending. Optional output bypasses the
chooser; otherwise its title is Export retained checkpoint predictor and default
name trained-predictor.py, not an unsanitized run ID. Validated exact source is
captured before the nested modal loop, then the actual workspace object, UUID and
busy state are rechecked before file creation. Cancel returns null. Advancing
mutable heads cannot replace that captured source. Neither action records undo
history or needs workspace Save; the exported file is separate.

Preview replies check workspace UUID, current exact selection/page and request
generation. Changed-workspace replies cannot populate the replacement UI. This
does not prove general same-UUID/ABA or crash/detached protection. Export receipts
do not change selection. Source panes inherit themes/base fonts/menu spacing and
scroll internally without shrinking type to fit long source lines.

CLI `export-run-trained RUN_ID RUN_SHA256 [--attempt N]
[--memory-limit-percent P]` returns the identical source artifact under value,
creating no source file and preserving workspace bytes. export-task-trained still
accepts a direct task-training-result value object, not a stored independent run.

## Evidence and limits

tests/test_run_export.py checks histories/no fallback, actual cancellation/initial
checkpoints, immutable heads, hashes, SQL/RNG isolation, no worker/new split/
inference during generation, RAM policy, stdlib-only CLI equality/error preservation,
exclusive/symlink creation and bitwise standalone/restored weighted multi-input
and ragged length/mask predictions. JS pins the displayed exact attempt. Real
sandboxed Qt checks literal preview, chooser/export, standalone numerical inference
(separately from generation), unread input, cancel/overwrite refusal and busy/
post-modal/late workspace guards, with no host Torch. This is not GUI notebook
execution, scalable full transport/report convergence, symbolic/recurrent or
detached execution, or Stage C completion.
