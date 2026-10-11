# Agentic changelog

Earlier product decisions and validation history are retained in
`docs/development.md` and commit messages; this file adds timestamped change
summaries without replacing them.

## 2026-10-11 03:25 -03:00 — Selectable-axis Softmax across the product

Added Softmax to the existing Activation block, with explicit sample-axis
validation and native/exported PyTorch mapping, because the user requested
N-way probability outputs. Inspector edits, compact labels, history, `.vtw`,
CLI and frozen notebook closures share the contract. Existing activation
documents and batch independence are preserved. New bundles capture the added
runtime module; an immutable pre-Softmax closure verifies real legacy notebook
execution without rewriting inputs. Full588 Python/245 JavaScript tests and
actual sandboxed Qt Softmax plus fit/retry/report/generation/recovery-export
checks pass. The first full run exposed a missing notebook dependency; this was
fixed, not suppressed. Local C development only, not a published release.

## 2026-10-11 03:03 -03:00 — Exact saved-attempt curves and CLI parity

Implemented a read-only head/point projection and actual SVG/table viewers in
run evidence and saved-report cards, because notebook figures alone did not make
retained training diagnostics available in the editor. Native/CLI identities,
failed retries, missing-value gaps, fixed retained-history axes, independently
acknowledged navigation and large-font scaling remain explicit. Twelve new
backend cases, fourteen new JavaScript cases, full571-Python/243-JavaScript
suites and actual Qt curve/report/predictor checks pass. No trainer, storage
grammar or schema change; complete native validation/metadata costs remain.
New disposable test state uses `.scratch/`; retained evidence stays in the private
campaign. C remains local/incomplete, not integrated or published.

## 2026-10-11 02:34 -03:00 — Implemented and validated request-local preview reuse

Removed redundant successful dependency, model/task and split preparation during
one immutable-source preview read, because candidate replicas repeated complete
native work. Exact results, errors, ordering, ownership and public direct-call
validation are preserved. Eleven new tests plus the559-Python/229-JavaScript
regressions and real Qt candidate/eight-fit/run/report/export flows pass. Matched
identical-wire preview measurement gives16.25× on the recorded fixture, not a
universal performance claim. Specifications and detailed limits are in
`docs/preview-resolution.md` and `docs/development.md`. C remains local/incomplete;
D/E and public publication are not claimed.

## 2026-10-11 02:35 -03:00 — Canonical development checkout and targeted retirement

The user confirmed sole product development and found hidden staging inconvenient.
After a clean verified source checkpoint, switched the canonical visible checkout
to `task/stage-c-experiments` and removed its redundant clean staging worktree
through Git. Source/history and test evidence remain; other worktrees, public
refs, user files/config and running app were not changed. The existing desktop
launcher targets this visible checkout without re-registration. This changes the
local launchable version, not Stage C integration or public release status.
