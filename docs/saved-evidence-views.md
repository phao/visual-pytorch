# Saved-run presentation contract — branch-only Stage C

This is a read-only projection of existing version-1 RunRecord/ReportRecipe
objects in `.vtw` v6, not another evidence store or training protocol. Python,
Qt bridge and CLI share `app/run_views.py`. The initial public prerelease does
not include this interface.

## Exact references, attempts and pages

References have exactly kind/id/revision, with an exact lowercase SHA-256 hash,
never null or an implicit browsing head. Missing/corrupt evidence or dependencies
fail visibly; no replacement model, task, data, split or result is inferred.

`run_view(workspace, reference, attempt=None, offset=0, limit=25)` reads a
one-based retained attempt. Omission selects the latest attempt of THAT exact
revision, not a newer revision of its logical run. A failed latest attempt with
no result shows no numerical counts/checkpoints/metrics; it never borrows them
from earlier successes. Reading another retained attempt is explicit.

Offsets are exact nonnegative integers through `2**53 - 1`; metric limits are
1..25. Booleans are rejected. Attempts must lie in the selected revision's
retained range. Offsets beyond the total return an empty page. Invalid requests
fail without SQL changes.

## Run response, version 1

The top level has version 1, scope `read-saved-evidence-only`, exact reference,
name, attempt_count, attempt, selection, objects, split, protocol, execution,
populations and metrics:

- attempt: saved number, start/finish times, stage, status and error.
- selection: original exact experiment and ordered coordinates.
- objects: original model/dataset/task references and names, trainable parameter
  count and dataset observation count. An excluded candidate keeps its declared
  context; that does not mean training was admitted.
- split: null before resolution; otherwise captured manifest identity/method,
  ordered role-ID hashes and counts, excluded count/hash and manifest metadata.
  Observation IDs are not returned or regenerated.
- protocol: captured resolved TrainingProtocol, or null before resolution.
- execution: null without a numerical result; otherwise saved status/stop reason/
  failure/evaluation error, checkpoint identity, actual counters, elapsed seconds,
  history-retention facts and environment. Fitted preprocessing is represented by
  its hash. Recorded RAM admission retains estimated_bytes, actual_batch_size,
  available_bytes and limit_percent: this is NOT measured peak memory. Weights,
  fitted arrays and checkpoint payload are omitted.
- populations: train/validation/test in that order, with objective/status/rule.
  Missing evaluation has null objective/rule and status `not-retained`, not zero.
  Cancellation preserves actual partial counters without inventing test metrics.
- metrics: offset/limit/total/rows/next_offset (null at the final page). Rows follow
  population then original TaskSpec head order. Each contains population, head
  definition (id/port/field/loss/objective weight/transforms/mask/sample weight/
  source unit) and the exact scalar saved additive metric ledger. Observation
  IDs are replaced by their ordered hash; other metric values, denominators,
  numerators, counts, status and available accuracy facts are unchanged.

Responses are deep-owned. Reading never launches a worker/notebook, samples a
split, fits new preprocessing, consumes global RNG, changes heads, dirty/history
state or SQL. Normal registry validation reconstructs/checks saved fitted
preprocessing in pure Python; this is not a training/prediction request. These
queries import no Torch, NumPy, Qt or nbformat.

## Report responses and composition

`report_view(workspace, reference, offset=0, limit=5)` accepts an exact report;
limit is 1..10. It returns version/scope/reference/name/question/mode/analysis/
sections/execution_behavior/experiment and runs. The runs page has offset/limit/
total/rows/next_offset in recipe order. Each row projects the latest attempt of
its exact selected run revision and its first 25 metric rows. Open the exact run
for other attempts/additional metrics. New heads never update an existing recipe.

Result reports label behavior `read-saved-evidence-only`; rerun recipes label
`explicit-execution-required` and return zero saved rows. Neither trains on read.
Excluded, failed, cancelled and unavailable trials stay visible.

The collapsed GUI composer holds a transient ordered exact-run basket. Adding
the identical reference is idempotent. Another revision of the same logical run
is refused until explicitly removed; retries/revisions are not extra run units.
Create saves a normal descriptive/run-unit ReportRecipe, never training. Save
workspace is needed for disk persistence. Undo restores heads but retains bytes.
No pooled means, ranking, confidence intervals, replicate independence or
statistical significance are asserted.

The new composer supplies workspace_id to native admission. A foreign UUID fails
before dependency resolution/insertion; the legacy one-run action may omit it.
Late read/admission replies cannot select/populate another workspace UUID.
Already completed origin admission remains in the origin; this is NOT detached
or crash-safe persistence, same-UUID replacement or ABA protection. Workspace
change clears the basket. Active/pending-terminal work blocks native mutations;
reading remains available.

## Presentation and boundaries

Cards inherit themes/fonts/menu spacing. Tables keep numbers/statuses whole and
scroll horizontally inside the panel. Summary values use six significant digits;
expandable ledgers expose unrounded saved values. Unavailable differs from zero
and accuracy Not applicable. Regression metrics use transformed target space;
source units are metadata, not physical loss units. Head objective multipliers
differ from sample weights/masks. Population objective rules remain the trainer's,
without renormalizing missing heads. User text uses textContent, not HTML or
untrusted math. Raw stored JSON remains explicitly expandable.

Pages bound projected/rendered rows, NOT full dependency validation, inline data
cost or metadata bytes. Existing scienceShow/scienceReport materialize complete
documents. This is not lazy storage, predictor export UI, GUI notebook execution,
full scalability or Stage C completion. Refresh can reset the presentation page
while retaining the exact dropdown identity.

## Executable evidence

tests/test_run_views.py uses real weighted CE/MAE fits, partial cancellation,
all-masked supervision and nine output heads. It checks saved scalar equality,
null versus zero, retained histories, exact hashes, order/pages, SQL/RNG isolation,
stdlib-only CLI equality and unchanged file bytes. tests/run-evidence.test.js
checks explicit selection and literal formatting. The sandboxed
tests/qt_saved_evidence_smoke.py exercises seven-run GUI composition/drilldown,
unsafe-looking text, pages/fonts/themes, undo/reload and busy/late UUID guards
with numerical execution disabled during browsing/composition. The editor's
initial Fit save is explicitly admitted and its frozen model/registry verified
unchanged BEFORE taking the complete read-only SQL baseline.
