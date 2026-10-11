# Saved-attempt training curves — local Stage C

The Qt run view and each saved-report run card offer **Saved training curves**.
They read immutable evidence; opening, paging or refreshing never trains, fits
preprocessing, regenerates a split, evaluates a predictor or starts a notebook.
This interface is not in the initial public prerelease.

## Source and exact identity

`app/run_curves.py::curve_view` supplies both the native `scienceRunCurve` action
and the read-only CLI `run-curve`. The reference has exactly `kind`, `id` and
`revision`, with kind `run` and an exact lowercase SHA-256 revision. No browsing
head is followed. An omitted attempt selects the latest retained attempt of that
exact revision. An explicit attempt is one-based and must exist there. A failed
latest retry does not borrow an earlier successful curve or checkpoint.

The selected task, dataset, head definition and target unit come from that run's
exact experiment pairing. An omitted head selects the first **declared** head;
head order is scientific order, never alphabetical. A selected head may remain
outside the current head-choice page. Missing/corrupt references fail visibly.

## Shared projection, version 1

Arguments are `reference`, optional `attempt` and `head`, `offset=0`, `limit=100`,
`head_offset=0`, `head_limit=25`. Offsets are integers from zero through
`2**53 - 1`; point limits are 1..100 and head limits 1..25. Booleans, unknown
heads and invalid attempts are rejected. Past-end offsets return empty pages.

The response is deep-owned and contains:

- `version:1`, `scope:read-saved-curve-only`, workspace UUID and exact reference;
- saved attempt number/status/stage/error; exact dataset/task references;
- the full selected head definition, source target unit and
  `loss_space:transformed-target-space`;
- `execution:no-fitting-or-training`, `scale:linear`, `x_range`, `y_range`;
- exact saved history-retention facts and final checkpoint identity, or null
  when the selected attempt has no numerical result;
- independent `heads` and `points` pages, each with offset/limit/total/rows and
  `next_offset` (null at the end). Head rows contain ID and loss;
- each point's retained ordinal, successful update, attempted batches, examples
  processed, checkpoint identity, timing/mode and exact train/validation metrics
  for the selected head. Observation-ID arrays are omitted; saved scalar values,
  additive numerators/denominators, counts/status and accuracy facts remain.

There are no fitted arrays, weights, raw observations or per-update test metrics.
The native request requires `action`, `workspace_id`, `reference` and permits
only the optional arguments above. A foreign UUID is rejected before resolving
dependencies. Reads are permitted while a job is busy. CLI returns the ordinary
`ok`/`value` envelope and leaves source file bytes and modification time unchanged.

No SQLite schema, scientific object grammar or TrainingProtocol has changed.
The projection is derived, not another persisted evidence store. Complete registry
validation and arbitrary metadata sizes are **not** bounded by these page counts.
Axes scan the complete retained history of one selected head. This is not lazy
storage or a universal wire-byte, latency, RAM or OOM guarantee.

## Numerical and visual meaning

Each mark is the saved weighted population-mean head loss, not a mean of batch
means and not the combined objective. The horizontal coordinate is successful
optimizer updates. Zero denotes an initial checkpoint **if one was measured and
retained**; normal fits do not invent an initial point. Attempted batches and
processed examples remain separate counters in the exact table.

Both axes are linear. The x range is zero to the final retained update (at least
one). The y range starts at zero and ends at 1.05 times the maximum available
train/validation value across the entire retained head history; overflow padding
falls back to the finite maximum. All-zero/all-missing history uses upper bound
one. Axes remain fixed while paging that head. Only current page points are drawn:
there is no averaging, sampling, smoothing or connection across page boundaries.

Null/unavailable values break line segments and are not displayed as zero. A
page with no available values displays explicit absence instead of an empty
success chart. Straight lines guide the eye between measured checkpoints; they
are not extra evaluations. Training uses solid lines/circles; validation uses
dashed lines/squares. Theme colors are supplementary. Test metrics remain
final-only and are never plotted here. No confidence interval is asserted.

Captions expose exact identity, attempt/head/status, page/retention counts,
transformed target space and declared target transforms. The source target unit
is metadata, not automatically the physical unit of the plotted loss. Expand
**Exact plotted rows** for unrounded values, status, checkpoint, effective weight,
records, numerator, timing/mode and batch/example counters. SVG title/description
and the visible table supplement hover text; user strings remain literal text.

## Interaction and lifecycle

Each curve starts collapsed and makes no curve request until opened. Report cards
own separate viewers bound to their exact run and displayed attempt; report reads
do not follow a retry. Head and point pages have independent Back/Next histories.
Successful replies advance the acknowledged cursor. Failed navigation preserves
the old page/history; **Read / refresh curve** retries the accepted page. Failed
head changes restore the accepted selection. Pending reads disable duplicate
navigation. Switching heads resets the point cursor, not head-choice history.

Replies are checked against workspace/reference/attempt/head/cursors and saved
metric domains. Detached views or workspace changes ignore late replies/errors.
Replacing the results DOM also invalidates its old viewer. This is not durable
process isolation, general ABA protection or crash-safe detached execution.

The plot scales its text with the user's base font rather than squeezing a chart
to fit a narrow panel. Horizontal scrolling preserves readable text. Actual
dark/GitHub/Math fixtures check fonts 14/18/28 and menu spacing 1/4/10; these scoped
geometry checks are not universal accessibility certification.

## Verification and regeneration

`tests/test_run_curves.py` uses real weighted classification/regression fits,
failed retries, cancellation and all-masked supervision; it checks exact scalars,
axes/order/pages, no execution, SQL/history/RNG isolation and stdlib CLI parity.
`tests/evidence-curves.test.js` checks numerical geometry, gaps/zeros, literal text,
independent acknowledged cursors, malformed replies and stale delivery.
`tests/qt_run_curves_smoke.py` exercises actual visible controls and native bridge,
including a real 106-update/30-head fit, paging/failure recovery and report cards.
Run it with the documented Qt and numerical environments, one numerical thread
and Chromium sandboxing enabled; optional `VTW_CURVE_SCREENSHOT` retains a capture.
Private campaign evidence records the particular test environment and outcomes.
