# Stage C acceptance — datasets, experiments and reports

Status: integrated locally in `main` at `0a3bc8cd826eb11a18abc08892faad347f8b274b`,
after acceptance checkpoint `bfafccb416c71063e89bab0449217e7745ebcf70`;
not a public release. Merge ancestry and exact source-tree equality were verified.
This
implements C of the pinned nepr programme, not completion of the full A–E goal.
The original requirement and exit-example mapping is in
[stage-c-verification.md](stage-c-verification.md), based on nepr commit
`777dd1c25e11583877e607064ffe7e61dfdb54a4`.

## Acceptance evidence

| Requirement | Current executable evidence |
| --- | --- |
| Independent dataset/task/experiment/run/report identities in SQLite, synchronized CLI | Revision/object/CLI suites; strict hashes, immutable revisions, retained historical attempts, explicit bindings and v6 roundtrips. |
| Group/time splits, train-only fitted transforms, named structured losses and minibatches | Dataset/preparation/task-training suites, independent scalar and tensor gradients, masked/weighted denominators, unequal batches and exact budget/cancellation counters. |
| Two models × two datasets × two declared replicates, compatible/excluded combinations and exact retained manifests | Experiment-object tests and actual Qt experiment/real-kernel eight-candidate rerun; candidate browser verifies all96 bounded statuses and historical pins without SQL/history/RNG changes. |
| Practical typed dataset/task/experiment authoring | Actual owned-session Qt checks: historical bases, literal int64 precision, large hidden values, independent pages/local pins, child Finish/Discard, read retry, one history edit and save/reload. The billion-coordinate example is preview only, not billion fits. |
| Read-only evidence, comparisons, predictor export and curves | Actual saved-evidence/curve/recovery/revision browser checks; exact earlier attempts survive retries, duplicate selections and missing values stay explicit, standalone fitted predictor performs independent inference without the app. |
| Notebook generation is not execution; saved-result reports do not retrain | Kernel-local no-training fence and real fresh-ipykernel tests; actual Qt generation/inspection/confirmation/saved-result/rerun, failures, timeout, Stop, retained artifacts and late/corrupt receipt checks. |
| Existing A/B/editor behavior survives |613 Python tests191.285s and260 JS tests1410.552ms; actual Qt graph gestures/history/bindings/rewiring/Softmax and editor-built CNN fit, checkpoint reload and trained export. Native experiment/notebook lifecycle suites14/10 cases pass. |
| Crowded window controls remain accessible | Actual native window60 font/zoom/spacing/theme/one-two-tab configurations:780 focused toolbar hit tests,300 launcher clicks,60 Validate clicks and60 native Tab reveals. SQL/disk/history/config file unchanged. |
| Workspace presentation is portable without incidental writes | Explicit panel presets, scoped/CAS guards and CLI parity; independent block/menu snapping; parameter-relative layout, session camera, exact numerical identity and .vtw save/reload checks. |

These are executable observations, not claims inferred from test counts alone.
The suites run on exercised local Linux/Python/Qt/Torch environments with private
workspaces/profiles and bounded CPU work. Chromium sandboxing stays enabled.
Retained private campaign evidence includes initial failures and final logs:
`evidence/stage-c-window-acceptance-20261011/` and preceding C checkpoints.

## Reproduced defects and test migrations

The initial30-window sweep reproduced20 covered-control/overflow/canvas failures.
Independent wrapping and bounded scrolling fix these without hidden duplicate
commands, font reduction or scientific edits. Fit now uses shared rendered block
metrics with adaptive margins in every positive uncovered rectangle. It does not
place the graph behind panels or force a minimum zoom that overflows its region.
The60-window follow-up exercises physical clicks and native keyboard focus.
Maximum font/zoom captures prove reachable overflow controls, not that tiny
diagram labels are readable at every combination of preferences.

Earlier harnesses depended on a startup SQLite Fit write, which the user's
session-only camera requirement intentionally removed. They now wait for the
webview camera; saved-evidence startup explicitly requires zero writes. Read-only
fixtures may have no undo directory and clean up accordingly. Experiment revision
tests use the current literal labels, retain old pins and unfinished text, and
assert that actual reload freezes the owned draft instead of silently admitting
it against replacement origin. Failed assumptions were retained, not suppressed.

## Explicit boundaries, not unfinished C promises

Visible transport/page bounds do not bound all native registry decoding, inline
CSV/data reads, validation, manifest generation, history snapshots or copies.
Request-local preview reuse reduces repeated successful preparation, not every
initial cost. Constrained RAM admission is an estimate, not an OOM guarantee.
The original C programme did not require universal lazy storage or resource proofs.
Do not call these limitations implemented detached/heavyweight execution.

Training is bounded CPU float32 with supported losses/operators and explicit
denominators, not an arbitrary architecture/optimizer/device catalogue. Notebook
Python is trusted execution, not a hostile-code sandbox; hashes establish byte
integrity, not signatures. Local portability and fresh kernels are tested, not
every OS, GPU, Jupyter frontend or hardware's bitwise reproducibility.

Symbolic architecture families, recurrent composition and tied-versus-repeated
weights belong to D. Durable detached jobs, optimizer-resume lineage, distributed
reductions and remote resource adapters belong to E. They remain required by the
active overall goal, and are not delivered by completing C. No remote jobs or
publication are authorized merely by this acceptance checkpoint.
