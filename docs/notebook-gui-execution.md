# Explicit notebook work in the floating GUI — branch-only Stage C

This specifies `app/desktop_notebooks.py`, `app/notebook_execution.py`,
`app/notebook_worker.py` and `app/web/science-notebooks.js`. It extends the
[portable notebook contract](notebook-reports.md), not the SQLite schema or
the [RunRecord](experiment-objects.md) grammar. It is local development work,
not a claim that the initial prerelease contains notebook execution or that C–E
are complete.

## User workflow and honest modes

The existing floating Experiments & history tab has a Notebook execution
disclosure. Generate notebook bundle creates a new directory without executing
cells. The GUI then inspects that directory asynchronously; inspection launches
no kernel. Inspect bundle also accepts an existing complete bundle through a
native directory chooser. No source code or raw HTML is rendered from the bundle.

The inspected selection displays mode, absolute local bundle path, exact recipe,
requested notebook hash and input-index hash. The editable report/kernel Python
defaults to the detected numerical interpreter; timeout per cell defaults to
600 seconds and must be a positive integer. Generation needs nbformat in the
GUI interpreter; inspection/execution need the optional report requirements in
the selected interpreter. Qt need not import nbclient, jupyter_client or Torch.
The interpreter is resolved and checked as executable before admission.

The Run button names its behavior: saved-result execution reads retained evidence
without training; rerun execution launches the captured fresh CPU fits. A native
plaintext confirmation explains this distinction and trusted-code risk, defaults
to Cancel, and puts complete paths/hashes/interpreter/timeout in Show Details.
Cancel starts no execution. It does not discard a previous inspected selection.
Recheck actual workspace object/UUID and busy state after the nested dialog loop.
The runner then revalidates the entire input closure and captured identity before
creating an attempt or starting a kernel. A changed bundle fails, not a substituted
current report head.
Requested bytes are captured and hash-checked again during preparation, then
retained from that verified snapshot rather than rereading a mutable file at copy
time. This is not an atomic filesystem snapshot or hostile-writer sandbox.

Both inspection and execution run on a QThread supervising a separate local
runner process. Native edits, competing jobs, replacement and close remain
guarded through queued terminal delivery, not only while `isRunning()` is true.
The editor state distinguishes inspection/execution from architecture training.
Read-only domain browsing and receipt reads remain available.

## Exact selection and native bridge

An inspected selection has exactly:

```json
{
  "version": 1,
  "kind": "visual-pytorch-notebook-selection",
  "path": "/absolute/local/bundle",
  "recipe": {"kind": "report", "id": "report-id", "revision": "<64 lowercase hex>"},
  "mode": "result-report",
  "input_index_hash": "<64 lowercase hex>",
  "requested_sha256": "<64 lowercase hex>"
}
```

Mode is `result-report` or `rerun`; paths here are session-local, not portable
manifest fields. GUI requests contain the originating workspace UUID. There is
no implicit report head, execute-on-inspection, auto-import, optimizer resume or
execute-raw-JSON action.

| Action | Inputs | Effect |
| --- | --- | --- |
| scienceNotebookInspect | workspace_id, optional directory/python/request_id | Chooser if needed; async verified selection, no kernel/output creation |
| scienceExecuteNotebook | workspace_id, selection, optional python/timeout | Explicit confirmation and async real kernel; fresh attempt artifacts |
| scienceNotebookHistory | none | Read copied session execution rows; never execute |
| scienceNotebookEvidence | id | Recheck retained receipt/artifact bytes and return copied session row |
| stop | none | Stop the active inspection/execution through its normal job cancellation |

Inspection/execution admission returns `job_id,operation,selection,status`, with
status `started-not-completed`. Inspection has null selection until its terminal
event. Cancellation of either native chooser/confirmation returns null. State
`notebook_job` is null when idle, otherwise `id,workspace_id,selection,operation`
(`inspect` or `execute`). It does not fabricate `experiment_job` or a RunRecord ID.

`scienceNotebookInspected` contains `job_id,workspace_id,request_id,inspection,error,status`.
Successful inspection is `inspected-not-executed` and carries `selection,python`.
Stopped/failed inspection has no execution receipt. Actual origin/job replacement
suppresses automatic selection, including a different connection with the same
UUID. GUI request-generation/UUID fences reject late inspection data.

`scienceNotebookProgress` contains `job_id,workspace_id,progress`. Progress has
version 1, attempt_id, stage (`kernel-starting`, `cell-running`, `cell-replied`),
zero-based cell_index or null, total notebook cell_count and section or null.
A reply also includes reply_status. Running/replied cells are not claims of
successful notebook completion, successful candidates or gradient-update counts.

`scienceNotebookComplete` is a session row with exactly
`id,workspace_id,selection,python,cell_timeout_seconds,result,error,status`.
Result is the existing `path,execution` notebook terminal object or null. It is
validated against captured identity/policy, the retained execution.json, actual
artifact bytes/hashes, executed-notebook metadata and cell counts. A runner error
without verified receipt has status `runner-error-no-verified-receipt`; it is not
fabricated as completed/cancelled execution. Inspection makes no history row.
Terminals are handled once; external artifacts survive origin/job replacement.
Automatic GUI display is UUID-scoped. Session history remains explicitly readable
with origin labels; it is not automatic cross-workspace scientific admission.

## Stop and evidence retention

Execution uses the real nbclient/AsyncKernelManager adapter. A private temporary
control directory carries a single Stop marker, avoiding nbclient's own signal
handlers. The worker checks Stop before kernel launch/each next code cell and
polls during execution. Cancellation of a pending launch waits for the manager's
ready future rather than abandoning an unidentified child. Actual cell timeouts
and independent kernel death remain failures. Only our admitted cancellation
can translate nbclient's polling-cancellation DeadKernelError into Stop.

The asynchronous client cleanup and an independent shutdown path run before
writing executed notebook/receipt bytes. Ordinary Stop retains partial output,
existing numerical artifacts and actual execution counts, with status cancelled
and NotebookCancelled diagnostic. Later cells remain unexecuted. Counts label
cells with execution counts, not completed gradients or successful cells.
An interrupted resolution need not have report-data.json or a terminal run for
the interrupted candidate; no such evidence is invented.

The parent gives cooperative Stop 30 seconds before the existing direct-child
KILL deadline. A forced termination, cleanup/I/O failure or malformed protocol
may leave inspectable partial files without a verified receipt: the GUI reports
that distinction and bundle path. This is not a hostile-descendant sandbox,
atomic publication, crash-safe supervisor or Stage E detached-job guarantee.
Normal owned Linux kernel/worker process-group termination has a real fit test;
it is not a guarantee for arbitrary Python-created descendants/platforms.

Receipt cards use literal DOM text, base fonts and theme tokens. Long hashes/paths
wrap inside the same floating panel. The exact artifact byte/hash ledger is a
collapsed disclosure. Changed artifact bytes fail explicit reads rather than
being displayed as verified. Reading/generating/executing does not change working
or saved .vtw bytes, domain undo/history, current selections or source run records.
Numerical runs produced by rerun are external bundle artifacts, never auto-imported.

Receipts/notebooks remain durable on disk; session history is window-owned memory,
not stored in .vtw. Reopening/discovering arbitrary prior attempt histories,
detached recovery, full inline validation/transport bounds, richer rendered reports
and full C exit convergence remain separate work. A completed notebook means
its cells completed, not that every selected training candidate succeeded.

## Evidence and references

`test_notebook_execution.py` uses real kernels for both modes, precise input
identity/corruption gates, a genuinely running long cell, Stop during an actual
gradient-updating rerun fit with kernel/numerical-child shutdown, independent kernel
death, stdlib-only parent inspection/execution, and receipt/artifact checks.
`qt_notebook_lifecycle_smoke.py` checks queued terminal fences, read-only history,
modal/chooser origin and busy guards, duplicate/invalid terminals and same-UUID
inspection replacement. `qt_notebooks_smoke.py` exercises real sandboxed WebEngine
controls, generation/inspection, saved-result and eight rerun fits, cell failure,
timeout, Stop, kernel shutdown, SQL/file isolation, actual native confirmation,
safe literal content and Dark/GitHub/Math fonts 14/18/28 with spacing 1/4/10.

Implementation references: [nbclient execution and cell hooks](https://nbclient.readthedocs.io/en/latest/client.html),
[nbclient API](https://nbclient.readthedocs.io/en/latest/reference/nbclient.html),
[Jupyter client kernel lifecycle](https://jupyter-client.readthedocs.io/en/stable/api/jupyter_client.html)
and [Qt queued thread delivery](https://doc.qt.io/qt-6/threads-qobject.html).
The async Stop translation is our design, verified against installed nbclient
behavior and real processes, not a library-promised durable cancellation protocol.
