# Bounded recovery browser and terminal notices — local Stage C

An experiment result is first validated/admitted on the GUI thread. A stopped
thread alone is not an admitted terminal result. Failed admission retains the
complete native snapshot/result for explicit inspection/export. This document
specifies the actual webview path, not detached execution or crash recovery.

## Lifetime and identity

The native window owns `experiment_recoveries`, not the deleted QThread or the
currently open workspace. Its opaque `scope_id` is a random 32-hex-character
window-lifetime identity, with an exact integer `sequence` initially zero.
Retention and a successful export marker update each advance that sequence once.
Workspace change does not replace this collection. Process death does lose it.
The sequence fences pages and mutable export markers; it is not a content hash,
execution receipt, persisted event log or general ABA guarantee.

State's `experiment_recoveries` field is now an `experiment-recovery-header`
projection, not an unbounded array: version1, scope, scope_id, sequence, total
and byte_limit65536. A new state update does not resend every diagnostic/result.
No new SQLite rows, workspace schema, CLI session token or parallel file format
is added. Native complete `scienceRecoveries`/`scienceRecoveryShow` remain
explicit compatibility reads, never the actual automatic/browser paths.

## Notifications and closed page requests

`scienceComplete` carries version1/scope experiment-completion-notification,
origin workspace_id, exact retained run reference and `attempt` containing
number/status/stage. It does not repeat the RunRecord, weights, result, datasets
or manifests. Duplicate terminal delivery produces no second admission/notice.

`scienceRecovery` carries the header fields with scope
experiment-recovery-notification and one `recovery` metadata row. The full
envelope stays native. The following Qt reads use closed requests, including
the entire compact request limit; unsupported properties fail:

| Action | Required besides action | Optional | Result scope |
| --- | --- | --- | --- |
| scienceRecoveryPage | scope_id, sequence | offset, limit, query, pin | experiment-recovery-page |
| scienceRecoveryInspect | scope_id, sequence, id | path, part, offset, limit | experiment-recovery-inspection |

Both require the exact current scope/sequence. Reads remain available while
training is active. Offsets are exact nonnegative integers; limit defaults25,
range1..25; booleans/unsafe integers fail. Query is at most200 Unicode code points
with no NUL, a casefolded literal substring of run ID or full diagnostic.
Browsing sorts by (run_id.casefold(), recovery_id), not scientific execution order.

A metadata row contains exact 32-hex recovery ID, `run_id` and `error` text
descriptors, has_result and exported booleans. Descriptors contain offset/text/
total/next_offset and show at most128 Unicode code points, explicitly partial
when more exist. Full export paths and snapshot/result data are omitted.
Page fields are header identity plus offset/limit/query/filtered total/rows/pin/
pinned/missing_pin/next_offset. A valid exact pin is independent of search/page;
missing pins remain unavailable, never substituted with another recovery.
Rows may shorten to the byte budget; follow returned cursors, not assumed strides.

Inspection reuses the literal ordinal tree projection in
[scientific-browser.md](scientific-browser.md). Its identity is scope_id/
sequence/recovery ID, not a scientific registry revision. Dict ordinals follow
the native captured insertion order; arrays preserve order. Complete long keys
and strings are chunked; scalar JSON tokens preserve exact integers. It does
not validate, coerce, admit or fabricate a RunRecord from incomplete evidence.
Inspection has no Edit this value capability.

Each successful notification/read projection is at most65536 bytes of compact
ASCII JSON (ensure_ascii=True, separators(',',':'), finite values), excluding
QWebChannel envelopes/error messages. Metadata has at most25 rows; scalar chunks
at most2000 code points. Hidden values are not fully encoded just to preview them.
This does not bound native captured copies/storage, search/sort, deep traversal,
complete RunRecord/SQL admission, thread-internal terminal payloads, progress
messages, whole state payloads or explicit full export. No universal RAM guarantee.

## Actual UI and export

The existing Experiment panel contains independently searched/paged recovery
choices, exact outside pins, Refresh, Inspect recovery and explicit Export.
Inspection uses the existing read-only tree with Root/Parent/Back/Next/Retry.
Literal text is not HTML. Failed Back retains history; Retry restores the current
cursor. Generation/scope/sequence checks fence late responses/errors. A stale
navigation/search completion cannot restore a previous session's cursor/history.
Workspace changes fence an in-flight tree selection, but the window-owned evidence
remains selectable from the new workspace.

Exports remain complete version1 JSON with format
`visual-pytorch-experiment-recovery`, ID, snapshot, result, error and exported_to.
The actual UI sends id/scope_id/sequence; native export rechecks the same identity
after the chooser's nested event loop and refuses busy work or existing paths.
Legacy id-only export remains compatible. Cancellation/failed creation does not
mark evidence exported or advance sequence. A failed write may leave a partial
new file; select a different new path. The successful file records pre-export
state, then native exported_to/sequence update. It does not train, import, save a
workspace or create a running/completed-job receipt.

Snapshots contain potentially sensitive observations/parameters. Explicit agent
recovery still inspects complete exported bytes and admits dependency-first
specifications followed by a validated retained run using the existing CLI.
Do not import a browser projection or outer recovery envelope as a scientific
specification. Close still warns before discarding unexported session evidence.

## Evidence

Nine stdlib backend tests cover large hidden values, exact integer tokens, Unicode/
key chunks, byte-shortened cursor exhaustion, pins, invalid types/limits, foreign/
stale sequences and non-mutating export markers. Fourteen new JS cases cover
the actual recovery controller/tree: search, pins, failed navigation/retry,
wrong identities, stale navigation, literal text, read-only inspection and
guarded explicit export. Full suites pass535 Python (171.967s)/218 JS (1047.393ms).

Fourteen native lifecycle cases pass (3.228s), including actual queued admission,
cancellation, duplicate delivery, failed admission across workspaces, small state/
notifications and modal export sequence changes/no overwrite. Real sandboxed Qt
inspects80 recoveries with large hidden tensors/diagnostics, exact9007199254740993,
independent outside pins, injected list/tree Back failures, unchanged SQL/history
and an exact full export. Its24 projections peak33406 bytes; font14/18/28 checks
and font18 capture are scoped UI evidence, not whole-window accessibility proof.
No host Torch/console errors. The separate real fit/retry/report/generation/
failed-admission/export regression and the actual eight-candidate/all-split
authoring regression also pass; current details are in
[development.md](development.md).

Remaining native resource costs, figures/window geometry/C integration and the
unchanged D/E requirements are not completed by this checkpoint.
