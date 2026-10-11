# Exact stored candidate browsing — branch-only Stage C

The actual Qt Experiments & history tab uses `app/candidate_views.py` and
`app/web/science-candidates.js`, not the complete `sciencePreview` transport.
This remains on the local C task branch, not the published prerelease.

## Native read and agent command

`stored(workspace, reference, offset=0, limit=25)` requires an exact retained
ExperimentSpec reference, with precisely `kind`, `id`, `revision`. It never
substitutes a mutable head. Offset is an exact nonnegative integer at most
9007199254740991; limit is an integer in 1..25. Booleans are invalid.
The Qt `sciencePreviewPage` request requires `workspace_id` and `reference`,
with optional offset/limit and no other keys. The complete request passes the
existing bounded native JSON parser before workspace identity checking.

The agent equivalent is:

```sh
python vpw_cli.py lab.vtw preview-experiment-page EXPERIMENT_ID EXPERIMENT_SHA256 --offset 0 --limit 25
```

It returns the same projection in the ordinary CLI envelope, without saving or
changing file bytes/mtime. No persistent session token is required. Existing
`preview-experiment` remains a deliberate complete candidate read: default100,
limit1..1000, with full manifests and resolved protocols. It is row-bounded,
not byte-bounded. `run-candidate` still accepts the exact row coordinates.
No SQLite tables, stored grammar, version or scientific identity change.

## Projection, not a partial specification

Successful output has version1, scope `stored-experiment-preview`, workspace_id,
byte_limit65536, exact experiment reference, offset/limit, candidates,
next_offset, total_combinations_exact and execution `no-fitting-or-training`.
The total is a decimal string, not a rounded JavaScript number. The byte limit
uses compact ASCII JSON, `ensure_ascii=True`, finite values and separators
`(',', ':')`; enclosing CLI/WebChannel formatting, envelopes and errors are not
included. A row may contain at most:

- id, coordinates, status and exact dependency references, identical to the
  existing native candidate resolver in original scientific axis order;
- reason: null or an explicit128-code-point descriptor with offset/text/total/
  next_offset, never silently presented as a complete long diagnostic;
- counts: null without a manifest, otherwise each role's actual ID count and
  excluded count, never role ID arrays, datasets, tensors or resolved protocols.

Compatible, excluded and incompatible candidates remain present. Counts do not
reinterpret missing evidence as zero. Native source generation/validation is
unchanged; projection is shared with prospective owned-draft previews, whose
owner/sequence header and scope remain distinct.

Rows stop before exceeding the byte limit. The actual next_offset is offset plus
the returned row count, not the requested limit. A page at/past the end is empty
with a null cursor. An individually oversized candidate fails explicitly, rather
than fabricating an empty successful page. The UI does not issue unsafe rounded
offsets: reaching the native exact-offset domain's edge is visibly unavailable.

## Actual interaction

Selecting a retained experiment opens its first candidate page. Previous/Next
follow successful cursor history. Failed navigation does not advance that
history; Refresh candidates retries the current acknowledged cursor. Pending
reads clear obsolete cards and prevent duplicate navigation. Selection changes,
workspace replacement and notified same-UUID external reload clear the controller;
late data/errors/navigation continuations cannot restore an older selection.
Replies must match version/scope/workspace/exact reference/offset/limit/execution
and decimal total before installation. Scientific metadata reads also disable
candidate execution while selection resolution is pending.

Role counts and partial diagnostics are literal text, never HTML. Only compatible
rows can Run this candidate. That explicit click sends workspace_id, the exact
retained experiment reference and coordinates—not the preview as a specification
or execution bundle. Busy/read/launch-pending guards refuse competing clicks;
detached old row handlers cannot launch. The native train path checks a supplied
workspace_id before capturing dependencies or replacing a worker. Existing
native training/capture/admission gates remain authoritative. Compatibility
clients may still omit the optional train workspace_id. Run acknowledgements
and errors belonging to an obsolete view do not change its replacement.

## Evidence and limits

`test_candidate_views.py` covers legacy equality, historical pins, byte-shortened
cursor exhaustion, explicit exclusions/incompatibilities, full CLI compatibility,
SQL/history/dirty/RNG purity, hidden manifests, exact decimal totals and closed
arguments. `science-candidates.test.js` covers the actual controller's read/run
guards and failed/stale navigation; `science-browser.test.js` exercises its
real mount through `LoomScience` and same-UUID reload handling.

`qt_candidate_browser_smoke.py` runs the actual sandboxed Qt/WebChannel app on
private data: three models × two datasets × four splits × two seeds × two
protocols =96 coordinates, including all three statuses, long observation IDs
and reasons, exact historical selection, forced6000-byte pages, failed Back/
Refresh and unchanged SQL/history/RNG. It forbids fits and the complete preview
UI action. Font14/18/28 checks and a captured font18 view are scoped form evidence,
not whole-window/accessibility certification. Actual fits are tested separately.

This bounds successful projected wire bytes and displayed cards, NOT complete
native registry/dependency validation, inline dataset parsing, manifest generation,
storage/history/copies, error text, thread-internal payloads, memory or time.
Native preview now reuses successful exact dependency/contract/compatibility/split
preparation within one read; [preview-resolution.md](preview-resolution.md)
records its equivalence contract and same-fixture measurement. It is neither a
cross-request cache nor streaming complete preview construction.
The projection does not prove lazy storage, universal ABA safety, durable recovery,
detached execution or that the full Stage C/D/E roadmap is complete.
