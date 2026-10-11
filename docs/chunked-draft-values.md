# Chunked draft values — local Stage C implementation

The actual Qt Experiment/history tab offers **Edit this value…** for a non-root
value of an exact dataset/task/experiment revision, but not keys, models, run
evidence or reports. It uses [owned-authoring-sessions.md](owned-authoring-sessions.md),
not a parallel persistence store. The [typed dataset form](typed-dataset-sessions.md)
and [typed task form](typed-task-sessions.md) share their owned sessions with this
editor; experiment authoring has not migrated.

## Three distinct states

The immutable revision stays unchanged. Its owned JSON-shaped document draft
also stays unchanged while one raw UTF-8 value buffer is staged. Buffer splices
advance `value_revision`, not document `sequence`.

**Apply fragment** stages text, which may be malformed. **Use staged value**
parses the whole buffer and publishes only a successful replacement into the
unsaved document, advancing sequence once and closing the buffer. **Validate
draft** runs the complete ordinary scientific validator without writes. **Retain
revision** uses normal admission/history; explicit workspace Save persists it
to disk. A schema-invalid temporary value cannot pass Check/Commit. Stable
document ID/version cannot change and a revision cannot be forged.

Malformed JSON, duplicate/escaped-alias keys, nonfinite constants, exponent
overflow, identity/sequence and memory failures preserve the buffer and original
document. Pending buffers block Patch/Check/Commit. Value Discard closes only the
buffer; Close discards the whole draft. Finish/Discard/Close/owner teardown close
their owned temporary files. Reads/disposal do not parse the complete value.

In linked dataset mode, Use staged value closes the child and returns its advanced
header to the parent form. Discard staged value releases only the buffer and
returns the unchanged header. Child Check/Save are disabled and guarded: parent
validation/retention remain explicit. Failed Start can return without closing
the parent session. Header identity/sequence and late-reload delivery are tested.
Standalone mode still offers Validate/Retain and Discard draft as described above.

## Closed command grammar

All requests require `action`, `session`, exact document `sequence`; unsupported
keys fail. Compact ASCII JSON requests/projections are at most 65536 bytes,
excluding the existing bridge envelope.

| Action | Other required properties | Optional properties | Result |
| --- | --- | --- | --- |
| `scienceDraftValueStart` | `path` | `from_base` (boolean, default false) | Buffer header |
| `scienceDraftValueRead` | `value`, `value_revision` | `offset` (default 0) | JSON fragment |
| `scienceDraftValueSplice` | `value`, `value_revision`, `offset`, `delete_bytes`, `text` | none | Header, buffer revision + 1 |
| `scienceDraftValueFinish` | `value`, `value_revision` | none | Document header, sequence + 1 |
| `scienceDraftValueDiscard` | `value`, `value_revision` | none | Unchanged document header |

Start/Splice/Finish require the captured actual owner/Workspace/connection/UUID
and are blocked during busy execution, including queued terminal delivery.
Read/Discard are owner/sequence-fenced but permit inspection/disposal after
switches and during execution. Tokens are transient, not `.vtw` or CLI IDs.

Buffer headers have `version: 1`, `scope: staged-draft-value`, `session`,
`sequence`, opaque `value`, nonnegative exact `value_revision`, normalized draft
`path`, original requested `source_path`, `from_base`, `total_bytes`,
`offset_units: utf8-bytes`, `byte_limit: 65536`. Read adds `offset`, literal `text`
and actual `next_offset` or null; neither returns the complete parsed value.

### Exact-base paths

Paths use container ordinals at the exact sequence, not truncated property names.
Default Start indexes the draft. `from_base: true` accepts a non-root stored-base
path only at sequence 0. Canonical stored root keys include `revision`; Begin
removes that property. Start subtracts one from the first ordinal only if it
comes after revision; descendants remain unchanged. Editing revision itself
fails. Header `source_path`/`from_base` verify this translation before the client
adopts normalized `path`. Scientific arrays/keys are not truncated or sorted.

### UTF-8 and numeric precision

Read returns at most 2000 Unicode code points from at most 8192 source bytes,
using an incremental decoder so no partial trailing character is emitted.
Offsets/deletions are UTF-8 bytes, not JavaScript UTF-16 units or code points.
Both ends must be complete character boundaries within the buffer, including
EOF; negative, boolean and out-of-range counts fail. Follow actual next_offset.

Splice inserts at most 2000 code points of valid UTF-8 and copies prefix/suffix
to another owned temporary file in blocks of at most 65536 bytes. Publish follows
flush/header validation; raw JSON syntax need not yet be valid. Integer tokens
beyond 2^53 remain literal text until Python whole-value parsing, not rounded
through JavaScript Number.

The themed, font-inheriting textarea treats content literally. Dirty fragments
block navigation; failures preserve text and cursor history. Pending commands
block duplicates. Exact base/session/sequence/buffer/path/cursor responses are
verified. Workspace switches and notified same-UUID reload suppress late
successes/errors and disable admission while preserving visible text for
inspection/copy/discard. Already-admitted operations can advance native state;
stale cursors then fail explicitly, not substitute another value. Reopen is explicit.

## Resource admission and limits

Constrained Open/Start/Patch/Finish/Check/Commit use the existing user setting
`memory_limit_percent` (default 10), in addition to numerical preflight. Unknown
available RAM fails closed; unconstrained mode skips observation. Read/discard
remain possible after the budget shrinks. Current config is read per operation.

Version-1 `app/authoring_memory.py` estimates include 8 MiB headroom plus six
estimated JSON-shaped Python object equivalents. Before exact-base decoding,
SQL UTF-8 document length gets a coarse 64× multiplier. Finish adds 64× staged
file bytes before full reading/parsing. Other live session estimates are
aggregated. The native shell repeats Commit preflight before undo capture.

These are admission heuristics, not measured allocation bounds or an OOM
guarantee. Complete parsing/copying, long-string encoding, dependencies, registry
admission and undo still have host costs. Temporary storage can be RAM-backed;
there is no disk quota/lazy dataset storage guarantee. Available RAM can change
after checking. No crash recovery or detached/durable temporary discovery is promised.

## Evidence and remaining migration

Eleven new stdlib tests cover large values/int64, UTF-8, source translation,
malformed/nonfinite/duplicate/identity failures, cleanup, aggregate admission,
unknown RAM and percentages. Twenty async JS tests cover dirty text, actual
cursors, retries, pending commands, stale read/edit/commit deliveries and linked
parent/child boundaries. The prior standalone checkpoint passed 479 Python/129
JS; the subsequent dataset migration passes 490 Python/148 JS, including pending
typed Begin exclusion against the actual typed controller.

`tests/qt_draft_values_smoke.py` loads actual app HTML/sandboxed QWebChannel with
20000 values. Actual controls verify historical editing, malformed-text/SQL
retention, int64 precision, budget refusal before whole parsing and undo capture,
font14/18/28 in dark/GitHub/Math, one history entry, undo/redo and `.vtw` reload.
No full AuthorBegin/Check/Save, scienceShow/scienceIndex, host Torch/Jupyter or
worker is involved. The private capture was inspected. Existing native dataset
and owner/session regressions pass.

Next: experiment typed-form migration onto bounded sessions, local axis/cards,
recovery/completion costs and integrated C acceptance. A working standalone value
editor is not proof that the whole authoring UI is resource-converged. CLI
import/ordered apply still admit complete validated objects without Qt tokens.

## Primary references

[Python TemporaryFile](https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryFile)
supports owned binary temporary storage/cleanup on close rather than selecting
an in-memory spool. [Incremental codecs](https://docs.python.org/3/library/codecs.html#incremental-encoding-and-decoding)
informs complete-character reads; byte/cursor/Unicode behavior is tested on the
installed runtimes. Neither reference is an every-platform crash-safety claim.
