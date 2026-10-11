# Typed dataset sessions — local Stage C checkpoint

The actual Qt dataset form now uses an [owned authoring session](owned-authoring-sessions.md).
Opening transfers a header and bounded typed pages, not the complete DatasetSpec.
The task/experiment forms have **not** migrated. This is branch-only work, not
a new public release, SQLite format or independent persistence store.

## Interaction and preservation

New/Edit retains an incomplete private dataset draft, its stable ID, exact
historical base and captured current head. At most six fields and twenty
observations appear per page; the byte budget may shorten either page further.
Navigate using actual returned cursors, never assumed increments. Scientific
field/record/vocabulary order is unchanged. Hidden values stay on the host.

Small cell/vocabulary/metadata values are complete **literal JSON text**, not
parsed JavaScript numbers. The native strict parser rejects duplicates, escaped
aliases, nonfinite constants and exponent overflow. `int64` fields accept exact
integer tokens from −2⁶³ through 2⁶³−1, excluding booleans and floats. Numeric
training preparation still explicitly converts inputs/regression targets to
float32: exact storage does not promise exact arithmetic for such large integers.
Category labels retain their existing type-sensitive fixed-vocabulary contract.

A conservative bounded walk determines whether a value can be encoded for a
small control. Large values are represented as an explicit type/size marker,
with an empty readonly control and **Edit value…**, not a truncated JSON value.
This avoids fully stringifying hidden tensors to discover that they are large.
Small values also offer staged editing when a replacement needs more space.
Absent optional values are added through their typed control before staging.

Typed edits are queued by stable target. A later edit of the same control
replaces its older queued edit; distinct targets retain user edit order. Focused
dirty text is captured before controls are disabled. Navigation, validation,
retention, structural changes and staging flush the queue in bounded batches.
Native syntax failures mark the failing control, retaining its text/queue for
repair. Temporary schema-invalid drafts are allowed; Check/Commit remain strict.
Already accepted batches remain in the unsaved draft if a later batch fails.
Atomicity is **per batch**, not an entire multi-batch form transaction.

Failed page reads offer **Retry page** without opening another session. With no
initial page, page-dependent actions remain disabled, but Cancel/Retry are
available. Failed backward navigation does not consume its retry history.
Invalid local text blocks rereading; it is never discarded to retry transport.
Validation markers are scoped to dataset controls, not hidden task/experiment
controls. Explicit Cancel cleans only the discarded dataset controls.

## Closed bridge commands

All requests require `action`, `session`, exact document `sequence`; unknown
properties fail. Compact ASCII JSON requests/projections are at most 65536 bytes,
excluding the ordinary bridge envelope. Owner/sequence/kind and actual native
working-copy/connection guards are those of owned sessions.

| Action | Other required properties | Optional properties | Result |
| --- | --- | --- | --- |
| `scienceDraftDatasetPage` | none | `field_offset`, `row_offset`, `field_limit`, `row_limit` | Typed page |
| `scienceDraftDatasetEdit` | `operations` | none | Header, sequence + 1 |
| `scienceDraftDatasetValue` | `target` | none | Current value descriptor/path |

Offsets are exact nonnegative integers, excluding booleans. Limits are 1..6
fields (default 6) and 1..20 observations (default 20). End pages may be empty.
Page/Value are non-numerical reads, including after a workspace switch. Edit
requires the captured actual current working copy/connection/UUID, no pending
value buffer, and no busy numerical execution including queued delivery.
Current aggregate constrained draft admission runs before touched copies and
before publishing; accepted headers/identity are checked before sequence changes.

### Page DTO

`version: 1`, `scope: owned-dataset-page`, `session`, `sequence`, `workspace_id`,
`id`, `name`; `field_offset`, `row_offset`, `field_limit`, `row_limit`,
`field_total`, `row_total`, `next_field_offset`, `next_row_offset` (actual next
ordinal or null); arrays `fields`, `rows`, `metadata`; `byte_limit: 65536`,
`text_limit: 256`. Neither this DTO nor an edit batch is a persistable DatasetSpec.

Fields have stable `id`, current `ordinal`, `path`, and ordered `properties` for
name, dtype, shape, axes, nullable, vocabulary, modality, unit, coordinates.
Rows have stable `id`, current `ordinal`, `path`, and `cells` in the returned
field order. Metadata descriptors are provenance then usage.

A descriptor has `key` where relevant, `path`, `present`, `text`, `summary`.
`text` is a complete JSON string of at most 256 code points when eligible,
otherwise null. Budget pressure can lower eligibility to 64 code points.
Conservative eligibility can mark a smaller value as large; the bound is not a
promise to inline every short value. Absent values have `present: false`, null
text/summary and a container path, **not** a purported existing-member path.
Large summaries have type array/object/string/scalar, `size` for containers or
strings (otherwise null), and at most a 64-code-point string preview (otherwise
null). The marker never substitutes for data. Vocabulary describes its ordered
`values` array and nested path, not the enclosing versioned vocabulary object.

Field projections reserve at most 32768 bytes to leave row space. Each added row
is checked against the full page bound. Stable IDs are complete, never truncated;
a single projection that cannot fit fails explicitly. Paths are current
insertion-order ordinals at this sequence, not persisted IDs.

### Ordered edit grammar

`operations` contains 1..25 entries. Each operation has exactly these properties:

| `op` | Other properties | Meaning |
| --- | --- | --- |
| `name` | `text` | Parse and replace dataset name |
| `metadata` | `key`, `text` | provenance/usage; empty text removes property |
| `field-add` | `id` | Append unique stable field, float32 scalar; existing records gain explicit null |
| `field-remove` | `field` | Remove exact field and corresponding cells |
| `record-add` | `id` | Append unique observation with all fields explicitly null |
| `record-remove` | `record` | Remove exact observation |
| `field` | `field`, `key`, `text` | Replace supported property; empty text removes optional property |
| `cell` | `record`, `field`, `text` | Parse and replace exact cell |

`text` is strict literal JSON of at most 2000 code points. Empty text is only
the removal convention above, not a JSON null or empty string. Dtype/shape cannot
be removed; field IDs cannot be changed through property edits. Vocabulary text
is parsed as values and wrapped in `{version: 1, values: ...}`. Selecting category
initializes an empty vocabulary if absent; selecting another dtype removes it.
These changes do not coerce existing cells. Final validation explains conflicts.
Missing/ambiguous field/record IDs fail, including after structural changes.
IDs such as `__proto__` are ordinary native string keys, not object prototypes.

All operations are applied to touched container copies, then stable document
ID/version, finite JSON, memory admission and projected header are checked before
publishing. Failure leaves the old document/sequence unchanged. Operation errors
use a zero-based `Dataset edit #N:` prefix; post-batch errors need not identify
a single control. Successful batches advance sequence once, including no-ops.

### Stable value targets and shared child editor

Targets have exactly `{kind: metadata, key}`, `{kind: field, field, key}`, or
`{kind: cell, field, record}`. Property names/dtype rules match the edit grammar.
After all pending typed edits are accepted, Value resolves stable IDs against
the **current** sequence. Removing a property or column cannot silently redirect
an old ordinal to a different value. Object key insertion order is not identity.

Response: `version: 1`, `scope: owned-dataset-value`, session/sequence, complete
echoed target, descriptor fields and byte_limit. The linked raw editor uses the
same header/session with `from_base: false`. It opens no second parent draft.
**Use staged value** advances the document sequence, closes the child and
refreshes the typed page; it does not retain a revision. **Discard staged value**
closes only the buffer and returns the unchanged parent header. Child Check/Save
are disabled and guarded; validation/retention belong to the parent form. A
failed Start can be discarded without closing its parent. Header kind/ID/owner/
scope/version/sequence are verified before returning. See
[chunked-draft-values.md](chunked-draft-values.md) for UTF-8 splice semantics.

Late success/error after a workspace switch or notified same-UUID reload cannot
replace controls, clear unsaved text, select old results or update a stale parent.
The retained draft is marked stale; explicit cancel/reopen is required. Successful
parent Commit uses normal history and selects its exact reference; explicit Save
workspace persists it to disk. No-op/undo/redo/old immutable evidence remain intact.

## Verification and scope limits

Eleven new backend tests cover exact paths/precision, large-value encoding
avoidance, byte-shortened Unicode pages, stable IDs/order, batch rollback,
strict parsing/limits, memory/origin/buffer guards and full int64 boundaries.
Fifteen typed-controller tests and four linked-editor tests exercise actual source
with controlled asynchronous replies, including focus capture, invalid repair,
32 distinct ordered edits across byte-limited batches, retry, stale replies,
wrong headers and shared-child retention boundaries.

Actual sandboxed Qt exercises 200 × 512 tensor observations, seven fields,
large Unicode metadata, historical base/head, exact int64 edits, child finish/
discard, injected initial/return-page failures, font14/18/28 in dark/GitHub/Math,
single undo/redo and save/reload. It uses one session, no complete AuthorBegin/
Check/Save/Index/Show, no host Torch/Jupyter, and bounded result projections.
The native capture is reviewed; the table reserves a readable 12ch ID column and
scrolls locally instead of compressing IDs/headings vertically.

This bounds transport, small-value preview work and rendered control count. It
does **not** provide lazy SQLite storage, a universal allocation/OOM or disk bound,
bounded dependency/registry/history validation, crash recovery or detached jobs.
Constrained estimates remain heuristics. Field/port suggestions now have
[bounded context pages](task-context-pages.md), not whole-host cost bounds.
Task/experiment full forms, local selection cards/axes, recovery/completion costs, in-app figures
and integrated C acceptance remain open. D/E requirements are unchanged.
