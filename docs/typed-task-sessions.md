# Typed task sessions — local Stage C implementation

The native Qt bridge now supports bounded literal binding and transform pages
inside an existing owned task session, exact literal edit batches and an owned
compatibility check. The actual typed task form now uses these commands: Open
returns a header, inputs/heads and their visible pipelines page independently,
and edits queue as exact literal text. [Experiment forms](typed-experiment-sessions.md)
also use owned sessions. These locally integrated Stage C controllers do not
establish universal resource bounds or complete the A–E programme.

## Visible form and preservation

The new/edit task controls open one session from an incomplete template or an
exact historical base, distinct from its captured browsing head. At most six
input and six head cards appear; each visible pipeline separately shows at most
six steps. Existing nonempty pipelines open initially; empty ones remain folded.
Bindings and transforms keep scientific order. Follow returned cursors rather
than assuming increments, including after byte-shortened pages. Navigation
history changes only after a successful read; failed backward reads retain a
usable Back control, and Retry transforms repeats the current failed read.

Focused dirty text is captured before disabling controls. Small property edits
coalesce by named target/key, send native-parsed JSON tokens, and flush in ordered
batches of at most 25 operations within the byte limit. Optional removals cannot
shift a later property edit into another target. Invalid scoped text blocks
mutation/paging/admission without being discarded. A native failed-operation
ordinal identifies the field needing repair. Atomicity is per accepted batch,
not an entire multi-batch form transaction.

Flush before structural edits, then read fresh ordinal paths. An accepted
structure change followed by read failure blocks admission and page-dependent
edits until Retry page succeeds. Parallel reads settle before releasing the
pending-command guard; one early failure cannot permit navigation while sibling
reads remain in flight. Initial failures retain the same session, not a second
implicit Open. Stale replies/errors cannot paint controls or notices after a
workspace switch/reload.

Large properties show a disabled explicit marker, not truncated editable text.
Edit value flushes typed edits, resolves the current named property, and opens
the raw child in that same session. Finish/Discard returns an advanced/unchanged
verified header; neither admits nor discards the parent. Exact model/dataset
selectors and independent context pages preserve historical pins and unfinished
binding text; read-only suggestions do not flush or rebuild binding controls.
Check/compatibility/Commit verify returned exact identities before displaying
validity or selecting retained results.

Returning to the same UUID alone does not re-enable editing. The transient
`scienceDraftOrigin` proof must establish the original actual native Workspace,
connection, owner and unchanged sequence. A notified reload stays stale even
when UUID is unchanged; cancel/reopen is required. This is not general ABA or
durable crash recovery. Common actions remain exclusive with dataset/experiment
drafts and active value children, with typing shortcuts preserved.

## Closed native commands

All requests require `action`. Unsupported top-level properties fail. Compact
ASCII JSON requests/results have a 65536-byte maximum, excluding the bridge
envelope. Existing session owner/sequence/origin/admission contracts apply;
no additional draft store, persisted identity or SQLite kind.

| Action | Other required properties | Optional properties | Result |
| --- | --- | --- | --- |
| `scienceDraftTaskPage` | `session`, `sequence`, `collection` | `offset`, `limit` | Literal binding cards |
| `scienceDraftTaskTransforms` | `session`, `sequence`, `collection`, `binding` | `offset`, `limit` | Literal transform cards |
| `scienceDraftTaskEdit` | `session`, `sequence`, `operations` | none | Advanced ordinary owned-session header |
| `scienceDraftTaskCompatibility` | `session`, `sequence`, `model`, `dataset` | none | Existing exact interface-only compatibility DTO |

`collection` is `inputs` or `heads`. Offset defaults to 0, exact nonnegative
integer; limit defaults to 6, exact integer 1..6. Booleans are not integers.
`binding` is an existing zero-based scientific binding ordinal. Reads retain
order rather than sort port names/head IDs. Another owner or stale sequence fails.
Inspection remains possible after a workspace switch, while mutation and
compatibility admission require the original actual Workspace/connection/UUID.
Page/transform/compatibility reads are non-numerical and allowed during a job;
edits are blocked while a job or its terminal delivery is pending.

## Page DTOs

Common fields: `version: 1`, `session`, `sequence`, `workspace_id`, task `id`,
`collection`, `offset`, `limit`, `total`, `path`, `rows`, `next_offset`,
`byte_limit: 65536`, `text_limit: 256`. Scope is `owned-task-page` for bindings
or `owned-task-transform-page` for transforms. A transform page also carries
`binding`. `path` is the current ordered collection/transform-array path,
including for empty arrays. A binding page includes a literal descriptor for optional root
`description`. Name/base/current-head live in the ordinary session header.

A row has exact scientific `ordinal`, current ordinal `path` and `properties`.
Binding rows also have `transforms: {path, total}`, never the complete pipeline.
Input properties are port/field/derived/length; head properties are
id/port/field/loss/weight/mask/sample_weight. Transform properties are
kind/scale/offset/length/value. Unknown properties are not silently deleted;
explicit raw draft inspection remains available and final validation rejects
unsupported TaskSpec grammar.

Descriptors use the existing dataset literal contract: key/path/present plus
complete JSON `text` or null with an explicitly incomplete `summary`. An absent
optional property differs from present JSON null. Large strings/objects are
not fully encoded merely to obtain a preview. A summary is not a replacement
value, complete stable port identity or editable partial JSON. Large values use
the existing same-session staged value editor. Paths are valid only at their exact
sequence; flush edits and reload/resolve current paths before structural moves
or opening a child editor.

At most six rows are returned; byte admission includes the actual next cursor.
Follow that cursor, not assumed increments. Empty end pages return no rows and
null cursor. A single unrepresentable card fails explicitly rather than clipping
identity. Malformed draft containers explain that raw inspection is needed
for repair; they are not silently replaced with empty arrays.

## Literal ordered edits and compatibility

`operations` has 1..25 entries. Each has the exact grammar below:

| op | Other properties |
| --- | --- |
| `put-json` | `path`, `key`, `text` |
| `replace-json` | `path`, `text` |
| `insert-json` | `path`, `index`, `text` |
| `remove` | `path` |
| `move` | `path`, `from`, `to` |
| `property` | `target`, `key`, `text` |

`property` resolves the key in the current document after each preceding
operation. Its closed target grammar is `{kind: "document"}` for name/description;
`{kind: "binding", collection, binding}` for that collection's listed properties;
or `{kind: "transform", collection, binding, transform}` for listed transform
properties. All ordinals are existing exact nonnegative integers, not booleans.
Only the documented keys are allowed. `text: null` removes the optional named
property (a no-op if absent); the JSON text token `"null"` sets a present null.
Stable document ID/version cannot be changed. Named targets are not permanent
binding IDs: structure operations flush prior queues and refresh paths/order.

Literal text is complete strict finite JSON, at most 2000 code points. It is
parsed natively, retaining integer tokens such as 9007199254740993 without a
JavaScript Number round trip. Duplicate keys, NaN and exponent overflow fail
without publishing any part of the batch. Bigger values use staged editing.
Paths/index/key/move semantics match the existing owned ordinal patch protocol,
not RFC 6902. Operations use the same atomic session publication contract; all
operations/post-batch identity/header/admission must pass before publication.
Sequence advances once per accepted batch. Temporary schema-invalid drafts can
remain local; strict Check/Commit are explicit.

Compatibility checks the current owned task, captured browsing head and exact
model/dataset references, without sending the task document across the bridge.
A pending staged value, changed origin, admission refusal or intrinsic invalid
task fails. Valid TaskSpec/interface mismatches return explicit incompatible
reason, matching the legacy interface-only contract. There is no fitting,
execution, split regeneration, tensor allocation or execution RAM permission.

## Persistence, costs and evidence

No `.vtw` tables/grammar or CLI tokens are added. Stateless CLI import/validation
uses the same complete TaskSpec/registry. Qt session tokens are not agent-side
persistence or a second editor process.

Full base loading/validation, touched-array copies, registry dependency checks,
history and eventual admission remain native costs. Wire/card bounds are not
lazy storage, universal RAM guarantees, durable recovery or full-form acceptance.

Ten stdlib tests cover literal/ordinal/SQL/RNG purity, 80 binding/1000 transform
pages, hidden descriptions, byte-shortened cursor coverage, exact integer/staged
parent semantics, ordered rollback/stale moves, historical base/head and strict
admission, malformed cards/limits/owner/origin/buffer/RAM fences, no-fit exact
compatibility and closed literal edit batches. Extended actual sandboxed native
QWebChannel checks all four commands, literal integer precision, unknown/duplicate
requests, distinct owners, stale sequences, unchanged SQL/history and owner
teardown, plus actual 80-binding/1000-transform reads and pending-terminal edit
refusal. This is reusable TaskSpec transport, not an 80-port model compilation.
That native transport checkpoint passed 508 Python/156 JavaScript tests.
Three further stdlib cases cover named-property shifts/rollback/closed targets
and actual native origin proofs. Twenty actual-controller tests cover bounded
reads, focus/literal precision, 32 ordered edits split at 25, invalid correction,
failed batches/acknowledgements, structural reload fences, independent pipelines,
actual cursor/back retry, wrong/stale replies and same-session child return.
`tests/qt_task_sessions_smoke.py` exercises the actual form with 80 bindings and
1000 steps, historical large description staging, precision, injected initial/
pipeline/child-return failures, one history entry/undo/redo/save-reload and
font14/18/28. This reusable task fixture does not compile or fit an 80-port model.
Current exact test outcomes and limits are recorded in [development.md](development.md).

Next: experiment forms/local axes/cards, recovery/completion/full host costs,
figures and remaining C integration before unchanged D/E. Bounded task editing
is not universal RAM/whole-form acceptance.
