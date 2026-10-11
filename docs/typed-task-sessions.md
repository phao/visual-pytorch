# Owned task card/transform transport — local Stage C backend checkpoint

The native Qt bridge now supports bounded literal binding and transform pages
inside an existing owned task session, exact literal edit batches and an owned
compatibility check. **The actual typed task form has not migrated yet.** It still
uses the complete legacy draft path; these commands do not complete GUI/resource
migration. Experiment forms remain unmigrated too.

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
`collection`, `offset`, `limit`, `total`, `rows`, `next_offset`,
`byte_limit: 65536`, `text_limit: 256`. Scope is `owned-task-page` for bindings
or `owned-task-transform-page` for transforms. A transform page also carries
`binding`. A binding page includes a literal descriptor for optional root
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

Literal text is complete strict finite JSON, at most 2000 code points. It is
parsed natively, retaining integer tokens such as 9007199254740993 without a
JavaScript Number round trip. Duplicate keys, NaN and exponent overflow fail
without publishing any part of the batch. Bigger values use staged editing.
Paths/index/key/move semantics match the existing owned ordinal patch protocol,
not RFC 6902. Translations feed the same atomic session patch method; all
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
The final suite passes 508 Python/156 JavaScript tests; full details are in
[development.md](development.md).

Next is actual task-form/controller migration, then experiment forms, local
axes/cards and remaining C acceptance before unchanged D/E. Native transport
tests do not prove the visible form already uses it.
