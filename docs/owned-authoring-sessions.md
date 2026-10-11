# Owned authoring sessions — retained Stage C backend/Qt checkpoint

`app/authoring_sessions.py` and the Qt bridge provide bounded draft transport
without sending complete drafts for Open/Check/Commit. **The existing typed
forms have not migrated to these commands.** This is a tested prerequisite,
not completion of resource convergence or a published feature.

## Ownership and identity

Each window owns an in-memory store; each actual bridge/tab has at most one
active session. A second Open fails until explicit Close or successful Commit,
never silently discarding unfinished work. The random 32-character hex `session`
token is not a persisted object ID or content revision. `sequence` begins at 0
and advances once per accepted patch batch, including an accepted no-op patch.
Read/Patch/Check/Commit require the current exact sequence; Describe retrieves it.

`kind`, `id`, `workspace_id`, `base`, `expected_head` retain ordinary version-1
draft meaning. Historical base and captured current browsing head remain distinct.
The actual owner, Workspace object and SQLite connection are captured in Python,
not supplied by requests. Patch/Check/Commit require that same current object,
connection and UUID. A replacement working copy/connection with the same UUID
fails. Read/Describe/Close remain possible for the owner's retained bytes after
a switch, preserving unsaved inspection. Another tab cannot use the token.
Owner destruction discards its sessions; successful Commit closes the session.
These are trusted local editor controls, not authentication, durable crash
recovery, a general ABA detector or replacement for existing file Save conflicts.

## Closed request and response grammar

Every command requires `action`; unsupported top-level properties fail. Compact
ASCII JSON requests and result projections are at most 65536 bytes, excluding
the existing bridge envelope. The bridge rejects duplicate keys/nonfinite JSON
constants before new-session dispatch. Python requests are independently JSON
normalized/size-checked. Unknown tokens and stale sequences fail explicitly.

| Action | Other required properties | Optional properties | Result |
| --- | --- | --- | --- |
| `scienceDraftOpen` | `kind` | `reference`, `id` | Header, never `document` |
| `scienceDraftDescribe` | `session` | none | Current header |
| `scienceDraftRead` | `session`, `sequence` | `path`, `part`, `offset`, `limit` | Ordinal page |
| `scienceDraftPatch` | `session`, `sequence`, `operations` | none | New header/sequence |
| `scienceDraftCheck` | `session`, `sequence` | none | Prospective exact reference |
| `scienceDraftCommit` | `session`, `sequence` | none | Retained reference; session closes |
| `scienceDraftClose` | `session` | none | `session`, `status: discarded-unsaved-draft` |

Open accepts only dataset/task/experiment. `reference` selects exact immutable
base bytes and forbids a replacement `id`; without it, ordinary incomplete
template/new-ID checks apply. Headers have `version: 1`,
`scope: owned-authoring-session`, session/sequence, ordinary draft-envelope fields
except document, `name` (possibly null for incomplete edited drafts) and
`byte_limit: 65536`. No session/draft table is added to SQLite.

Read uses `scope: owned-draft-inspection-page` plus session/sequence, not immutable
reference. Other fields match [scientific-browser.md](scientific-browser.md):
at most 25 container entries, 128-code-point key/value previews, 2000-code-point
scalar chunks, exact returned offsets, `part: value|key`, paths of at most 1024
nonnegative exact ordinals. Dictionary order is the current draft insertion order;
array order is scientific order. Paths are valid only at their sequence. Keys
can be reconstructed in chunks and are not truncated into purported identities.

## Atomic ordered patches

`operations` has 1..25 entries applied sequentially. Each operation has exactly
the properties below. This custom ordinal protocol is **not RFC 6902 JSON Patch
or a partial scientific specification**.

| `op` | Other properties | Meaning |
| --- | --- | --- |
| `replace` | `path`, `value` | Replace an existing value; `[]` replaces the document |
| `remove` | `path` | Remove existing member/item; root removal forbidden |
| `put` | `path`, `key`, `value` | Set/add string-keyed property of an existing dictionary |
| `insert` | `path`, `index`, `value` | Insert in array at 0..length, including append |
| `move` | `path`, `from`, `to` | Move existing array item to final existing index |

Only touched containers are copied; untargeted subtrees/order remain unchanged.
Patch values are detached by JSON normalization. All operations, post-batch stable
document ID/version and response projection must pass before new bytes/sequence
are published. A failed second operation/oversized header rolls back the entire
batch. Scientific fields/records/ports/axes/transforms/selections are never sorted.

Temporary schema-invalid JSON-shaped drafts are permitted. Stable document ID/
version cannot change, and `revision` cannot be injected. Check/Commit run the
existing full grammar, dependency and captured-head checks; failure retains
bytes/sequence. Successful admission uses the existing history wrapper; unchanged
current content produces no SQL/dirty/history change. Undo restores heads while
retaining immutable evidence. Open/Patch/Commit are blocked during busy numerical
work, including queued terminal delivery; Read/Check/Describe/Close are non-numerical.

## Costs, lifecycle and unfinished migration

Wire bounds do not bound complete draft/SQL parsing, validation/dependencies,
touched-container copies, history snapshots, registry admission or RAM. Open
still validates/copies full base bytes on the host. There is no lazy storage,
parallel persistence, persistent cache, detached job or optimizer continuation.
Long values can be inspected in chunks; replacement of arbitrary large cells/raw
unfinished JSON still needs chunked value editing and explicit resource admission.
Do not truncate data/keys/ontology/text to meet the patch limit.

Next: large-value/resource safeguards and migration of actual typed forms onto
owner/sequence pages, preserving invalid local text and existing typed helpers;
paged field/port contexts and local selection cards; then in-app figures,
integrated C acceptance and unchanged D/E. Stateless CLI specification/import
commands continue to use the same registry: transient tab tokens are not `.vtw`
or CLI persistence features.

The bridge Python signal is now `notification` with Qt name `event`, preserving
JavaScript `host.event.connect` without shadowing native
[`QObject.event`](https://doc.qt.io/qt-6/qobject.html#event). The name parameter
was checked against installed PyQt6 `pyqtSignal.__doc__` and the actual channel,
not inferred from another binding. Direct bridge DeferredDelete reproduced
`native Qt signal is not callable`; both direct and view/tab teardown now release
sessions normally. This fixes a concrete lifecycle defect, not all shutdown risks.

## Evidence

Nine new stdlib tests cover incomplete templates, historical pages/long keys/
values/order, atomic rollback/detached values, ownership, malformed/missing/stale
sequences, forged identity, unchanged SQL/RNG, no-op admission and actual same-UUID
workspace/connection replacement. Thirty-five focused authoring/browser tests pass.

`tests/qt_authoring_sessions_smoke.py` exercises actual sandboxed QWebChannel with
1600 observations, bounded pages/scalars, distinct owners, duplicate/unknown-key
rejection, pending-job guard, pure check, undo/redo/no-op/retained old evidence,
save/reload, same-UUID replacement and direct bridge/view destruction. Channel
notification name delivery is tested; no host Torch or user state is involved.
This is transport/lifecycle evidence, not migrated-form or whole-memory acceptance.
