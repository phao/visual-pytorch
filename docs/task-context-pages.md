# Exact task field/port suggestions — local Stage C checkpoint

The actual Qt task form uses three independent bounded lists: dataset fields,
model input ports, model output ports. The same pure backend serves the agent
CLI. This replaces complete `scienceTaskContext` payloads on the actual GUI
path; that legacy read remains supported. Task/experiment draft editing has
**not** migrated to owned sessions.

## Interaction and identities

Each collapsed suggestions section has literal name/ID search, optional exact
stable-ID inspection, previous/next controls and an actual range. Pages contain
at most 25 rows, shortened by the byte budget. Follow returned cursors, never
assumed increments. Failed backward reads retain retry history. An inspected ID
appears independently of query/page, or is explicitly unavailable; it is never
substituted with the first option or a similarly named ID.

Options populate existing datalists. Binding controls retain actual manual
values, including incomplete/invalid text. Searching/refreshing never rerenders
bindings, parses them, sorts execution order or admits a task. Suggestions sort
by casefolded display name then ID and retain original scientific `ordinal`.
Browsing sort is not tensor packing order. Same exact-context refresh preserves
filters/inspection/cursor; selecting different revisions resets filters.
Stored-revision selectors retain their own pins and initial defaults.

Descriptors omit observations, complete vocabularies, unit/modality/coordinates
metadata and weights. Categories report only the fixed vocabulary count.
Shape/axes previews have at most four entries, complete flags and actual rank.
Incomplete shapes visibly say **preview, rank N**; a preview is not an execution
shape. Full specification/tree inspection remains explicit. Complete IDs and
explicit name previews use literal text/option values, never innerHTML. Names
have at most 128 code points with a complete flag; shortened names visibly say
name preview. Existing non-string JSON field names fall back to stable IDs for
suggestions without changing stored metadata.

Read success/error is fenced by current form/context generation, workspace UUID
and validity callback. Returned exact hashes, collection/query/cursor/pin,
scope/version and workspace are checked before replacement. Duplicate pending
navigation and busy/invalid-owner user reads are guarded and disabled. A notified
same-UUID reload clears old suggestions before new reads; old replies cannot
overwrite newer controls/notices. This is editor fencing, not general ABA,
durable isolation or compatibility validation.

## Closed native request

`scienceTaskContextPage` requires exactly `action`, `workspace_id`, `model`,
`dataset`, `collection`; optional `query`, `offset`, `limit`, `pin`. Unsupported
properties fail. References are exact closed kind/ID/revision objects, not heads.
Workspace UUID must match the native working copy. Compact ASCII JSON requests/
projections are at most 65536 bytes, excluding the bridge envelope. It is a read,
including during execution; the actual GUI disables context controls while busy.

Collection is `fields`, `input_ports` or `output_ports`. Query defaults to empty,
has at most 200 code points and no NUL. Offset defaults to 0, exact nonnegative
integer; limit defaults to 25, 1..25. Booleans are not integers. Pin is null/
omitted or a valid complete stable ID. Matching casefolds name or ID. Absent exact
model/dataset revisions fail rather than choose a head; absent member pins are
returned explicitly unavailable.

## Response DTO

`version: 1`, `scope: task-context-page`, `workspace_id`, exact model/dataset,
collection/query/offset/limit, `total` (matches), `source_total` (whole collection),
`record_count`, `rows`, `pinned`, `missing_pin`, `requested_pin`, actual
`next_offset` or null, `shape_limit: 4`, `name_limit: 128`, `byte_limit: 65536`. End/search-empty
pages may have no rows and null next offset.

A row has complete id, bounded name/name_complete/name_source (`name` or `id`),
zero-based scientific ordinal, dtype, shape preview,
shape_rank/shape_complete, axes preview or null, axes_complete, vocabulary_count
(category fields only, otherwise null). Port dtype defaults to float32. Preview
arrays are bounded copies. Pinned is an independently resolved row or null;
missing_pin is the exact unavailable ID or null. No observation/vocabulary values
or arbitrary metadata are returned. This DTO is not a persistable DatasetSpec,
full model contract or execution binding. An unrepresentable single row/pin fails
instead of truncating identity. Next-cursor encoding is included in byte checks.

## CLI and persistent format

```sh
python vpw_cli.py lab.vtw task-context-page fields \
  --model MODEL_ID MODEL_SHA256 --dataset DATASET_ID DATASET_SHA256 \
  --query temperature --limit 25 --pin EXACT_FIELD_ID
```

`input_ports`/`output_ports` use the same command. Offset/limit/query/pin map
directly to the pure backend. CLI envelopes are outside the projection limit.
The command is read-only: no save, fitting or worker. No new `.vtw` schema/kind/
token or scientific semantics; TaskSpec IDs/ordered bindings remain unchanged.

## Costs and evidence

Wire/DOM bounds are not host cost bounds. Exact registry reads decode/validate
full model/dataset documents; port pages additionally compute the pure model
interface. Filtering/sorting scans full native collections, and independent
pages may repeat those costs. No lazy storage/cache/RAM guarantee is implied.
Final task compatibility still validates actual full selected interfaces.

Eight backend tests cover order/SQL/RNG purity, hidden large vocabulary/metadata,
partial shape/axes, historical refs, search/pins/missing IDs, Unicode shortened
cursor coverage, arbitrary JSON/long name compatibility, grammar/limits and actual
CLI/file parity. Eight asynchronous
tests cover independent pages/pins/cursors, retry/pending/blocked controls,
wrong identities, invalidated same-UUID contexts, stale success/error, preserved
filters and literal preview labels. Native large-context and task-fit evidence
is recorded after verification in [development.md](development.md).

Task/experiment complete drafts/cards/local axes, recovery/completion/inline
host costs, figures/C integrated acceptance and unchanged D/E remain open.
