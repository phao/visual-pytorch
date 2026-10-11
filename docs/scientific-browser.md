# Exact scientific browsing — Stage C

The scientific Experiment/history tab and `vpw_cli.py` share the read projections
in `app/science_browser.py`. They inspect retained v6 registry objects, not a
second database. None of these operations fits transforms, allocates a numerical
model, trains, generates a notebook or changes stored bytes/heads/history. This
workflow is locally integrated; it is not in the initial published prerelease.

Its **Edit this value…** control is a separate mutation workflow, not a read
action: it opens an owned draft for a non-root dataset/task/experiment value.
[chunked-draft-values.md](chunked-draft-values.md) specifies staging, explicit
acceptance, admission and retained-text semantics; browsing itself remains pure.

## Identity and limits

An exact reference has exactly `kind`, `id`, `revision`. Kind is one of `dataset`,
`experiment`, `model`, `report`, `run`, `task`; revision is a retained SHA-256.
Summary/inspection never interpret missing/null revisions as a browsing head.
Names label objects; they are not execution identities.

Each successful projection has `version: 1`, an explicit non-document `scope`,
and `byte_limit: 65536`. This limit is measured by compact JSON with
`ensure_ascii=True`, separators `(',', ':')` and finite values. It does NOT
include CLI indentation/envelopes, QWebChannel envelopes or error messages.
These projections cannot be admitted as partial specifications by `put-spec`.

Container/metadata requests use exact nonnegative integer `offset`, and exact
integer `limit` in 1..25. Booleans and integers beyond JavaScript's exact range
are rejected. Returned rows may be fewer than requested to fit the byte budget;
always follow the actual `next_offset`, not `offset + limit`. `null` means no
next page. A cursor at/past the end returns an empty page; it invents no values.

These are transport/rendering limits, NOT lazy storage or universal resource
admission. Summary/tree reads still validate the entire immutable document and
its dependency closure. Registry loading, inline JSON, SQLite count/sort/offset,
and visiting a deep dictionary member can have substantial memory/time cost.
No time deadline, external dataset adapter, detached job or OS isolation claim
is implied. Actual owned authoring and recovery/completion have separate scoped
bounds; see [recovery-browser.md](recovery-browser.md). These do not bound
native full validation/admission/history/storage costs.

## Metadata pages

`index_page(workspace, kind, offset=0, limit=25, query='', preferred=None,
heads_only=False, allow_missing_pin=False)` returns `scope: scientific-metadata-page`, `kind`, `query`,
`heads_only`, `offset`, `limit`, `total`, `rows`, `pinned`, `missing_pin`, `next_offset` and the
version/byte fields. Each row contains `kind`, `id`, `revision`, `name`,
`created_at`, `selected`. `selected` means this revision is the current browsing
head; historical/unselected revisions are still retained.

SQLite reads metadata columns only, not full document text. The alphabetical
order is `(name.casefold(), id, revision)`, implemented using a connection-local
collation/function. Query is a literal case-insensitive substring of name or ID,
at most 200 Unicode code points and no NUL, not SQL LIKE/glob/regex syntax.
`heads_only` is strictly boolean. No persistent index/schema migration is added.

The optional `preferred` exact reference is resolved independently of the query,
page and heads-only filter. It is returned as `pinned`, even if historical or
outside this page. It does not alter `total`, row offsets or selection in SQLite.
A nonexistent pin fails explicitly by default, never silently substituting another
object. With strictly boolean `allow_missing_pin=True`, it instead returns the
exact absent reference as `missing_pin`, with `pinned: null`. This is an explicit
unavailable selection, not a fabricated metadata row. The two fields cannot both
be non-null; both are null without a pin. Identities are still strictly validated.
The main read browser keeps strict missing-pin behavior. Authoring pickers request
the unavailable-marker variant to preserve incomplete choices honestly. The GUI
adds at most one pinned/unavailable option, plus a Choose explicitly option in
authoring. CLI `spec-page --pin ID REVISION [--allow-missing-pin]` uses the same
projection, independent of search/page/head filters.

## Typed summaries

`summary(workspace, reference)` returns `scope: exact-spec-summary`, exact
`reference`, `id`, `name`, `counts`, and version/byte fields, with these additions:

| Kind | Summary contents |
| --- | --- |
| dataset | Storage kind; counts of fields and observations; no records/cells |
| task | Counts of input bindings and loss heads; no full binding arrays |
| model | Root, architecture hash, reachable architecture/block counts |
| experiment | First 1024 question code points and text cursor; selection-axis counts |
| run | Exact selection, attempt count, latest number/stage/status/timestamps; error type and first 1024 message code points when present |
| report | Mode, first 1024 question code points/cursor, exact experiment or null, selected-run count |

Latest run status is not a replacement for selected-attempt evidence. Structured
run/report views still query their explicit retained evidence pages. No raw
history, weights, observations or manifests are eagerly copied into the summary.

## Complete contents, explicitly paged

`inspect_page(workspace, reference, path=[], part='value', offset=0, limit=25)`
returns `scope: exact-spec-inspection-page`, `reference`, `path`, `part`, `type`,
`offset`, `limit`, `total`, `entries`, `scalar`, `next_offset` and version/byte
fields. `type` is `object`, `array`, `string`, `number`, `boolean` or `null`.

`path` is a JSON array of at most 1024 exact nonnegative ordinals, not key strings.
An object ordinal selects a member in its immutable canonical stored key order;
an array ordinal selects the original scientific order. Thus `[]` is the root,
and `[3, 0]` enters root member 3, then its member/element 0. Read member labels
before choosing ordinals; do not assume illustrative paths fit another object.
Arrays of observations, features, ports, vocabulary, transforms, attempts and
experiment axes are NEVER alphabetically sorted.

`part: key` reads the key of the addressed dictionary member as a chunked string;
an array element/root has no key and is rejected. `part: value` is the default.
Keys are not embedded in paths, so extremely long keys do not inflate requests.

For a container, `total` is member count, `scalar` is null, and each entry has:

* `ordinal`: absolute member/element ordinal;
* `key`: null for arrays; otherwise `{offset, text, total, next_offset}` with
  at most 128 key code points, explicitly a preview;
* `preview`: `{type, size}` for containers, or `{type, value}` for scalar values,
  where `value` is the same text-cursor shape with at most 128 code points.

For a scalar, `entries` is empty; `scalar` contains `{offset, text, total,
next_offset}` with at most 2000 code points. `total`/offset count Unicode code
points, not bytes, UTF-16 units or grapheme clusters. `representation` is
`native-string` for stored strings/key text, or `json-token` for numbers,
booleans/null. Therefore false, null, numeric 0 and string `"0"` remain distinct.
No missing cell or unavailable metric is manufactured as numeric zero.

## GUI interaction and delivery

Browse/search uses a 25-row metadata page and optional pinned exact selection.
Refresh retains that identity rather than advancing to a newer head. Previous
stores actual visited offsets, including byte-shortened pages. Pending navigation
is disabled and duplicate same-turn actions are guarded.

The Specification/evidence disclosure initially shows a typed summary. Opening
it reads the root; Open value enters a member; Read full key appears for long
keys. Root, Parent and Contents previous/next navigate explicit pages. Parent
from key text opens the same member's value. Status labels include exact
revision, type, ordinal path and member/code-point range; previews are never
labelled complete JSON or presented as an editable specification.

All member labels, values and diagnostics use text nodes, not interpreted HTML.
Tables can scroll locally; scalar text wraps inside the panel and inherits the
base font/theme. The existing floating panel and keyboard/typing behavior remain.

New bridge reads are `scienceBrowse`, `scienceSummary`, `scienceInspect`, with
`workspace_id` required to match the current UUID. Their arguments match the
helper contracts above. Frontend responses/errors are fenced by captured
workspace and generation/selection. Tree replies additionally must match exact
reference/path/part/offset. Navigating clears old page content while reading.
An `externalReload` notification invalidates pending/content state even if UUID
is unchanged, then refreshes visible browsing without advancing the retained pin.
This is not a general ABA, multi-editor transaction or crash-safety guarantee.

## CLI and reproduction

```sh
python vpw_cli.py lab.vtw spec-page dataset --query measurements --offset 0 --limit 25
python vpw_cli.py lab.vtw spec-page run --heads-only
python vpw_cli.py lab.vtw spec-page dataset --query measurements --pin DATASET_ID DATASET_SHA256
python vpw_cli.py lab.vtw spec-page dataset --pin MISSING_ID DATASET_SHA256 --allow-missing-pin
python vpw_cli.py lab.vtw spec-summary dataset DATASET_ID DATASET_SHA256
python vpw_cli.py lab.vtw spec-inspect dataset DATASET_ID DATASET_SHA256 --path '[]'
python vpw_cli.py lab.vtw spec-inspect dataset DATASET_ID DATASET_SHA256 --path '[3,0]' --part key --offset 2000
```

All use the existing CLI envelope and read-only working copy. Complete
`spec-list`, `spec-show` and `report-data` remain available explicitly; those
commands are not bounded projections. No save or schema/version change occurs.

Backend/CLI tests cover all six kinds, exact revisions, hostile/long Unicode,
byte-shortened pages, original tensor/array order, key/scalar reconstruction,
literal search, invalid cursors, SQL/file/RNG isolation and a stdlib-only host.
Asynchronous JavaScript tests cover stale success/error delivery, unchanged-UUID
reload, response identity/cursor checks and rapid navigation. The actual Qt
fixture `tests/qt_science_browser_smoke.py` uses a private 1250 × 512 inline
dataset, historical/head revisions, large keys/values and dark/GitHub/Math themes
through font 28. See `development.md` for results and remaining acceptance work.
