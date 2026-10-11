# Authoring revision choices — retained Stage C branch

Task and experiment authoring now share `app/web/revision-choice.js`, using
`scienceBrowse` and the same read-only metadata projection as `spec-page`.
They no longer fetch `scienceIndex` and repeat every stored revision in every
selector. This is not yet in the published initial prerelease.

## Interaction and identities

Each stored-choice selector has its own literal name/ID search, previous/next
buttons and visible actual page range. Pages contain at most 25 metadata rows,
with at most one exact pin outside the page/query and one Choose explicitly
option. Byte-shortened pages follow the server's actual next offset; previous
stores the actual visited cursor, never assumes exactly 25 returned rows.
Alphabetical registry browsing does not sort the draft's execution selections.

Refresh, search and paging never advance an existing selected revision to a
new browsing head. Missing exact pins remain selected with an Unavailable label;
the user must choose another deliberately. This uses `allow_missing_pin: true`,
whose `missing_pin` reference is separate from actual metadata rows. Default
strict main browsing and CLI reads still fail on a missing pin unless requested.
See [scientific-browser.md](scientific-browser.md) for shared exact schemas.

Task model/dataset compatibility contexts retain their existing initial default:
the first returned current head when no selection exists, only on that picker's
first load. If no head is in the initial page, the choice remains explicit.
Clearing a selection later is not undone by refresh. These are suggestions,
not dependencies stored in TaskSpec or automatic execution.

Experiment architecture/dataset/task selections always begin incomplete, not
guessed from list position. Selectors for pair whitelists/exclusions still refer
to the draft's explicitly selected architectures in their existing order;
dataset/task pairing-ID choices also use the draft. These local choices and axis
cards are NOT paged by this checkpoint, unlike remote stored revision choices.

Refreshing suggestions does not rebuild other controls, discard unfinished or
invalid text, renumber IDs, reorder selections or admit a draft. Existing
validation, no-op/history, optimistic exact base/current head and explicit Save
workspace rules are unchanged. Complete large-draft/long-cell editing remains
the next owned authoring transport task, not secretly solved by these pickers.

## Delivery, controls and layout

A picker captures workspace UUID and owner validity, fences read success AND
failure by generation, and ignores detached/disposed form replies. It checks
returned scope/kind/query/offset and pin identity before rendering. Pending
navigation is disabled AND guarded against duplicate same-turn clicks. Busy,
workspace-changed and invalid-owner controls cannot issue user page requests.
Current errors remain visible; stale errors do not overwrite the newer form.
Task context errors also use their existing selection/workspace-generation fence.

An externalReload notification invalidates/re-requests visible authoring
suggestions even with an unchanged UUID, clears old task compatibility/context
and experiment draft previews, and retains draft text/exact references. This is
notification-scoped refresh fencing, not complete ABA/file-save/durable isolation.

Native selects retain their one-control labels. Search/paging buttons live in a
sibling group with explicit accessible names; they are not extra labelable
descendants inside that select label. A compact grid keeps arrow buttons on the
same line, with range/status beneath and themed inherited base fonts. Select
tooltips contain complete exact ID/hash when narrow labels truncate visually.
Disposal restores the original labelled select structure and invalidates replies.
The accepted floating-panel/keyboard/typing/undo design remains unchanged.

This structure follows the primary [HTML label content model](https://html.spec.whatwg.org/multipage/forms.html#the-label-element):
the labelled select is the label's only labelable descendant. Search and paging
remain separately named controls rather than acquiring that select's caption.

## CLI and costs

```sh
python vpw_cli.py lab.vtw spec-page model --query encoder --offset 0 --limit 25
python vpw_cli.py lab.vtw spec-page dataset --pin DATASET_ID DATASET_SHA256
python vpw_cli.py lab.vtw spec-page task --pin MISSING_ID TASK_SHA256 --allow-missing-pin
```

Metadata pages do not fetch document columns. Compact ASCII JSON projections
remain <=65536 bytes; CLI/bridge envelopes are not included. No new SQLite
schema, persistent cache, alternate draft database or numerical-worker call.
This does not bound all source validation/SQLite costs, whole draft transport,
task field/port contexts, arbitrary metadata, local draft choices/cards or
recovery/completion payloads. Do not describe this as lazy dataset storage or a
universal memory guarantee.

## Evidence and reproduction

Two additional Python tests cover strict versus explicit absent pins, no invented
replacement, boolean validation, exact CLI pin parity and unchanged file bytes.
Ten JavaScript interaction tests cover pending/duplicate navigation, stale success
and errors, disposal/workspace/owner changes, default versus explicit choices,
cursor/pin mismatch, labels and safe literal metadata.

`tests/qt_revision_choices_smoke.py` uses a private workspace with 112 retained
revisions each of model/dataset/task, real search/page/historical pins, incomplete
task/question text, same-UUID reload and dark/GitHub/Math fonts 14/18/28. It asserts
unchanged SQL, no full metadata-index call, no run and no host Torch. Existing
task/experiment Qt suites additionally exercise actual multihead/ragged/eight-run
and split-protocol fits and exact/conflict/history behavior. Metadata-unavailable
negative controls explicitly mock the page provider; they do not claim that
retained bytes were deleted. Current totals/results are in `development.md`.
