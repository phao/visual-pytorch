# Owned experiment editor — retained local Stage C checkpoint

The family execution selector offers Runtime sequence T (v1) and Runtime sequence
T with LSTM (v2). Selecting them queues explicit experiment version5 or6 before
the exact execution property; older concrete/spatial choices require4 and never
downgrade an acknowledged/queued newer version. The visible exact version control
accepts1–6. Owned sequence/workspace/head/job/child-editor fences remain unchanged.
See [temporal-task-execution.md](temporal-task-execution.md) and
[retained runtime-v2 experiments](runtime-temporal-v2-experiments.md).

The native owned-session service now provides experiment selection cards, named
literal edits, current-property lookup, local selection choices and a projected
prospective preview. The actual experiment form now uses this service for Open,
ordered editing, preview, Check and Commit; it no longer transfers a complete
ExperimentSpec in those requests. The legacy complete CLI/bridge authoring APIs
remain deliberate compatibility paths. This is not full Stage C completion or
universal native resource convergence. Dataset/task forms use the same ownership
and staged-child machinery.

## Actual typed form

`app/web/experiment-session-form.js` is mounted after dataset/task forms in the
real scientific panel. Only one editor owns a session at a time. It has independent
six-card pages for models, dataset/task pairings, splits, initialization seeds,
training configurations, whitelist pairs and exclusions. Back history follows
returned byte-shortened cursors, not a guessed stride. Structure actions flush
queued literal edits, read current paths, mutate and reload cards. Failed initial
or post-acceptance reads retain the session and fence admission until Retry.
Parallel reads settle before releasing pending guards; failed Back retains its
navigation history. No automatic axis sorting or omitted-entry deletion occurs.

The complete supported CPU/Adam controls remain: optimizer constants, batch size,
evaluation cadence, retained history, all four independent budget bounds, shuffle
and its seed, protocol initialization seed and the overriding initialization
axis. Seed cards preserve explicit integer order; append/remove/reorder replace
the former complete-array text control. Blank budget fields set JSON null, not
the string "null" or omitted/moved properties. Integer controls retain native
literal tokens beyond JavaScript Number's exact range. Decimal integers are
explicit, without floating exponent conversion; seeds keep their declared range.
Row/group/temporal/external split fields, actual role IDs, comparison modes,
pair whitelists and exclusion reasons remain editable. Mode changes never erase
existing conflicting entries. IDs retain literal spaces/commas and duplicate IDs
remain available to the authoritative validator, not silently deduplicated.

Focused dirty text is captured before disabling controls. Invalid fields and
failed native batches retain their text; only correction replaces that property's
queued operation. Batches have at most 25 operations/65536 compact ASCII bytes.
Atomicity is per batch, not a multi-batch form transaction. Questions/reasons/
role arrays too large for typed descriptors have labelled partial previews and
explicit same-session raw children. Current Value/path lookup follows a flush;
child Finish/Discard adopts only the same parent's advanced/unchanged header,
then reloads pages without retaining/discarding the entire experiment. Entry and
complete-experiment JSON repair are also explicit staged actions. Values, staged
files and session tokens do not add a second persisted format.

Stored revision choices keep exact immutable pins, without defaulting an experiment
to a browsing head. `experiment-local-choice.js` independently pages/searches
selected models and pairing IDs. Reference key order is normalized for dropdown
identity only; retained reference bytes are not rewritten on refresh. Pairing
dropdown values are literal IDs, not JSON-quoted strings. Missing/duplicate/
incomplete choices remain visible; refresh never coerces them. Queries, cursors,
session/sequence/workspace/selection and controller generation fence late results
and errors. Tool groups live outside their select's label and have accessible
search/navigation labels. Invalid non-model selections are not sent as model pins.

Preview uses the exact projections below, displaying explicit partial reasons,
available role counts and decimal-string totals without run buttons. Edits invalidate
previous previews. Paging follows actual cursors and retained Back history.
Wrong/stale page/edit/check/preview/Commit results cannot replace current controls
or select a foreign identity. Returning UUID alone does not resume editing: native
Origin must prove the original actual owner/Workspace/connection/sequence. Notified
reload stays stale. These are transient fences, not durable recovery/general ABA.

## Closed native commands

Version-2 family choices reuse these commands and ownership fences. Add family
selection explicitly upgrades the draft to version 2; historical version-2 edits
preserve that version. Family model cards expose id, family, bindings and axes;
they also expose optional execution. Explicitly choosing either concrete baseline
or runtime image H/W upgrades to version4 before the named policy edit; omitted
policy stays concrete without any automatic write. Acknowledgement updates the
visible version and controller; later edits cannot silently downgrade it. See
[runtime-task-execution.md](runtime-task-execution.md) for exact semantics.
fixed/axis JSON stays raw exact text with staged large-value repair. Root version
is a named property used for the explicit upgrade, not a new persistent field.
See [family-experiment-axes.md](family-experiment-axes.md) for grammar, coordinates,
conditional points, paired-family editing and terminal/native/CLI behavior.

All requests require `action`; unsupported keys fail. Compact ASCII JSON
requests/result projections are at most 65536 bytes, excluding the existing
bridge envelope. Owner, document sequence, native origin, active buffer and
constrained draft admission follow [owned-authoring-sessions.md](owned-authoring-sessions.md).
No new persisted kind/table, second draft store or CLI token is added.

| Action | Other required properties | Optional properties | Result |
| --- | --- | --- | --- |
| scienceDraftExperimentPage | session, sequence, collection | offset, limit | Ordered entry/property cards |
| scienceDraftExperimentValue | session, sequence, target, key | none | Current named property descriptor |
| scienceDraftExperimentChoices | session, sequence, collection | offset, limit, query, pin | Local selection metadata/pin page |
| scienceDraftExperimentPreview | session, sequence | offset, limit | Exact prospective candidates without manifests/protocols |
| scienceDraftExperimentEdit | session, sequence, operations | none | Advanced ordinary session header |

Card/value/choice reads retain unsaved inspection after a workspace switch and
during a job. Edit/Preview require the original actual Workspace/connection/UUID;
Preview is non-numerical and allowed during a job, while Edit is blocked until
the job and terminal delivery are finished. Edit/Preview require no active staged
buffer and current constrained draft admission. That estimate is not execution
RAM permission. Check/Commit keep the ordinary complete validators/head/history.

## Ordered cards and descriptors

`collection` is models, data_tasks, splits, initialization_seeds, protocols,
pairs or exclusions. The latter two resolve under selection. Offset defaults to
0, exact nonnegative integer; card limit defaults to 6, exact integer 1..6.
Booleans are not integers. Arrays retain scientific order, never alphabetical.

Card DTO fields: version1, scope owned-experiment-page, session, sequence,
workspace_id, experiment id, collection, offset, limit, total, path, rows,
next_offset, byte_limit65536, text_limit256, reference_text_limit8192 and roots.
Each row has ordinal, current absolute ordinal path, and properties. Each
property has key, path, present, text and summary. Root properties are name,
question, selection.kind and version. Version edits permit monotonic explicit
upgrades through v4, never downgrade of an owned draft. Descriptor path is null
when an intermediate/leaf is
missing; missing does not mean a present JSON null. A complete `text` is literal
strict JSON, not a JavaScript Number. Otherwise an explicit type/size summary
marks an incomplete value; it is not editable partial JSON.

Reference properties use an 8192-character conservative eligibility budget so
valid exact references with long Unicode IDs remain complete selector values.
Other fields use 256 characters. These are bounded eligibility walks, not full
encoding of huge questions, external role arrays or arbitrary hidden values.
The complete card byte budget may shorten a page; follow next_offset. Empty
end pages have no rows/null cursor. An unrepresentable first card fails explicitly.
Malformed containers require explicit raw repair, not silent replacement.

The supported named entry keys and their literal paths are:

| Collection | key → path relative to its entry |
| --- | --- |
| models | reference → entry itself |
| data_tasks | id, dataset, task → same-named member |
| splits | id → id; method/seed/field/excluded → protocol.member; fractions.0/.1/.2 → protocol.fractions[index]; roles.train/.validation/.test → protocol.roles.member |
| initialization_seeds | value → entry itself |
| protocols | id → id; batch_size/evaluate_every/shuffle → protocol.member; history.max_points → protocol.history.max_points; budget.epochs/.updates/.examples/.seconds → protocol.budget.member; optimizer.lr/.eps/.weight_decay → protocol.optimizer.member; optimizer.betas.0/.1 → protocol.optimizer.betas[index]; seeds.initialization/.shuffle → protocol.seeds.member |
| pairs | model, data_task → same-named member |
| exclusions | model, data_task, reason → same-named member |

Fixed backend/optimizer kind/augmentation/checkpoint/test policies are not new
typed editable fields. Existing explicit raw editing and final closed grammar
remain; silently changing policy or deleting incompatible entries is forbidden.

Value lookup returns version1/scope owned-experiment-value plus the common
session/sequence/workspace/id/byte_limit, exact target/key and property descriptor.
Flush pending edits before lookup and opening a same-session raw child; resolve
paths again after changes. Child Finish/Discard returns verified advanced/
unchanged parent headers, never implicitly admits/discards the experiment.

## Literal ordered edits

Each batch contains 1..25 operations. Structural put-json/replace-json/
insert-json/remove/move use the [task literal operation grammar](typed-task-sessions.md)
and current ordinal paths; this is not RFC6902. An additional property operation
has exactly op:"property", target, key, text.

Closed targets are {kind:"document"} for name/question/selection.kind/version, or
{kind:"entry",collection,ordinal} for listed entry keys. Ordinals are existing
exact nonnegative integers. Named paths resolve against the current draft after
each preceding operation, not dictionary insertion positions captured earlier.
Structure changes must flush queued named edits and refresh order/paths; ordinals
are not permanent IDs.

Literal text is complete finite strict JSON of at most 2000 code points, retaining
integer tokens without a Number round trip. Duplicate/nonfinite/overflow literals
fail. text:null removes an optional named dictionary property; the literal text
"null" sets a present null. Array/whole-entry removal uses explicit structural
Remove, not a null property shorthand. Missing/malformed intermediate containers
require raw repair; the service does not invent them. Failed operation numbers
are reported as Experiment edit #N. All operations/identity/header/admission must
pass before publication; failures preserve bytes/sequence. Successful publication
advances once. Multi-batch atomicity is not a whole-form transaction. Final
Check/Commit reject temporary invalid scientific combinations.

## Independent local choices

Only selected models and data_tasks are choice collections. Defaults: offset0,
limit25 (1..25); query is literal casefold search of at most 200 characters/no
NUL. Model searches include full revision hashes. pin is an exact model reference
or stable pairing ID, independently retained outside search/page. Native choice
reads do not consult mutable registry heads, fit, mutate or rewrite selections.

DTO scope owned-experiment-choices includes common session/header fields,
collection, query, offset, limit, total, rows, next_offset, pin, pinned,
missing_pin and ambiguous_pin. Each row has original scientific ordinal, value
(exact reference/pairing ID or null), label and valid. Incomplete entries remain
visible as incomplete, not dropped or coerced into choices. Duplicate exact
pins are explicit via ambiguous_pin; final validation still rejects duplicates.
Alphabetic metadata browsing retains original ordinal and never sorts stored
axes. Pins do not silently advance or substitute an unavailable identity.
Follow actual byte-shortened cursors. Current host implementation scans/sorts
local choices; wire bounds do not make that work lazy.

## Prospective preview without large manifests

Preview retains the existing check/dependency/candidate semantics, Cartesian
scientific order, incompatibilities/exclusions and no-fit/no-training contract.
Default offset0/limit25 (1..25), exact integers. Full manifest generation and
dependency/base validation are native costs; they are not bounded by this DTO.

DTO scope owned-experiment-preview has common session fields, exact prospective
experiment reference, offset, limit, total_combinations_exact (decimal string),
candidates, next_offset and execution:"no-fitting-or-training". It deliberately
does not send a numeric Cartesian total that might round in JavaScript.
Each candidate has exactly id, coordinates, status, reason, references and counts.
Reason is null or an explicit 128-code-point text projection with offset/text/
total/next_offset, not a purported complete reason when more text exists.
Counts are null without a manifest, otherwise exact train/validation/test/
excluded observation counts. Unavailable counts are not zero-valued successes.
No full manifest, role IDs, training protocol, tensors or fitted values cross
this preview channel. Partial reasons require deliberate full inspection/export.
Byte-shortened pages preserve actual candidate order/cursor; unrepresentable
first candidates fail. Existing explicit complete CLI previews/exports remain.

## Evidence and remaining work

Fifteen stdlib tests cover closed cards/targets/nested paths, hidden20k role IDs/
question text, cursor shortening, exact literals and rollback, shared child
staging, alphabetic local choices/exact and missing/duplicate pins/incomplete
entries, legacy candidate equality/counts/SQL/RNG purity, projected reasons/
large decimal totals and owner/origin/sequence/buffer/RAM/head/limit guards.
Additional cursor cases exhaust byte-shortened local choices and actual candidate
previews without skipped/duplicated entries, retaining independent outside pins.
A staged raw document preserves absent-local exact model references with
200-character supplementary-Unicode IDs across model/pair/exclusion cards and
local pins; Check rejects unavailable dependencies rather than replacing them.

Extended real sandboxed QWebChannel exercises all five commands with 80 pairings,
1000 seeds and a 160000-coordinate experiment, closed requests/distinct owners,
literal9007199254740993, failed atomic batches/stale sequences, staged-value and
pending-terminal guards, unchanged SQL/history before explicit admission, exact
old evidence retention and owner teardown. No host Torch/console errors.
These are native transport tests; separate actual-form evidence follows.
Current full checks are recorded in [development.md](development.md).

Twenty-eight controller/choice cases verify the actual implementation's ordered
pages, focus/precision/invalid text, batch splitting, fresh structural paths,
failed-read retry/settlement, linked child adoption, exact decimal previews,
comparison conflicts, wrong/stale responses, independent pins/key normalization,
owner/origin/job/editor exclusivity and late-error suppression.

`qt_experiment_sessions_smoke.py` opens a historical exact experiment with 80-entry
axes, 1000 seeds, 20000-code-point question and large reasons. Actual sandboxed
controls exercise independent pages/search/outside pins, literal precision,
same-session child Finish/Discard, injected initial/Back/child-return failures,
exact retained bytes, one history entry/undo/redo/save-reload and fonts14/18/28.
Its 163 projections peak39205 bytes; capture inspected. Preview shows 1024000000
declared coordinates while resolving only its current page, with fitting/dispatch
forbidden. This is a large reusable selection editor, not billion-run execution,
full-window wire bounds or universal host-memory proof. The actual fit regression
separately checks eight candidate fits, all split methods, full optimizer/seeds,
historical choices, exclusions/incompatibilities, 96-coordinate preview, invalid
text, no-op/history/head/workspace/pending guards and reload.

Actual recovery/completion webview projections now have separate bounds in
[recovery-browser.md](recovery-browser.md). Remaining C native inline/retention
costs, figures/window geometry and
integration work still precede unchanged D/E. No full Stage C or universal RAM/
crash/detached-safety claim follows here.
