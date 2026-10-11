# Owned experiment transport — local Stage C backend checkpoint

The native owned-session service now provides experiment selection cards, named
literal edits, current-property lookup, local selection choices and a projected
prospective preview. **The actual experiment form has not migrated yet.** It
still uses the complete legacy draft path. Native tests are not visible-form
acceptance or Stage C completion. Dataset/task forms already have separate
owned-session implementations.

## Closed native commands

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
question, selection.kind. Descriptor path is null when an intermediate/leaf is
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

Closed targets are {kind:"document"} for name/question/selection.kind, or
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
These are native transport tests, not actual experiment-form migration.
Current full checks are recorded in [development.md](development.md).

Next connect the actual form: bounded local axes/cards, independent local and
stored exact-choice controllers, literal queues/focus/invalid text preservation,
flush/current lookup/linked staging, actual preview projection consumption,
owner/generation/sequence/retry fences and existing eight-fit/all-split/history
gates. Then remaining C resource/figure/window/integration work and unchanged
D/E. No full Stage C or universal RAM/crash/detached-safety claim follows here.
