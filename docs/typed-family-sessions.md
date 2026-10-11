# Owned architecture family editor — local Stage D checkpoint

The existing Experiments & history page now has **New architecture family**.
It captures the current valid drawing's complete reachable concrete closure
without editing its drawing, dataset, learned results or weights. The family
starts with that root/templates and empty symbols/assignments/repeats.
**Edit selected revision** opens an exact retained family, including historical
revisions, using the existing owner/sequence/origin/head safeguards.

This is executable typed family authoring, not full D acceptance or publication.
The grammar is [architecture-families.md](architecture-families.md). Independent
and explicitly [tied finite chains](shared-parameters.md) are implemented;
runtime recurrence is not finite copies. [Explicit experiment axes](family-experiment-axes.md)
are implemented, not automatically inferred from symbol ranges.

## User workflow

1. Draw a valid architecture; open Experiments & history and create a family.
2. Concrete templates list stable model IDs, names and graph descriptors. The
   root refers to one of those IDs.
3. Add symbols with stable IDs, model-local scopes, readable names, optional
   integer defaults/bounds or version-3 boolean/finite enum choices. Type selectors
   are alphabetic; defaults and enum values use literal JSON. Changing type does
   not silently remove incompatible bounds/defaults. A blank default means unresolved, not zero.
4. Add assignments to existing template node fields. Values are integers,
   expression ASTs or arrays of expressions for shapes/pairs. Examples:
   {"symbol":"width-id"} or {"op":"mul","args":[2,{"symbol":"width-id"}]}.
   Version3 also permits existing constructor fields such as bias and activation,
   with matching literals or symbol references; see [constructor-choices.md](constructor-choices.md).
   No Python text is executed.
5. Add finite repeats: stable repeat ID, model ID, ordered node-ID array,
   count expression and explicit mode "independent" or "tied". Selecting tied
   explicitly upgrades a v1 owned family document to version2 in the same edit
   batch, preserving v3 when already selected. Type/domain edits explicitly
   upgrade to v3. All owned version downgrades are denied. Current unary-chain
   ingress/egress/no-crossing constraints still apply.
6. Preview with a JSON stable-symbol-ID binding map. Omitted values use defaults.
   Conditional/incompatible results explain missing/invalid choices. Preview
   never allocates a numerical model, fits transforms or trains.
7. Retain the family revision. It can remain unbound; it is not an executable
   instance. Normal workspace Save persists it to disk.
8. Select an exact retained family and create an instance with explicit bindings.
   **Add editable drawings** distinguishes materialization from registry-only
   retention. This uses atomic family/specialization/frozen-model admission and
   refuses implicit replacement. Open the drawing through Architectures;
   training remains explicit.

Name/root and small scalar/expression controls are typed cards. Template IDs
cannot be renamed separately from dictionary keys. Graph/large values use the
owned staged JSON editor; only the parent retains the family. Complete JSON is
an explicit escape hatch, not the normal wire payload. Arrays preserve order;
remove/move are explicit, never automatic sorting. Integer text is sent verbatim
so JavaScript Number rounding cannot silently alter a literal. Invalid text
remains visible and blocks admission.

## Native transport

scienceDraftOpen accepts optional model, only for a new kind "family"; it cannot
combine a drawing seed with an exact historical base or another kind. Without
seed/base it creates an incomplete draft for explicit staged repair. Headers
contain ownership metadata, not complete source/model closures. Drawing capture
reads only reachable architecture documents, not datasets or unrelated models.
Owned admission checks their accumulated source-size estimate before each
document is retrieved/parsed, then checks the complete family draft. Stored
legacy result fields are not template content; native SQL/parsing costs remain
distinct from the small wire projection.

All family draft requests require exact session/sequence identities and share
the 65536-byte compact-ASCII JSON request boundary:

| Action | Required fields besides action | Optional |
| --- | --- | --- |
| scienceDraftFamilyPage | session, sequence, collection | offset, limit |
| scienceDraftFamilyEdit | session, sequence, operations | none |
| scienceDraftFamilyPreview | session, sequence | bindings |

Collections: templates, symbols, assignments, repeats, constructors. The optional
constructors array is available in family v4/v5/v6; absence reads as an empty collection with
null path and no draft mutation. Add constructor explicitly upgrades to v4 and
inserts in one owned batch. Each card exposes parent model, scoped reference node, direct
child symbol and typed JSON value. See [scoped-constructors.md](scoped-constructors.md).
Pages require nonnegative
exact offsets, 1..6 cards and explicit totals/next offsets. Responses pin
workspace/object/session/sequence/collection/cursor with scope owned-family-page.
Scalar descriptors return complete text or a large summary/path, never clipped
JSON masquerading as a complete value. Root properties are name/root/version/runtime_inputs/runtime_policy;
version uses exact numeric text. Optional runtime_inputs uses a JSON textarea:
nonblank JSON queues explicit version5 first, blank removes the field without
downgrading, and invalid text stays visible while blocking admission. Its keys
are stable root Input IDs and values are H/W sample axes1/2, not constructor
bindings. See [runtime-spatial-contracts.md](runtime-spatial-contracts.md).
The family6 `runtime_policy` JSON textarea explicitly queues version6 before
setting a nonblank value. Blank text retains JSON null, rather than removing this
required field. Its only nonnull grammar is version1/kind runtime-temporal-v1
with stable root Input IDs mapped to sample axis [0]. Invalid JSON remains visible
and blocks actions. Older `runtime_inputs` is not silently converted or removed;
an explicit migration must remove it and acknowledge the new policy. A drawing
with reachable Recurrent seeds family6 with runtime_policy:null, never inferred
dynamic time. Family6 constructor cards may target Recurrent's direct child scope.
See [temporal-families.md](temporal-families.md).
Template
graph/name may be edited explicitly; template ID is read-only.

Edits are 1..25 ordered operations, published once after all succeed: property
with target/key/raw JSON text (null removes the property), or insert/remove/move
on symbols/assignments/repeats/constructors. Targets are document or collection+ordinal under
the exact sequence. Text is at most2000 characters; larger values use staging.
Each operation has closed fields. Structural template identity/closure changes
require staged JSON or a drawing seed, never an implicit reference rewrite.

Preview scope is owned-family-preview with exact owner/sequence. Verified
results include family/specialization/architecture identities, root, architecture/
block/occurrence/parameter counts and input/output shapes. Resolved bindings are
a complete descriptor or an explicit large summary plus binding_count.
Family v4/v5/v6 also returns constructor_contexts and a bounded resolved_contexts
descriptor, not an unbounded concrete closure. Add-card page replies are fenced
by exact owner/sequence/cursor and fresh grammar; workspace switches cannot insert
late cards into another draft.
Family v5/v6 additionally returns bounded runtime/runtime_specification descriptors
and runtime_status (conditional-per-call-guards or fixed-shape). Baseline verified
status does not mean every runtime resolution will execute. The displayed grammar
and controller grammar both follow acknowledged explicit upgrades.
Conditional missing lists show at most25 with missing_total/missing_next_offset;
use symbol cards for remaining IDs. Projections are not retained objects or full
receipts. Native validation/copying/expansion costs are NOT bounded by wire size;
current expansion/graph limits remain, with no lazy-storage claim.

Retention uses scienceDraftCheck/scienceDraftCommit and the immutable v7 registry.
Staging, head/origin/owner/sequence guards, memory admission, duplicate/nonfinite
JSON rejection and no-op retention match [owned sessions](owned-authoring-sessions.md).
Edits may be incomplete; explicit check/retention enforce the full grammar.

## Retained-instance action and CLI parity

scienceFamilyResolve requires action, workspace_id, exact family reference;
bindings and boolean materialize (default true) are optional. It is undo-framed
and blocked during active work. Strict JSON/closed fields and workspace UUID are
checked. Results return exact family/model/specialization refs, materialized
flag, root and parameter count, not a complete concrete closure. A failed
admission leaves no partial rows. Undo may remove heads/drawings; immutable
bytes remain available by exact revision. Missing heads are not lost provenance.

CLI uses the same persisted grammar: put-spec admits complete family definitions;
preview-family/preview-stored-family inspect bindings; specialize-family admits
an exact referenced or supplied family with identical binding/materialization
semantics. Agents may edit JSON and apply atomic CAS-protected batches; Qt reload
observes the retained revisions. Transient Qt tokens are not another SQLite or
CLI persistence format.

**Export checked family…** beside Create concrete instance exports the selected
exact retained family as a standalone guarded PyTorch class, without creating a
model, retaining a new object or using the instance-bindings textarea. It requires
a new file; cancellation/late workspace changes cannot claim completion. The
native source remains off the bridge. See [checked-family-export.md](checked-family-export.md).

## Controller and verification boundaries

The form uses the existing floating panel, palette tokens and menu spacing.
Labels/cards wrap without horizontal overflow. It checks exact headers, pages
and preview identities. Failed card reads lock typed fields/admission while
keeping Retry/complete staged repair available. Text survives rejected edits,
switches and reloads; stale origins cannot be admitted. Binding edits invalidate
previews. Only acknowledged retention closes the parent. Other scientific
editors cannot open a second draft.

tests/test_family_drafts.py covers seed capture, historical heads, ownership,
sequence/working-copy fences, atomic edits, integers/AST errors, staging, long
128-symbol projections and fresh standard-library no-Torch/NumPy/Qt operation.
tests/family-session-form.test.js covers ownership, queued text, invalid text,
stale acknowledgements/replies, repair paths and exact instance selection.
tests/qt_family_authoring_smoke.py uses actual controls/bridge: width3, two
independent Linear layers (21 parameters), no pre-retention SQL writes, bad-text
preservation, retained provenance, undo/redo and .vtw save/reload. Its layout
check is local/default-font evidence, not all-theme/platform or accessibility
certification. Original independent family tensor/gradient/fit/export and real
notebook suites remain required regressions.
