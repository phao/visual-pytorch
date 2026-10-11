# Scoped Module constructors — family version 4

This is the executable contract for the local Stage D checkpoint in
`app/families.py`, `app/family_records.py`, `app/family_drafts.py` and their Qt/CLI
consumers. It extends [family specialization](architecture-families.md) and
[typed scalar choices](constructor-choices.md); it is not runtime recurrence or
a Python constructor-expression interpreter. Old family v1–v3 resolution paths,
hashes and receipt shapes remain unchanged.

## Why local bindings matter

A reusable template can have a local `width` symbol. Two parent Module blocks
can call that template with different widths, activations or bias settings.
Same-named symbols in parent/child scopes are not equal unless explicitly bound.
A child with required symbols and no defaults is executable if each actual call
supplies those arguments. No meaningless global default is required.

The runnable [scoped-branches.json](../examples/families/scoped-branches.json)
shares the input `[B, 2]` between two calls of one SLP template. The left call
resolves to Linear(2,3,bias=False) → Tanh, the right to
Linear(2,5,bias=True) → ReLU. Ordered Concat produces `[B,8]`, then
Linear(8,1,bias=True) produces `[B,1]`. This is 30 trainable parameters:
6 left + 15 right + 9 head. The input/output port identities remain stable.

## Closed grammar

Family v4 preserves all v3 required fields and adds one OPTIONAL `constructors`
array; absent means no explicit call bindings. It contains at most 512 records,
each with exactly `model`, `node`, `symbol`, `value`:

```json
{"model":"parent", "node":"left", "symbol":"child-width",
 "value":{"symbol":"left-width"}}
```

`model` names an existing parent template. `node` identifies a Module block in
that template, before finite-repeat expansion. `symbol` identifies a symbol
whose scope is exactly that Module's directly referenced child template. The
triple (model,node,symbol) must be unique. Cross-scope name guessing, binding a
grandchild directly, arbitrary node annotations and unknown fields fail.

For an integer child symbol, `value` uses the existing bounded integer JSON AST:
literal integer, local symbol reference, add/mul/sub/div/floordiv. All referenced
symbols must be integer symbols in the PARENT scope. Bounds, exact division,
positive denominators, intermediate integer range, depth24/node256/arity limits
are unchanged. A boolean/enum child accepts a matching JSON literal or a matching
typed parent-local symbol reference. No float, coercion, bool-as-int, Python text,
function call or data-dependent inference is accepted.

Target-domain checks happen after evaluation. A parent domain may be broader than
its child's: out-of-child-domain points remain incompatible, not clamped or
filtered. Symbols must have assignment, repeat-count or constructor-expression
dependents; unused declarations remain errors. Merely being a child target does
not make an otherwise unused symbol meaningful.

## Binding and resolution procedure

1. Validate the full finite concrete seed closure and family grammar.
2. Validate every explicitly supplied fallback, including values later overridden.
   Build canonical fallbacks from supplied values and all available defaults.
3. At a template call, for each local symbol choose the explicit call argument
   first, otherwise its global stable-ID fallback, otherwise its declared default.
   Defaults have already been included in the fallback map. No argument means
   unresolved, even if a same-named parent symbol happens to exist.
4. Apply local field assignments and bounded finite repeats. Expanded Module
   occurrences inherit their original template-node constructor annotations.
5. Evaluate child call expressions using this instance's local values. Descend
   into each direct child and replace refs with its resolved concrete model ID.
6. Validate concrete shapes, routing and explicit sharing; return the concrete
   closure, shape/parameter contract, architecture hash and provenance receipt.

Preview preserves `verified`, `conditional`, `incompatible`. A conditional result
names the missing stable symbol IDs, scope, readable names and first encountered
constructor path, such as `parent/left`. Resolving that path may expose further
missing calls; this is not an exhaustive enumeration of all missing contexts.
Malformed grammar/wrong supplied scalar types raise rather than being mistaken
for a selectable incompatible numerical point. No numerical runtime is imported
for preview, specialization or source generation.

Memoization keys contain template identity and ALL effective local values.
Equal contexts reuse a concrete MODEL SPEC, not a parameter object. Every ordinary
Module call still constructs independent weights. Only explicit local
`share_with` or finite tied-repeat mode reuses a registered module. Different
resolved child refs cannot share: incompatible constructors fail, never become
fresh independent copies to hide a sharing error.

The resolver allows at most 64 DISTINCT resolved constructor contexts, reserving
each before copying its graph. Graphs retain the existing 64-block limit; there
are at most64 templates/128 symbols/512 assignments/32 finite chains. Variant
fan-out may reach the context limit even with a small source closure; an error
names the call path and asks for reduced fan-out. These are current prototype
bounds, not a universal memory guarantee or the user's RAM safety setting.

## Identity, receipts and durable provenance

Family-v4 resolution returns receipt `version:2`. `bindings` is the canonical
available fallback map; required child symbols satisfied exclusively by calls
need not appear there. `model_ids` maps each template ID to an ordered array of
resolved model IDs. `instances` is an ordered array of exactly:

```json
{"model":"family-<SHA256>", "template":"child",
 "bindings":{"child-width":3,"child-activation":"Tanh","child-bias":false}}
```

There is one row per distinct resolved context, in child-before-parent traversal
order; repeated callers of equal contexts do not duplicate rows. These are NOT
parameter-instance records. Parameter ownership remains in the concrete graph
and existing checkpoint schema. `occurrences` names expanded nodes using resolved
model IDs and adds their original template ID. Root/models/contract/architecture
hash remain present. Family v1–v3 retain receipt v1 and template→single-ID maps.

Specialization v3 REQUIRED fields are version,id,name,family,bindings,model,
instances, with only the verified optional revision. Instances must have unique
model IDs, 1..64 closed context records and bounded typed local binding maps.
Admission/load/save re-specialize the exact pinned family using retained
fallbacks and compare canonical fallbacks, every context and the entire frozen
model document. Missing defaults, forged local values/IDs/scopes/rows or a
grammar downgrade fail even if output shape/parameter count would match.

Family v4 MUST pair with specialization v3. Family v3 remains specialization v2;
family v1/v2 remains specialization v1. Experiment/run v3 already carries exact
typed family coordinates and can select v4; it does not need a new version.
Concrete ModelRevision versions and portable runtime closure remain unchanged.
`.vtw` container11 gates family v4/specialization v3 without new tables. Earlier
headers cannot contain them; opening older valid files upgrades only the private
working copy until explicit Save. Old document identities are not rewritten.

## Qt, CLI and agent collaboration

Owned family cards add “Scoped Module constructor bindings”, paged six at a time.
An absent optional array reads as an empty projection, not an implicit draft
mutation. Add constructor explicitly upgrades the family to v4 and inserts the
first card in ONE owned edit batch; old experiments are not upgraded to v4.
Cards expose model/node/symbol and exact JSON value, with staged large-value
editing, retained invalid text and existing sequence/workspace/history fences.
An insertion page must match the complete owner, sequence and cursor; a late page
after a workspace change cannot mutate the new working copy.

Native preview includes the distinct context count and a bounded descriptor of
resolved contexts; large values remain summarized, not shipped unboundedly.
Ordinary instance creation materializes child-before-parent concrete drawings;
Module navigation opens the actual resolved child. `vpw_cli.py preview-family`,
`specialize-family` and atomic `apply` use the same resolver and records, not
agent-only schema shortcuts. New families can be seeded from drawings and staged
or edited through the existing owned cards. No hidden SQLite draft tables.

## Verification and limits

Pure tests cover missing arguments, same-name scopes, scalar forwarding, nested
derived expressions, defaults/overrides, invalid targets/types/domains, sharing,
finite repeated Modules, 65th-context pre-copy refusal, canonical provenance,
atomic rollback, owned editor upgrades, `.vtw` gates, CLI and experiment axes.
Independent numerical tests cover branched outputs and input/parameter gradients,
native/generated parity, independent/shared optimizer state, worker/checkpoint/
retry/fresh predictor and actual Jupyter reruns. Native sandboxed Qt checks edit
and explicit-upgrade paths, read-only preview, retention, Save/reload and no host
Torch import. See tests/test_scoped_constructors.py, test_scoped_execution.py and
qt_scoped_constructors_smoke.py. This is bounded local evidence, not broad OS or
hostile-file certification.

Runtime-variable spatial axes/guards, recurrent state/length/masks/reset, dynamic
loops are NOT implemented by this checkpoint. A subsequent
[checked family export](checked-family-export.md) now emits standalone PyTorch
classes with constructor guards and semantic warm starts, alongside concrete
export. Full Stage D and Stage E remain unfinished.

Primary reference: [PyTorch Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html),
particularly registered children and duplicate module/parameter traversal. This
informed the spec-versus-object distinction; independent local tests establish
the actual behavior of the installed runtime rather than assuming doc-version
parity. No PyTorch dependency was upgraded.
