# Internal parameter-policy foundation — not yet workspace semantics

`app.parameter_policy.resolve(schema, policy, aliases=...)` is an internal pure
normalizer, not a persisted graph field or automatic compile/train hook. Existing
workspace versions, compilers, optimizers, exports and GUI behavior are unchanged.
This foundation does not deliver `lora.vtw` or exact neural MS-VAR.

## Internal numerical application checkpoint

`graph_aliases(models, mid)` now inventories concrete graph occurrence aliases
by traversing reference and sharing scopes structurally. It does not split IDs.
Canonical and occurrence slash-spelling collisions refuse instead of silently
guessing ownership. This is not a standalone-family schema adapter: those use
their own exact prefixes/component IDs. Nothing calls the helper automatically.

`app.parameter_execution.configure(module, schema, policy, aliases=...)` applies
the explicit internal policy to a freshly constructed, memory-admitted CPU
float32 model before optimizer construction. It requires exact unique owner
coverage, registered paths/shapes, ordinary contiguous leaf Parameters and no
existing gradients. Native tied-object aliases count once; different objects
sharing storage refuse. All initial values/receipt fields prepare before mutation,
with rollback on application failure. Frozen weights use `requires_grad_(False)`;
forward operations are not detached or put in `no_grad`.

Private per-owner generators use the declared seed literally. Equal seeds/shapes
can produce equal values; generator order/native paths do not own meaning. The
actual Torch version and materialized owner-value digest are recorded. Float32
rounding, not exact real arithmetic, defines realized values; std/bounds collapsing
in that representation refuse. No global RNG is consumed by configuration.
Constructor draws occurred earlier and are captured, not reconstructed from a
policy seed. Receipt is initial parameters only, not optimizer continuation.

`trainable_parameters(module)` selects unique trainable objects and refuses an
empty training selection; all-frozen inference still works. The normal trainers
do not yet call it. Frozen values still consume storage and can require activation
memory for input/adapter gradients; trainable counts are not total RAM estimates.

Seventeen pure policy cases and14 native numerical cases cover these additions;
106 focused cases pass in2.062s. Six bounded LoRA fits (ranks1/2/4, ordinary and
temporal inputs,200 updates) match an independent PyTorch expression, preserve
frozen base values bitwise and have adapter-only Adam state. There is still no
persisted policy grammar, automatic graph compile/train hook, full regression,
native/family/export/worker/notebook/CLI policy parity or delivered LoRA workspace.
An initial overflow fixture used a finite seed7 draw; changing the fixture to
seed0 exercises the intended rejection. A focused command typo named a missing
test module; the corrected106-case log is the acceptance evidence, not the failed
command. Raw evidence is retained privately in the owning campaign.

Input `schema` is the existing `parameter_schema` path-to-`{id,shape}` mapping.
Stable IDs identify unique owners; paths are separately recorded execution
addresses. Duplicate/ambiguous owner IDs refuse. Caller-supplied aliases must
map directly to known canonical owners, without cycles or shadowing another
owner. Automatic graph alias discovery remains part of integration, not this API.

Optional policy has exactly integer `version:1` and an object `parameters`, keyed
by canonical or declared alias IDs. Each entry has optional strict Boolean
`trainable` (default true) and `initializer` (default constructor). Omitted owners
retain those defaults. Equal normalized alias declarations coalesce; conflicts
refuse independent of insertion order. There is no implicit fuzzy-label policy.

Initializer vocabulary:

- `{"kind":"constructor"}` or `{"kind":"zeros"}`.
- `{"kind":"values","dtype":"float32","values":...}` with exact nested
  shape and finite float32-range numeric leaves; Boolean/string leaves refuse.
- `{"kind":"normal","seed":7,"mean":0,"std":0.05}` with positive std.
- `{"kind":"uniform","seed":7,"low":-0.1,"high":0.1}` with ordered
  float32-range bounds/width.

Seeds are integers0..2147483647. Numeric initializer fields/leaves normalize to
Python float, so equal integer/float alias declarations cannot make identity
depend on dictionary order. This is declaration normalization, not float32
materialization or random drawing. No RNG, tensor allocation, file write or
Torch/NumPy/Qt import occurs. Returned values are detached private copies.

Output contains all owners with ID/path/shape/dtype/normalized policy, separate
stored/trainable/frozen element counts and a semantic declaration digest. The
digest excludes execution paths, includes shapes/values/trainability, and is not
an executed initial-state fingerprint or a training checkpoint.

Fifteen new pure tests cover defaults,256 frozen/64 trainable LoRA declarations,
bias selection, all-frozen counts, alias equivalence/conflicts, identity stability,
malformed fields/types/versions/shapes/seeds/values, schema ambiguity and fresh
no-numerical-import discovery.74 focused foundation/arithmetic/discovery/history/
revision checks pass. No new numerical LoRA fit, full frozen regression or
GUI/worker/export/notebook policy support is claimed.

Next: graph owner/alias resolution, actual initializer and selected-optimizer
behavior with independent values/gradients/fits; then explicit persisted version/
identity fences, native/core/family/worker/export/notebook/CLI/Qt route parity.
Opening the future policy must not reinterpret old ignored `trainable:false`
metadata, cut frozen-branch input gradients or confuse a new initialization with
optimizer-state resume. Exact retained base references and derived merged
inference remain required by the full LoRA request.

Primary references:
[PyTorch autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html),
[Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html),
[LoRA paper](https://arxiv.org/html/2106.09685#S4.SS1).
Documentation2.14 informs the intended behavior; actual execution must be tested
against the installed runtime separately, not inferred from these pages.
