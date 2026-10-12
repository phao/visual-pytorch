# Internal parameter-policy foundation — not yet workspace semantics

`app.parameter_policy.resolve(schema, policy, aliases=...)` is an internal pure
normalizer, not a persisted graph field or automatic compile/train hook. Existing
workspace versions, compilers, optimizers, exports and GUI behavior are unchanged.
This foundation does not deliver `lora.vtw` or exact neural MS-VAR.

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
