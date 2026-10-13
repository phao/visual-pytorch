# ParameterDerivation v1: explicit LoRA inference merge

Container26 admits the independent `parameter-derivation` kind in existing
immutable `spec_revisions` and mutable `spec_heads`. No SQL table, graph operator
or initializer meaning is added. Any retained derivation, including an archived
one with no selected head, requires container26. Policy and binding introduction
fences remain24 and25; deferred21/22 containers remain rejected.

The closed record has `version`, `id`, `name`, `operation`, `sources`,
`destination`, `optimizer_resume: false` and optional verified `revision`.

- `operation`: `kind: lora-merge`, finite positive nonboolean `alpha ≤ 1e6`,
  positive integer `rank`, exact `scale = alpha/rank` and
  `arithmetic: binary32-sequential-products-sum-scale-add-v1`.
- `sources`: exactly `base`, `A`, `B`. Each is the same closed retained-source
  descriptor used by ParameterBinding1: exact run/model revisions, explicit
  attempt and checkpoint hash, native path plus semantic owner ID, shape/dtype,
  shape-domain-bound value hash, value format, actual status/update count and
  no optimizer resumption.
- `destination`: exact frozen model reference, relative component selector and
  absolute owner ID for its `linear1` weight, matrix shape, float32 dtype, exact
  value hash and value format.

All three sources must be distinct owners from one actual checkpoint. Dependency
validation rereads their values and full descriptors, proves the complete source
topology, recomputes the merge, and checks its frozen output model and bytes. A
shape-compatible arbitrary triple of tensors is not sufficient.

The v1 source must have exactly one Input, one Output, three bias-free Linear
nodes, one Scale and one ordered Add. Its only paths are Input→base→Add and
Input→A→B→Scale→Add, then Add→Output. Base is frozen; A/B are trainable. The
Scale coefficient must equal alpha/rank. There is no hidden nonlinear factor
path, additional owner or ported routing. Native owner identities are components,
not ambiguous slash-separated paths. This deliberately excludes nested MLP
merging; each supported flat adapted layer can be merged independently.

Output is a new three-node bias-free Linear inference map with exact value
initialization and frozen ownership. Stable input/output IDs, sample shapes and
axes are preserved. It is not silently substituted into the original drawing or
training run. Existing model IDs refuse. Atomic admission saves drawing, frozen
model and receipt together, rolls back on failure and does not modify source
evidence. Receipt dependencies retain exact source run, source model and output
model revisions even if browsing heads move.

Arithmetic rounds every scalar product and sequential sum to binary32, then
rounds scale multiplication and base addition; nonfinite/overflow values refuse.
The local v1 operation refuses more than one million scalar products before
output allocation. Different GEMM kernels may round differently: equivalence of
merged and unmerged inference is approximate, not promised bitwise.

## CLI and browsing

`preview-lora-merge SOURCES.json --alpha 4 --id NEW_ID --name NAME` is read-only.
The atomic edit operation is `derive-lora-merge`, with `sources` and optional
`alpha`, `id`, `name`. Both use the shared service, not a CLI-specific kernel.

The scientific revision browser includes Parameter derivations, permitting
bounded metadata, summary and exact tree inspection. That does not imply a new
graph-editing merge wizard. A distinct portable exact-base+adapter export is
specified in [lora-inference-package.md](lora-inference-package.md).
