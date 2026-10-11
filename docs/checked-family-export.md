# Checked standalone architecture families

This Stage D capability exports a retained family as a standalone Python source
file containing a real `torch.nn.Module` class. It extends
[scoped constructors](scoped-constructors.md) and all earlier family grammars,
not the fixed-instance or saved-checkpoint predictor exporters. It does NOT train
at export time or turn a family into a saved inference checkpoint.
Explicit runtime task predictors now reuse this checked constructor rather than
a second lowering; their distinct execution policy is specified in
[runtime-task-execution.md](runtime-task-execution.md).

## Qt and CLI

In Experiments & history, select an EXACT retained architecture-family revision
and choose **Export checked family…** beside Create concrete instance. Export is
disabled while a scientific draft/current action is active. Retain a draft first;
the exported family is the retained revision, not unsaved card text, a newer head,
an instance-binding textarea, or a mutable drawing. Instance bindings are chosen
later when constructing the exported Python class.

The CLI uses the same service:

```sh
python vpw_cli.py workspace.vtw export-family FAMILY_ID REVISION_SHA256 \
  --output architecture_family.py
```

`--memory-limit-percent` explicitly overrides the read-only user preference. The
workspace's constrained flag and user percentage are captured in the source.
Invalid percentages fail closed; the CLI does not save a preference or workspace.
The destination must be NEW; neither an existing file nor a final symlink is
overwritten/followed. Qt captures exact validated source before its modal picker,
then rejects a workspace/UUID/active-job change before writing. Cancellation
writes nothing. The native return is a compact path/hash/provenance receipt, not
the ~100KB Python source sent over the webview bridge. Source-generation/native
host cost remains distinct from the bounded bridge payload.

## Using the exported module

Install a compatible PyTorch in the consuming Python environment. Keep the
exported file on its ordinary import path, independently of the Visual PyTorch
repository:

```python
import architecture_family as family

print(family.preview_family({"left-width": 3, "right-width": 5}))
model = family.ArchitectureFamily(bindings={"left-width": 3, "right-width": 5})
y = model(x)
```

Constructor signature:

```python
ArchitectureFamily(*, bindings=None, constrained=None, memory_limit_percent=None,
                   runtime_inputs=None, initial_shapes=None, runtime_temporal=None)
```

`bindings` maps stable symbol IDs to exact integers, booleans or enum strings.
Omitted values follow the captured family's defaults/direct-child call bindings;
required unresolved root/child symbols fail with actionable paths. Explicit call
overrides retain family-v4 semantics. No symbol is inferred from tensor data.
None policy arguments use the captured policy. Explicit `constrained=False` is
an ordinary caller choice, not a hidden exporter bypass.

Import defines the class; it does NOT create a `model` instance, allocate its
parameters, or consume initialization RNG. It imports Torch as a dependency.
`preview_family` checks contracts without constructing layers or consuming Torch
RNG. Actual construction resolves and validates the family, concrete routing,
shapes, types, domains, finite-repeat/context limits and sharing, then performs
the existing whole-model current-RAM preflight BEFORE creating any layer.
Numerical construction is CPU float32 regardless of process default dtype/device.
Ordinary subsequent `.to(...)`, `.double()`, `.train()`, `.eval()`, optimizers and
PyTorch module calls are available to the consuming Python user; this checkpoint
does not certify every accelerator or `torch.compile`/ONNX/tracing backend.

The class registers one `network` child. State keys are `network.layer_...`, while
`model.network.state_dict()` retains the ordinary concrete compiler's keys.
Ordinary family instances have independent parameters. A resolved graph's
explicit share_with/tied mode still reuses one registered module internally.
`forward` invokes `self.network(x)` so normal module hooks apply at this boundary.
Tensor/ordered named-input and output dictionary contracts match the concrete
compiler. `forward_tensor` and `intermediate_values` retain its packed-output and
diagnostic conventions, not a separate diagram interpretation.

Optional `runtime_inputs` explicitly enables the checked H/W domain described in
[runtime spatial contracts](runtime-spatial-contracts.md). It defaults to the
ordinary fixed-shape path for family v1–4 or v5 without a declaration. For v5
with runtime_inputs, omitted/None uses that exact captured declaration; an explicit
map instead supplies a separately checked instance override recorded in provenance.
To export a fixed v5 domain, retain a version without the optional declaration;
None does not silently remove a stored policy. No input data infers constructor bindings. Runtime
specification/shape previews and instance provenance retain the explicit domain.
Persisted declarations/typed GUI authoring are implemented in `.vtw`12. Task
datasets and worker/notebook fits remain concrete fixed-shape; merely capturing
the runtime module does not change their data contract. Warm starts transfer compatible weights, not a promised
runtime-domain continuation.

Family6 instead has required nullable `runtime_policy`. A nonnull policy defaults
the constructor's `runtime_temporal` to its exact retained semantic-T declaration.
An explicit map supplies a separately checked instance override; null policy is
fixed unless the caller explicitly opts in. Nonnull runtime_temporal on family1–5
is rejected. Spatial and temporal overrides cannot be selected together.
`initial_shapes` supplies an admitted first actual case, not new learned dimensions.
Recurrent transition parameter IDs use stable family/template/recurrent/node
identities, independent of baseline T and generated scoped model IDs. Compatible
strict warm starts transfer weights across those constructor points, not optimizer
state or hidden recurrent state. [Temporal families](temporal-families.md) gives
the complete retained grammar and native exact-specialization construction API.
Explicit [temporal experiments](temporal-task-execution.md) now reuse this same
checked constructor for their runtime architecture and saved predictor exports.

## Captured runnable closure and provenance

The source contains the exact family definition and an isolated eleven-module
runtime: activations, core, evidence, families, graph, memory, operators,
runtime_shapes, recurrent, recurrent_nodes and temporal_shapes. Capturing the
temporal helper preserves the optional compiler closure; family1–5 still reject
Recurrent and do not implicitly acquire a temporal policy. See
[runtime-temporal-contracts.md](runtime-temporal-contracts.md).
Core/evidence are selected AST declarations covering the reachable resolver,
contracts, compiler and semantic schema, not the Workspace/SQLite/training/UI
machinery. Other nine files are captured unchanged. No application installation,
Qt, database, dataset observations, sidecar package or executable user expression
is required. Resolution/compiler behavior is reused, not reimplemented from a
visually similar drawing. The private package is initialized lazily under a lock
and uses a fresh private namespace, never replacing an installed `app` package.
Failed initialization removes only its own module registrations.

Metadata records export_format1, operator_version1, exact family grammar,
family/root identity, container version and SHA-256 of the captured runtime
source map. The file byte hash includes this metadata, literal definition and
entry code. `export_metadata()`, `family_definition()`,
`ArchitectureFamily.constructor_specification()`, `model.provenance()`,
`model.contract` and `model.semantic_parameter_schema()` return private copies.
Instance provenance adds canonical bindings, scoped contexts, architecture hash
and actual constructor policy. Family/runtime byte changes are detected when
using captured data. Hashes are integrity/provenance, NOT authenticity, a sandbox
or permission to execute untrusted Python. Runtime executes only the captured
project source; typed family inputs never become Python expressions.

Original generated runtime/helper code is GPL-3.0-only; preserve notices and
provide the GPLv3 license when redistributing. Dependencies keep their own
licenses. Export is not a claim that generated helpers are license-exempt.

## Semantic warm starts

```python
source = family.ArchitectureFamily(bindings={"hidden-depth": 1})
# Train source explicitly, if desired.
snapshot = source.semantic_state()
target = family.ArchitectureFamily(bindings={"hidden-depth": 3})
receipt = target.warm_start(snapshot)  # explicit compatible partial transfer
```

This example uses a width/depth family, not the scoped-branches symbol names.
Snapshots are caller-owned dictionaries with version1, exact family reference,
runtime hash, canonical source bindings, semantic schema and cloned detached CPU
tensors. They contain no optimizer, dataset, random-state or training-resume
claim. No pickle loader or workspace persistence is introduced. A caller may
serialize these in their own Python workflow; do not deserialize untrusted pickle.

Parameter IDs are canonical JSON arrays of typed path components, e.g.:

```json
["family","scoped-branches","template","parent","call","left","owner","linear","parameter","weight"]
```

They retain family/root, Module call path, parameter-owner block and weight/bias
identity, with repeat occurrence IDs where applicable. A slash inside a node ID
is one component, not an accidental nested call. These IDs do not depend on
layer-registration ordinals or the whole-specialization hash. Shared aliases
have no second parameter entry. The constructor-state schema names each physical
state path, semantic ID and exact tensor shape.

Warm-start admission requires the same EXACT family revision and captured runtime,
then re-resolves canonical source bindings and reproduces its entire semantic
schema before inspecting weights. This prevents forged identity/shape maps from
masquerading as another constructor point. Current-RAM preflight applies to both
source/target contracts using the target instance's actual constructor policy,
before tensor finite checks/copies. This remains an admission heuristic, not an
OOM/whole-process memory guarantee. Every source tensor must be dense,
finite and match its declared shape. Matching semantic IDs with exact shapes
and dtypes are copied; missing/shape-or-dtype-incompatible/unused entries are
reported. Reordering source dictionary entries does not change meaning. A head
whose registration index moved after repeat expansion is matched by its semantic
ID, not whichever layer used to occupy that slot.

Default partial loading is explicit through `warm_start(...)`; `strict=True`
requires every target/source parameter to be compatible and fails before writes
otherwise. All eligibility/metadata/value checks happen before target writes.
This is not a rollback guarantee against a device failure during tensor copying.
Use this before creating an optimizer; it does not clear/repair an already-owned
external optimizer's state. Cross-family/revision/compiler migration, arbitrary
architecture-to-architecture transfer and training continuation remain separate
capabilities, not inferred by matching a few shapes.

## Verification and limits

tests/test_family_export.py checks read-only deterministic generation, all four
grammars, unresolved drafts, exact historical selection, policy, exclusive files,
actual CLI output and a fresh no-Torch/NumPy/Qt generation process. Its native
dispatch tests run under system Python/PyQt and exercise cancellation, modal
workspace changes, closed fields and bounded receipts. Tests requiring this Qt
installation are explicitly skipped in the separate numerical Python environment
and run there through the system interpreter, not reported as numerical passes.

tests/test_family_export_execution.py independently checks branched/Conv2d outputs
and input/parameter gradients; compares all four grammars with native compilation;
tests invalid constructors/RAM refusal before layer allocation, ownership/hooks/
Adam/input shapes, moved-head/partial/strict/dtype warm starts, source-schema and
nonfinite rejection, component escaping and private-copy behavior. A fresh
app-free process imports/previews without RNG mutation, performs3 updates and
strictly transfers state with equal prediction. The real Qt smoke tests select,
export and verify exact source bytes without host Torch or SQL writes.

The current finite graph/family/context bounds and CPU RAM heuristic remain.
Exported classes support checked runtime H/W, not arbitrary channel/rank changes,
ragged task datasets, recurrent state/sequence masks/reset policies or heavyweight
execution. D/E remain incomplete.

Primary reference: [PyTorch Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html),
especially registered submodules, forward calls/hooks and state dictionaries.
Local tests, not documentation version alone, verify the installed behavior.
