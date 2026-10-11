# Runtime spatial contracts — Stage D retained family checkpoint

This is the executable contract of `app/runtime_shapes.py`, `app/graph.py` and
`app/core.py`, with retained family declarations and owned Qt/CLI authoring. The native compiler and
checked standalone family constructor can explicitly accept varying image H/W
without creating new weights. Family v5 and specialization v4 require `.vtw`12,
without new SQL tables. Ordinary diagrams, frozen ModelRevisions, worker tasks
and dataset bindings remain concrete. Retaining a runtime domain does not make
those fixed-shape execution paths dynamic. Recurrent state/length/mask/reset and
mixed composition remain subsequent D work; D/E are not complete.

## Retained declarations and exact provenance

Family v5 extends v4 with optional `runtime_inputs`, for example
`{"input": [1, 2]}`. Keys are root template Input node IDs, not labels or
constructor symbol IDs. Axes index sample dimensions, excluding B: 0=C, 1=H,
2=W. The field must be a nonempty map of valid CHW root inputs to nonempty,
unique subsets of `[1,2]`; omission means fixed shapes. Declaration order is
preserved in the family bytes; resolution sorts axes canonically. Earlier family
versions reject this field. v5 without it is explicitly legal and fixed-shape.

Concrete templates and symbols still define a baseline specialization. H/W
runtime axes are NOT additional constructor bindings and never inferred from
tensors. After ordinary scoped specialization, the allocation-free dependency
pass proves parameter independence. Unresolved or incompatible constructor points
remain visible, but cannot create a retained specialization. A valid declaration
with a fixed Flatten→Linear head can therefore be retained as a family draft;
resolution refuses it with the affected node and a reduction/binding repair.

The resolution receipt has version3 and includes `runtime` and
`runtime_specification`. The persisted specialization is version4, with exactly
the ordinary v3 fields plus mandatory `runtime`: null for fixed-shape v5, or
`{"version":1,"inputs":{...},"specification_sha256":"<64 lowercase hex>"}`.
The SHA-256 uses the existing canonical finite-JSON digest of the complete pure
runtime specification for that exact family/bindings/resolved context. The
specification describes a conditional domain, not a proof that all resolutions
execute. Admission/load/save reproduce and compare this witness alongside the
exact frozen model and constructor contexts. Forged axes, missing/null witnesses
or hashes fail atomically. Older grammar and receipt hashes remain unchanged.

No new table or runtime sidecar is introduced. The existing persistent `.vtw.lock`
still provides save synchronization. Older files upgrade only in memory until
explicit Save; an earlier container claiming v5 family/v4 specialization rows
is rejected. Undo may remove drawings and browsing heads, but exact historical
runtime provenance remains retrievable by revision.

In the owned family editor, `runtime_inputs` is a bounded JSON textarea with
an explanatory tooltip. Nonblank JSON explicitly queues version5 before the
property in the same batch. Blank removes the property, never downgrades the
grammar. Invalid text remains visible and blocks preview/admission. Acknowledged
upgrades update the displayed version and the controller's subsequent edits.
Preview exposes bounded runtime/specification descriptors and
`runtime_status: conditional-per-call-guards` or `fixed-shape`; `status: verified`
alone describes the baseline specialization, not every runtime case.

`preview-stored-family ID REVISION` and `export-family ID REVISION` use the same
exact retained source, never the mutable head. `examples/families/spatial-adaptive.json`
is a runnable default179-parameter grouped image encoder with adaptive reduction,
constructor C/d/K/g and explicit runtime H/W. C=6,d=12 gives375 parameters,
including both learned biases. CLI commands do not run tensors or modify SQL.

## Explicit construction and identity

```python
from app.core import compile_model

model = compile_model(
    models, root_id,
    constrained=True,
    memory_limit_percent=10,
    runtime_inputs={"input": [1, 2]},
)
# Input node ID, NOT its editable label. Sample axes omit batch B.
y = model(x)  # x: [B, fixed_C, actual_H, actual_W]
```

`runtime_inputs=None` retains the exact ordinary concrete path. A supplied
declaration must be a nonempty dictionary mapping existing root Input node IDs
to nonempty unique subsets of `[1, 2]`: H and/or W. Axis0 is C, not batch B.
The declared root sample shape must have rank3 and logical axes `[C,H,W]`
(omitted names have that initial interpretation). Channels, rank, undeclared
spatial axes and other input ports remain fixed. Batch must be a positive exact
integer. Sample dimensions and their product stay in the editor's exact-integer
range. Arbitrary runtime channel counts, general symbolic shapes, sequence length
and ragged within-batch samples are NOT supported by this contract.

The compiler constructs once on CPU float32 after ordinary whole-model admission.
Calling `.double()` explicitly supports CPU float64 inputs. Calls reject mixed
input/parameter dtypes, non-CPU devices, sparse tensors, wrong names/ranks/fixed
dimensions and mismatched batches before packing or executing the first layer.
The source's reachable graph closure and declaration are copied privately;
datasets, presentation view state and unrelated models are not captured. The
registered parameter/state-dict paths match the ordinary concrete compiler. No
layer, parameter or optimizer is recreated between resolutions. Explicit
share_with owners remain shared, including different resolutions through the
same nested encoder. Ordinary independent Module calls remain independent.

The baseline public `model.contract` remains the construction shape, unchanged
by calls. Use `model.runtime_specification()` for a private copy of the conditional
runtime domain and `model.preview_shapes(...)` for a private actual-case plan.
Neither mutates view/save status, source dictionaries, weights or RNG.

## Parameter-shape proof and actual-case guards

This is not an inference from a successful example batch. An allocation-free
data-flow pass propagates each runtime dimension's dependency through the current
operators. Linear's final input width and Conv2d's channel count must be statically
independent of runtime axes. Fixed Flatten feeding Linear is rejected at the
affected node with the repair: bind image resolution or add adaptive/spatial
reduction before Flatten. Unknown symbolic relations are not silently accepted.

Known rules are deliberately small:

- Activation preserves dimension dependencies, including Softmax's fixed axis.
- Conv2d/fixed pooling retain spatial dependencies; their normal concrete rules
  guard positivity, axes, kernel/stride/padding/dilation and groups per call.
- AdaptiveAvgPool2d replaces H/W with fixed output sizes and removes their
  dependencies; subsequent Flatten/Linear can therefore have fixed weights.
- Permute reorders both data dimensions and logical axis names. `[C,W,H]` cannot
  feed NCHW pooling/Conv merely because it still has rank3; use an explicit
  restoring adapter. Batch is never permuted.
- Flatten unions dependencies of merged dimensions. Reshape's inferred `-1`
  depends conservatively on input dimensions; literal dimensions are fixed.
  Actual product preservation/divisibility still must pass for every call.
- Ordered Concat propagates dependencies and checks equality on unjoined axes;
  no alphabetical reordering occurs. Splitter checks its declared last-axis sum;
  Replicator preserves data/alias identity and each named output boundary.
- Module propagates actual shapes/dependencies into its single tensor input.
  Runtime-dependent calls into a multiple-input Module are explicitly rejected:
  a packed vector alone cannot recover variable internal port shapes. Fixed
  multiple-input Module boundaries retain their old concrete behavior.
- Explicit Output shapes stay literal and are checked each call. A declaration
  is conditional, not a promise that every pair of positive H/W is accepted.

Root multi-input runtime calls require a dictionary of exact declared port names,
each containing its own tensor. Packing happens only after all ports pass
shape/dtype/batch/RAM checks. Flat packed input is rejected when variable port
shapes cannot be identified. Nested actual shape plans are per-call immutable
data; two shared calls can have different H/W without changing a layer's stored
contract. Dependency specs memoize by exact dimension-dependency context; actual
plans cache by exact input shapes. Each is bounded to64 distinct Module contexts.
The reachable source closure has at most64 checked model specs, with the existing
per-model64-node/128-edge limits. There is no unbounded cross-call plan cache.

Example: C=4, d=8, K=3, groups=2, Conv3×3 with padding1 → Tanh → adaptive1×1
→ Flatten → Linear has `8×2×3×3 + 8 + 3×8 + 3 = 179` learned parameters.
The same parameter objects accept 1×1,3×9,11×4 and6×7 in the regression oracle.
Replacing adaptive pooling with a fixed Flatten makes the Linear allocation
depend on H×W and therefore prevents this runtime declaration.

## Preview and memory admission

```python
preview = model.preview_shapes(
    {"input": [4, 17, 19]},  # every root Input ID, including fixed ports
    batch_size=3,
    gradients=True,
    dtype_bytes=8,
)
```

Preview is allocation-free and returns `status="verified-for-supplied-shapes"`,
the actual concrete operator/DAG contract, and a current-RAM safety report. That
status proves this supplied case only; `runtime_specification()` has
`status="conditional"` and lists the guards. Invalid shapes or admission fail
with a visible error. Native admission uses `torch.is_grad_enabled()` and actual
CPU float32/64 tensor metadata; it runs before `_pack_inputs`, `torch.cat`, reshape,
convolution, pooling or any other tensor computation in the network.

The version1 runtime-forward estimate charges the full actual batch, actual
activation/convolution-workspace sizes, all learned parameters, boundary-packing
copies, dtype size and fixed headroom. Gradient-enabled calls include conservative
Adam/copy headroom, not just weights. Double precision doubles size-dependent
components. Constrained execution blocks above the configured percentage of
current Linux/cgroup available RAM, defaults10%; equality is admitted. Missing
available RAM fails closed. Explicit unconstrained execution still checks shapes,
types and parameter invariants. This is a heuristic, NOT an allocator cap, OOM
guarantee, GPU budget, hostile-code sandbox, or accounting of arbitrary caller
optimizers/caches/concurrent allocations. Caller-created inputs already exist
when the network is invoked; use pure preview before allocating large inputs.

## Checked export and portable runtime

```python
import architecture_family as family

model = family.ArchitectureFamily(
    bindings={...},
    runtime_inputs={"input": [1, 2]},
)
```

Checked family exports capture the actual runtime_shapes module alongside the
resolver/compiler: eight isolated modules now. They expose the same optional
runtime constructor argument/spec/preview methods and record the caller's
declaration in instance provenance. Binding values still choose the concrete
construction example; input data does not infer constructor symbols. Baseline
architecture hashes retain their concrete meaning. Semantic warm starts transfer
only compatible weights within the exact same family/runtime, not a runtime-domain
or optimizer continuation claim. The snapshot remains its existing closed format.

New notebook bundles capture25 runtime files, including runtime_shapes.py. The
verifier preserves earlier complete sets only when captured source does not
import missing dependencies. Omitting runtime_shapes from new core source is
rejected before a kernel is launched, even if a fixed-shape execution was planned.
This closure gate is compatibility/integrity, not code authenticity or sandboxing.
Current report/task dataset contracts remain fixed-shape; this dependency capture
does NOT claim an authored variable-resolution dataset/experiment/report workflow.

The initial native/standalone API is ordinary eager PyTorch. Existing root and
operator hooks, autograd, state_dict and optimizers are exercised. Nested Module
execution retains the original compiler's forward_tensor convention. This is NOT
a `torch.compile`, tracing, ONNX, accelerator or complete native platform claim.

## References and evidence

PyTorch documents adaptive output sizes independent of input resolution in
[AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html)
and channel/group/kernel shape rules in
[Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html).
Those are operator contracts, not proof of this implementation; installed-runtime
independent functional output/input/parameter-gradient tests provide the evidence.
See `tests/test_runtime_shapes.py`, `tests/test_runtime_shapes_execution.py` and
the existing checked-export/notebook suites. Exact run counts/failed probes and
remaining D/E work are recorded in `docs/development.md` and campaign evidence.
