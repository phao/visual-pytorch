# Closed Add and fixed Scale blocks

These are shared architecture primitives for residual paths and the forthcoming
LoRA/neural MS-VAR work, not completed LoRA adapters or a probabilistic model.
The executable references are `app/operators.py`, `app/graph.py`,
`app/core.py`, `app/revisions.py`, `app/families.py` and `app/family_records.py`.

## Operator documents and forward computation

```json
{"id":"sum","type":"Add","op_version":1,"sources":["base","delta"]}
```

The operator requires at least two incoming tensors with identical full shapes,
sample-axis declarations, CPU device and float32 dtype. `sources` determines
left-to-right addition order, not graph-node order or alphabetical order; float32
rounding makes that distinction observable. It uses ordinary differentiable,
out-of-place PyTorch addition. There is no broadcasting, flattening, learned
weight, implicit dtype promotion or implicit device transfer. Named multiport
sources use the existing physical `node@port` key and explicit edge `from_port`.
Each incoming physical source appears exactly once in `sources` and the wires.

Structural drafts may have fewer than two sources while the user adds wires.
They can be edited/saved but fail semantic execution checks. Source keys must be
unique, nonempty, trimmed strings. The extra input socket permits a new branch;
the inspector reorders the declared sources. Removing Add never silently bypasses
it as an identity. Reconnecting arrow endpoints uses the same atomic wire rules
as other blocks.

```json
{"id":"scaled_delta","type":"Scale","op_version":1,"coefficient":-1.5}
```

Scale has exactly one incoming CPU float32 tensor and computes `x * coefficient`,
preserving sample shape, axes and B. The coefficient is fixed architecture data;
it is neither an `nn.Parameter` nor a trainable buffer. Zero and negative values
are valid. The value is numeric, finite, float32-range and editor-representable;
booleans, strings, missing values, NaN and infinities refuse. Numerically integral
values must lie in ±(2^53−1), regardless of integer/float JSON spelling, so the
Python validator and JavaScript editor agree. GUI entry accepts
signed decimal/scientific notation, not hexadecimal or empty text. No finite
output guarantee is implied: multiplying large tensor entries can overflow.

Both operators have integer `op_version:1` (defaulted when omitted), accept only
their closed fields plus existing common node identity/geometry/folding fields,
and reject `share_with`, `trainable` and `bias`. They own zero parameters but
participate in execution and activation-memory estimates. Their blocks have
distinct type icons, compact summaries, inspector explanations and About help.
The Scale coefficient is editable through the same Qt/CLI history mechanism.

## Storage and retained versions

New saves use `.vtw23`; SQL tables are unchanged. This development reader accepts
1–20 and 23 only. 21/22 retain their deferred-E meaning and refuse here. Opening
a supported old file upgrades only its in-memory copy; it never rewrites the
original or fits a model. Earlier headers cannot carry arithmetic or the new
retained document versions, even if an arithmetic family is unmaterialized.

Reachable arithmetic selects ModelRevision5. ModelRevision1–4 refuse these
operators. Family9 explicitly admits arithmetic templates and requires nullable
`runtime_policy`: null for fixed shapes, or the existing temporal-v2 policy.
Family1–8 refuse arithmetic templates; old non-arithmetic hashes/defaults remain
unchanged. Family9 retains Specialization8 with exact family/bindings/model,
constructor contexts and nullable re-proved temporal witness. Scalar coefficients
are concrete JSON numbers, not a new symbolic floating-point binding language.

Add/Scale are pure graph operations, not parameter-owning layer constructors.
Shared model capture, concrete/family standalone export, task workers, checkpoints
and saved predictors carry the same implementation. Direct concrete tasks still
use TaskBundle1 with their explicit ModelRevision5; temporal-v2 family tasks use
TaskBundle4 with Family9/Specialization8 provenance. DatasetSpec, TaskSpec,
training protocol and experiment/run/report document grammars are not widened.

## Evidence and boundaries

Tests compare independent values and all input/weight/bias gradients for signed
and zero scales on one- and multidimensional samples; verify actual float32 sum
order; reject shape/dtype/broadcasting/version/field forgeries; and test atomic CLI
edits and real `.vtw` round trips. Native and generated models load identical
state dictionaries. A real isolated 12-update worker reproduces local checkpoint
and history exactly, with saved predictor parity. The sandboxed Qt harness
actually clicks sockets, edits Scale, performs undo/redo and saves/reopens, without
importing Torch/NumPy in the desktop host. These are bounded tests, not claims
of unrestricted GPU/dtype support or completion of either feature report.

PyTorch's official [add](https://docs.pytorch.org/docs/2.14/generated/torch.add.html)
and [mul](https://docs.pytorch.org/docs/2.14/generated/torch.mul.html) documentation
informed the operation choice. PyTorch itself permits broadcasting/type promotion;
this application's explicit graph contract intentionally refuses them. The
documentation redirect identified version2.14; locally executed tests, not that
version label, establish the behavior of the installed numerical environment.
