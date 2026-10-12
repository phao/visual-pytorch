# Versioned initialization and trainability

Current development branch: `.vtw`24, ModelRevision6, Family10 and
Specialization9. This is under construction, not an announcement of a public
release or complete LoRA/MS-VAR implementation.

A graph may contain a closed `parameter_policy` object with integer `version:1`
and `parameters`. Keys are canonical JSON component arrays **relative to that
architecture**, serialized as strings with no spaces. The final component is
the native parameter name. Components are atoms, so a block named `a/b` is
different from a Module `a` containing block `b`.

```json
{
  "version": 1,
  "parameters": {
    "[\"base\",\"weight\"]": {
      "trainable": false,
      "initializer": {
        "kind": "values", "dtype": "float32",
        "values": [[0.6, 0.0], [0.0, 0.6]]
      }
    },
    "[\"A\",\"weight\"]": {
      "initializer": {"kind": "normal", "seed": 17, "mean": 0.0, "std": 0.05}
    },
    "[\"B\",\"weight\"]": {"initializer": {"kind": "zeros"}}
  }
}
```

This illustrative policy requires matching actual 2-input/2-output base and
adapter boxes; it is not a complete architecture by itself. For the requested
16×16/rank2 showcase, the base values are a16×16 array and A/B shapes2×16/16×2.
The forward remains ordinary Linear→Linear, fixed Scale and elementwise Add.
No nonlinearity is inserted between the adapter factors.

## Scope and ownership

`["projection","linear","weight"]` selects a nested parameter through the
Module/Recurrent block `projection`, without using a referenced architecture ID
or guessing its registration index. A child architecture can declare an
initializer while its parent declares freezing. Explicit unequal values for the
same owner/field refuse. Defaults apply once after combining explicit fields:
trainable=true, initializer=constructor. Shared aliases resolve structurally to
one owner, so copies neither initialize nor update it twice. Old ignored node
metadata such as `trainable:false` retains its old lack of execution meaning.

Policy-bearing concrete execution schemas use canonical full JSON component
owner IDs beginning with the exact root architecture ID. Older policy-free models
keep their existing slash IDs and hashes. Standalone family semantic states keep
their separate family/template/call/owner component convention; native addresses
are not semantic IDs. Frozen evidence captures the exact applicable convention.

Family10 retains the same nullable temporal-v2 runtime policy as Family9, and
requires an explicit graph policy somewhere in its reachable templates.
Specialization9 re-proves the exact family/model/policy/runtime. A declaration on
a repeated template node applies to **all** expanded copies. Independent repeats
own separate Parameters; tied copies resolve to the same canonical owner.
Initializer shape checks occur against the resolved constructor, not a caption.
Selectors are translated as components; generated occurrence IDs are not split
on `/`. Older Family1–9 and Specialization1–8 never gain these policy semantics.

## Execution and provenance

Supported initializers: constructor, zeros, exact nested float32 values,
normal(seed,mean,std), uniform(seed,low,high). Declarations/shapes/numerics are
checked before numerical construction. Materialization checks finite actual
float32 values and distributions that remain nondegenerate in float32. A private
generator uses the explicit seed literally per owner, independent of registration
order and global RNG. Equal seeds and shapes may intentionally give equal draws.
Constructor values are captured, not claimed to follow a policy seed.

Apply once at the root after construction and before optimizer creation. All
owners and receipts prepare before mutation; application failures roll back.
Freeze only `requires_grad`: do not detach outputs or wrap frozen branches in
`no_grad`, since upstream inputs/adapters may need their gradients. Adam receives
only unique trainable Parameters. All-frozen inference remains valid; training
refuses even when constrained RAM mode is disabled.

Receipts include declaration and materialized owner-value digests, separate
stored/trainable/frozen counts, actual native addresses and Torch/device/dtype/
private-seed rule. They describe initial parameters, **not optimizer resume**.
Policy checkpoints use legacy inference checkpoint_version2 or task checkpoint
version2; policy-free checkpoints retain version1. Exact model6/family10/spec9
revisions fence these changes. Compatibility validation does not authenticate a
manually forged training history or recompute predictions without Torch.

Frozen parameters still consume storage and can require backward activation
memory. RAM admission conservatively retains existing whole-model terms and adds
256 bytes per stored parameter element plus4096 per unique owner for policy
preparation/provenance. This is an admission allowance, not measured peak RSS.
Task RAM evidence reproduces this component and its owner counts exactly.

Native, real worker, plain app-free source, captured family runtime, checkpoint
restoration and actual saved/rerun notebook kernels have bounded checks. Captured
closures include parameter_policy.py, policy_graph.py and parameter_execution.py;
new closures missing them refuse, while genuine older closures remain readable.
Family warm-start cannot silently overwrite frozen supplied base values.

## CLI authoring

`vpw_cli.py workspace.vtw parameter-policy MODEL_ID` reports resolved owners,
counts and initializers without importing Torch/NumPy/Qt or drawing values.
Ordered `apply` edits accept:

```json
{"op":"set-parameter-policy","id":"MODEL_ID","policy":{"version":1,"parameters":{}}}
```

The policy is replaced explicitly and validated against the prospective graph;
`policy:null` removes that graph's explicit declaration. Disk replacement remains
atomic for the full CLI edit batch. Session undo is owned by the desktop's
snapshot wrapper, not persisted or fabricated by the CLI process. Policy changes
invalidate incompatible attached results while historical revisions survive.

## Remaining work

Dedicated native policy controls/badges, compatible immutable retained-base
binding, complete requested LoRA workspace/comparisons/merged inference artifact,
broader frozen regression/integration and exact Neural MS-VAR remain. Explicit
values do not yet implement a source-checkpoint selector. Existing Experiment UI
is preserved; the separate Loss-workbench study is a design prototype, not a new
production board. Stage E remains deferred.
