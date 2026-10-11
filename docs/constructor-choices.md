# Explicit constructor choices — family version 3

This local Stage D checkpoint extends the checked family grammar, not the Python
operator registry or a general executable expression language. The same resolver
feeds Qt, CLI, immutable specialization records, experiments, workers and exported
notebooks. `examples/families/activation-bias.json` is a runnable example: five
activation choices and bias on/off produce ordinary concrete MLPs with ten or six
parameters. This does not finish D or introduce runtime recurrence.

## Closed symbol types

Family version 3 preserves the existing model closure, assignment and finite
repeat fields. Symbol definitions still require stable `id`, model-local `scope`
and readable `name`. A symbol's optional `type` is one of:

| Type | Domain | Optional fields |
| --- | --- | --- |
| `integer` (also the absent-type default) | Exact editor-range integers | `default`, `minimum`, `maximum` |
| `boolean` | JSON `false` or `true`, not 0/1 or strings | `default` |
| `enum` | 1–64 distinct explicit strings in required `values` | `default` |

Enum strings use the existing identifier rule: nonempty, trimmed, at most 100
characters. The finite string list is data, not Python, module names to import or
calls to evaluate. Enum/boolean symbols cannot contain arithmetic bounds;
boolean/integer symbols cannot contain `values`. Defaults must match the exact
type and declared domain. Every symbol must have an assignment or repeat-count
dependent. Scope/name uniqueness and all existing closure/cost bounds remain.

Examples:

```json
{"id":"activation-kind","scope":"mlp","name":"activation",
 "type":"enum","values":["GELU","ReLU","Tanh"],"default":"Tanh"}
```

```json
{"id":"learn-bias","scope":"mlp","name":"bias",
 "type":"boolean","default":true}
```

Family versions 1/2 reject the new `type`/`values` keys and choice assignments;
they retain their original integer-only meaning and hashes. Version 3 also
supports the explicit shared owners/tied repeats from version 2. An absent type
does not infer a type from a default, seed or dataset.

## Assignment targets and expressions

All previous integer fields/arithmetic ASTs remain. Arithmetic and repeat counts
accept only integer symbols; `true` is never promoted to 1. Additional version-3
targets are:

| Block | Field | Type |
| --- | --- | --- |
| Linear | `bias` | boolean |
| Conv2d | `bias`, `padding_mode` | boolean, enum |
| Activation | `activation` | enum |
| Concat | `flatten` | boolean |
| AvgPool2d | `count_include_pad`, `ceil_mode` | boolean |
| MaxPool2d | `ceil_mode` | boolean |

A target must already exist in the concrete seed. A constructor value is either
a matching literal or exactly `{"symbol":"stable-id"}` with a matching type and
model-local scope. Arithmetic ASTs, Python strings-as-code and arbitrary metadata
targets are rejected. Duplicate targets remain errors. No absent constructor
field is silently added, removed or normalized to repair an instance.

The existing operator contracts decide which concrete choices are executable.
For example, Conv2d op_version 1 still permits only zeros padding and pooling
op_version 1 still permits only ceil_mode false. Selecting another declared enum
or boolean value yields an incompatible point, not a new unsupported operator.
Activation choices pass through the existing five-name registry. A Softmax
choice without `dim` uses its existing last-sample-axis default, excluding batch
B. An explicit `dim` on a non-Softmax seed is still rejected; there is no guessed
conditional field removal. Shared owners must still agree on constructor arguments
after binding. Equal choices never imply sharing.

Bindings use stable IDs and actual JSON values:

```json
{"activation-kind":"GELU","learn-bias":false}
```

An unresolved symbol yields a conditional preview. Supplied wrong JSON scalar
types raise GraphError; otherwise out-of-domain or invalid concrete operator/shape
choices remain visible as incompatible, without mutation or numerical allocation.
The binding map in a successful receipt includes every resolved default. Existing
family/specialization/model identity algorithms are unchanged and hash the actual
typed values; names, enum labels and bools are not converted into numeric codes.

## Retention and file compatibility

`.vtw` version 10 gates family v3, specialization v2 and experiment/run v3. No SQL
tables, extra sidecar, executable code or pending session are added. Readers accept
container versions 1–10 and migrate only the working copy. An older header
containing new semantics fails without changing the source file; Save is explicit.

A family v3 instance retains specialization `version:2`; its exact family and
frozen-model references are checked by re-specializing every typed binding.
Specialization v1 remains integer-only and applies to family v1/v2. A forged
same-shaped model, omitted default, coercion, or mismatched specialization version
is rejected. Concrete independent model captures remain ModelRevision v1;
actual shared ones remain ModelRevision v2. Typed family choices alone do not
change the concrete forward model grammar.

## Experiment and execution parity

Experiment/run v3 have the same closed fields as v2, but fixed bindings, ordered
axes and exact coordinates additionally accept booleans and bounded strings.
Experiment v2 remains integer-only and cannot select a family v3 revision. All
axis values are distinct by exact type and value: false and integer zero are
different coordinates, never equal membership tests. Float/list/object values
remain errors. Intrinsic axes need not all have one type: explicitly incompatible
members are shown when the exact family resolves them, not filtered or repaired.

Each axis still contains 1–1000 entries; axes/fixed bindings are disjoint, stable
IDs distinct, and the last axis changes fastest. Arithmetic paging/counts, exact
decimal projections, comparison whitelists/exclusion reasons and candidate IDs
are preserved. Derived model/specialization provenance is retained atomically at
terminal run admission; retries and portable result/rerun notebooks resolve exact
typed selections, not current browsing heads. No fit occurs in preview.

## Qt and agent workflow

The owned family editor exposes an alphabetically ordered type selector, enum
`values` JSON and a literal JSON default field. Integer defaults stay exact
decimal text; enum defaults are quoted strings, boolean defaults true/false.
Blank default/domain/bound fields mean absent properties, not zero/false/empty
strings. Switching types does not secretly discard old defaults or bounds:
explicitly repair them, then Check/Preview. Selecting a type/domain queues an
explicit family version-3 upgrade in the same owned edit batch. Choosing tied
mode preserves an already v3 document instead of downgrading it to v2. Owned
document versions can advance 1→2→3 (or 1→3), never decrease.

The experiment editor exposes the exact selection version as a numeric control.
Use version 3 for a typed family selection and its existing bindings/axes JSON
controls. Adding another family selection preserves an existing version 3.
This is not automatic product enumeration or an inferred choice from strings.
Existing ownership, raw invalid text, stale sequence/origin, staged large values,
memory admission, Check, retention, undo/redo and explicit Save fences remain.

`vpw_cli.py preview-family`, `preview-stored-family`, atomic `specialize-family`,
spec admission/preview/execution and recovery commands use this same resolver.
No new CLI-side editor state or executable hooks are introduced.

## Verification and limits

`test_constructor_choices.py` tests grammar/scopes/types, old-version denial,
conditional/incompatible choices, type-exact coordinates/order, immutable records,
forgeries, SQLite roundtrip/header gates, actual agent edits and owned upgrades.
`test_constructor_execution.py` independently checks all ten activation/bias
instances against functional output/input/parameter-gradient oracles, actual
constructor arguments for convolution/pooling, native/generated model parity,
actual worker fits/retry/checkpoints/fresh predictors, and a two-point real
Jupyter rerun with exported models, curves and predictors.
`qt_constructor_choices_smoke.py` uses the real sandboxed WebEngine, native bridge
and private `.vtw`: typed defaults, retained invalid scalar, train-free/read-only
preview, retention, concrete creation and Save/reload. JavaScript checks exercise
actual forms and version queues. These are scoped tests, not universal native
cost, hostile-file, platform, accessibility or screenshot claims.

Cross-scope constructor bindings, nested family instantiation, arbitrary operator
switches, dynamic spatial guards and structured recurrent state/length/mask/reset
remain separate D work. E durable/heavyweight execution remains later work.

Primary references consulted: PyTorch's
[activation registry](https://docs.pytorch.org/docs/stable/nn.html),
[Softmax dimension contract](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.Softmax.html),
[exact GELU default](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.GELU.html)
and [Linear bias implementation](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py).
They informed constructor mapping; local numerical/runtime tests, not source
inspiration, establish the behavior verified here.
