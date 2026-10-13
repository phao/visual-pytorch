# Shared application and graph discovery

Read-only development commands for agent/user authoring, not weight-freezing,
latent filtering or losses. JSON envelopes use version1 tags.
This is an application-description contract, **not** a JSON Schema dialect or a
replacement for actual validators.

```sh
python vpw_cli.py capabilities
python vpw_cli.py operator-schema Conv2d
python vpw_cli.py operator-schema Recurrent
python vpw_cli.py format-schema
python vpw_cli.py lab.vtw ports ARCHITECTURE_ID
```

Commands return the existing `{"ok": ..., "value": ...}` envelope. Unknown or
wrong-case types refuse rather than return an empty successful schema.

## Application scope versus a workspace

The first three commands need no workspace argument. Workspace-first
`lab.vtw capabilities` also works, but that path is **not read**. Their scope is
`application-contract-discovery; no-workspace-read-or-execution`. Supplying
`--expect-revision` refuses because it would falsely imply checking a file.
Use `inspect` to inspect an actual file and its version.

Queries import no Torch/NumPy/Qt, construct no tensors/models, read no user config
and perform no fitting/inference. `format-schema` constructs/closes only the
actual in-memory SQLite creator and queries its SQL definitions; it writes no
file and emits no ephemeral workspace IDs/timestamps. `ports` opens the workspace
normally into a working copy, honors revision conflicts, does not save, and
describes the current editable graph, not an implicitly chosen frozen revision.

## Shared sources and validated witnesses

`app.operators.BLOCK_TYPES` is the accepted registry used by `check_document`
and discovery. The registry now includes closed `Add` and fixed `Scale`:
[arithmetic blocks](arithmetic-blocks.md) require container23 and explicit retained
document versions. This extension is not silently valid in older containers.
Closed `Constant` and `TensorParameter` roots require container27/Model7/Family11/
Specialization10. Their examples have no fictitious Input, their ports report
no incoming sockets and their literal/value-owner semantics are distinct.
The exact contract is [tensor sources](tensor-source-kernel.md).
Shared tensor/Recurrent/LSTM field sets and normalizers supply constructor
metadata/defaults. Legacy descriptions state permissive-extra-field limitations:
accepting `trainable:false` does not mean training implements it. Softmax's
stricter fields are conditional on that activation choice.

Each description contains required/optional fields, example values separately
from defaults, shape/axis/batch rules, cardinality/ports, sharing scope, route
limitations and a complete tiny `{root, models, node}` example. Referenced
models/cells are included. All closures pass the real structural and tensor
validator at query time. Registry additions without discovery metadata fail
closed instead of silently disappearing. Examples are construction witnesses,
not opaque source-code recipes, datasets or training results.

Sample axes exclude B. Splitter divides the last sample axis and preserves
earlier axes; it does not flatten multidimensional inputs. Concat's flatten
setting has different semantics. Replicator aliases can intentionally have
overlapping packing offsets.

## Complete port contracts

`ports` includes stable boundary IDs/names, shapes/axes, float32 dtype, packing
offsets, node shape contracts, original wires/source-port IDs, explicit role/
control bindings, resolved execution-plan order and referenced model/cell
interfaces. Browsable reference names are sorted, but packing/bindings/source
order are not. Geometry/folding do not change the numerical contract.

Representation is explicitly not declared in the current grammar. Shapes and
labels do not establish logits, probabilities, distribution parameters,
likelihoods or posteriors. Cross-entropy expects logits; the current checker does
not prove an upstream output is logits. These commands report that gap; they
do not add loss/representation refusal semantics.

## Limits and next work

This first layer does not provide a complete formal serialized DatasetSpec/
TaskSpec/experiment schema, nested-field type constraints or every historical
feature's minimum reader version. Current container version and actual accepted
reader range are exposed; feature-specific compatibility remains in the format
specification/load gates. A current example is not valid in every older container.

Complete replies and validation can grow with the reachable graph: no constant
cost/large-workspace latency claim. Templates, edit planning, packs, semantic
outputs/tasks, initialization policy and LoRA/MS-VAR primitives remain separate
requested slices. Neither the reports' full discovery priority nor their full
feature request is declared complete by these commands.

Tests cover real frozen-model/workspace round trips, shared fields/defaults,
malformed refusals, no mutation/heavy imports, fresh `-I -S -B` CLI queries without
a workspace, named multiports/reference interfaces and unequal ordered heads.
All19 examples match native/exported PyTorch outputs after strict state-dict
loading. Existing family export and CNN/recurrent worker/kernel tests exercise
the shared registry's complete captured-source closure, not only its names.
