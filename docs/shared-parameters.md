# Explicit parameter sharing — local Stage D checkpoint

Executable references are app/graph.py, app/core.py, app/evidence.py,
app/families.py and app/revisions.py. Sharing is actual module identity, not equal
initial values, a copied checkpoint or an implicit consequence of equal widths.
This implements finite tied uses, not runtime recurrence, arbitrary partial or
transposed parameter tying, or full Stage D acceptance.

## Concrete graph grammar

A Linear, Conv2d or Module node may additionally have share_with: a nonempty
string naming another node of the same type in this exact model scope. It is a
parameter-ownership reference, not a data-flow wire. Ordinary input/output wires
and topological execution order remain unchanged. Owners can be used later in
the computation than an alias. Chains resolve to their terminal owner; cycles,
self-reference, missing/different-type targets and annotations on unsupported
operators fail explicitly. Omitting the field means a fresh independent module.

Every use retains its own complete constructor fields; nothing is silently
overwritten to fit its owner. Linear uses must agree on final-axis input/output
width and bias presence, while earlier sample dimensions can differ. Conv2d uses
must agree on normalized complete constructor kwargs, including channels,
kernel/stride/padding/dilation/groups/bias; spatial H/W can differ. Modules must
refer to the same exact architecture in this closure. Whole Module reuse shares
its entire already-validated subtree. Ownership does not cross model scopes:
two independent referring Modules construct separate subtrees unless those
parent occurrences explicitly share.

```json
{"id":"second-use","type":"Linear","features":3,"bias":true,"share_with":"first-use"}
```

Both uses execute on their incoming tensors. The compiler and standalone source
register only the canonical owner at its existing topological layer_N path;
each alias's execution plan refers to that same attribute. No duplicate state
keys or alias-attribute registration is needed. parameters(), optimizer state,
state_dict(), pure parameter_schema and checkpoint stable IDs contain each owner
once. Stable IDs use the canonical owner node path, not the alias path. A later
owner can leave gaps in layer_N names; these are intentional, not missing layers.
Changing sharing changes architecture identity and invalidates old checkpoints.
Extra alias checkpoint keys are rejected, not loaded in an order that could
silently overwrite a shared tensor.

Unique parameter counts are deduplicated; execution/activation/convolution
workspace counts include every use. RAM admission therefore does not pretend
shared uses have no activation cost. Parameter-relative Layout counts the owner
once and gives aliases zero newly owned parameters. The Inspector provides a
same-type local owner selector with explicit Independent parameters. IDs are
alphabetically listed; computation/packing order is not sorted. Canonical owner
and gradient/checkpoint meaning are explained. Diagram notes/hover text mark
sharing while retaining bias information. Editing uses existing undo, stale-result
invalidation, persistence and validation. Missing/cyclic ownership fails document
admission; a constructor mismatch can remain an explicit invalid draft diagram.
Deleting an owner does not silently detach aliases: detach/retarget them first.

Affine serial-vector analysis follows execution-plan occurrences recursively,
not unique module enumeration: using a Linear three times composes its matrix
three times even though only one parameter set is learned.

## Family grammar version 2

Version1 remains independent-only. Family version2 preserves its exact closed
top-level fields, symbol/assignment/binding/count rules and existing finite chain
ingress/egress constraints. It additionally accepts explicit share_with in
templates and repeat mode tied. Runtime recurrent/unknown modes remain errors.

Tied repetition uses occurrence0 as the canonical owner for each parameterized
template node. Later uses point to that owner; parameter-free operations execute
normally on each occurrence. Independent mode still creates separate owners,
except explicit sharing already declared inside a template remains explicit:
internal template references are remapped within each occurrence. For tied mode,
those internal references point into occurrence0. Explicit references outside a
body remain in their model scope; other nodes referencing an expanded body are
retargeted to its occurrence0. Consecutive bodies retain these references.
Non-square or otherwise incompatible repeated constructors produce an
incompatible specialization with origin information, not independent copies.
The current64-block materialization bound is unchanged, not final large-model
support. Finite tied repetition is not hidden/cell state or a runtime loop.

The runnable examples/families/width-depth-tied.json uses this grammar. With
width3 and two input features it has25 learned parameters for depths1,2,3,…
while the number of finite forward operations grows. Compare the independent
width-depth.json example, whose learned parameter count grows with depth.

The owned family mode selector explicitly upgrades a v1 draft to v2 together
with selecting tied. Root version is an exact numeric text control; v2 downgrade
is denied by the existing owner/head/sequence/origin/memory/staging service.
Drawing-seeded families preserve the captured model's sharing/version semantics.
Preview and retention remain separate and train-free. Concrete instances,
family experiment axes, terminal provenance and CLI recovery reuse the same
resolver and canonical checkpoint paths.

## Versions, SQLite and agent CLI

New containers use .vtw version9, with unchanged SQL tables. Load of an older
header containing explicit sharing or v2 family/model revisions fails without
rewriting its source. Genuine independent v1–8 contents migrate in working
memory only; old hashes remain unchanged. Frozen ModelRevision version2 carries
shared graphs; version1 may not contain share_with. Independent captures remain
version1, so an older validator cannot silently ignore sharing in a standalone
frozen revision. Family v2 is also explicitly versioned. Experiment/run v2 keeps
its existing family-coordinate grammar from v8; Specialization remains v1 and
pins exact family/bindings/model relationships.

Agents use ordinary ordered set-node values:{share_with:owner}, remove_fields
to detach, freeze-model, put-spec and specialize-family edits. There is no
parallel sharing store or inferred tie-by-name rule. Removing a field explicitly
constructs independent parameters on the next compilation; it does not copy
trained weights or change an immutable prior frozen model. CAS/atomic disk Save
and existing reference/recovery services remain authoritative.

Portable runtime remains the complete24-file closure because these changes live
in already captured modules. Old immutable20/21-file bundles run with their own
captured independent semantics, not this implementation injected into them.

## Evidence and limits

tests/test_shared_parameters.py covers strict ownership/chains/cycles/constructor
validation, counts/paths, identity/version fences, seed/owned upgrade, template
reference remapping, SQLite save/load/no-rewrite, CLI edits and stdlib execution.
tests/test_shared_execution.py uses independent functional forward and gradient
oracles for repeated Linear/nested Module and grouped Conv2d branches with
different H/W, checks Adam against one independent shared-parameter oracle,
actual worker retry/checkpoints and fresh standalone prediction, and executes a
real two-point Jupyter rerun with exported sources/predictors/curves.
tests/qt_shared_parameters_smoke.py exercises actual sandboxed WebEngine Inspector
selection, undo/redo, explicit family upgrade/preview/retention/instance creation
and save/reload, without host Torch or preview SQL writes. These bounded local
checks are not all-platform/accessibility, hostile-file, training-quality or
durable lifecycle certification.

Design references: PyTorch's [Module API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
defines registered module/parameter identity, duplicate removal and strict state
loading. Its [weight-sharing tutorial](https://docs.pytorch.org/tutorials/beginner/examples_nn/dynamic_net.html)
demonstrates reuse during multiple forward operations. Our single-owner storage
and occurrence-based plan are implementation choices motivated by those semantics;
the numerical and checkpoint claims are verified locally, not inferred from the
documentation. Future partial/transposed ties must define their own shape/identity
and checkpoint contracts rather than overloading share_with.
