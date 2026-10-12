# Explicit spatial-runtime task execution

Reference: `app/runtime_tasks.py`, `app/task_execution.py`,
`app/experiment_objects.py`, `app/worker.py`, and the owned experiment form.
This is a bounded Stage D extension, not completion of recurrence or durable jobs.
The separate [temporal execution policy](temporal-task-execution.md) uses
ExperimentSpec5/TaskBundle3. It does not reinterpret this spatial policy or its
closed family5/specialization4 proof.

## Selection, storage and compatibility

ExperimentSpec and RunRecord integer version 4 extend the preserved v3 grammar.
Each family choice optionally adds `execution`, exactly `concrete-v1` or
`runtime-spatial-v1`. Omission means `concrete-v1`, including for a family v5 with
a retained runtime declaration. A concrete ModelRevision choice has no execution
field. Earlier experiment grammars reject the new field. Neither opening an old
file nor adding a runtime family implicitly changes a saved experiment's behavior.

`runtime-spatial-v1` requires a family v5 with a non-null specialization v4 runtime
witness. The exact selected family, bindings, scoped contexts, baseline frozen
ModelRevision and specification digest are reproduced before admission. Conditional
or incompatible choices remain visible, but cannot run. A mutable head is never
substituted. Experiment/run v4 records require a `.vtw`13 header; no SQL table or
sidecar is added. Older containers open into an upgraded private working database
without rewriting their source. Save is explicit. Existing family v5 and
specialization v4 records still have their minimum header requirement of 12.

In the owned experiment form, each family card has **Family execution**. It starts
at Concrete baseline for an omitted policy. Explicitly selecting either mode queues
the monotonic version4 upgrade before its named property edit. Acknowledgement
updates both controller and visible grammar. Owner/sequence/workspace, staged-value,
pending-worker, exact revision and atomic Commit fences are unchanged. Preview
does not fit, allocate tensors or persist revisions. Retain before Run.

CLI `apply` batches with `put-spec` operations use the same complete closed ExperimentSpec. Existing
`preview-experiment`, `preview-experiment-page`, `run-candidate`, run evidence,
`export-run-trained`, notebook export and explicit execution resolve this policy.
The CLI does not need a parallel mutable runtime database or editor session.

## Frozen execution bundle and identity

TaskBundle v1 remains unchanged. TaskBundle v2 has exactly those v1 keys plus
`runtime`, exactly `{family, specialization}`. Both are complete frozen revision
documents, not IDs, a mutable lookup or arbitrary code. The ordinary frozen
`model` remains the constructor-baseline ModelRevision. The runtime witness must
reproduce that same model, family, bindings and conditional domain. Forged fields,
revisions, axes, witness hashes or version types fail before preparation/allocation.

For v2, `run_identity` has integer version2, the same model/dataset/task references,
manifest/protocol hashes and `task_runtime: masked-additive-v1`, plus
`compiler: native-runtime-spatial-v1` and `runtime: digest(bundle.runtime)`.
Version1 identity is byte-semantically unchanged. Thus equal parameter shapes do
not make concrete and runtime executions interchangeable. The checkpoint/result
envelopes remain version1; their identity comparison covers the exact v2 proof.
This is local integrity evidence, not authentication or proof of training origin.

Workers accept closed requests with kind `task-v1` or `task-v2`, whose exact integer
bundle version must match. Preparation, trainer, checkpoint restore and exporter
all use the same validated contract/compiler path. Protocols, seeds, masking,
weighted objectives, minibatching, history retention and fitted transforms keep
their existing [task-training semantics](task-training-contract.md).

## Dataset-derived actual case

Dataset fields and TaskSpec bindings stay concrete and explicit. For each declared
runtime Input ID, the bound numeric field has a fixed rectangular CHW shape; its
post-transform sample shape supplies
the actual execution case. Ordinary [spatial guards](runtime-spatial-contracts.md)
check fixed C/rank/undeclared axes and every operator/output. Ordinary task checks
then check port IDs, logical axes, target shapes, losses, vocabulary, masks,
transforms and split membership. Derived length/mask inputs cannot impersonate
runtime image fields. No constructor argument is inferred from data.

Two dataset/task pairs can specify different rectangular H/W for the same exact
family specialization and weights schema. Named inputs may have different shapes
from each other, including through explicitly shared encoders. Each dataset field
and batch still has one rectangular shape. Ragged/mixed-size images within one
field/batch are not padded or resized implicitly. Output dimensions may vary only
if both the architecture's actual contract and explicit targets permit them.

Contract reuse keys include runtime proof and dataset/task revisions, avoiding
reuse of a previous dataset's actual shapes. Pure actual-case planning precedes
typed preparation and current-RAM run admission. Initial runtime parameter
allocation is checked against the actual case, not unused baseline activations;
every forward subsequently checks actual batch/dtype/gradient RAM before packing
or layers. Heuristic admission is not an allocator limit or OOM guarantee.

## Saved inference and portable reports

An opted-in saved predictor validates new field-valued records before allocating
tensors. Each runtime port requires one common positive rectangular CHW shape in
the incoming batch. Fixed dimensions, finite values, actual output plan and current
RAM are checked; weights are never recreated and preprocessing is never refitted.
Different incoming H/W is currently supported only for inputs with no transforms.
Inputs with fitted transforms may use their exact saved field shape; changing
resolution is rejected with an explicit resolution-independent-adapter repair.
This deliberately avoids silently reinterpreting per-feature fitted statistics.

The standalone predictor captures the actual checked family/compiler and exact
constructor bindings, initial dataset case, fitted state and learned weights. It
imports Torch but no application package and contains no raw training observations.
Helper capture checks loaded bytecode against the current source and fails closed
if implementation files changed after import; restart before exporting, rather
than emitting a different function under an old source location.
`compile_model`/exported `ArchitectureFamily` accept optional `initial_shapes` only
with an explicit runtime domain; it controls construction-time safety admission,
not constructor bindings, parameter shapes or the conditional domain.

New portable report bundles capture 27 runtime files, including `runtime_tasks.py`
and `family_export.py`. Missing dependencies in current source are rejected before
kernel launch. Genuine earlier captured closures remain executable without byte
rewrites. A runtime report adds `runtime-architecture-<run revision>.py` for each
runtime run, alongside its trained predictor. Existing `architecture-<model
revision>.py` files explicitly remain concrete constructor-baseline exports.

## Verified boundaries

Tests cover pure multi-dataset planning, old-mode preservation, closed grammar,
historical references, proof forgery, actual RAM refusal, owned monotonic edits,
header gates and actual CLI preview. Numerical tests compare independent functional
Adam weights and elementwise population means; execute shared multi-input fits,
fresh app-free predictions at new H/W, actual CLI/worker save/reload, and real
Jupyter two-resolution reruns. A sandboxed private Qt WebEngine test edits and
retains the explicit mode and delivers three native-worker updates without
importing Torch in the GUI process. No user app/config/workspace is changed.

Unfinished: ragged spatial batches, resolution-independent fitted adapters,
recurrent transition/state/length/mask/reset contracts, mixed CNN/RNN composition,
GPU execution and detached durable execution. Neither this extension nor finite
tied repeats constitutes completion of Stage D/E.
