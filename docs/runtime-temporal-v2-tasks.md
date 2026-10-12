# Runtime-temporal-v2 typed task execution

TaskBundle4 explicitly executes a frozen Family8/Specialization7 runtime-v2
contract. This is a retained Stage D checkpoint, not full D/E completion.
[ExperimentSpec/RunRecord6](runtime-temporal-v2-experiments.md) now explicitly
retains this route in container20, through owned authoring and real worker/kernel
paths. TaskBundle4 itself adds no workspace table. Older contracts stay closed.

## Closed provenance and identity

`task_bundle(model, dataset, task, manifest, protocol, runtime=...)` selects4
only for exact Specialization7. `runtime` has exactly `family` and
`specialization`, both frozen complete documents, not mutable head references.
`runtime_tasks.proof(..., execution='runtime-temporal-v2')` requires Family8,
Specialization7 and a nonnull witness. It re-resolves exact family/bindings,
constructor contexts, model reference/closure, witness3 and proof2 hash through
the same retained-specialization validator. Null does not opt in; other family
versions cannot use this route. Existing TaskBundle1–3 checks remain closed.

TaskBundle4 has exactly version/model/dataset/task/manifest/protocol/runtime.
DatasetSpec1, TaskSpec1, TrainingProtocol1 and ModelRevision4 are unchanged.
RunIdentity4 records the exact model/data/task references and manifest/protocol
hashes, compiler `native-runtime-temporal-v2`, task runtime `masked-additive-v1`
and the digest of complete frozen runtime provenance. New data or transforms
change the run identity even when the family/model owners are identical.
Checkpoint1 remains inference-only: fitted preprocessing, exact identity,
stable parameter IDs and finite CPU float32 weights; no optimizer continuation,
freezing policy or authenticity claim is added.

## Actual case and data meaning

The constructor's time symbol retains a baseline drawing. Explicit field-to-port
bindings determine a task's actual rectangular sample shapes. The v2 dependency
proof permits only declared leading semantic T axes; learned widths, ranks,
hidden/cell state dimensions and undeclared axes remain fixed. Every Input must
be bound; each Output has its explicitly selected objective. State and length
data are not synthesized from the drawing or guessed from labels.

DatasetSpec1's existing ragged-leading shape and explicit `pad` transform can
produce a rectangular task. A separate explicit derived-length binding supplies
execution lengths from the raw ragged field. Length controls determine LSTM
execution, not supervision: loss masks/sample weights remain separate TaskSpec
bindings and are never inferred from those lengths. Group/temporal splits use
the retained observation IDs/fields and manifest, not constructor ordering.

`compile_task_model` passes actual shapes and the exact
`runtime_temporal_v2` declaration to the shared compiler. It does not instantiate
the possibly larger baseline T first. The same compiler serves local training,
isolated workers, checkpoint restoration and exported inference.

RAM admission before tensor allocation includes the existing task/data/output
and metric-history allowance, whole-graph parameter owners, recurrent unrolls
and v2 LSTM gate/copy charges for actual batch/T and nested call contexts.
LSTM calls are charged individually while tied owners stay unique. Terminal
result admission independently recomputes these components; deleting the LSTM
charge and repairing the total is still rejected. Estimates are conservative
admission policies, not measurements or guarantees of allocator peak usage.

## Workers, native Qt adapter and portable inference

Worker request kind is exactly `task-v4` and must match integer bundle version4.
Malformed old/new combinations fail before importing Torch. The native
`TaskTrainingJob` validates the frozen bundle and selects its exact admitted
kind; QThread progress/terminal signals carry unchanged task evidence. This
does not itself infer a retained experiment choice; that requires explicit
ExperimentSpec6/runtime-temporal-v2 selection in the existing editor.

`restore_task_predictor` and `generate_task_predictor` check exact checkpoint
identity, fitted transform and parameter keys/shapes. Plain rectangular inputs
may change T between calls, including unsorted zero/full execution lengths,
while state and feature axes stay fixed. Each port's samples in a batch must be
rectangular; different row lengths require explicit padding. Supplied initial
states are independent for each call; no hidden carry is added.

Fitted transforms preserve their admitted execution shape and train-only fitted
state. A standardization vector is not reinterpreted at a new length. Explicit
padding accepts raw ragged samples only up to its captured padded length; longer
samples refuse rather than truncate. Standalone inference lowers the same
helpers and captured checked-family runtime, needs Torch but not the installed
application, and excludes raw training observations. Existing captured source
archives are not rewritten.

## Verification

`tests/test_temporal_v2_tasks.py` checks exact versions/provenance, shape and RAM
guards, private/RNG-neutral/no-heavy-import preparation, explicit padding and
derived lengths, loss-mask independence, fitted-shape restrictions and disjoint
group/ordered-time split populations.
`tests/test_temporal_v2_task_execution.py` checks all16 constructor combinations
against independently scanned gate equations and Adam updates, actual worker
versus local checkpoint/metric/progress equality, unavailable/cancelled cases,
saved variable-T predictors, fresh application-free source, bad controls/states,
RAM refusal before construction and forged identity/memory evidence.
`tests/qt_temporal_v2_task_smoke.py` checks real native QThread/signals for three
data adapters with no Torch/NumPy import in the GUI process. Legacy probes compare
TaskBundle1/2 and3 identities/preparation/memory/shape plans against committed
modules. Full suite acceptance and integration state are recorded separately
in `docs/development.md` and campaign evidence; these tests alone are not D exit.
