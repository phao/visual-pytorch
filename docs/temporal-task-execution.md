# Explicit temporal experiment execution — local Stage D checkpoint

Reference implementation: `family_axes.py`, `runtime_tasks.py`,
`experiment_objects.py`, `task_execution.py`, `task_result.py`, `worker.py`,
`report_runtime.py` and the owned experiment form. This integrates retained
[temporal family6](temporal-families.md) with actual task execution. It is not a
claim that all Stage D or durable/heavyweight Stage E work is complete.

## Explicit choice, not inferred execution

ExperimentSpec5 and RunRecord5 extend the preserved version4 grammar. A family
choice may explicitly set `execution:"runtime-temporal-v1"`. The existing values
`concrete-v1` and `runtime-spatial-v1` remain valid; omission still means concrete.
Concrete ModelRevision choices have no execution field. Experiment versions1–4
reject the temporal choice. A retained dynamic family or materialized drawing does
not automatically change an experiment's execution mode.

Temporal selection requires the exact family6 revision and nonnull specialization5
witness. Before admission, re-resolution reproduces fallback bindings, constructor
contexts, baseline frozen model and the complete temporal specification digest.
Null temporal policy or a spatial policy selected on a recurrent family is
incompatible, not silently repaired. Browsing heads never substitute for exact
references. Conditional/unresolved points remain visible but cannot run.

Experiment/run5 records require a `.vtw`16 container, with no new tables. A header
below16 containing them is rejected. Family6/specialization5 retain their minimum
header15, and older ordinary/spatial documents keep their previous semantics.
Opening upgrades only the private working database; Save is explicit.

## Frozen task execution and checkpoint identity

TaskBundle3 has exactly the version1 keys (`version`, `model`, `dataset`, `task`,
`manifest`, `protocol`) plus `runtime`, exactly `{family,specialization}` containing
complete frozen revision documents. The `model` is still the constructor baseline;
the dataset-derived actual contract is planned separately. Bundle2 remains spatial
family5/specialization4 only; a temporal witness cannot be relabeled as bundle2.

Run identity3 has the same model/dataset/task references, manifest/protocol hashes
and `task_runtime:"masked-additive-v1"`, plus
`compiler:"native-runtime-temporal-v1"` and `runtime:digest(bundle.runtime)`.
Concrete identity1 and spatial identity2 remain unchanged. Equal learned shapes
do not make these execution modes interchangeable. Existing checkpoint/result
envelopes stay version1 and compare this complete execution identity. Checkpoint
parameter IDs/paths come from the exact frozen model; saved prediction does not
promise cross-constructor optimizer continuation or hidden recurrent state.

Workers accept `kind:"task-v3"` with exact integer bundle version3. Kind/version
mismatch, forged proof/model/bindings/witness or checkpoint identity fails before
numerical training. Native construction, worker fitting, restore, checked runtime
architecture export and predictor export use the same validated compiler path.
Actual task dataset input shapes are passed as the first admitted case; unused
baseline T is not substituted. One set of learned owners is reused at valid T.

## Data, state, controls and objectives

For each declared temporal Input ID, a task must explicitly bind a numeric field.
Its post-transform rectangular sample shape determines the actual case. The pure
temporal proof checks rank, undeclared dimensions, common T within each recurrent
call and fixed learned/frame/state dimensions. Task checks then verify stable port
IDs, semantic axes, target/head shapes, loss masks, transforms and split membership.
No constructor symbol is inferred from data.

Sequence fields can be `[T,D]` or `[T,C,H,W]`; masks/reset are `[T]`; initial states
and lengths remain their declared fixed dimensions. Controls use the graph's
explicit exact-bool/exact-int64 adapters. Probabilities, fractional lengths and
out-of-range values are rejected rather than cast silently. Every exposed task
Output still requires an explicit positive-weight head under TaskSpec1. Execution
masks/lengths and loss masks are distinct bindings; one never implies the other.
Empty supervision retains unavailable evidence and skips optimizer updates.

Different dataset/task pairs may have different rectangular T with the same exact
family specialization and learned schema. No ragged batch is padded implicitly.
Existing explicit pad transforms may collate a ragged numeric field to their
literal fixed length; matching controls and targets remain explicit. The runtime
shape planner does not infer a derived length/mask for a dynamic port. Existing
derived bindings and their explicit pad rules remain ordinary task semantics;
direct runtime ports require explicit numeric field bindings. A fully dynamic
derived-control adapter is not introduced here.

Task run admission includes typed preparation, parameters/optimizer/evidence,
history, actual-minibatch activations/workspaces and conservative nested recurrent
unroll copies/controls. Stored result validation independently reproduces the
unroll charge. Every actual forward/predict additionally checks current-B/T RAM
before packing/control conversion/unrolling. This is a conservative heuristic,
not an allocator limit or OOM guarantee. Full BPTT, active-row-only evaluation,
zero padded trajectories, state holds and resets to explicit call-initial state
remain the recurrent executor's semantics.

## Saved inference, preprocessing and portable reports

Saved native/app-free predictors accept new rectangular T for untransformed
runtime fields, preserving all fixed frame/state dimensions and one common
sample shape per field/batch. Pure shape planning and RAM admission precede tensor
construction. All supplied fields/controls and finite values are validated.

For a runtime input with fitted transforms, new records must satisfy its saved raw
field shape. The captured fitted execution shape is used after applying those
captured transforms; neither per-feature statistics nor pad lengths are refitted
or reinterpreted for a new T. Explicit ragged pad input is validated as ragged raw
data, then collated to the saved literal length; overlong data is refused rather
than truncated. Changing transformed-input T needs a separately specified
resolution-independent preprocessing adapter, not silent reuse of feature means.

Generated report bundles capture the existing complete30-file runnable closure
and exact temporal selections. Both saved-result and explicit rerun notebook
kernels can produce temporal runtime architecture files and checked saved
predictors. Saved mode reads retained evidence; rerun mode explicitly trains.
Both preserve exact checkpoint IDs, metrics, frozen references and split manifests;
generation and execution remain separate actions. No durable background scheduler,
accelerator backend or automatic external execution is introduced.

## Qt and CLI

The owned experiment family card has **Family execution → Runtime sequence T**.
Choosing it queues version5 before the property edit; choosing another mode never
downgrades5. Preview stays read-only/allocation-free. Retain before Run; normal file
Save persists the revision/result. Existing owner/sequence/workspace/head/busy/
staged-child fences remain. The CLI uses the same ExperimentSpec in `put-spec`,
preview/run-candidate/run evidence/export-run-trained and notebook commands, with
no separate database/editor session.

## Verification scope

Tests cover two dataset T cases over one frozen baseline, closed/versioned proof,
forgery/null/spatial refusal, actual-case controls/targets/state incompatibility,
pure imports/CLI/load gates, owned upgrades and current memory. Numerical tests
compare an independent per-observation recurrence/Adam oracle at float32 tolerance;
native/worker checkpoints, saved/app-free predictions and retained/replayed notebook
checkpoint IDs/metrics remain exact. CNN transition fitting and explicit pad/
standardization lowering are tested. Actual Qt/WebEngine performs an explicit
temporal edit, read-only Preview, Retain, three-update worker and .vtw reload without
host Torch or JavaScript console errors. The unchanged spatial Qt workflow is
also checked. These are concrete verified workflows, not full D/E acceptance.

For a portable GUI-ready example, run
`python -m examples.temporal_experiment /path/to/new-temporal.vtw`.
Creation uses only standard-library Python and refuses existing destinations.
Add `--train` in a PyTorch environment to explicitly execute both T=3/7 trials;
the default constrained policy remains enabled. Each trial constructs its own
independent instance of the same exact family, not shared weights across trials.
Open the file, then Experiment → Experiments & history → Two explicit time cases.
