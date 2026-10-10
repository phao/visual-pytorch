# Concrete experiments, terminal runs and report recipes — version 1

This is the executable Stage C task-branch contract, not a completed experiment
GUI or notebook generator. Reference implementations are
`app/experiment_objects.py`, `app/task_result.py` and `app/revisions.py`.
All three object kinds require a `.vtw` container with `user_version=6`.
The SQL tables and reference JSON hash convention are in
[revision-store.md](revision-store.md). A revision is local integrity evidence,
not authentication or proof that an experiment happened.

## Common rules and exact references

Objects have exactly the keys described below; additional keys are rejected.
Every object has integer `version: 1`, nonempty string `id` and `name` (maximum
200 characters, no characters with code point below 32), plus an optional verified
`revision`. Whitespace-only strings are not separately rejected by this grammar.
Storage computes
that revision, returns an owned copy and never modifies a retained revision.
Question/reason text is nonempty, at most 20,000 characters and has no NUL.
An exact reference has exactly `kind`, `id`, `revision`; the latter is 64
lowercase hexadecimal SHA-256 characters. References must resolve to retained
documents with that same kind, ID and revision. Display names and mutable
browsing heads never resolve an execution dependency implicitly.

Dependencies are acyclic by kind:

```text
ExperimentSpec → ModelRevision, DatasetSpec, TaskSpec
RunRecord      → ExperimentSpec → those exact revisions
ReportRecipe   → exact RunRecord revisions OR one exact ExperimentSpec
```

The application checks JSON dependencies on put/get/load/save. The SQL FK only
checks head→revision. Authoring architecture deletion, head archival and undo do
not delete these retained documents. No garbage collection is implemented.
An ordered CLI edit batch must create dependencies before their consumers.

## ExperimentSpec

Required keys besides the common header:

| Key | Exact representation |
| --- | --- |
| question | Explicit research question text |
| models | Nonempty ordered array of distinct exact `model` references |
| data_tasks | Nonempty array of `{id,dataset,task}`; pairing IDs are unique; dataset/task are exact references |
| splits | Nonempty array of `{id,protocol}`, with unique selection IDs |
| initialization_seeds | Nonempty array of distinct integers in 0..2³¹−1; booleans rejected |
| protocols | Nonempty array of `{id,protocol}`, with unique selection IDs and closed TrainingProtocol v1 |
| selection | Exactly `{kind,pairs,exclusions}`, defined below |

Each split protocol has exactly the fields associated with its method:

| method | Keys |
| --- | --- |
| row | method, seed, fractions |
| group | method, seed, fractions, field |
| temporal | method, fractions, field |
| external | method, roles, excluded |

Fractions are [train, validation, test], finite in [0,1], sum to 1 within
absolute tolerance 10⁻¹², with positive train/test. Row/group seeds are integers
in 0..2³¹−1. Group/time `field` is a stable field ID. External roles have exactly
train/validation/test ID arrays; excluded is another ID array. All IDs across
these four arrays are distinct; train/test must be nonempty. Dataset-dependent
membership, shape, group isolation and chronology are checked at candidate
resolution, not guessed when storing an intrinsically valid selection.
See [dataset-task-contract.md](dataset-task-contract.md) for actual allocation,
ordering, retained manifest fields and split edge cases.

Selection `kind` is `crossed` or `paired`. Crossed requires `pairs: []`.
Paired requires a nonempty whitelist of exact `{model,data_task}` pairs.
Exclusions are distinct `{model,data_task,reason}` entries referring to selected
axes; they take precedence over the whitelist. A pairing describes selected
model/data combinations, not a statistical independence claim. Selecting a
matching initialization integer for different models does not align their
random variables. The current analysis is descriptive, not a confidence interval.

TrainingProtocol is specified completely in
[task-training-contract.md](task-training-contract.md). Each candidate's
initialization seed comes from the initialization axis, overriding the template's
`seeds.initialization`; shuffle seed and all other protocol fields remain as
declared. Generation/provenance belongs to the dataset; split randomness belongs
to each split; stochastic augmentation is explicitly unsupported (`none`).

## Pure, bounded preview

`preview_experiment(experiment, resolve, offset=0, limit=100)` returns exactly
`version,experiment,offset,limit,total_combinations,candidates,next_offset,scope`.
Offset is a nonnegative integer and limit an integer in 1..1000.
The declared product order is model × data/task × split × initialization ×
training configuration, last axis fastest. These are ordered protocol axes,
not alphabetically sorted browsing lists. Mixed-radix indexing jumps directly
to an offset without constructing skipped combinations. A page can be empty
past the end. Counts/offsets are Python integers; consumers using JavaScript
must avoid lossy Number conversion for values beyond its safe integer range.

Each candidate has exactly `id,experiment,coordinates,status,reason,references,
manifest,protocol`. Coordinates have exactly `model,data_task,split,
initialization,protocol`: model is an exact reference; data_task/split/protocol
are selection IDs. Candidate ID is the reference hash of the experiment
reference with the additional `coordinates` key. Status is `compatible`,
`incompatible` or `excluded`; unsuccessful combinations stay in the page with
a visible reason. References identify model/dataset/task exactly. Compatible
candidates contain the actual SplitManifest and resolved TrainingProtocol.
Preview checks trainable parameters, bindings, shapes and split membership, but
does not fit transforms, train, launch a worker or change global RNG state.
Compatibility is not a RAM admission promise. A report selected from mutable
heads would not have this reproducibility contract.

## RunRecord and captured resolution

Required keys besides the header are `selection,resolved,attempts`.

Selection has exactly `experiment,coordinates,candidate_id`, with the exact
experiment reference and candidate coordinates/identity above. Resolved is null
for a compatibility/exclusion failure; otherwise it has exactly:

```text
version: 1
references: {model: exact reference, dataset: exact reference, task: exact reference}
manifest: the actual validated SplitManifest, including observation IDs
protocol: the actual closed TrainingProtocol with resolved initialization seed
```

Resolution is frozen before admission/worker launch. Reads, retries and report
views use this captured manifest, not a freshly regenerated split. Its protocol
must correspond to the declared selection; row/group manifests add
`algorithm: python-mt19937-fisher-yates-v1`, and group/temporal add
`allocation_unit: distinct-field-values`. Manifest validation checks exact
dataset revision, IDs, roles/exclusions and applicable grouping/chronology.

Attempts is a nonempty ordered array. Each attempt has exactly:

| Key | Contract |
| --- | --- |
| number | Consecutive integer starting at 1 |
| started_at, finished_at | Explicit UTC ISO timestamps; finish ≥ start; next start ≥ previous finish |
| stage | compatibility, admission, worker or execution |
| status | completed, failed, cancelled or unavailable |
| error | null, or exactly `{type,message}`; type follows the ID grammar, message the text grammar |
| result | null, or the validated task-training-result with `bundle` removed |

A checkpoint-free attempt cannot claim execution/completion/cancellation. It
must be failed/unavailable with an explicit diagnostic. Compatibility failures
have null resolved; admission/worker failures have captured resolution.
Execution result.status must equal attempt.status; failed execution requires
an error, other execution states require null error. All prior attempt bytes,
selection and resolution are immutable across revisions of a logical run.
A new revision appends without truncating the longest retained history; retries
read that history even if undo moved the head backwards. Explicitly selecting
an already retained older revision is allowed.

`execute_candidate` makes one explicit synchronous attempt through the real
bounded JSONL worker. An existing run ID requires `retry=True`; selection
substitution is rejected. A retry is a fresh training attempt, not optimizer
continuation. Failure at compatibility, RAM admission, missing executable,
import/initialization, transport or task execution is retained with its stage.
All-mask unavailable and cooperative cancellation are terminal task states.
The caller performs the atomic workspace Save; closing/killing the caller before
Save is not durable execution. Queued/running receipts, detached reconnection,
preemption/resume and heavyweight backends remain Stage E work.

Stored result keys are exactly the task result keys documented in
[task-training-contract.md](task-training-contract.md), except `bundle`.
Pure admission validates execution/checkpoint hashes, fitted state, parameter
IDs/keys/shapes/finite weights, actual counters/budget boundary, history retention,
checkpoint/population timing, train/validation-only curves and completed-only
test evaluation. Per-head observation IDs, evaluated/included counts and weighted
denominators are checked against the frozen task and actual populations, using
the runtime's float32 mask/weight conversion. Means, accuracy and weighted
objective must match their additive terms. Environment, sampler and version-2
RAM components must match the stated runtime/policy.
This checks internal evidence and compatibility, not neural prediction truth
or authenticated execution origin. It does not import Torch/NumPy/Qt, but fitting
validation reconstructs expected preprocessing in pure Python; it is not a
hostile-file sandbox or constant-memory verifier.

## ReportRecipe and read-only materialization

Required keys besides the header are `question,mode,runs,experiment,sections,analysis`.
Mode `result-report` requires nonempty distinct exact RunRecord references and
null experiment. Duplicate logical run IDs are rejected, even at different
revisions: retries/snapshots are not additional independent run-level units.
Mode `rerun` requires `runs: []` and one exact ExperimentSpec reference.
No mixed implicit execution mode exists.

Sections is a nonempty ordered unique subset of question, protocol, models,
datasets, training, evaluation, failures, environment, checkpoints.
Analysis is exactly `{kind: descriptive, inferential_unit: run}`.
No search/CI/bootstrap algorithm or undeclared inference is accepted.

`report_data` returns exactly `version,recipe,question,mode,sections,analysis,
execution_behavior,experiment,runs,notebook_status`.
Result views use `execution_behavior: read-saved-evidence-only`; rerun views
use `explicit-execution-required` and do not launch merely by viewing them.
`notebook_status` is `not-generated`: this API materializes evidence, not a
notebook. Report rows have exactly `reference,name,selection,resolved,
latest_attempt,status,stage,error,checkpoint_id,counts,metrics,environment,
attempt_history`. A row summarizes the latest attempt of that selected immutable
run revision, not an automatically newer head. Attempt history keeps
number/start/finish/stage/status/error for every attempt. Failed runs remain visible;
missing numerical evidence is null/empty, never fabricated. A future retry cannot
change a result recipe pointing at an older revision.

## CLI, recovery and executable example

[vpw-cli.md](vpw-cli.md) specifies preview-experiment, run-candidate and report-data.
On a concurrent Save conflict, run-candidate returns the complete recoverable
record with `persisted: false` and `save_error`, without overwriting external
edits. Do not interpret the command's `ok: true` envelope as a completed fit or
successful Save. Inspect both terminal attempt status and persisted flag.
Re-inspect the current file, then explicitly put the retained record through a
revision-checked edit batch; do not rerun the fit just to recover its evidence.
All exact dependencies must still exist in the receiving workspace.

From the product repository root:

```sh
python -m examples.experiment_workspace /path/to/new.vtw
python -m examples.experiment_workspace /path/to/another-new.vtw --run-first
```

The example refuses overwrites and creates two frozen models × two deterministic
datasets × two initialization seeds, one split/configuration, and a rerun recipe.
By default it only saves and previews. The explicit flag executes one bounded
candidate and stores a result-report recipe including any terminal failure.
It prints exact references and preview IDs; no user config or existing workspace
is modified. The example does not execute a generated notebook.

## Verification boundaries and design references

Tests execute every combination of a separate 2×2×2 fixture and explicitly retry
each: exact checkpoint/metric replay is checked in the same local environment.
They exercise visible incompatibility/exclusion, train-free paging, immutable
dependency corruption, fixed manifests under a disabled split generator, honest
weighted/classification metrics, actual RAM failure, real missing executable,
concurrent Save recovery, undo/old-head retry and read-only report views.
Fresh stdlib-only readers require no numerical/Qt imports. These are bounded CPU
tests, not distributed/heavyweight validation or full Stage C acceptance.

[Python reproducibility notes](https://docs.python.org/3/library/random.html#notes-on-reproducibility)
motivate recording actual manifests instead of assuming all seeded procedures
remain stable across versions. [SQLite foreign keys](https://www.sqlite.org/foreignkeys.html)
support the exact relational head key; JSON dependencies need the additional
application checks described above. These references explain design choices,
not proof of untested behavior.
