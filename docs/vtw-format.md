# Visual Torch Workspace (.vtw): format specification, version 14

This is a first-class product specification for readers, writers, lab exchange
and agent tools. The executable reference is app/core.py, app/graph.py,
app/file_sync.py, app/revisions.py and vpw_cli.py. The reusable GUI has a separate specification:
[UI-AGENT-HANDOFF.md](UI-AGENT-HANDOFF.md).

The local Stage D [family specialization checkpoint](architecture-families.md)
retains immutable `family` and `specialization` revisions in the existing registry
tables, alongside the exact frozen concrete model. Atomic CLI admission can also
create ordinary editable drawings. These new closed document kinds require a
version-7 container; they add no SQL table or sidecar. The
[typed family editor](typed-family-sessions.md) uses transient owned sessions;
[explicit shared parameters](shared-parameters.md) now require version9;
[typed constructor choices](constructor-choices.md) require version10;
[scoped Module constructors](scoped-constructors.md) require version11;
[retained runtime H/W contracts](runtime-spatial-contracts.md) require version12;
[explicit runtime task execution](runtime-task-execution.md) requires version13;
[recurrent graph references](recurrent-graph-contract.md) require version14.
Runtime recurrent execution is not implied by finite tied uses.

## 1. Container, versioning and compatibility

Extension: .vtw; container: ordinary SQLite 3, not a ZIP or executable Python.
metadata.format must equal visual-torch-workspace. New saves use
PRAGMA user_version = 14. The current reader accepts versions 1–14; opening
an older file copies it into memory and upgrades the working version to 14.
Opening alone never rewrites the source; explicit Save writes version 14.

Other versions are rejected. Version 2 introduced tensor/port/DAG semantics;
version 3 pins input packing and records dataset bindings and checkpoint identity,
without changing SQL tables. Version 4 adds versioned structured operators,
optional named sample axes and a checked recorded structured dataset adapter,
without changing SQL tables. Version 5 adds independent immutable dataset/task/
model revisions and mutable browsing heads, with guarded SQL and exact content
validation. Version 6 adds closed experiment/run/report semantics to those same
revision tables, checks exact dependencies and append-only terminal attempts,
and captures actual split manifests. These are application semantics, not new
SQL tables. A version-1–5 header containing experiment/run/report rows is rejected.
Version 7 adds exact family/specialization provenance to the same immutable
registry. A version-1–6 header containing either new kind is rejected. Load/save
re-resolve each specialization from its exact family revision and all explicit
bindings, then compare its exact frozen model, not merely matching shapes/counts.
Version 8 adds closed experiment/run document version 2 for ordered family
binding axes and exact terminal specialization provenance, without SQL changes.
An earlier container containing these version-2 records is rejected. See
[family-experiment-axes.md](family-experiment-axes.md) for the complete grammar.
Version9 adds explicit local share_with module identity and version-2 family/model
documents without new SQL tables. Earlier headers containing shared drawings or
these v2 documents are rejected. ModelRevision v1 cannot contain sharing fields;
independent captures preserve v1 hashes. [shared-parameters.md](shared-parameters.md)
specifies constructor/owner/unique checkpoint paths and finite tied-repeat rules.
Version10 adds closed scalar constructor choices: family v3, specialization v2
and experiment/run v3, without changing tables or ordinary concrete model
versions. Earlier headers containing these documents fail; older integer-only
grammars/hashes remain unchanged. See [constructor-choices.md](constructor-choices.md).
Version11 adds family v4 scoped Module constructors and specialization v3's exact
local context records, without table or concrete ModelRevision changes. Typed
experiment/run v3 can already select these families. Earlier headers containing
family v4 or specialization v3 fail; old grammar/hashes stay unchanged. See
[scoped-constructors.md](scoped-constructors.md).
Version12 adds family v5's optional root runtime_inputs declaration and
specialization v4's mandatory canonical runtime witness (or null for fixed shapes).
Load/save re-resolve that witness from the exact family and bindings; earlier
headers containing these records fail. SQL tables, concrete ModelRevisions and
fixed-shape worker/task contracts do not change. See
[runtime-spatial-contracts.md](runtime-spatial-contracts.md) for the closed fields,
dependency proof and actual-call guards.
Version13 adds experiment/run v4's explicit family execution selection and exact
runtime TaskBundle v2 provenance/identity. Earlier headers containing those v4
records fail. Existing concrete experiments/bundles retain their old execution;
there are no new SQL tables, ModelRevision semantics or report recipe version.
See [runtime-task-execution.md](runtime-task-execution.md).
Version14 adds closed `Recurrent` multiport references and ModelRevision v3's
reachable transition closure. Earlier headers containing either new semantic
feature fail. No SQL tables, family1–5 grammar, TaskBundle1, experiment/run or
report recipe grammar changes. Existing concrete task execution can capture the
new model revision. See [recurrent-graph-contract.md](recurrent-graph-contract.md).
Older binaries are not expected to open version 14;
retain an original copy if an old reader must remain usable. This is not an
unversioned operator extension.

Open checks marker/version, SQLite integrity, foreign keys, document structure,
row/document ID/name consistency and reference-index correspondence. Semantic
drafts can open without compiling. Opening never executes model code or launches
training. These checks are not a hostile-file sandbox or memory guarantee.

Dataset int64 values use exact JSON integer tokens in the full signed 64-bit
range; see [dataset-task-contract.md](dataset-task-contract.md). Explicit float32
training preparation does not promise exact arithmetic on large integers.
The Qt [typed dataset session](typed-dataset-sessions.md) and raw-value buffers
are transient editor state, not `.vtw` rows, retained revisions or CLI sessions.
Only successful parent admission followed by workspace Save persists edits.
Task field/port suggestion pages are also read-only projections, not partial
model/dataset documents or persistent bindings. Their shared GUI/CLI protocol is
in [task-context-pages.md](task-context-pages.md); stored scientific order/IDs
and full metadata remain unchanged.
Owned task binding/transform cards, literal edit batches and compatibility
commands are also transient native transport, specified in
[typed-task-sessions.md](typed-task-sessions.md). They add no SQL rows or CLI
session tokens; the visible task form now uses that transport, including linked
child staging. Only explicit parent Commit followed by workspace Save persists
edits. Native transient origin proofs are not additional file identities.
Native experiment card/local-choice/value/edit/preview commands likewise add no
persisted rows or parallel format; [typed-experiment-sessions.md](typed-experiment-sessions.md)
specifies their transient grammar and actual form. Experiment cards/raw-child
buffers now share one transient session; only explicit parent Commit and file
Save persist an ordinary complete ExperimentSpec. No parallel SQLite store is added.

Window-owned recovery scope/sequence and browser pages are also transient;
[recovery-browser.md](recovery-browser.md) specifies their exact boundary. They
add no .vtw tables or durable jobs. Only explicit full recovery JSON export
creates external evidence; no browser projection is a persisted scientific spec.

## 2. Exact relational schema

TEXT payloads contain UTF-8 JSON without NaN/Infinity. The implemented DDL is:

```sql
PRAGMA user_version = 13;
PRAGMA foreign_keys = ON;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE architectures(
    id TEXT PRIMARY KEY, name TEXT NOT NULL,
    document TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE architecture_refs(
    owner_id TEXT NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    target_id TEXT NOT NULL REFERENCES architectures(id),
    PRIMARY KEY(owner_id,node_id));
CREATE TABLE datasets(
    architecture_id TEXT PRIMARY KEY REFERENCES architectures(id) ON DELETE CASCADE,
    payload TEXT NOT NULL);
CREATE TABLE runs(
    id INTEGER PRIMARY KEY,
    architecture_id TEXT NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, graph_snapshot TEXT NOT NULL,
    result TEXT NOT NULL);
CREATE TABLE spec_revisions(
    kind TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
    name TEXT NOT NULL, document TEXT NOT NULL, created_at TEXT NOT NULL,
    PRIMARY KEY(kind,id,revision));
CREATE TABLE spec_heads(
    kind TEXT NOT NULL, id TEXT NOT NULL, revision TEXT NOT NULL,
    PRIMARY KEY(kind,id),
    FOREIGN KEY(kind,id,revision)
        REFERENCES spec_revisions(kind,id,revision));
CREATE TRIGGER spec_revisions_no_update BEFORE UPDATE ON spec_revisions
BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be updated'); END;
CREATE TRIGGER spec_revisions_no_delete BEFORE DELETE ON spec_revisions
BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be deleted'); END;
CREATE TRIGGER spec_revisions_no_replace BEFORE INSERT ON spec_revisions
WHEN EXISTS(SELECT 1 FROM spec_revisions
            WHERE kind=NEW.kind AND id=NEW.id AND revision=NEW.revision)
BEGIN SELECT RAISE(ABORT, 'Immutable revisions cannot be replaced'); END;
```

Independent revision kinds are dataset, experiment, family, model, report, run,
specialization and task
(integer document version 1; experiment/run additionally support integer version
2 in a version-8 or newer container; family/model additionally support integer
2 in a version-9 or newer container; family/experiment/run additionally support
integer version3 and specialization supports integer version2 in a version-10
container; family additionally supports integer version4 and specialization
integer version3 in a version-11 container; family additionally supports integer
version5 and specialization integer version4 in a version-12 container);
other kinds/versions are rejected.
Revisions are content-addressed finite JSON and never overwritten or cascaded
from authoring architecture deletion. Browsing heads are mutable references to
retained revisions. Exact schemas, frozen reference closures, hash/guard checks,
working-copy-only migration, archive/reselection and undo retention are specified
in [revision-store.md](revision-store.md). Exact experiment, terminal run and report
recipe schemas are in [experiment-objects.md](experiment-objects.md).
Their JSON references are checked by the application on admission/load/save;
only head→revision references are SQL foreign keys. The locally accepted Stage C
GUI is specified in [experiment-gui.md](experiment-gui.md); the new D family
grammar has read/import integration and a [typed family editor](typed-family-sessions.md).
Notebook artifacts are separately specified in
[notebook-reports.md](notebook-reports.md). Legacy datasets/runs keep their
original meaning and existing APIs; no automatic typed-data interpretation occurs.

### Metadata

| Key | Value |
| --- | --- |
| format | Exact visual-torch-workspace marker |
| workspace_id | Generated UUID hex string identifying the collection |
| created_at | UTC ISO-8601 string with timezone |
| name | Workspace display name; initially Untitled workspace |
| constrained_mode | Text true or false; absent legacy key means true |
| layout_preferences | Optional finite JSON, closed layout profile version 1; absent means independent defaults. See [layout.md](layout.md). |
| panel_layout | Optional closed floating-panel preset version 1, CSS-pixel ordinary rectangles/flags/stacking; not computational history. See [panel-workspace.md](panel-workspace.md). |

Unknown metadata keys are preserved, not executed. Timestamps are write/run
evidence, not graph order or semantic versions. SQLite rowid determines the
current collection display order, not graph identity or a portable ordering API.

### Architecture document and dataset separation

architectures.document is a JSON object:

| Field | Contract |
| --- | --- |
| id | Stable string equal to its relational row ID |
| name | 1–100 nonblank characters; row name is the stripped document name |
| graph | Object with nodes and edges arrays |
| training | Optional object: epochs/lr/seed default to 400/0.03/7 |
| view | Optional legacy camera object: x, y, zoom, optional fitted. GUI opens ignore it and center/fit; current camera is session-only and GUI edits submit neutral view. See [layout.md](layout.md). |
| result | Latest result object or null |
| dataset_binding | Null, or a version-1 binding object identifying input/output groups by stable node ID, sample shape and canonical start/stop offsets, plus output_mode |

dataset is NOT duplicated in that column. Workspace.models() materializes it
from datasets.payload; Workspace.put() accepts the materialized document.
dataset: null deletes the current dataset. Unknown JSON fields survive normal
round trips but do not acquire execution semantics.

architecture_refs must exactly index every Module and Recurrent as
(owner ID, node ID, ref ID).
Targets must exist. A stale independent index is rejected; deleting a referenced
target is blocked. Deleting an unreferenced architecture cascades its legacy
data/runs, not independent revisions/frozen closures.
The current API keeps at least one architecture once a collection is populated.

## 3. Graph identity, wires and ordering

Each node has a unique string id and an allowlisted type. Optional finite x,y
are its diagram top-left coordinates (default zero, magnitude at most 100000).
They do not affect computation. Optional width,height are preferred diagram
dimensions in world units, finite positive numbers at most 100000 (booleans
rejected). Omit either to use its automatic dimension. The renderer takes the
maximum of the stored value and the readable typography/port minimum; small
stored values remain valid, and extra ports can grow a box without rewriting it.
These are presentation fields, not tensor dimensions. Changes preserve dataset,
run evidence and forward semantics. Optional collapsed must be a boolean;
true renders a compact icon-and-summary box with usable ports. Small labels show
input/output dimensions, Linear last-axis widths and bias state, or activation
name; long summaries truncate with full tooltip text. Folding is presentation-only,
retains expanded width/height preferences and participates in undo/redo.
Optional Input/Output label contains 1–60
nonblank characters; absent label means node ID. Names are unique separately
on each boundary. There are 2–64 nodes and at most 128 edges per graph; these are
prototype grammar limits, not constrained-mode limits.

Each edge has from/to IDs and an optional string from_port:

```json
{"from":"split","from_port":"coordinates","to":"coordinate_mlp"}
```

New router wires should specify from_port explicitly. It selects a declared
Splitter/Replicator port, not a destination node. Legacy router edges without
it select the port whose ID equals to. Ordinary tensor producers must not use
from_port. Duplicate (from,to,from_port) wires are rejected.

Ordinary producers may feed multiple consumers: they receive the same tensor
and autograd accumulates their paths. Replicator makes this reuse visible, but
is not mandatory. Input receives no wire; Output sends none. Ordinary boxes
receive exactly one wire; Concat alone receives multiple ordered wires.
Non-output nodes require a consumer; unused router ports may remain unconnected.
Complete graphs must be acyclic and reachable from Inputs. Dead/disconnected
drafts can be stored but cannot compile.

Inputs may follow independent paths and join later or terminate in separate
Outputs. There is no one-chain/one-join/terminal-router restriction.
Stable topological order comes from wires, never visual positions. Input roots
follow graph.input_order, an ordered list containing each Input ID exactly once.
Workspace.put pins this order when absent, deriving legacy meaning from a Concat
listing all Input IDs directly, or otherwise node-array order. Existing stored
order survives harmless node-array reordering; newly added roots append. Older
files are pinned in the in-memory upgrade, without rewriting the original.
Changing port identities/shapes/order does not implicitly rebind existing data.

## 4. Tensor semantics and operators

B is the runtime batch axis. Shapes omit B: [2,3] means a [B,2,3] tensor.
Sample shapes contain 1–8 positive integer dimensions. Reshape alone permits
one -1. Integers and sample element counts must be exactly representable by
JavaScript, maximum 2^53 - 1. This is not an allocation allowance.
Current training tensors are float32.

| Type | Computational fields | Meaning |
| --- | --- | --- |
| Input | shape; legacy features; optional axes | Exact named sample shape; explicit shape takes precedence; features=d means shape=[d]. axes lists unique sample-axis names (one per dimension), never B. |
| Linear | features, bias (default true) | nn.Linear(in_last,out_features); last axis only, shared weights across all preceding sample positions. |
| Activation | activation; Softmax dim=-1 | ReLU, Tanh, Sigmoid or exact GELU are elementwise. Softmax normalizes a slice along one sample axis, excluding B; dim indexes sample dimensions, negative indices count from the end. Runtime nn.Softmax gets the resolved index plus 1. Shapes and axes are preserved. |
| Reshape | shape | x.reshape(B,*resolved_shape); sample product preserved; one -1 inferred; B never reshaped. |
| Concat | sources, flatten (default true) | Ordered incoming tensors; flatten each sample then concatenate dim 1, or concatenate dim -1 with matching preceding dimensions when flatten=false. |
| Splitter | parts=[{output:port_id,features:size},...] | torch.split on last axis; positive sizes sum to incoming last-axis width; other axes preserved. |
| Replicator | outputs=[port_id,...] | All ports alias the same full tensor; no clone or parameters. |
| Module | ref | Independently instantiated graph reference; exact packed input shape match, canonical forward_tensor output. |
| Recurrent | ref, sequence_inputs, states, sequence_outputs, optional controls | Explicit transition graph unroll; named state/sequence output ports, constant shared step weights, validated control adapters. See recurrent-graph-contract.md. |
| Output | optional shape, label | Identity boundary. Shape asserts incoming dimensions; it never implicitly reshapes. |

Structured operators use integer op_version=1. Missing version is interpreted
as 1; writes normalize it explicitly along with all defaults and scalar spatial
parameters to two-element JSON arrays [height,width]. Unknown versions and
unsupported fields/options are rejected, not silently ignored. Logical Conv2d
layout is NCHW; this is separate from physical strides/memory format. Unnamed
rank-3 samples are interpreted as CHW. Named incoming axes must be C,H,W.
Permute, not Reshape, changes axis order and values; Reshape discards axis names.

| Type | Version-1 computational fields/defaults | Meaning |
| --- | --- | --- |
| Conv2d | out_channels; kernel_size; stride=[1,1]; padding=[0,0]; dilation=[1,1]; groups=1; bias=true; padding_mode=zeros | Derive in_channels from C. Positive channels/kernel/stride/dilation/groups; both channel counts divisible by groups. Weight [out,in/groups,kH,kW], optional bias [out]. Spatial output floor((size+2p−d(k−1)−1)/s)+1 must be positive. Numeric zero-padding only; strings same/valid and other boundary modes are not version-1 options. |
| AvgPool2d | kernel_size; stride=kernel_size; padding=[0,0]; ceil_mode=false; count_include_pad=true | CHW average windows; padding ≤ floor(kernel/2); floor spatial sizes. Include or exclude zero-padding from divisor as declared. No divisor_override option. |
| MaxPool2d | kernel_size; stride=kernel_size; padding=[0,0]; dilation=[1,1]; ceil_mode=false | CHW maxima, negative-infinity padding, floor spatial sizes; padding ≤ floor(kernel/2). No return_indices option. |
| AdaptiveAvgPool2d | output_size | Positive scalar/pair output H,W; channels preserved. [1,1] is global spatial average; PyTorch adaptive bin semantics, including overlapping bins. |
| Flatten | start_dim=0; end_dim=-1 | Axes index SAMPLE dimensions, excluding B. Negative indices count from sample rank. Merge an ordered contiguous axis range; runtime nn.Flatten gets resolved sample axes plus 1. Batch never merged. |
| Permute | dims | Permutation of sample axes 0..rank−1, excluding B. Runtime prepends batch axis 0 and shifts sample axes by 1. Example HWC→CHW: dims=[2,0,1]. |

Shape contracts report propagated output_axes and operator constructor/kwargs,
local parameters and weight_shape. Conv2d and averaging are affine for fixed
weights; MaxPool2d is nonlinear. Structured operators conservatively disable the
serial-vector matrix_affine report; they never build a dense global convolution
matrix. See [CNN contract and verification](cnn-workflow.md).

Router ports have at least two unique IDs. The UI Add port helper currently caps
each router at eight; there is no eight-input-path limit. Concat.sources lists
ordinary source node IDs, or node_id@port_id for router source edges. Entries
uniquely identify its incoming wires and fix order. Avoid @ in new node/port IDs
to keep these composite keys unambiguous. Two ports of one router can rejoin
directly by declaring both keys. Reshape may copy storage; zero-copy is not promised.

These operator rules follow [PyTorch Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)
and [torch.reshape](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html).

## 5. Callable interfaces and canonical prediction

forward(x) accepts either:

- One Input: its exact [B,*shape] tensor, or an exact one-name dictionary.
- Multiple Inputs: a flat [B,total_input_values] tensor, or an exact dictionary
  of named tensors with declared sample shapes and equal B. Packed parts follow
  input_groups, flattening each sample row-major; each part is restored to its
  declared shape before its independent path executes.

It returns the single Output tensor, or a dictionary of terminal names to actual
tensors when there are multiple Outputs. Native/exported modules evaluate the
same validated execution plan and agree with equal state dictionaries.

forward_tensor(x) is the canonical prediction for training and references:

1. One Output: its tensor, including sample dimensions.
2. Complete direct terminal Splitter/Replicator: the pre-router tensor where
   available, preserving legacy training behavior. Higher-rank split targets use
   original row-major order, not concatenated separately flattened slices.
3. Other multi-output graphs: concatenated flattened terminal samples.
   Exact aliases traced through Replicators share packed offsets and count once;
   distinct computations are not deduplicated by coincidentally equal values.

The derived contract reports input_shape/output_shape for canonical tensors;
inputs/outputs count scalar values per sample, NOT just last-axis width.
Per-node shapes contain full input_shape/output_shape plus legacy input/output
last-axis widths. It also reports parameters, affine, matrix_affine, order,
plan, tensor_source, resource summaries, groups and output_mode
(tensor/split/replicate/branches).

Groups include id/name/shape/features/start/stop. features is sample value count.
Direct split groups additionally specify axis_start/axis_stop. Extract higher-rank
split slices on the last axis; flat start/stop is not sufficient. Replicas share
canonical offsets. Other terminal groups use concatenated flat offsets.

Module references currently expose one bundled socket pair, not independently wired
named ports. Each occurrence has independent weights. Reference cycles are
rejected even when each internal DAG is acyclic. Its output presentation does not
replace its canonical adapter.

Recurrent instead has explicitly bound incoming wire keys and named outgoing
ports. Feedback is internal to its loop; the surrounding graph remains acyclic.
Recurrent bodies occupy a registered `transition` submodule below the owner.

State-dict paths are layer_0, layer_1, ... for Linear/Activation/Reshape/Module/Recurrent
and the six structured operators above,
in topological order, nested through references. Boundary/Concat/router nodes
do not occupy these operation indices. Paths can change after graph insertion/
reordering; restore weights against their graph snapshot, not another architecture.

## 6. Data, training and evidence

This section describes the unchanged architecture-owned legacy adapter. Independent
dataset/task/model revisions use the revision-store schema introduced in v5. Their
task-aware worker returns a separate frozen JSON training artifact and fitted
predictor, specified in [task-training-contract.md](task-training-contract.md).
It does not silently write that artifact into legacy `runs` or `result`.
The v6 `run-candidate` wrapper explicitly stores its validated terminal evidence
as an independent RunRecord, with exact dependencies and captured actual manifest.
See [experiment-objects.md](experiment-objects.md). GUI/history
integration remains a subsequent Stage C gate.

The local Stage C `scienceRunCurve`/CLI `run-curve` interface derives paged head
curves from these immutable RunRecord attempts. It adds no tables or persisted
objects, changes no existing evidence and never trains on read. The exact read
projection, fixed axes and missing-value rules are in
[saved-run-curves.md](saved-run-curves.md).

datasets.payload contains matching x/y arrays with at least two flat rows.
x has inputs values; y has outputs canonical values. Values are finite numbers,
not booleans, with magnitude at most 1e6. Optional truth/source/seed/noise fields
record provenance. CSV uses x columns then y columns, an optional header, and
the app's 8 MB import guard.

Training reshapes x to the canonical input, executes the DAG, then flattens the
canonical prediction for MSE. This is a training adapter, not an implicit graph
Reshape. Full-batch CPU Adam uses one worker thread and seeded shuffled 80/20 split.
Epochs are positive integers; lr is finite in (0,1]; seed is 0 through 2^31-1.
Parameter-free graphs compile but cannot train with this optimizer.

Synthetic targets use a flat affine A,b, deterministic Tanh teacher, or a spatial
Conv2d→Tanh→global average→Linear teacher. Eligible CNNs choose spatial in Auto;
other models keep the affine/nonlinear choice. Structured single-input synthetic
datasets record adapter={version:1,kind:declared-row-major,sample_shape,axes}.
Its exact shape/axis meaning is checked before runs and trained export; a
mismatch cannot be repaired by column rebinding, which does not move values.
Legacy flat datasets without this optional adapter remain supported through
their recorded port/shape bindings. Bindings add axes only when named; old
unnamed bindings/checkpoint identities keep their original meaning.

A shared-
position higher-rank Linear model may not represent an arbitrary flat target.
Low loss is not guaranteed. Same graphs/seeds do not guarantee bitwise numerical
reproducibility across Torch versions/hardware.

Latest result includes learned state-dict weights, loss history, held-out mse,
flat previews, row counts, stop state, layer descriptors, shapes/groups/mode.
Post-update train/held-out observations describe the same evaluation-mode
checkpoint. requested_epochs and completed_updates are distinct; epochs is the
actual completed update count, including cancellation between plot samples.
split_indices identifies the training/held-out rows; metric_protocol states
MSE, post-update timing, evaluation mode and float32. Stop reason, elapsed time
and examples_processed are explicit. Layer descriptors include stable node-ID
paths alongside state-dict paths. Optional affine analysis reports available,
unavailable or error independently of successful training evidence.
Nonlinear graphs do not report effective A,b. Current A,b reporting is limited
to serial vector-affine compositions (matrix_affine); valid affine branching or
higher-rank geometry can lack that compact report.

runs.graph_snapshot stores the whole materialized architecture collection used
by that run (possibly including datasets); runs.result is its result evidence.
Semantic graph edits clear latest results in the graph and transitive users.
Dataset or training changes clear that graph's latest result. Layout, resizing,
folding and boundary display-label edits preserve computational identity.
Results with identity record version-1 SHA-256 digests of the reachable graph
closure, dataset, binding and training configuration. Checkpoint restore/export
requires matching identity, checkpoint_version integer 1, the supported identity
preprocessing adapter/binding, exact stable parameter IDs and state-dict keys,
exact tensor shapes and finite float32-representable numeric weights (booleans
and string coercion are rejected). Validation does not allocate a Torch model.
Native and exported parameter constructors explicitly select CPU float32 instead
of inheriting a process-wide default dtype. Strict state-dict loading follows
admission. This is an inference checkpoint, not an optimizer/resume checkpoint
or authenticated proof of how a manually supplied result was obtained. Historical runs are
not rewritten. Regular source exports instantiate fresh weights; export-trained
embeds the matched trained state for evaluation-mode inference. A history browser
is not implemented. Exact-revision result/rerun notebook bundles are implemented
separately; they do not silently reinterpret this legacy result format.

Existing datasets retain their binding when only the graph changes. Training
rejects an incompatible binding rather than silently assigning old columns to
new ports. Replacing data creates a binding to the current valid contract;
CLI rebind-dataset explicitly declares that existing columns now mean the current
port layout. This does not reorder or transform those columns. Old datasets
without bindings are bound once to their migrated legacy contract when valid.
Rebinding requires valid current x/y widths/values; it clears the current result
but does not alter historical run evidence. The Qt Experiment panel shows both
recorded and current bindings, blocks incompatible/unbound data before launching,
and offers a confirmed, undoable Use current port order action. Cancel or a draft
change during the modal confirmation does not rebind.

Checkpoint display status is current, unverified (legacy without identity), stale,
invalid or none. Current means snapshot-compatible, not an independently
authenticated scientific claim. Legacy metrics are labelled unverified. Stale or
invalid results do not supply current displayed metrics/loss plots. Source export
is explicitly labelled a fresh-weight template; trained inference uses the
validated trained export API/CLI. Worker completions are checked against their
captured snapshot before entering current/history storage.

Notebook bundles use `.ipynb` v4 plus a closed version-1 input manifest and exact
revision closure, outside the SQLite container. Generation and actual-kernel
execution are distinct explicit commands; neither writes the source workspace.
The GUI's inspected bundle selection, active job and session receipt history are
not SQLite tables/revisions. Durable requested/executed notebooks and execution
receipts live in external bundle attempt directories; rerun results are not
automatically admitted as workspace RunRecords. No v6/schema change is introduced
by explicit GUI execution. See [notebook-reports.md](notebook-reports.md) and
[notebook-gui-execution.md](notebook-gui-execution.md) for exact contracts and limits.

## 7. Run policy is not graph meaning

constrained_mode defaults on and can be disabled/re-enabled without shrinking
models or deleting data/results/runs. Checks happen before allocation, synthetic
generation and training; host and isolated worker enforce the captured mode.
Missing worker mode means true. Policy cannot change during an active job.

The execution criterion is estimated whole-run bytes ≤ a user percentage of
currently available RAM, default 10%. Parameter tensors, expanded reference
occurrences, activations/batch, optimizer, data/copies, evidence and runtime
headroom count. Linux MemAvailable excludes swap and observable cgroup-v2 limits
can reduce it. Unknown RAM blocks constrained execution. The exact versioned
estimator is specified in [memory-policy.md](memory-policy.md); it is a heuristic,
not an OOM guarantee or a CPU-time limit. Former fixed width/parameter/row/epoch
execution caps are removed. Prototype grammar/transport limits remain separate.

Large valid specs remain editable/saveable/exportable in constrained mode.
Executing exports outside the app is outside these checks. Unconstrained mode
does not disable grammar, finite-value, bridge-size or browser-security rules.
A saved opt-out travels with a shared file; review before running.

## 8. Writing, revisions and desktop synchronization

The desktop edits an in-memory database. Save writes and flushes a temporary SQLite
snapshot beside the destination, then atomically replaces it. Successful saves
record SHA-256 content revisions. There is no persistent WebEngine model store.

Cooperative writers use <absolute workspace path>.lock with exclusive advisory
flock. This persistent sidecar inode survives database replacement; never delete
it during normal operation. Acquisition is nonblocking: report busy/retry.
Check the expected revision while holding it before replacing the workspace.
App/CLI refuse stale saves rather than silently overwriting another writer.

This is single-host cooperative locking, not distributed synchronization.
Arbitrary SQL writers ignoring it are unsupported and can race the protocol.
It provides atomic replacement, not an undo archive or power-loss durability
guarantee for all filesystems.

The desktop polls inode/size/mtime once per second and checks changed content.
Clean, idle workspaces reload and hydrate existing tabs; missing models navigate
to a remaining architecture/home. Unsaved edits cause a warning and are retained,
not overwritten. Save As preserves them separately; File → Reload asks before
discarding. Active runs postpone reload; subsequent results remain local evidence,
so conflicts are preserved rather than auto-merged. There is no three-way merge.

Use [vpw-cli.md](vpw-cli.md), not raw SQL, for agents: it maintains reference
indices/results and applies an edit batch with one save. --expect-revision makes
read/plan/edit explicit. Share saved .vtw snapshots, not unsaved GUI buffers.

Stored candidate browsing has a separate non-document projection, also exposed
as CLI `preview-experiment-page`: exact retained identity/coordinates/status/
references/counts, explicit partial reasons and decimal totals, no full manifests
or resolved protocols. It is not an admitted ExperimentSpec or persisted SQL
record; the complete `preview-experiment` read remains available. No v6 schema
change follows. See [candidate-browser.md](candidate-browser.md).

Session undo/redo stores at most 50 edit steps and 128 MiB of combined SQLite
snapshots in a private temporary directory, NOT extra .vtw tables. Save retains
history; opening/external reload starts fresh history. Undo marks the workspace
dirty without rewinding the known disk revision. Run evidence for surviving
restored architectures is preserved; merged run row IDs need not remain stable.
Training completion is not a workspace edit-history step. Oversized edits clear
history with a notice; history is not a durable recovery/backup facility.

User settings are separate: ~/.tensorloon/config.json stores ui_zoom (0.5–3),
font_size (integer 12–28), theme, spacing (integer 1–10), box_spacing (integer
1–100), memory_limit_percent, grid_snap and menu_grid_snap (strict booleans).
Defaults are 1.15, 14, dark, 4, 35, 10, false and false. grid_snap applies only to
blocks; menu_grid_snap applies to panels/pack. Ctrl inverts each independently
during its movement/resize; Shift retains menu-layer movement. See [panel-workspace.md](panel-workspace.md).
Older string spacing values compact/spaced load as 4/10 in memory, without a
file write; the next explicit preference Save writes numeric spacing. New
preference changes accept only integers in range, not strings, floats or booleans.
Theme/spacing/box geometry do not affect forward semantics; Preview is transient,
Save persists preferences and Cancel restores them. Unknown config keys survive;
malformed config is preserved and must be repaired before preference writes.
The Visual PyTorch rename preserves ~/.tensorloon/config.json and the
VisualTorch/TensorLoom QSettings namespace; no workspace or preference schema
change is introduced by branding.
QSettings stores recent paths. Floating-menu geometry is transient per view;
graph top-left positions, preferred block width/height and diagram zoom are .vtw
presentation data. User box spacing changes automatic/readable-minimum dimensions
and port geometry, never these saved values or computational fields. GUI resizing
is a session undo/redo step. CLI resize sets both dimensions; set-node with empty
values and remove_fields=["width","height"] restores automatic sizing. Themes
and unordered architecture summaries are sorted alphabetically; execution order,
boundary packing, source/port arrays and edit batches are never alphabetized.
