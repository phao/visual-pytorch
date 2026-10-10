# Visual Torch Workspace (.vtw): format specification, version 3

This is a first-class product specification for readers, writers, lab exchange
and agent tools. The executable reference is app/core.py, app/graph.py,
app/file_sync.py and vpw_cli.py. The reusable GUI has a separate specification:
[UI-AGENT-HANDOFF.md](UI-AGENT-HANDOFF.md).

## 1. Container, versioning and compatibility

Extension: .vtw; container: ordinary SQLite 3, not a ZIP or executable Python.
metadata.format must equal visual-torch-workspace. New saves use
PRAGMA user_version = 3. The current reader accepts versions 1, 2 and 3; opening
an older file copies it into memory and upgrades the working version to 3.
Opening alone never rewrites the source; explicit Save writes version 3.

Other versions are rejected. Version 2 introduced tensor/port/DAG semantics;
version 3 pins input packing and records dataset bindings and checkpoint identity,
without changing SQL tables. Older binaries are not expected to open version 3;
retain an original copy if an old reader must remain usable. This is not an
unversioned operator extension.

Open checks marker/version, SQLite integrity, foreign keys, document structure,
row/document ID/name consistency and reference-index correspondence. Semantic
drafts can open without compiling. Opening never executes model code or launches
training. These checks are not a hostile-file sandbox or memory guarantee.

## 2. Exact relational schema

TEXT payloads contain UTF-8 JSON without NaN/Infinity. The implemented DDL is:

```sql
PRAGMA user_version = 3;
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
```

### Metadata

| Key | Value |
| --- | --- |
| format | Exact visual-torch-workspace marker |
| workspace_id | Generated UUID hex string identifying the collection |
| created_at | UTC ISO-8601 string with timezone |
| name | Workspace display name; initially Untitled workspace |
| constrained_mode | Text true or false; absent legacy key means true |

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
| view | Optional presentation object: x, y, zoom, optional fitted |
| result | Latest result object or null |
| dataset_binding | Null, or a version-1 binding object identifying input/output groups by stable node ID, sample shape and canonical start/stop offsets, plus output_mode |

dataset is NOT duplicated in that column. Workspace.models() materializes it
from datasets.payload; Workspace.put() accepts the materialized document.
dataset: null deletes the current dataset. Unknown JSON fields survive normal
round trips but do not acquire execution semantics.

architecture_refs must exactly index every Module as (owner ID, node ID, ref ID).
Targets must exist. A stale independent index is rejected; deleting a referenced
target is blocked. Deleting an unreferenced architecture cascades its data/runs.
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
| Input | shape; legacy features | Exact named sample shape; explicit shape takes precedence; features=d means shape=[d]. |
| Linear | features, bias (default true) | nn.Linear(in_last,out_features); last axis only, shared weights across all preceding sample positions. |
| Activation | activation | ReLU, Tanh, Sigmoid or exact GELU; axes preserved. |
| Reshape | shape | x.reshape(B,*resolved_shape); sample product preserved; one -1 inferred; B never reshaped. |
| Concat | sources, flatten (default true) | Ordered incoming tensors; flatten each sample then concatenate dim 1, or concatenate dim -1 with matching preceding dimensions when flatten=false. |
| Splitter | parts=[{output:port_id,features:size},...] | torch.split on last axis; positive sizes sum to incoming last-axis width; other axes preserved. |
| Replicator | outputs=[port_id,...] | All ports alias the same full tensor; no clone or parameters. |
| Module | ref | Independently instantiated graph reference; exact packed input shape match, canonical forward_tensor output. |
| Output | optional shape, label | Identity boundary. Shape asserts incoming dimensions; it never implicitly reshapes. |

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

References currently expose one bundled socket pair, not independently wired
named ports. Each occurrence has independent weights. Reference cycles are
rejected even when each internal DAG is acyclic. Its output presentation does not
replace its canonical adapter.

State-dict paths are layer_0, layer_1, ... for Linear/Activation/Reshape/Module
in topological order, nested through references. Boundary/Concat/router nodes
do not occupy these operation indices. Paths can change after graph insertion/
reordering; restore weights against their graph snapshot, not another architecture.

## 6. Data, training and evidence

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

Synthetic targets use a flat affine A,b or deterministic Tanh teacher. A shared-
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
requires matching identity and strict state-dict loading. Historical runs are
not rewritten. Regular source exports instantiate fresh weights; export-trained
embeds the matched trained state for evaluation-mode inference. A history browser
and notebook generation are not implemented.

Existing datasets retain their binding when only the graph changes. Training
rejects an incompatible binding rather than silently assigning old columns to
new ports. Replacing data creates a binding to the current valid contract;
CLI rebind-dataset explicitly declares that existing columns now mean the current
port layout. This does not reorder or transform those columns. Old datasets
without bindings are bound once to their migrated legacy contract when valid.

Future notebook generation may create .ipynb directly or via an agent prompt.
No notebook schema/command/protocol is claimed or reserved in this version.

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

Session undo/redo stores at most 50 edit steps and 128 MiB of combined SQLite
snapshots in a private temporary directory, NOT extra .vtw tables. Save retains
history; opening/external reload starts fresh history. Undo marks the workspace
dirty without rewinding the known disk revision. Run evidence for surviving
restored architectures is preserved; merged run row IDs need not remain stable.
Training completion is not a workspace edit-history step. Oversized edits clear
history with a notice; history is not a durable recovery/backup facility.

User settings are separate: ~/.tensorloon/config.json stores ui_zoom (0.5–3),
font_size (integer 12–28), theme, spacing (integer 1–10), box_spacing (integer
1–100) and memory_limit_percent. Defaults are 1.15, 14, dark, 4, 35 and 10.
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
