# Optional example packs

`examples/neural-msvar-pack.json` describes the existing committed
`examples/neural-msvar.vtw`. It adds no workspace registry kind or SQLite version.
Opening the workspace, reading the JSON, or calling `check` never runs a fit.

This bounded version-1 format supports the original neural MS-VAR example only.
Its producer source commit is
`e086c5157d0a5fa221c53d7f3d896e34bbdf4f3a`, which introduced the committed workspace
and its builder. The workspace SHA256 is computed from the actual file bytes.
These fields provide reproducibility provenance, not a signature or proof of
authorship. No Git command or network lookup is needed by the checker.

## Meaning and claims

The stable editable architecture ID is `neural-msvar`. Its declared use is
`exact-model`: the graph implements causal lag features, a learned transition
network, affine state means, positive diagonal variances, and exact conditional
Gaussian filtering and sequence likelihood. “Exact” qualifies inference under
the specified model, conditional on explicitly supplied observed history and
valid prefix lengths. The twelve-update parameter estimate does not establish
an exact fit, true regime recovery, identifiability, stationarity, or calibration.

The committed version uses K=5 states, p=3 observed lags, D=3 coordinates,
hidden=32, T=12, and controlled=false. All six recipe bindings carry an explicit
JSON type and value; booleans, floats, strings, and integers are not coerced.
The fixed family has no free symbols. These are recipe constructor bindings,
not invented family-axis bindings. Its existing runtime policy allows variable
observation time while history and state dimensions remain fixed.

The dataset contains 24 actual synthetic trajectories and no regime targets.
The task optimizes conditional marginal sequence negative log likelihood. The
saved experiment pins the grouped split, initialization/shuffle seeds, Adam
protocol, and actual run. The manifest names exact immutable dataset, task,
family, specialization, model, experiment, and run revisions. It neither
generates labels nor manufactures fit results.

## Shared Python API

`app.example_packs.check(workspace, manifest)` accepts a saved canonical
`Workspace` object or a `.vtw` path, and a JSON dictionary or manifest path. It
returns JSON-compatible saved provenance with `status="checked"` and
`executed=false`. Invalid inputs raise `GraphError` with a JSON field location
or a workspace location.

The checker uses `Workspace.load`, `Workspace.get_spec`, the closed construction
recipe service, graph signatures, and canonical candidate/bundle/result
validation. It verifies the source byte hash, reconstructs the typed recipe,
compares its computation with the editable graph and both frozen family/model
graphs, and verifies the dataset/task meaning and actual saved run selection.
The expected checkpoint hash, complete metrics hash (including population IDs),
and completed update count must match retained evidence. Existing unrelated
references do not satisfy the check. Mutable browsing heads are never substituted
for exact revisions.

There are no configuration, numerical, or GUI imports; `/usr/bin/python3 -S`
can perform the check. The checker never saves, edits SQL/history, allocates
tensors, or executes Python text from the manifest. Existing-workspace arguments
retain their path, dirty state, SQL contents, and history. Path arguments use the
canonical read-only disk loader and discard its in-memory copy afterward.

`app.example_packs.replay(workspace, manifest, *, python, timeout,
memory_limit_percent=10.)` is a separate explicit action. The trusted interpreter
is supplied by the caller, never by the manifest. The timeout must be finite and
in (0, 60] seconds; memory admission must be in (0, 10] percent. The existing
admission service remains a RAM estimate rather than an operating-system limit.

Replay first performs the same check, copies the canonical database into a
private in-memory `Workspace`, and calls the existing `execute_candidate` →
`run_json_worker` → `app.worker` route. It uses the saved split, candidate,
protocol, and independent seeds. Only the existing updates-only budget of 12 is
accepted; the declared hard ceiling is 24, not an authorization to increase the
original run. Unknown policies, larger budgets, and executable manifest fields
are rejected before execution. Neither source nor private copy is ever saved;
the private terminal run is discarded after returning its actual result.

Replay compares actual checkpoint weights/hash, complete final metrics, execution
identity, and counters against the saved result. It returns `status="matched"`
and `verified=true` only when all match exactly. A completed but different result
returns `status="mismatch"`; a worker/admission failure returns `status="failed"`
with the recorded error. Returned provenance includes the saved environment;
the actual result includes the replay environment. Timings and available-memory
observations are not equality criteria. Floating-point equality across other
Torch/platform versions is not promised and is never replaced with fabricated
success or silently loosened tolerances.

## Closed JSON version 1

The required top-level fields are `version`, `id`, `producer_source_commit`,
`workspace_sha256`, `architectures`, `references`, and `verification`. Every
object is closed. The single architecture declares `id`, `intended_use`,
`assumptions`, and `recipe`; the recipe declares `name`, `version`, and typed
`bindings`. The supported assumptions are the explicit semantic tokens in the
checked-in manifest. The recipe catalogue determines the correct
exact/approximate/component classification; this pack cannot be relabelled as
an approximate estimator or a parameter component.

Each reference has `kind`, `id`, and a lowercase SHA256 `revision`.
`verification` contains the fixed `saved-candidate-replay-v1` kind, the actual
saved candidate `coordinates`, `budget={updates:12,max_updates:24}`, and
`expected` checkpoint/metrics SHA256 hashes plus `completed_updates=12`.
Duplicate keys, nonfinite numbers, future versions, unknown keys, commands,
arbitrary source/executable declarations, and manifests over 1 MiB are refused.
Changing the workspace legitimately requires a separately reviewed pack version
or supported profile; this format is not a general experiment workbench.

Validation is bounded to `tests/test_example_packs.py`. Pure checks run with
`/usr/bin/python3 -S`; real native replay uses
`/home/pedro/miniforge3/bin/python`, CPU affinity 4,6 and
`OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. The fixture expectations
come from the committed workspace bytes and retained run, not a new fit used to
rewrite reference evidence.
