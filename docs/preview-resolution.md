# Request-local candidate preparation

`app/experiment_objects.py::preview_experiment` reuses successful preparation
only within one synchronous preview read of immutable sources. This is a native
implementation optimization, not a persistent cache, schema change or new CLI
action. The closed candidate and projected browser contracts are unchanged.

## Lifetime and identities

The experiment is intrinsically validated once before resolving its requested
mixed-radix page. A private `_PreviewResolution` holds one last successfully
resolved document per dependency kind (model, dataset, task), keyed by exact
ID/revision; one model contract; one successful model/dataset/task compatibility
key; and one validated split keyed by exact dataset ID/revision and the full
split protocol hash. An old slot is dropped before loading its replacement.
Failures are never memoized. Axis lookup maps scale with the validated experiment,
not the number of candidates visited. An empty past-end page resolves no candidate.

The resolver must provide stable immutable revisions for this synchronous read.
Native registry reads still perform integrity and dependency checks; the first
read is not replaced by a hash-only trust shortcut. Model/dataset/task heads are
not identities. No state is shared between preview calls, workspaces or workers.
There is no cache of fitted transforms, numerical models, optimizers, checkpoints
or mutable authoring sessions. Public `candidate` still fully validates each
direct call, including training, captured manifests and retries; it has no
argument permitting callers to inject the private preview context.

Coordinate order, candidate IDs, original references, seeds, protocols, split
membership and incompatible/excluded reasons retain their existing meaning.
Each complete returned row owns independent coordinate/reference/protocol/manifest
copies. Mutating one row cannot alter another or a later read. Preparation fits
nothing, launches no worker, changes no SQL/history and uses no global RNG.

## Measured evidence

The retained comparison loads the previous `experiment_objects.py` from commit
`893192a5753f24b3ba0e1b9a3d36e1a91ca25fed`, with the same remaining runtime closure.
Both versions read the **same** immutable temporary workspace: 500 records per
dataset, 1000 initialization seeds, 4000 total coordinates, a 25-candidate
projected page. Three paired measurements alternate baseline/current order.
Setup/startup are excluded; native preview plus projection serialization is timed.
Python 3.13.7, CPU affinity to logical CPU 0, stdlib-only preparation, no fitting.
Other bounded regression activity existed, so this is not an isolated-hardware
or universal application-performance claim.

| Measurement | Previous implementation | Request-local preparation |
| --- | ---: | ---: |
| Median seconds | 0.937987 | 0.057727 |
| Top-level dependency resolver calls | 76 | 4 |
| Model/task/split preparation calls, each | 25 | 1 |
| Intrinsic experiment validation calls | 27 | 2 |
| Python tracemalloc peak bytes | 3330399 | 1355043 |

The ratio of medians is 16.25 on this fixture. Every measured response has the
same compact wire bytes: 20589 bytes, SHA-256
`e0e600d2f0b72794dadb8c10e4eb4e3b2b414156b7a3a68df5f9ebc5b6eb587c`.
The additional native experiment load accounts for the fourth resolver call and
second intrinsic validation. Traced allocation is not physical RAM; the process
RSS high-water mark is shared by both variants and is not comparative evidence.
Earlier separately-created pilot fixtures have different random identities and
are not used as proof of byte equality.

`tests/test_preview_resolution.py` has eleven cases covering complete direct
candidate equality across seeds/protocols/models/datasets and all split methods,
invalid split/compatibility reasons, exclusions without dependency reads,
substituted identities, retried transient errors, slot capacity/replacement,
independent result copies, fresh-call/workspace/head isolation, direct public
validation and unchanged SQL/history/dirty/RNG/no-fit behavior. These complement
the existing native registry hash-validation tests; injected source errors are
not described as direct SQL-tamper tests.

The matched measurement script/logs and GUI regression evidence are retained in
the private campaign handoff, not installed as runtime dependencies.

## Limits and continuation

This removes repeated same-coordinate-family preparation, not all native costs.
Initial registry decoding/integrity/dependency checks, inline dataset validation,
split generation, full row manifests/copies, full JSON reads, admission, storage,
history and recovery retention can still scale with complete scientific data.
Complete preview still constructs up to its requested candidate count before
browser projection; this is not streaming or lazy SQLite storage. A different
model/data/split per row will reuse much less. No universal time/RAM bound, OOM
guarantee, crash-safe detached execution or Stage C completion follows.

See [candidate-browser.md](candidate-browser.md) for projected wire/cursor
contracts, and [stage-c-verification.md](stage-c-verification.md) for the remaining
GUI/integration work before D/E.
