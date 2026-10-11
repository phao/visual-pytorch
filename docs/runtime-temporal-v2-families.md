# Retained runtime-T LSTM families

Family8, Specialization7 and container19 extend the separately versioned
[runtime-temporal-v2 backend](runtime-temporal-v2.md). This is retained D work,
not full D/E completion. [TaskBundle4](runtime-temporal-v2-tasks.md) admits direct
typed training/worker/inference; ExperimentSpec/RunRecord v2 execution choices
remain unsupported. Existing versions and retained source closures are
not silently widened, upgraded or rewritten.

## Explicit definition and witness

Family8 preserves Family7's template, scoped constructor, finite repeat, sharing
and integer-selection grammar. Its required runtime_policy is either JSON null
(fixed shapes), or this exact closed document:

```json
{"version":2,"kind":"runtime-temporal-v2","inputs":{"sequence":[0]}}
```

Input IDs are stable root IDs, not names or learned constructor symbols. The
declared leading sample axis must have semantic T. Learned widths, ranks, frame
dimensions and supplied hidden/cell dimensions remain fixed for each resolved
constructor. Null does not infer runtime execution. Family8 does not accept v1
or spatial policies; those older family versions remain available unchanged.
Both Recurrent and LSTM follow the v2 proof, including nested references and
the three explicit LSTM roles/ports. No implicit hidden state, masks or labels.

Specialization7 requires exact family/model references, full canonical bindings,
constructor contexts and nullable runtime. Nonnull runtime is a closed witness
with version3, kind runtime-temporal-v2, canonical inputs and specification_sha256.
Resolution6 carries dependency specification2. Its hash is recomputed from the
exact resolved closure on admission/load/compile; existence, matching counts or
matching shapes alone do not prove meaning. The frozen model stays ModelRevision4.

The time constructor symbol chooses a baseline drawing/specialization. Runtime
calls may use different positive T without reallocating owners. A fixed Output
shape remains a fixed constraint: explicitly remove it if the intended output
should follow T. The example removes only its sequence Output annotation; hidden
and cell annotations remain constructor-derived. Changing a parameter dimension
still requires a different constructor/model, not an actual-shape override.

## Storage and host routes

Container19 is required even for an unmaterialized Family8/Specialization7.
Older headers carrying these revisions fail closed. Opening an older valid file
upgrades only its in-memory working header; only explicit Save writes19. No SQL
table, sidecar, graph/operator, data, task, experiment, run or report grammar change.
Retention plus optional drawing materialization is atomic; late errors restore
all rows and dirty state. Historical references never become mutable head aliases.

The owned family editor still seeds concrete LSTM drawings as Family7/null.
Entering a v2 policy queues explicit version8 before that exact policy, using
the same host grammar as CLI. Host patches admit upgrades through8, not downgrades
or9. Invalid JSON remains editable; unsupported versions are not coerced. Preview
does not allocate/train/write SQL. Retain, exact export, materialize, undo/redo and
reload preserve provenance. This is not a new runtime training control panel.

```sh
python examples/runtime_lstm_family.py NEW.vtw --bindings CONSTRUCTOR_BINDINGS.json
python vpw_cli.py NEW.vtw preview-stored-family FAMILY_ID REVISION
python vpw_cli.py NEW.vtw export-family FAMILY_ID REVISION --output NEW_FAMILY.py
```

Bindings are optional; commands refuse target overwrites and create no data/task,
fit or implicit initial state. Existing atomic apply specialize-family/import
services retain the same grammar. Native compile_specialization selects v2 only
from the exact checked witness and writes no canvas/SQL state. initial_shapes is
optional: without it construction checks baseline T; with it the supplied actual
case is checked before allocation. Each call still checks current actual B/T/RAM.

## Checked export and parameter state

ArchitectureFamily automatically selects its captured Family8 policy; it cannot
override the admitted policy or invent one for a null-policy family. Its optional
runtime_temporal_v2 argument must equal the captured declaration. Other runtime
arguments refuse before layers allocate. Older family exports cannot select v2.
Standalone runtime specification, constructor specification and provenance are
caller-owned copies; preview_shapes/forward use the same current-case compiler.
Import resolves no instance; constructing still imports/uses PyTorch as in the
existing checked-family format. Source generation and pure preview need no Torch.

T-only changes preserve semantic parameter IDs, shapes, counts and ownership.
Strict warm start validates the complete source snapshot and plan before copies;
partial loading remains explicit and does not imply freezing or optimizer resume.
For v2, warm-start RAM planning uses source/target static shapes and the target's
admitted initial T, rather than rejecting a huge unused baseline T. Shape and RAM
guards remain active even when checkpoint schemas happen to agree. Current-RAM
heuristics are not allocator limits or OOM guarantees.

## Verification scope

Pure tests cover64 constructor combinations at3 actual T values, independent
counts/ports, proof hashes, version/axis/role refusals, exact retained history,
atomic rollback, container fences and actual CLI operations. Numerical tests
compare16 constructor combinations through retained native and checked exports
at3 T values against independent gate equations, including every input/state/
parameter gradient, zero-length/NaN padding, private proofs, unchanged owners,
current-RAM refusal before allocation/copy, huge baseline/small actual cases,
strict time-only warm start and fresh app-free checkpoint prediction.
Each native/export route also completes35 variable-T Adam updates matching an
independent student at every step. Shared-owner family branches independently
vary T, count156 elements once and match all combined input/state/weight gradients.
Actual sandboxed Qt verifies explicit7→8 editing, invalid text, SQL-free preview,
exact export, materialization, undo/redo and .vtw19 reload. Old family1–6 and
Family7 resolution/record values are compared with committed grammar modules;
that comparison is not a whole historic deployment replay. Campaign evidence
records exact runs and first failures. None of this completes the still-required
runtime-v2 scientific task/worker/notebook routes or original D convergence.
