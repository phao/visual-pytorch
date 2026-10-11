# Independent revisions in `.vtw` v7

This contract includes the accepted local Stage C base and the unfinished
Stage D family checkpoint. Concrete experiment selections, terminal attempts,
typed dataset/task/experiment authoring and both notebook modes are implemented
within their documented scope; full D/E and public publication are not implied.
Concrete experiment/run/report closed
contracts are in [experiment-objects.md](experiment-objects.md). The task
trainer/export component is specified separately in task-training-contract.md. The legacy
architecture-owned dataset/checkpoint path stays supported unchanged.

Owned authoring sessions on this branch are transient tab/host state, not an
additional vtw kind/table or a document revision. Their exact sequence/page/patch/
admission contract is in [owned-authoring-sessions.md](owned-authoring-sessions.md).
The typed dataset form now uses bounded stable-ID pages/edits in those sessions;
the [task form](typed-task-sessions.md) now uses bounded ordered cards/literal
edits in the same machinery. The experiment form likewise uses owned sessions;
see [typed-experiment-sessions.md](typed-experiment-sessions.md).
See [typed-dataset-sessions.md](typed-dataset-sessions.md).
CLI retained-object grammar and content identities remain unchanged. Raw value buffers are owned temporary files,
not SQLite rows or retained revisions; explicit Finish publishes only to an
unsaved draft. [chunked-draft-values.md](chunked-draft-values.md) specifies their
byte/cursor/admission contract. CLI import/apply still targets complete validated
retained objects; it does not persist Qt tokens or unfinished text.

## Documents versus browsing heads

`spec_revisions` stores immutable, content-addressed versioned documents. The
implemented kinds are `dataset`, `experiment`, `family`, `model`, `report`, `run`,
`specialization`, `task`. Dataset/task shapes
and transform syntax are specified in [dataset-task-contract.md](dataset-task-contract.md).
Task storage validates intrinsic syntax independently of any model or dataset;
execution validates the selected dataset/model shapes and field/port meaning.
Storing a TaskSpec is not proof that every combination is compatible.

Every object has string `id`, `name` and integer `version`. Version1 remains
supported; experiment/run v2 adds explicit family coordinates (container≥8),
family/model v2 adds explicit sharing (container≥9), and family/experiment/run v3
plus specialization v2 add typed constructor choices (container≥10). Family v4
and specialization v3 add scoped Module constructor contexts (container≥11),
as specified in [scoped-constructors.md](scoped-constructors.md). A document's
`revision` is SHA-256 of UTF-8 encoded JSON using Python `json.dumps` with
`sort_keys=True`, `separators=(',', ':')`, `allow_nan=False`, and default
`ensure_ascii=True`, excluding its own `revision` field. Ordered arrays remain
ordered. This is the reference encoding, not a claim of RFC 8785 canonicalization
across arbitrary serializers; an external writer must match it or use the CLI.
Storage returns a private
copy with that revision. Supplying a mismatched revision is an error, not a way
to update an existing revision. Duplicate content is idempotent. Immutable rows
cannot be updated, deleted or replaced, even via ordinary direct SQLite writes.
This is local integrity, not authentication or proof of a scientific claim.

`spec_heads` maps `(kind,id)` to the revision currently selected for browsing.
Selection can move to any retained revision or be archived (remove the head).
Neither action deletes document bytes. No model/dataset deletion cascade reaches
independent revisions. Lists are alphabetic by case-folded name, ID and revision;
field, tensor, input-port and transform ordering is never sorted.

## Frozen models

ModelRevision keys: `version`, `id`, `name`, `root`, `models`, `architecture_hash`,
with version1 for independent captures or version2 for explicit local sharing,
as specified in [shared-parameters.md](shared-parameters.md), plus
optional verified `revision`. `root` equals `id`. `models` contains exactly the
reachable reference closure, each entry with its original `id`, `name`, `graph`.
Frozen graphs privately pin input packing and normalize versioned operator
defaults. Computational fields, boundary labels/names and internal references
are retained. Position/size/folding and architecture-owned data/results/training
are not embedded. Closure graphs are structurally and semantically validated
without Torch/Qt or parameter allocation. Extra unrelated models are rejected.
The architecture hash uses the same presentation-independent reachable-closure
identity as Stage A. It differs from the document revision: a name change may
produce a new document revision while preserving scientific architecture hash.
Deleting/editing an authoring architecture does not change this frozen closure.

## Exact schema

Version 5 introduced these tables without altering existing v4 tables.
Version 6 added experiment/run/report; version 7 adds family/specialization.
Both use the same DDL:

```sql
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

The table columns, composite keys/FK and all three canonical guard triggers are
validated on load; missing or changed guards are not silently recreated for a
v5/v6/v7 file. Every document's kind/schema, hash, relational ID/name, UTC creation
time and head references are checked. Experiment→model/dataset/task,
run→experiment, report→run/experiment and specialization→family/model references
are exact retained revisions. Specialization validation reproduces its complete
concrete closure from the pinned family and explicit bindings; equal shapes or
parameter counts alone do not establish equality. See
[architecture-families.md](architecture-families.md) for the closed new kinds.
These polymorphic JSON references are application-validated dependencies, not
additional SQL foreign keys. Load/save validate the complete retained closure,
including archived documents. Unknown kinds/versions are rejected.
For each logical run ID, every retained revision must have equal selection and
captured resolution; overlapping attempt prefixes must be byte-equivalent under
the reference JSON hash. A new revision cannot truncate the longest retained
history. Idempotent storage/reselection of a retained old revision remains valid.

## Migration and history

New files/saves use `PRAGMA user_version=11`. Readers accept v1–11. Opening an older
file backs it up into memory, preserves the old tables/data/run/checkpoint bytes,
performs the already-defined v1/2 packing migration when needed, then adds empty
revision tables/guards when absent and sets the working version to 11. Existing
v5 tables are validated, not recreated. The source file stays
unchanged until explicit Save. Legacy `dataset.x/y` is not guessed into a typed
ontology or rebound. A header-lowered fixture containing revision objects is
accepted only for the v5 dataset/task/model semantics, if its complete existing
revision schema and content pass the same checks. New experiment/run/report
semantics require at least a v6 header; family/specialization semantics require
v7; experiment/run v2 requires v8, explicit sharing and family/model v2 require
v9; typed constructor choices in family/experiment/run v3 or specialization v2
require v10; scoped family v4/specialization v3 require v11. Content is not
silently discarded. Partial schemas fail. Versions above11 fail. See
[scoped-constructors.md](scoped-constructors.md) and [constructor-choices.md](constructor-choices.md) for exact
typed provenance/version rules.

Save validates revision integrity before the existing atomic, revision-checked
file replacement. Session undo/redo restores mutable authoring state and heads
while unioning all retained immutable revisions from the current working copy.
Matching keys must have equal documents; conflicts fail instead of choosing a
winner. This preserves run evidence roots and their dependencies. Legacy
`runs` keeps its prior architecture-dependent history semantics for compatibility.
Independent terminal RunRecords retain frozen dependencies after authoring model
deletion, head archival or session undo. Explicit retries append attempts from
the longest retained history, not whichever head is currently browsed.
Archive is recoverable by selecting a retained revision; no GC is implemented.

## CLI and Python API

Workspace delegates `put_spec(kind, document, select=True)`,
`get_spec(kind,id,revision=None)`, `list_specs(kind=None, all_revisions=False)`,
`select_spec(kind,id,revision)`, `archive_spec(kind,id)`, and `freeze_model(id)`.
The last method stores a frozen closure and returns its exact model reference.
References are `{kind,id,revision}`, never a human name or an implicit head at
execution time. Inspect/read does not migrate disk bytes.

Ordered `apply` operations: `put-spec` with `kind`, `document`, optional boolean
`select`; `select-spec` with exact reference; `archive-spec` with `kind`, `id`;
`freeze-model` with authoring architecture `id`. Operations share the existing
atomic file-save/revision conflict protocol. A failure anywhere leaves on-disk
bytes unchanged; `--allow-invalid` does not bypass revision or schema checks.
CLI `spec-list`/`spec-show` read selections or retained revisions; `prepare-stored`
requires exact model/dataset/task revision references and explicit split protocol
or manifest. It executes preparation, not training, and never follows a changed
head to substitute another dataset/model. `preview-experiment` is read-only and
paged; `run-candidate` explicitly executes and atomically saves a terminal run;
`report-data` reads exact selected evidence without execution. Full experiment
GUI integration follows [experiment-gui.md](experiment-gui.md). Notebook export consumes exact revisions without
changing SQLite bytes; see [notebook-reports.md](notebook-reports.md). Filesystem revision polling continues
to use the existing protocol.

The GUI/CLI [scientific read browser](scientific-browser.md) adds metadata pages,
typed summaries and ordinal tree/scalar inspection of exact existing revisions.
These are non-document projections, not new SQL tables or partial-spec admission
format. Registry/dependency validation remains unchanged. Complete legacy reads
and explicit exports are retained; source arrays/ports/observations retain order.

## References

[SQLite composite foreign keys](https://www.sqlite.org/foreignkeys.html) informed
the exact head→revision relation; foreign keys are enabled per connection.
[SQLite trigger semantics](https://www.sqlite.org/lang_createtrigger.html) informed
the immutable guards, including a duplicate insert guard independent of
[REPLACE delete-trigger behavior](https://www.sqlite.org/lang_conflict.html).
These constraints complement application
hash/semantic checks; they do not make a supplied database a hostile-file sandbox.
