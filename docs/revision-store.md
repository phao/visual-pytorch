# Independent revisions in `.vtw` v6

This contract is implemented on the unfinished Stage C task branch. It does not
mean that notebooks or the full task GUI are complete. Concrete experiment
selections, terminal attempts and report recipes are executable; their closed
contracts are in [experiment-objects.md](experiment-objects.md). The task
trainer/export component is specified separately in task-training-contract.md. The legacy
architecture-owned dataset/checkpoint path stays supported unchanged.

## Documents versus browsing heads

`spec_revisions` stores immutable, content-addressed version-1 documents. The
implemented kinds are `dataset`, `experiment`, `model`, `report`, `run`, `task`. Dataset/task shapes
and transform syntax are specified in [dataset-task-contract.md](dataset-task-contract.md).
Task storage validates intrinsic syntax independently of any model or dataset;
execution validates the selected dataset/model shapes and field/port meaning.
Storing a TaskSpec is not proof that every combination is compatible.

Every object has string `id`, `name` and integer `version: 1`. A document's
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
Version 6 uses the same DDL with three additional closed document kinds:

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
v5/v6 file. Every document's kind/schema, hash, relational ID/name, UTC creation
time and head references are checked. Experiment→model/dataset/task,
run→experiment and report→run/experiment references are exact retained revisions.
These polymorphic JSON references are application-validated dependencies, not
additional SQL foreign keys. Load/save validate the complete retained closure,
including archived documents. Unknown kinds/versions are rejected.
For each logical run ID, every retained revision must have equal selection and
captured resolution; overlapping attempt prefixes must be byte-equivalent under
the reference JSON hash. A new revision cannot truncate the longest retained
history. Idempotent storage/reselection of a retained old revision remains valid.

## Migration and history

New files/saves use `PRAGMA user_version=6`. Readers accept v1–6. Opening an older
file backs it up into memory, preserves the old tables/data/run/checkpoint bytes,
performs the already-defined v1/2 packing migration when needed, then adds empty
revision tables/guards when absent and sets the working version to 6. Existing
v5 tables are validated, not recreated. The source file stays
unchanged until explicit Save. Legacy `dataset.x/y` is not guessed into a typed
ontology or rebound. A header-lowered fixture containing revision objects is
accepted only for the v5 dataset/task/model semantics, if its complete existing
revision schema and content pass the same checks. New experiment/run/report
semantics require a v6 header. Content is not silently discarded. Partial schemas
fail. Versions above 6 fail.

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
GUI/notebook integration remains pending. Filesystem revision polling continues
to use the existing protocol.

## References

[SQLite composite foreign keys](https://www.sqlite.org/foreignkeys.html) informed
the exact head→revision relation; foreign keys are enabled per connection.
[SQLite trigger semantics](https://www.sqlite.org/lang_createtrigger.html) informed
the immutable guards, including a duplicate insert guard independent of
[REPLACE delete-trigger behavior](https://www.sqlite.org/lang_conflict.html).
These constraints complement application
hash/semantic checks; they do not make a supplied database a hostile-file sandbox.
