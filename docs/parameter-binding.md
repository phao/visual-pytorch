# Retained initial-parameter bindings — version1 / container25

A binding copies one exact retained checkpoint tensor into one explicitly selected
canonical parameter owner's initial values. It preserves destination trainability
and rejects shape/dtype mismatch, transpose, broadcasting, guessed names and stale
destination identity. It is initialization, **not optimizer resume**. Model6,
Family10, Specialization9 and policy1 retain their existing numerical meaning.

## Independent immutable record

Kind `parameter-binding` joins the alphabetical `KINDS` registry. Store its closed
JSON in `spec_revisions`, with optional current browsing head in `spec_heads`.
Existing immutable triggers apply; no SQL table, sidecar or executable payload is
added. Any binding row, including archived/unselected history, requires a
container25. Policy-only containers24 remain readable without disk mutation;
deferred containers21/22 remain refused.

Top-level fields are exactly `version:1`, stable `id`, `name`, `source`,
`destination`, `optimizer_resume:false`, and optional content-derived `revision`.
The revision is the existing SHA256 of canonical document content without the
`revision` field. Both name and ID obey the existing scientific identifier rule.

`source` is the checked retained descriptor from
[exact parameter sources](retained-parameter-sources.md): version1,
kind `retained-run-parameter`, exact `run` and `model` references,
integer `attempt`, `checkpoint_id`, `parameter:{path,id}`, `shape`,
`dtype:"float32"`, `values_sha256`, `value_format`, terminal `status`,
`completed_updates`, and `optimizer_resume:false`. Exact references are closed
`{kind,id,revision}` with lowercase SHA256 revisions. No latest/head fallback is
permitted; checkpoint-free attempts refuse rather than selecting earlier success.

`destination` has exactly `model` (exact ModelRevision reference), `selector`
(canonical relative JSON component array), `id` (full component owner ID),
`shape`, `dtype`, `values_sha256`, and `value_format`. Its owner ID equals the
canonical JSON array formed by prepending the destination model ID to the
selector components. Shapes, dtype, value digest and representation must equal
the source. Owner shapes follow the existing1–8-positive-integer rule.

The digest representation is `ieee754-binary32-le; shape-domain-v1` as specified
in the source-reader document. The destination stores normal `values` policy
initialization with detached float32-roundtripped values. Provenance is **not** an
undocumented extra field in that closed initializer.

## Full dependency proof and atomic admission

Dependencies are the exact source RunRecord, source ModelRevision, and destination
ModelRevision. Validation reconstructs the source's pinned experiment/candidate
bundle and saved split manifest, checks the full retained result/checkpoint, and
recomputes deterministic train-only preprocessing. It then proves descriptor
agreement and the destination's exact resolved owner, shape, `values` initializer
and float32 value digest. Source and destination model identities may differ;
each is checked against its own snapshot. This is structural lineage validation,
not cryptographic proof that arbitrary external evidence was genuinely trained.

Preview validates the existing `{id,node,architecture}` owner token first, reads
the selected source, proves the prospective scoped policy and derives the exact
frozen destination model and binding revision. It performs no SQL writes, RNG
draws, inference, model fitting or Torch/NumPy/Qt imports.

Admission uses one SQLite savepoint for the mutable drawing, retained frozen model
and binding/head. Refusal/exception rolls them back and restores the dirty flag.
Reapplying an unchanged candidate with a fresh token and already selected exact
revisions is a true no-op, including no SQL/history writes. An old token refuses.
Changed source lineage or a changed frozen model may legitimately create a new
receipt even when its tensor bytes happen to coincide. Desktop admission is one
normal undo step; immutable historical revisions survive undo, while restored
heads/drawing identify the prior working state.

## CLI and Inspector

Read-only commands:

```text
vpw_cli.py file.vtw prepare-parameter-binding MODEL_ID NODE_ID
vpw_cli.py file.vtw retained-parameters RUN_ID RUN_REVISION --attempt 1
vpw_cli.py file.vtw preview-parameter-binding token.json '["linear1","weight"]' source.json
```

`source.json` uses the closed run/attempt/checkpoint/path+ID selection described
in the source-reader document. Apply an ordered JSON array with this edit:

```json
{"op":"bind-parameter","token":{"id":"MODEL_ID","node":"NODE_ID","architecture":"EXACT_ARCHITECTURE_SHA256"},"selector":"[\"linear1\",\"weight\"]","source":{"run":{"kind":"run","id":"RUN_ID","revision":"EXACT_RUN_SHA256"},"attempt":1,"checkpoint_id":"EXACT_CHECKPOINT_SHA256","parameter":{"path":"layer_0.weight","id":"EXACT_SOURCE_OWNER_ID"}}}
```

Optional edit fields `binding_id` and `name` customize receipt identity/name; no
other fields are admitted. The existing CLI disk-revision guard and atomic whole
batch replacement remain in force. A later failing edit leaves the disk unchanged.

The parameter-owner Inspector card adds a collapsed retained-tensor picker.
Paged/searchable exact run revisions include historical choices; the user enters
an explicit attempt, reads a bounded tensor page, chooses a compatible row and
binds. No default latest/attempt/tensor is silently chosen. Metadata pages are at
most25 rows/64KiB, not bounds on the underlying full checkpoint validation.
Incompatible tensor shapes are visible but disabled. Workspace UUID, reload scope,
architecture token, selected owner, busy guard and serialized ordinary saves
protect the edit. Binding returns the authoritative model for reconciliation;
concurrent diagram moves cannot overwrite accepted values with stale policy.
The scientific JSON codec preserves signed zero in cloning, textarea disclosure
and bridge serialization. Ordinary editor `updateModel` calls explicitly preserve
the server's opaque parameter policy, avoiding both zero-sign loss and accidental
float/int literal recoding during geometry edits. Parameter-policy services remain
the explicit route for its changes. Python no-op/conflict comparisons likewise
distinguish signed zero. Backend refusal is notified before rebuilding the card,
so it remains visible even when that old card is detached.
Pending initializer fields/open state survive a validation redraw only when
workspace scope, architecture identity and selected box still match. They remain
unapplied form data, never new stored parameters. A changed model clears them.

Pinned tensor-source notes prove that **current initial-value bytes** match a
selected retained receipt. They do not claim the whole current model equals the
receipt's original destination snapshot. Thus copying a second owner or adding an
upstream block does not hide an unchanged first owner's lineage; changing its
values removes the match. Tooltip includes the original receipt/model/source
references. Historical receipts remain available in scientific browsing even if
the mutable drawing changes. Matching currently scans selected small receipt
metadata, then validates matches; it is not a lazy/on-disk tensor adapter or a
large-registry performance guarantee.

## Export and remaining work

Ordinary native/generated PyTorch initializes the copied values through policy1,
without needing a live source run or optimizer. Captured report runtime includes
binding/source/owner-edit dependencies and rejects missing new imports before
kernel startup, while genuine older closures remain supported. Independent
receipts are not implicitly injected into unrelated report dependency closures.

The full requested LoRA workspace/rank comparisons/derived merged artifact and
faithful Neural MS-VAR are separate remaining deliverables. This binding feature
does not implement those, a legacy integer-run snapshot adapter, LoRA artifact
packaging, production/HPC training, Stage E or a new Loss workbench in the app.
