# Exact retained parameter sources — internal source-half contract

`app.parameter_sources.inventory(workspace, run_reference, attempt=number)` lists
the chosen checkpoint's native state-dict paths, exact semantic owner IDs, shapes
and float32 dtype. Attempt is mandatory: no mutable latest head or attempt is
selected. `read(workspace, selection)` returns a detached JSON tensor and an
identity descriptor. Selection is a closed object:

```json
{
  "run": {"kind": "run", "id": "RUN_ID", "revision": "EXACT_SHA256"},
  "attempt": 1,
  "checkpoint_id": "EXACT_CHECKPOINT_SHA256",
  "parameter": {"path": "layer_0.weight", "id": "EXACT_SEMANTIC_OWNER_ID"}
}
```

Resolve the exact immutable run and pinned experiment/model/dataset/task with its
saved split manifest; validate the entire retained result/checkpoint. Require the
captured frozen model to match its retained exact ModelRevision. A source path and
owner ID must both match the same checkpoint entry. Legacy slash IDs are not
split or guessed, and component-based IDs are not rewritten. Failed/cancelled
attempts with a valid retained checkpoint may be selected explicitly; an attempt
without one refuses, never substitutes another attempt.

The returned `source` descriptor includes exact run/model references, attempt,
checkpoint ID, source path/owner, shape/dtype, value digest, attempt status and
completed-update count. `optimizer_resume` is always false. Neither declaration
nor structural validation proves that arbitrary externally forged evidence came
from training; normal application admission and actual test execution supply the
separate provenance context.

## Float32 value representation and digest

Owner shapes are lists of one to eight positive integers, excluding booleans.
Values are shape-validated finite float32-range numbers, then roundtripped through
little-endian IEEE754 binary32. Booleans, wrong dimensions and nonfinite numbers
refuse. Return a new nested list; caller edits do not mutate retained evidence.

SHA256 hashes, in this exact order:

1. ASCII bytes `Visual PyTorch float32 tensor v1` followed by one zero byte.
2. The exact integer shape array as compact ASCII JSON, followed by one zero byte.
3. Every leaf packed with Python `struct.pack('<f', value)` in nested-list order.

The representation is named `ieee754-binary32-le; shape-domain-v1`. Shape is part
of the digest; identical flattened values at different shapes differ. Signed
zero is preserved. This is a value digest, not the original checkpoint JSON hash,
which is pinned separately.

## Deliberate limits

This component reads only. Validation recomputes deterministic train-only
preprocessing to check the captured checkpoint; it does not mutate persisted
preprocessing. It does not initialize a destination model, alter
trainability, create a binding specification, restore preprocessing or optimizer
state, launch inference/training, create a split, or write SQL. It imports no
Torch/NumPy/Qt in a fresh reader process. There is no CLI/GUI source selector yet,
and this descriptor is **not** admitted parameter-policy/`.vtw` specification
syntax. Next: version-fenced immutable binding receipt plus atomic guarded
destination edits, persistence, CLI and Inspector controls. The existing exact
values initializer remains independently valid but does not claim source lineage.

Six bounded tests use one actual three-update retained worker fit and an explicit
failed retry, plus SQL/RNG/no-execution guards, changed heads, refusal, fresh
no-numerical reader and independent byte-digest checks. Full final-tree regression
and completed LoRA/MS-VAR acceptance remain separate gates.
