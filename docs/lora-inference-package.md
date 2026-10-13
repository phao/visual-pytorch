# Exact-base LoRA inference package, version 1

This is a research-prototype export for a checked, flat, bias-free LoRA graph:

\[
y = W_0 x + (\alpha/r)B(Ax).
\]

PyTorch uses batch-leading row tensors, so the implementation applies three
`torch.nn.functional.linear` calls. Earlier sample axes are preserved. There is
no activation between A and B. The supplied base is frozen, and both adapter
owners are trainable in the retained source model. The exported predictor freezes
all three tensors as buffers, but it still permits gradients with respect to its
input.

## Export

`vpw_cli.py WORKSPACE.vtw export-lora-package SOURCES.json NEW_DIRECTORY --alpha 4`

`SOURCES.json` has the same exact `base`, `A`, `B` selections accepted by
`preview-lora-merge`: each identifies a full run revision, explicit retained
attempt, checkpoint hash, native parameter path and semantic owner ID. No latest
checkpoint or positional owner guessing is allowed. All three sources must
belong to one exact checkpoint, and the complete frozen source graph must prove
the corresponding LoRA wiring, base freezing, bias absence and scale.

Export and integrity verification do not import PyTorch, NumPy or Qt or change the
workspace. The directory must not exist. A filesystem failure may leave a partial
directory; preserve it for diagnosis and choose a new destination. Existing
directories and final-file symlinks are never overwritten.

## Files and identity

| File | Meaning |
| --- | --- |
| `manifest.json` | Closed v1 contract, exact source descriptors, stable input/output IDs and sample shapes/axes, operation, file hashes and merged-value witness |
| `base.f32` | Required exact base weights, included rather than implicitly fetched |
| `A.f32`, `B.f32` | Adapter factors from the same retained checkpoint |
| `predictor.py` | Self-contained CPU float32 PyTorch inference loader |

Matrices use row-major IEEE-754 binary32 little-endian bytes. The manifest
records each file's byte count and SHA-256, as well as the shape-domain-bound
value hash of its retained tensor. Verification checks these against each other
and independently recomputes the deterministic merged-weight witness. The
generated loader embeds the canonical manifest SHA-256 and checks manifest and
tensor bytes before importing PyTorch or allocating tensors. Pure verification
also compares the generated source against its deterministic expected bytes.

Hashes establish identity and detect corruption; they are not signatures or proof
of authorship. Do not execute untrusted Python packages. Offline verification
cannot establish that the referenced workspace exists or that its source record
is truthful; export obtains those records through the validated workspace reader.

## Inference contract and limits

Import `predictor`, call `model = predictor.load()`, then
`model({INPUT_PORT_ID: x})`. The return value is a dictionary keyed by the exact
output port ID. Input must be CPU float32 with shape `[B, *sample_shape]`.

This package takes **raw architecture inputs**. It does not include fitted task
transforms, target transforms, optimizer state, training resumption, metric
semantics, or a training dataset. For task-aware inference use the separately
available retained-run predictor export. The base is mandatory: removing it does
not produce a standalone adapter. Use the manifest's exact base identity if a
future storage system separates base and adapter artifacts.

The current export deliberately accepts only the flat, single-input/single-output,
bias-free LoRA structure accepted by the explicit merge service. It is not a
general Transformer/attention adapter exporter. The local merge witness remains
bounded to one million scalar dot products. Merged and unmerged execution can
differ through float32 rounding; tests declare `atol=2e-6, rtol=1e-5` on the
bounded teaching fixtures, not universal error guarantees.

The structural design follows the original [LoRA paper](https://arxiv.org/abs/2106.09685).
The actual inference operations use PyTorch's
[linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.linear.html)
and [registered buffers](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer).
