# Softmax — explicit shape-preserving normalization

Choose **Softmax** in Building blocks → Nonlinear activation, then add an
Activation block. Double-click it to open the Inspector and choose its
**Normalization axis**. The default is −1: the last sample dimension.

For input `[B,N]`, the output is `[B,N]`; each row sums to one. Softmax couples
values within the selected slice, unlike elementwise Sigmoid. It preserves shape,
axis names, batch size and all other dimensions and has no trainable parameters.
It uses PyTorch's numerically stable Softmax, not an exported naive exponential
sum. Finite-precision underflow can yield zero probabilities.

## Sample-axis convention and PyTorch mapping

Like Flatten and Permute, axis numbers refer to **sample dimensions**, excluding
batch B. Positive indices start at zero; negative indices count from the end.
For sample rank r, valid axes are −r through r−1. Normalizing across B is not
offered: observations must not affect each other's probability distributions.

| Tensor | Declared sample dim | PyTorch dim | Meaning |
| --- | --- | --- | --- |
| `[B,N]` | −1 or 0 | 1 | Normalize N features within each observation |
| `[B,T,N]` | −1 or 1 | 2 | Normalize N features at each position |
| `[B,T,N]` | 0 or −2 | 1 | Normalize T positions separately for each feature |
| `[B,C,H,W]` | 0 or −3 | 1 | Normalize C classes at each spatial site |

The resolved dimension is `(dim mod r) + 1` in PyTorch. The Inspector shows that
dimension and the propagated axis name when available. Expanded and compact
blocks expose the declared axis. Shape/rank validation happens before allocating
a Torch layer; unknown configuration fields, booleans, fractions and out-of-range
axes fail rather than silently coercing to a different model.

## Storage, execution and agent edits

The persisted node is `{"id":"probabilities","type":"Activation",
"activation":"Softmax","dim":-1}`. Normalization fills an omitted dim with −1
without changing the caller's dictionary. The original four activation documents
are unchanged. Existing presentation fields remain available; `dim` is a
computational setting, so changing it invalidates incompatible result identity.
Equivalent axis spellings still represent different declared specifications.
Folding/repositioning remain visual-only.

Both native compilation and standalone exported Python use
`nn.Softmax(dim=RESOLVED_PYTORCH_DIM)`. SQLite save/reload, frozen model closure,
undo/redo, CLI validate/export and trained checkpoint restoration preserve it.
Agents can use the ordinary ordered `set-node` operation to edit `dim` or
`activation`; batch validation rejects an invalid final model atomically. No
SQLite tables or workspace format version changed. Older app versions reject the
new activation rather than reinterpret it; existing models retain their meaning.
New generated notebook bundles include its pure activation runtime dependency;
previous immutable 20-file runtime bundles remain untouched and are verified
through an actual pre-Softmax closure/fresh-kernel regression. An incomplete new
closure cannot masquerade as an old one to omit the activation dependency.

## Loss semantics and verification

Softmax does not select a loss. Cross-entropy task heads expect **raw logits** and
already perform their own stable normalization; do not place Softmax before that
loss. Explicit probability outputs can be useful for inference or an intentionally
chosen probability-space regression objective. The app does not silently insert
or remove Softmax or change your task. See the primary
[PyTorch Softmax documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)
and [CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

`tests/test_softmax.py` compares independent scalar probabilities and analytic
vector–Jacobian products, positive/negative multidimensional axes, extreme logits,
batch independence and singleton axes. It checks allocation-free validation,
standalone export, independent input paths joined into five probabilities,
registry/history/CLI identity and an actual probability-target
fit with exact trained export. `tests/activations.test.js` checks teaching text,
alphabetical choices and literal axes. `tests/qt_softmax_smoke.py` exercises actual
palette/inspector controls, rejected edits, undo/redo, folding, native code and
`.vtw` reload without loading Torch in the UI host. Private evidence records exact
outcomes and the inspected rendered capture; this is a local prototype change,
not publication of a new release.
