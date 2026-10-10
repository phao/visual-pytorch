# First structured CNN workflow: contract and verification

This is a bounded CPU float32 regression slice, not a complete vision/task
platform. It extends the same DAG, .vtw collection, CLI, worker, checkpoint
validation and standalone export while preserving the accepted MLP UI.

## Use it

Create a reproducible workspace without Qt or Torch imports:

    python -m examples.cnn_regression /path/to/new-cnn.vtw
    python run.py /path/to/new-cnn.vtw

The example is Input [1,6,5] (C,H,W) → Conv2d(2 filters, kernel [3,2]) →
Tanh → AdaptiveAvgPool2d([1,1]) → Flatten → Linear(1,bias) → Output.
It includes 128 synthetic structured samples through a recorded row-major
adapter. Open Experiment and Train. Add --train to the generator to run its
bounded 450-update, one-thread fit and save both checkpoint and historical run.
It refuses an existing target and saves inputs before optional training.

To build manually, double-click Input, set sample shape and optional C,H,W
axis names. Select it and choose Conv2d in Building blocks; edit kernel,
channels, stride, padding, dilation, groups and bias in Inspector. Insert an
activation, spatial pooling, Flatten and a Linear readout in order. Palette
entries are alphabetical with type icons and explanatory tooltips. Choose
Spatial Conv2d teacher (or Auto for eligible CNNs), generate data and train.
Arbitrary student graphs are not promised to represent this teacher exactly.

Inspector displays input/output sizes, axes, weight shape and spatial-size
formula. Flatten and Permute index sample axes, never B. HWC data needs Permute
[2,0,1], not Reshape. Compact Conv2d blocks retain channel/kernel summaries and
bias state; icons remain drag surfaces. Learned parameters show actual operator
types/tensor dimensions, not a mislabeled Linear matrix for a convolution.

## Persistence and execution

[vtw-format.md](vtw-format.md) specifies version-4 fields, defaults, unsupported
options, axes and adapters. app/operators.py owns pure normalization, shape and
constructor specs; native execution and generated code consume these specs.
Grouped divisibility, layout and positive spatial sizes are checked before
Torch allocation. Writes normalize new operators; legacy operators are unchanged.

The v3→v4 working-copy migration preserves old graph/data/binding/checkpoint
meaning. Opening is read-only; explicit Save upgrades the container. Optional
axes enter bindings only when present. A recorded tensor-adapter mismatch cannot
be repaired by column rebinding, which does not reorder values.

Forward uses actual [B,C,H,W] tensors; flat storage is an explicit input adapter.
Only the loss packs predictions for existing MSE regression. Conv weight/bias
shapes and stable IDs participate in pure checkpoint admission, including
independent nested reference instances. Restore/export load exact weights and
evaluation mode. Convolution never allocates a dense effective A. RAM admission
includes convolution workspace allowances; [memory-policy.md](memory-policy.md)
specifies this heuristic, which is not an OOM guarantee.

## Acceptance evidence

tests/test_cnn.py uses independent Python scalar cross-correlation and analytic
vector-Jacobian loops, not F.conv2d or the graph executor as its oracle. Cases
cover groups 1/2, depthwise multiplier, bias off/on, batches one/two, unequal
kernel/stride/padding/dilation and exact input/weight/bias gradients. There are
32 additional bounded shape/parameter configurations.

Independent pooling loops check negative-infinity MaxPool padding, AvgPool
include/exclude-padding divisors and gradients, and unequal overlapping adaptive
bins. Identifiable Permute values differ from Reshape. Partial/negative Flatten
axes preserve B/value order. Unsupported versions/options, numeric coercion,
groups, dimensions, axes, permutations and adapter rebinding are rejected.

Native/export equivalence includes all pooling types, Flatten/head, bias states,
multiple batches, global default float64, exact state schemas and independent
nested CNN references. The spatial teacher is independently evaluated from stored
kernels. A 350-update fit reached held-out MSE 0.00027903161 vs 0.014173648 for
the training-only mean baseline. Metrics were recomputed from restored predictions;
SQLite reload and actual CLI trained export reproduce predictions exactly.
Corrupt Conv weights are rejected.

Actual CLI subprocess checks create/normalize a CNN and reject an invalid grouped
edit without changing disk bytes. The runnable example is executed with --train;
its checkpoint/run are reloaded and a repeated invocation refuses overwriting.
The complete regression suite passed 195 Python and 57 JavaScript tests, including
the inherited Stage A gates. Native Qt binding/history/policy and the inherited
real WebEngine gesture/theme/font/undo checks also passed.

tests/qt_cnn_smoke.py uses real sandboxed Qt/WebEngine controls to build/edit,
undo/redo, collapse with visible bias, generate data, train through the actual
worker, display Conv2d results, save/reload and produce trained export. Its
450-update fit reached held-out MSE 0.000021214712 vs 0.026966474 for the training
mean baseline, without JS errors. Temporary profiles/config/workspaces do not
touch user state. Local Torch 2.13.0 was tested; numerical evidence is not a
cross-runtime bitwise promise.

Commands:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
    node --test tests/*.test.js
    QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu /usr/bin/python3 -m tests.qt_cnn_smoke
    QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu /usr/bin/python3 -m tests.qt_ui_smoke

Use your Qt interpreter elsewhere; the smoke discovers the separate Torch
interpreter. Chromium sandboxing stays enabled.

## Sources and deliberate limits

Primary references:
[Conv2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv2d.html),
[AvgPool2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.AvgPool2d.html),
[MaxPool2d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.MaxPool2d.html),
[Flatten](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Flatten.html),
[Permute](https://docs.pytorch.org/docs/2.8/generated/torch.permute.html).
Our sample-axis numbering intentionally differs from raw Torch Flatten/Permute;
the batch-axis translation is part of the file contract and independently tested.

No classification task/loss, arbitrary image storage, BatchNorm, transpose
convolution, symbolic/recurrent state, optimizer resume or detached backend is
claimed. These require subsequent C/D/E contracts and tests.
