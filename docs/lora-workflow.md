# A small, executable LoRA teaching workspace

Open [examples/lora.vtw](../examples/lora.vtw). Use the Architectures panel to
inspect **LoRA expanded · r=2**. The lower branch contains Adapter A, Adapter B
and a fixed alpha/r scale; the upper branch contains the frozen supplied Base W₀.
Add combines equal-shaped tensors, rather than concatenating features.

This is a synthetic linear/MLP example, not a language model or pretrained
Transformer reproduction. The teacher deliberately uses a rank2 perturbation;
the small comparison demonstrates execution and parameter policy, not unbiased
evidence that LoRA generally outperforms full fine-tuning.

## What is inside

The workspace has16 related drawings: base-only and full-fine-tuning linear
maps; expanded rank1/2/4 adapters; a reusable adapter wrapper; base/full/adapted
two-layer MLPs and their six layer definitions; and an explicitly derived merged
inference map. MLP ReLU is **between complete adapted layers**, never between A
and B. Baseline layer matrices are 0.6I and, for the second MLP layer, 0.8I.

For row-batched tensors the rank2 layer computes

\[
Y = XW_0^\mathsf{T} + 2(XA^\mathsf{T})B^\mathsf{T}.
\]

The base is16×16, A is2×16 and B is16×2; all biases are off. A starts from
seeded normal values (seed17, standard deviation0.05), B from zeros. The stored
base is frozen without detaching the input-gradient path. Scale uses alpha4/r;
rank changes regenerate factor shapes and initializer provenance.

| Drawing | Frozen elements | Trainable elements | Stored elements |
| --- | ---: | ---: | ---: |
| Base-only | 256 | 0 | 256 |
| Full fine-tuning | 0 | 256 | 256 |
| LoRA r=1 | 256 | 32 | 288 |
| LoRA r=2 | 256 | 64 | 320 |
| LoRA r=4 | 256 | 128 | 384 |
| Two-layer LoRA MLP | 512 | 128 | 640 |

The mutable drawings describe **initialization**, not automatically the trained
weights of a retained run. Display labels make the base/A/B roles clearer without
rewriting historical frozen models or results. To use actual trained values,
export an exact retained-run predictor or inspect the merged inference drawing.
Folding, dragging and resizing change presentation, not computation. Space toggles
the selected block; Ctrl+Z/Ctrl+Shift+Z undo/redo edits.

## Real checks, not full training infrastructure

The Experiment panel's scientific browser contains two fixed comparisons:
linear full/rank1/2/4, and MLP full/adapted. Each comparison uses the same supplied
base values,96 observations, task, split and Adam protocol. Split seed31 gives
60 train/12 validation/24 test observations; initialization seed7, batch24,
learning rate0.02,200 updates, evaluation every20 updates.

Base-only results are separate **zero-update native evaluation artifacts**:
[linear](../examples/lora.linear-baseline.json) and
[MLP](../examples/lora.mlp-baseline.json). They do not invent an optimizer or a
completed training run. [Comparison evidence](../examples/lora.comparisons.json)
identifies the six genuine retained attempts, their initialization snapshots,
metrics, exact checkpoint references and merge provenance. Failures are retained
as failures; the generator never relabels an incomplete comparison as complete.

| Synthetic held-out population | Test MSE |
| --- | ---: |
| Linear base-only | 1.190317520500921e-4 |
| Linear full fine-tuning | 4.516446295477032e-5 |
| LoRA r=1 | 4.803040277109452e-5 |
| LoRA r=2 | 1.4556105602904373e-10 |
| LoRA r=4 | 2.73537835163041e-10 |
| MLP base-only | 3.2998473434060993e-4 |
| MLP full fine-tuning | 3.523507075413371e-6 |
| MLP LoRA | 9.752724317898744e-7 |

These numbers concern one deliberately low-rank synthetic teacher and a fixed
seed/protocol. Do not treat them as a model-selection study or confidence interval.
For larger training, export the report's explicit-rerun notebook and run it in an
appropriate external environment. Generating a notebook does not execute it.
The saved-result notebook reads retained evidence without retraining.

## Inference and lineage

The rank2 merged drawing is an **explicit new derived artifact**:

\[
W_{\mathrm{merged}}=W_0+2BA.
\]

Its ParameterDerivation record pins all three owners to one exact retained run,
attempt and checkpoint. Original drawings/runs remain intact. The inference
weight is frozen and preserves stable input/output IDs and sample shape.
Float32 kernels may round differently; the bounded example uses2e-6 absolute and
1e-5 relative prediction tolerance, not a universal error guarantee.

The [exact-base inference package](lora-inference-package.md) includes mandatory
base/A/B bytes and a checked standalone CPU float32 predictor. It takes raw
architecture inputs and does not include fitted task transforms or optimizer
resumption. The separate retained-run predictor includes fitted preprocessing.
[Merge](parameter-derivation.md), [policy](parameter-policy.md),
[binding](parameter-binding.md) and [format](vtw-format.md) specifications describe
the actual closed schemas and compatibility fences.

## Generate and replay

`python -m examples.lora_workspace /path/to/NEW.vtw` prepares related definitions
and recipes only. It does not train or fabricate a merged map.

`python -m examples.lora_workspace /path/to/NEW.vtw --run-comparisons` explicitly
executes six fits, evaluates the frozen baselines and admits the actual merge.
Existing destinations refuse. Use a fresh basename for every generation.

The bounded acceptance probes take explicit existing-example and new-output
paths: `tests/qt_lora_example_smoke.py`, `tests/probe_lora_cli.py` and
`tests/probe_lora_notebooks.py`. The desktop probe uses a private config/profile
and in-memory workspace copy, never the user's running app. With GPU rendering
disabled but Chromium sandboxing enabled, actual compact-icon drag/undo/Space
expansion,200-update GUI worker, save/reload and matching CLI execution pass.
Fresh saved/rerun kernels reproduce four original checkpoint IDs, metrics and
checked predictions. Qt's synthetic mouseDClick did not generate a browser
double-click in the diagnostic trace; native double-click injection is therefore
not claimed by these probes. The ordinary browser handler remains unchanged.

Broader final regression acceptance is recorded separately in development
history; targeted checks alone do not establish complete product conformance.
Exact latent-state Neural MS-VAR and a production Loss workbench are not implied
by this LoRA example. Architecture/Dataset/Loss/Optimizer retain separate
workbench types; the standalone Loss design study is not installed in the app.
