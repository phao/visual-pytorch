# A–D development checkpoint: scope and verification

This is a research-prototype source checkpoint, not a new packaged release or
a promise that every neural architecture is supported. The initial
`v0.1.0-alpha.1` tag remains unchanged. A–C were previously published; this
checkpoint adds the accepted bounded Stage D work. Original code is
GPL-3.0-only and remains openly AI-coded and human/AI-directed.

## What Stage D adds

- Scoped integer/enum architecture constructors, checked derived dimensions,
  nested bindings and explicit specialization provenance.
- Repeated independent layers versus genuinely shared parameter owners, with
  unique parameter counts and summed gradients/one optimizer update per owner.
- Explicit recurrent cell interfaces and returned state ports, lengths, masks
  and reset controls. No hidden state carry across unrelated calls.
- Built-in projected/bidirectional LSTM with explicit role/state ordering, plus
  allowed runtime temporal dimensions without changing learned parameter widths.
- Retained dataset/task/family/experiment/checkpoint/report routes and standalone
  PyTorch generation, including actual fresh saved-result/rerun notebook kernels.
- A complete CNN-to-recurrent example with ragged image sequences, explicit
  collation, independently tested gradients and bounded real-worker training.

Start with [the CNN/recurrent guide](cnn-recurrent-example.md):

```sh
python -m examples.cnn_recurrent_experiment /path/to/new-cnn-recurrent.vtw
```

The generator refuses existing files. Open the new workspace in the desktop;
training is a separate explicit action. See that guide for constructor options,
controls, losses and the distinction between fitting predictions and recovering
unique underlying parameters.

## Evidence underlying acceptance

The accepted original-D tip is
`501e5bf1a2aa4fcb6fa179ead1debb9b78de4a05`. Its executable code, examples and
tests match the frozen implementation
`fe8cb80ddaa6c7960ca1ed247c6a68fbe86bdbff`; subsequent acceptance commits changed
documentation only. Verification included:

- Full1274-case Python suite:1271 executed, three Qt-dependent skips;
  elapsed630.759s. Separate real native Qt checks cover the skipped environment
  routes; a skip is not counted as a passing executed test.
- Full359-case JavaScript suite and a143-case original-D-focused selection.
- Fourteen new pure/numerical CNN-recurrent cases: six constructors, independent
  forward/input/initial-state/all-parameter gradient checks, variable T1/3/9,
  zero/full/unsorted lengths, NaN padding, masks and resets.
- Real CLI and CPU workers, saved checkpoints, fresh app-free predictors and
  actual saved-result/rerun kernels. Two bounded60-update fits demonstrate the
  complete numerical route, not a real-world benchmark or convergence guarantee.
- Sandboxed Qt displays both materialized diagrams/five named inputs, runs
  three real worker updates and reloads its exact saved result without host Torch.
- Original-D post-integration checks:14 Python/359 JavaScript cases and55 exact
  legacy experiment points. Fresh publication checks are recorded separately
  in development.md; old evidence is not relabeled a new full-suite run.

Relevant executable tests are `test_scoped_constructors`, `test_scoped_execution`,
`test_shared_parameters`, `test_shared_execution`, recurrent/LSTM contract and
execution modules, temporal-v2 tests, `test_cnn_recurrent_example`,
`test_cnn_recurrent_execution` and `qt_cnn_recurrent_smoke.py`.

## Compatibility and deliberate limits

Current workspaces use container version20. Old supported workspaces open into
a working copy; explicit Save is required to update a file. The historical
Family7/temporal-v1 typed proof restriction is retained; Family8/runtime-v2 is
explicit rather than a silent reinterpretation. Model/data/task/parameter/port
identities remain part of checked execution provenance.

Temporal splits allocate distinct time values and preserve timestamp ties.
Group splits allocate whole groups/subjects. These are separate tested policies,
not chronological splitting separately inside each subject.

Stage E is deferred and isolated on development branches. This source does not
include durable detached/remote/distributed jobs or optimizer-state Resume.
Experimental E version21/22 files are not readable here: keep their originals
for the E branch rather than downgrading headers or discarding evidence.
Ordinary retained inference checkpoints are not optimizer-resumable states.

LoRA parameter freezing/explicit initialization and exact latent-regime Gaussian
HMM/neural MS-VAR sequence likelihood are not implemented in this checkpoint.
Generic recurrence, Softmax and supervised heads do not imply those statistical
models. Those requested additions require their own versioned contracts and
independent forward/gradient/filter/worker/export/kernel checks.

Linux/Python/CPU are the exercised environments. Cross-platform installers,
GPU/AMP equivalence, elastic training and broad performance claims remain
unverified. RAM admission is a conservative estimate, not an OOM guarantee.
