# Application captures

These are actual offscreen Qt WebEngine captures using private synthetic test
workspaces and an off-the-record browser profile, not AI-generated UI mockups.

Reproduce with a Python environment containing Qt WebEngine:

```sh
QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
VTW_SMOKE_SCREENSHOT=assets/screenshots/interface.png \
python -m tests.qt_ui_smoke
```

The harness captures GitHub/Math/Metallic examples and a resized Linear block.
It also captures `interface-collapsed.png` and `interface-expanded.png`, showing
the Layout profiles and their actual bulk state/geometry changes.
The final `interface.png` can show a deliberately edited draft; use
`interface-resized.png` for the README's valid model. It demonstrates diagram
editing and shape inspection, not a completed training run. Future training and
diagnostic images must come from real, reproducible runs as well.

The initial public preview also includes `cnn-training.png`,
`cnn-training-weights.png` and `cnn-training-code.png`. These are three real views
of one editor-built, completed CNN regression run: 128 synthetic observations,
17 parameters, 450 updates; held-out MSE 0.0000212147115235 versus training-mean
baseline 0.0269664740140. They are bounded workflow evidence, not a benchmark.
The source-code view is clearly a fresh-weight template, not a trained export.
Regenerate using the private offscreen harness:

```sh
QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VTW_SMOKE_SCREENSHOT=assets/screenshots/cnn-training.png \
python -m tests.qt_cnn_smoke
```

The harness verifies current-checkpoint status, visible tabs and loss traces,
then waits for Chromium to paint before capturing; a DOM change alone is not a
painted screenshot. All data/config/browser profiles are temporary and synthetic.
