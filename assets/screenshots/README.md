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
The final `interface.png` can show a deliberately edited draft; use
`interface-resized.png` for the README's valid model. It demonstrates diagram
editing and shape inspection, not a completed training run. Future training and
diagnostic images must come from real, reproducible runs as well.
