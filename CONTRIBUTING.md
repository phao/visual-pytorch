# Contributing

This is a research prototype with human and AI contributors. Both are welcome.
Explain the problem, a small reproducible case, and the behavior you expect.
For interface changes, provide real screenshots and describe keyboard behavior.
Do not post private datasets, workspaces, credentials or identifiable lab material.
Keep issue reports small; a synthetic example is often enough.

Use a task branch and open a pull request against `main`. Preserve the floating
tools and compact canvas unless a redesign is explicitly agreed. Keep the CLI,
workspace format and portable UI specifications synchronized. Describe AI
assistance honestly, and review generated code before presenting it as reliable.

## Checks

Python tests require PyTorch; native smoke tests additionally require PyQt6 and
PyQt6-WebEngine. JavaScript tests use Node's built-in test runner.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.js
python -m compileall -q app tests run.py vpw_cli.py install_desktop.py
QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
  python -m tests.qt_ui_smoke
QT_QPA_PLATFORM=offscreen python -m tests.qt_worker_smoke
```

Do not disable the Chromium sandbox to make tests pass. Qt system packages and
offscreen rendering support vary by platform; report unavailable checks rather
than claiming they passed. Use temporary private files/profiles, bounded fits and
conservative CPU thread counts. Never test by changing a user's live workspace or
restarting their app. Changes to scientific computation need independent tensor,
gradient or checkpoint comparisons appropriate to the risk.

Contributions to original project code/documentation are under GPL-3.0-only.
Preserve third-party attribution and explain any new dependency or license.
