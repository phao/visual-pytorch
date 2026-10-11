# Stage A: trustworthy computation and evidence

This is an exit-gate audit of the bounded first review milestone, not a claim
that the neural experiment platform or roadmap A–E is finished. Source, tests and
specifications live in this canonical product repository. The accepted GUI and
old MLP workflows are preserved. Subsequent CNN, dataset/experiment, symbolic/
recurrent and detached execution work must maintain these gates.

## Requirements and authoritative evidence

| Requirement | Executable evidence and outcome |
| --- | --- |
| F1: valid independent affine branches retain training evidence even without a serial A,b summary | `test_evidence.py` tests equal/unequal-width branches; result weights/metrics survive with explicit unavailable analysis. `test_checkpoints.py` injects an analysis exception and restores/exports the resulting valid checkpoint. |
| F2: displayed train/held-out metrics identify the same update/checkpoint, mode and population | `test_evidence.py` computes the one-update training MSE independently from returned weights and seeded indices; final held-out observation equals result MSE. `test_checkpoints.py` independently checks zero-update and router held-out metrics. Result stores post-update/eval/dtype and split indices. |
| F3: completed updates do not equal decimated plot length | `test_evidence.py` cancels after exactly 13 updates in an 800-update request. `test_checkpoints.py` verifies cancellation before any update (0 updates/examples, valid initial checkpoint/evaluation). |
| F4: changed datasets/drafts do not display stale results as current; preserve history | `test_evidence.py` replaces data and checks current invalidation/history preservation. `qt_binding_smoke.py` rejects invalid worker checkpoint contents and archives a valid late completion against its original snapshot without presenting it as current. GUI verifies current/unverified labels and hides incompatible evidence. |
| F6: node storage/display labels cannot silently change column meaning | `test_bindings.py` pins legacy direct-root Concat order, verifies offsets, preserves binding/checkpoint under storage/label edits, blocks same-width port/order and same-flat-width shape changes. v1/v2 migration is in-memory until explicit Save. |
| Explicit rebind is intentional and atomic, not a numerical transformation | `test_bindings.py` exercises real CLI/SQLite rebind, unchanged numbers/history and byte-identical disk preservation on wrong-width failure. `qt_binding_smoke.py` checks cancel, confirmation title, undo/redo, training guard and stale modal confirmation. Real WebEngine verifies the button and binding displays through the bridge. |
| L8: concurrent pipe drainage, bounded observations and termination | 26 private-child/admission tests in `test_transport.py`: simultaneous large pipes, streamed progress, exact line limits, corrupt protocol, bounded stderr, timeout, callback/startup cleanup, cooperative TERM and ignored-TERM KILL. See worker-protocol.md for limits/assumptions. |
| Saved/reloaded/exported trained inference matches independent computations | `test_checkpoints.py` uses direct tensor algebra, not only native/generated compiler agreement: bias on/off Linear, MLP, last-axis tensor Linear, independent tanh/ReLU paths joined into sigmoid, fresh reference instances, Splitter/Replicator outputs, SQLite save/reload and actual CLI export. |
| Checkpoint representation, identity and preprocessing are checked before use | `test_checkpoints.py` rejects wrong version, identity, keys/shapes/IDs, unsupported preprocessing, NaN/Infinity/overflow, bool/string/null values. Export admission runs in fresh Python without Torch. |
| Public Module API and precision are honest | `test_checkpoints.py` verifies nn.Module rather than fake Sequential slicing, declared intermediate values, and unchanged CPU float32 training/restoration/export under global float64 default dtype. |
| Optional teacher challenge is not mistaken for an optimizer defect | Auto selects affine versus nonlinear target families, not guaranteed representability. The UI says this explicitly; deliberate teacher-family/capacity challenges remain permitted rather than being banned. |
| Existing GUI/CLI/MLP checks continue to work | Full Python/JS suites plus native WebEngine, Qt binding/history/policy and real Qt/PyTorch worker checks remain passing. No user application, workspace or configuration is restarted/changed by these tests. |

## Verified test boundary

The final bounded checkpoint has 173 Python tests and 54 JavaScript tests. The
real sandboxed WebEngine smoke covers gestures, compact icon drag, layout,
history, three themes × three font/spacing settings, binding display/rebind,
legacy/current result labels and no JavaScript console errors. Native Qt
dispatcher checks cover admission before `TrainingJob` creation, undo/redo,
confirmation races and stale completions. Separate-interpreter Qt linear and
MLP fits passed (about 1.19e-12 and 7.74e-5 held-out MSE respectively). These are
bounded fixtures, not claims about learning quality on arbitrary data.

Commands (run from repository root; use a Python with Torch for Python tests
and a Python with PyQt6/WebEngine for native Qt tests):

```sh
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.js
PYTHONPATH=. /usr/bin/python3 tests/qt_binding_smoke.py
PYTHONPATH=. /usr/bin/python3 tests/qt_history_smoke.py
PYTHONPATH=. /usr/bin/python3 tests/qt_policy_smoke.py
PYTHONPATH=. /usr/bin/python3 tests/qt_worker_smoke.py
PYTHONPATH=. /usr/bin/python3 tests/qt_worker_smoke.py --mlp
QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
  PYTHONPATH=. /usr/bin/python3 tests/qt_ui_smoke.py
```

The `/usr/bin/python3` executable above is the tested local Qt interpreter, not a
requirement on every installation. Tests discover a Torch interpreter through
the regular `VTW_TORCH_PYTHON`/PATH discovery mechanism. Chromium sandboxing stays
enabled; no `--no-sandbox`, GPU-intensive workload or external job is needed.

## Remaining platform work and limitations

The test counts above describe the historical A checkpoint, not the current
suite. The first B CNN slice and C dataset/task/experiment/run/report workflows
were subsequently implemented; see [cnn-workflow.md](cnn-workflow.md) and
[stage-c-acceptance.md](stage-c-acceptance.md). Those stages must continue to
preserve A's evidence/binding gates. D/E are separate later milestones, not
implied by this A audit. Newer task predictors retain fitted preprocessing;
that does not change the legacy identity-preprocessing checkpoint below.

The stored v1 checkpoint is for inference: identity preprocessing and CPU
float32 parameters. It does not store optimizer state or promise training resume,
authenticate caller-supplied evidence, support arbitrary fitted preprocessing,
or guarantee numerical reproducibility across devices/Torch versions. Local
transport assumes no pipe-inheriting hostile descendants; memory admission is
an estimate, not a hard resource sandbox. Scientific APIs still operate on
concrete graphs; no future operator or notebook feature is implied by these tests.

Explicit dtype/device construction follows the supported Linear constructor:
[PyTorch Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).
The global-default counterexample reflects documented dtype inference behavior:
[PyTorch default dtype](https://docs.pytorch.org/docs/stable/generated/torch.set_default_dtype.html).
