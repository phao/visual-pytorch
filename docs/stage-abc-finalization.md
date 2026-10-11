# A–C finalization and publication gate

The user requested: finish A–C → push that verified checkpoint to GitHub → only
then resume D–E. This checkpoint is based on `main`
`bc85e57254083bf969a189d9685574b4d8598669`, not the unfinished D branch. Existing D
work remains at `ab0502c9d318be6ebb46fed96b455bd230b4b00a`; it is not included here.

## What A–C means

- **A: trustworthy semantics and evidence.** Stable input identities and explicit
  dataset bindings; exact data/model/preprocessing revisions; actual work
  counters and cancellation; optional analysis cannot destroy a valid learned
  checkpoint; exported predictors retain their fitted preprocessing and weights.
  See [A verification](stage-a-verification.md).
- **B: a bounded usable CNN/tensor workflow.** Explicit tensor shapes and axes,
  versioned operators, Conv2d/pooling/Flatten/Permute, independent numerical and
  gradient checks, structured training and standalone prediction/export.
  See [CNN workflow](cnn-workflow.md). This is not every CNN/operator/device.
- **C: independent scientific objects and reproducible reporting.** Versioned
  datasets/tasks/experiments/runs/reports, group/time splits, train-only fitted
  transforms, structured losses/minibatches, explicit candidates and exclusions,
  saved immutable evidence, portable reports and actual fresh-kernel execution.
  See [C requirement mapping](stage-c-verification.md) and
  [C acceptance and boundaries](stage-c-acceptance.md).

These are the pinned nepr programme's scoped stages, not an assertion that every
suggestion in its report or the full A–E product has been delivered.

## Reproduced defect and correction

Fresh sandboxed Qt checks reproduced a missing-nbformat error during generation
with `/usr/bin/python3`, despite successful notebook execution through the chosen
report environment. Generation now captures all selected revision/data/runtime
bytes in the host, constructs and inspects the bundle in the selected report
Python, and verifies the returned exact closure locally. CLI `generate-report`
offers `--report-python`; omitting it retains direct in-process generation.

Generation never fits a model, executes notebook cells or starts a kernel. It
refuses existing destinations and does not modify SQL/history. The generation
process has a 30-second deadline and a bounded result line; complete inline
capture/validation/copy costs remain explicit. It is synchronous, not a new
cancellable or detached-job feature. No `.vtw` or report-format bump is needed.

## Final verification

Fresh unchanged-source checks pass:

- 628 Python tests in 257.810 seconds, including 15 new generation regressions;
  260 JavaScript tests in 1213.211708 milliseconds.
- Real sandboxed Qt graph gestures/history, stable bindings, editor-built CNN
  training, checkpoint reload and trained export. The CNN's held-out MSE is
  0.0000212147115 versus its 0.0269664740 train-mean baseline; this is a small
  synthetic test, not a general benchmark.
- Actual Qt dataset/task/experiment authoring, candidate browsing, saved evidence
  and predictors/curves; real worker fits, retry, recovery export and save/reload.
- Actual notebook generation with an invalid-interpreter refusal and selected
  Python success; fresh saved-result and eight-fit rerun kernels; failure,
  timeout, Stop, kernel shutdown and exact retained artifact checks. The GUI
  host imports neither Torch, nbformat, nbclient nor jupyter_client. SQL unchanged.
- 14 native experiment lifecycle cases and 10 notebook lifecycle cases.
- 60 actual native-window configurations: 780 focused toolbar hit tests, 300
  launcher clicks, 60 Validate clicks and 60 native Tab reveals. Private SQL,
  disk, history and configuration are unchanged.

- Fresh report fixtures execute both saved-result and eight-fit rerun notebooks;
  actual sandboxed rendering verifies 11 retained/synthetic curve geometries,
  notebook cell counts, no remote requests and zero JavaScript errors. Synthetic
  layout probes are explicitly distinguished from retained numerical curves.

Whole-suite checks used numerical Python 3.12 and native
Qt checks used system Python 3.13 with separate report dependencies. Workers use
one numerical thread; Chromium sandboxing is enabled. No user application,
workspace or configuration is used. Initial generation failure and a render
harness invocation missing its required fixture argument remain in private
campaign evidence along with the successful repeats. Compositor diagnostics on
stderr are retained; zero JavaScript errors does not mean empty stderr.

## Boundaries and continuation

The A–C checkpoint reads `.vtw` version 6. Existing D source supports newer
version-16 workspaces. Do not open or rewrite a user's newer workspace through
the A–C checkout. After the authorized A–C push, bring the correction into the
retained D branch and restore that current-format everyday checkout before
continuing D. No silent workspace downgrade or user application restart.

The initial `v0.1.0-alpha.1` tag stays unchanged. A source push is not a new
packaged release, tag, cross-platform certification or completed D/E promise.
