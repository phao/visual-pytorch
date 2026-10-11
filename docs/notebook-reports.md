# Portable notebook reports — branch-only Stage C checkpoint

This contract describes executable code on `task/stage-c-experiments`, not the
initial public release. A bounded [experiment GUI](experiment-gui.md) now browses
selection/history, generates bundles and explicitly inspects/executes them;
full Stage C gates remain open. [GUI notebook execution](notebook-gui-execution.md)
specifies asynchronous origin/busy/Stop and receipt presentation.
The CLI and helpers generate reports from exact stored
ReportRecipe revisions; browsing heads never substitute for those revisions.

## Two honest modes

- `result-report` reads the recipe's saved immutable run revisions, including
  unsuccessful attempts. Generating or executing this mode launches no training.
- `rerun` captures a bounded experiment selection. Generation does not train.
  Executing the notebook's resolution cell explicitly launches fresh CPU fits.
  Its mandatory preceding preview displays all captured compatible, excluded
  and incompatible rows. Failed/excluded rows remain visible with reasons and
  unavailable metrics, never zero-valued substitutes.

Questions and descriptive summaries are conditional on the selected observations,
split populations and protocols. Retries are not independent scientific run units.
Inference, architecture rankings and generalization guarantees are not implied.

## CLI and optional dependencies

Install `requirements-report.txt` into the authoring/execution environment.
The ordinary application/CLI does not eagerly import Jupyter, Torch or Qt for
these commands. Authoring requires nbformat; real automated execution also needs
nbclient, jupyter_client and ipykernel. The numerical interpreter needs PyTorch.

```sh
python vpw_cli.py example.vtw generate-report REPORT_ID EXACT_REVISION --output NEW_DIRECTORY
python vpw_cli.py example.vtw execute-report NEW_DIRECTORY --kernel-python /absolute/path/to/python --cell-timeout 600
```

Generation accepts `--offset`, `--limit` (1–1000), `--allow-partial`,
`--memory-limit-percent` and `--worker-timeout` for explicit rerun coverage and
execution policy. Without partial permission it rejects incomplete selection.
Saved-result mode rejects candidate paging flags. Neither command changes the
source workspace. The output directory must be new; partial writes are left
inspectable and are not reused or silently overwritten.

## Version-1 bundle and provenance

Transfer the complete directory, not only its `.ipynb` file:

```text
bundle/
  manifest.json
  payload.json
  requested.ipynb
  LICENSE
  README.txt
  requirements.txt
  runtime/app/             complete 20-file numerical source closure
  executions/<attempt>/   fresh artifacts from each explicit execution
```

`manifest.json` has exactly version, format, recipe, files and requested_notebook.
Version is 1; format is `visual-pytorch-report-bundle`. Recipe is an exact
kind/id/revision reference. `files` maps each expected relative input path to
byte count and SHA-256. `requested_notebook` identifies `requested.ipynb` and its
SHA-256. The manifest is published last. Verification rejects missing/unexpected
closure entries, unsafe paths, symlinks, changed bytes and mismatched identities
before starting a kernel. Hashes detect mutation; they do not authenticate authors.

`payload.json` has exactly version, recipe, documents, preview, execution_policy
and generation. Documents contain the complete immutable dependency closure in
dependency-before-consumer order, each as kind/document. Duplicate or additional
dependencies are rejected. Preview is null for saved results, otherwise the
captured bounded page, coordinates, status and actual split manifests. Replay
validates these without regenerating splits. Execution policy records
`cpu-float32-v1`, constrained mode, RAM percentage and optional worker timeout.
Generation records Python/platform. There are no absolute private source paths.

The numerical runtime is copied, not imported from the producer checkout. Its
closed file list is `app.notebooks.RUNTIME`, including presentation contracts.
The GPL license travels with it. The bundle contains raw inline observations and
weights: inspect privacy before transferring or publishing it.

## Notebook and kernel execution

The notebook is nbformat 4 with stable unique cell IDs. The visual_pytorch
metadata records version, exact recipe, mode, input_index_hash, execution_behavior
and requested state. A bootstrap cell rechecks input bytes/identity and refuses a
preloaded app package from a different checkout. Each fresh manual execution
creates its own attempt directory. Rerun preview precedes resolution; selected
question/models/datasets/protocol/training/evaluation/failures/environment/
checkpoints sections have explicit section metadata and order.

`execute_bundle` uses an actual fresh ipykernel via AsyncKernelManager/nbclient,
not a Python-cell simulator. Kernel specification, connection secret and runtime
directories are temporary and outside published artifacts; no user kernel is
registered. Kernel Python is explicit. Cell timeout must be a positive integer.
On completion, timeout or cell failure it shuts down the kernel and retains:

- the exact requested bytes in the attempt directory;
- `executed.ipynb` with real outputs, execution counts and failing-cell output;
  later unexecuted cells retain null counts;
- `execution.json`, a version-1 notebook-execution receipt with attempt_id,
  status (completed/failed/cancelled), recipe, requested_sha256,
  input_index_hash, started_at, finished_at, elapsed_seconds, error,
  engine (`nbclient-real-ipykernel`), interpreter, cell_timeout_seconds,
  executed_code_cells and an artifact byte/hash index.

The root requested notebook is never rewritten. Explicit Stop supports a cancel
event with asynchronous cleanup and partial-output retention; actual timeout or
independent kernel death stays failed. Kernel Stop before launch retains zero-cell
evidence. Stop during startup waits for manager readiness rather than abandoning
an unidentified process. KeyboardInterrupt/SystemExit preserves cancelled evidence
before reraising. Receipt artifacts exclude the
receipt itself. A completed notebook means cells completed, not that every
training candidate succeeded. Per-run statuses remain authoritative. Manual
Jupyter execution writes numerical artifacts but does not automatically save
the executed notebook or this receipt; the adapter supplies those guarantees.

## Numerical artifacts and figures

Each terminal candidate is saved immediately as immutable run JSON. Reports
also retain resolved report/result data, metrics CSV, fresh architecture source
and fitted standalone predictor source for admitted checkpoints. Exported
predictors use Torch/stdlib, not the app package, and replay saved preprocessing
without refitting. Fresh architecture source is not a trained predictor.

Metric rows identify run/revision/attempt/status, model/dataset/task, checkpoint,
population, head/loss, source units, target transforms, value, denominator,
records, parameters and updates. Missing metrics remain blank. Every head has
separate post-update train/validation curves: linear axes, labelled ticks,
solid/circle versus dashed/square encodings, no smoothing, no test curve and no
invented confidence intervals. Missing points break curves. Companion CSV
retains each point's update/checkpoint/population/value/denominator/record count;
captions identify transformed target loss, retained/discarded history and run.

Newly generated training cells display a semantic HTML figure companion with the
same retained numerical SVG, a readable full caption and a collapsed first-100
exact CSV-row preview. Full CSVs remain available in the attempt directory;
unavailable values are labelled, not replaced by zeros. Larger wrapped SVG text,
inward endpoint labels and scoped title/description IDs prevent reproduced
clipping/collisions. See [report-figure-presentation.md](report-figure-presentation.md)
for precise layout, trusted-input, format, resource and rendered-test limits.

## Verification and limits

The original portable-bundle checkpoint's sixteen notebook tests include actual kernels for both modes, all eight bounded
2×2×2 fits and exact local replay, portable closure and standalone inference,
real failing cells/timeouts, corruption gates, exclusions/incompatible rows,
failed metrics, captured split reads/retry-substitution rejection and independent
SVG-coordinate/CSV checks. At that historical checkpoint the branch passed 342 Python tests and
57 JavaScript tests plus native Qt and sandboxed WebEngine layout/history checks.
Later GUI execution and cancellation evidence is recorded in
[development.md](development.md) and [notebook-gui-execution.md](notebook-gui-execution.md).
These are fixture/environment-specific results, not cross-platform bitwise promises.

The [Stage C requirement audit](stage-c-verification.md) separates computational
exit evidence from remaining GUI/transport/integration work. Its saved-result
no-training probe runs inside an actual fresh kernel, guarding candidate/task/
worker entry points and Torch imports while all selected report sections execute.
The same guarded rerun fails at its first candidate dispatch as a negative
control. Parent-process patches alone cannot prove this subprocess property;
these test guards are not a hostile-code sandbox.

Bundle execution runs trusted Python. Integrity checks are not a hostile-code
sandbox, signatures, remote job receipts or optimizer resume. Source workspaces
remain read-only and generated runs are bundle artifacts, not automatically
imported into the lab workspace. Explicit local GUI notebook inspection/execution
is implemented separately; richer report UI, remote execution and full Stage C
completion are not claimed. A real sandboxed browser now renders both complete
executed notebook modes and their SVGs through the optional nbconvert basic
acceptance fixture. The report-figure contract records visual/keyboard/narrow-view
checks and their scope; numerical tests and this bounded rendered review do not
certify every frontend/viewer or close the full C exit audit.
