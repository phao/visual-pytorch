# Frozen task training and fitted inference — Stage C component

This task-branch component is executable through Python, CLI and a native Qt
worker adapter. Independent terminal RunRecord storage now wraps this component;
see [experiment-objects.md](experiment-objects.md). It is **not** the final
experiment selection GUI, symbolic families or detached jobs. Concrete notebook
reporting is specified separately in [notebook-reports.md](notebook-reports.md).
Legacy regression workers and identity-preprocessing checkpoints are unchanged.

## Frozen execution input

`TaskBundle` integer version 1 has exactly `version`, `model`, `dataset`, `task`,
`manifest`, `protocol`. ModelRevision, DatasetSpec and TaskSpec include verified
content revisions; the complete reachable model closure and actual inline records
are present. Compatibility checks cover stable input/output IDs, shapes, axes,
loss/vocabulary, transforms and split IDs before Torch import. A browsing head is
never an implicit execution reference. `task_bundle` makes owned frozen values;
`validate_bundle` rejects unversioned/tampered/incompatible closures.

`run_identity` integer version 1 captures the three `{kind,id,revision}` references,
manifest/protocol hashes, `compiler: native-graph-v1` and
`task_runtime: masked-additive-v1`. These reuse the canonical JSON hash convention
in [revision-store.md](revision-store.md), not RFC 8785. Environment is recorded
separately; hashes do not prove execution origin or cross-version bitwise replay.

`TrainingProtocol` version 1 is closed. An example with all explicit fields:

```json
{
  "version": 1,
  "backend": "cpu-float32-v1",
  "batch_size": 16,
  "budget": {"epochs": 20, "updates": null, "examples": null, "seconds": null},
  "optimizer": {"kind": "Adam", "lr": 0.03, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
  "seeds": {"initialization": 7, "shuffle": 11},
  "shuffle": true,
  "evaluate_every": 10,
  "history": {"kind": "latest-window", "max_points": 200},
  "augmentation": "none",
  "checkpoint_selection": "last-successful-update",
  "test_evaluation": "final-completed-only"
}
```

Integer counters must be positive, seeds in 0..2^31−1, shuffle strictly boolean.
At least one budget is non-null; optional seconds is positive finite. Adam settings
are finite: 0 < lr ≤ 1, both betas in [0,1), eps > 0, weight_decay ≥ 0. No hidden
schedule, early stopping, augmentation, optimizer resume, CUDA or precision policy
is accepted. The numerical backend uses CPU float32 constructors, one Torch CPU
thread, Adam's non-fused/non-foreach implementation, and an initialization RNG
fork. Epoch shuffles use a private Python MT19937 instance. Initialization/shuffle
do not alter caller RNG state; matching seed numbers across differently shaped
models does not imply identical initial random variables. Generation and split
identity already belong to the retained dataset/provenance and actual manifest.

## Batches, updates and objective

Each finite epoch traverses each train ID once, optionally shuffled, without
replacement. Final short batches are retained. An example budget truncates its
last batch exactly; it never silently overshoots by a full batch. Boundaries check
cancel, then update/example/epoch budgets, then elapsed time. Multiple budgets stop
at the first observed boundary (the stated order breaks simultaneous ties).
Elapsed seconds begin after model/optimizer setup, include evaluation, and end
after final evaluation; the time budget is cooperative, not a hard process deadline.
It can be exceeded by one batch/final evaluation. Transport deadlines are separate.

For each head, the unreduced loss uses the task's actual mask and sample weights.
MSE/MAE elements are scalar components; cross-entropy uses fixed-vocabulary class
indices and logits [B,C,...]. A head mean is its weighted numerator divided by its
positive effective-weight sum. The minibatch objective is the sum of
`head.weight × available_head_mean`; unavailable heads contribute no term, and
remaining head weights are **not** renormalized. This is a minibatch objective,
not a claim that its gradient equals a globally weighted whole-dataset objective.
Population reporting sums numerators/denominators, never means of batch means.

No supervised head → skip the optimizer update. An entirely unsupervised full
epoch stops with `status: unavailable`, reason `no_training_supervision`; update-only
budgets cannot loop forever in this case. `attempted_batches` and
`examples_processed` include skipped/failed batch attempts. `completed_updates`
and `updated_examples` count only finite successful optimizer steps; the latter
counts their batch rows, including masked rows, not effective supervised elements.
`skipped_batches` counts wholly unsupervised batches. `completed_epochs` counts
fully traversed epochs, not partial passes. Per-head effective element counts
remain in metrics. Finite gradients, parameters and optimizer tensor state are
checked; a corrupt step restores the last finite inference parameters and fails.

## Timing, evidence and terminal results

Curve points carry actual update/batch/example counts, checkpoint SHA, eval mode
and `timing: post-update` (`initial` at update 0). Each contains only train and
validation metrics at that exact checkpoint. The first successful update, every
declared interval, and a necessary final point are observed. Each head records
actual observation IDs, population/checkpoint, additive terms, evaluated/unique
counts and unavailable zero denominators. Classification accuracy uses the same
masked weighted population as loss. Empty validation is explicitly unavailable.
Test is evaluated once, at the final checkpoint of completed budget execution;
failed/cancelled/unavailable attempts do not inspect test. There is no best-test
selection or validation-selected checkpoint yet.

Curves retain the declared latest window, with `observed_points` and
`discarded_points` made explicit. Every observation can be emitted as progress;
the final artifact does not falsely claim to retain discarded curve points.

`task-training-result` version 1 returns: `status`, `stop_reason`, `failure`,
`evaluation_error`, `identity`, `bundle`, `checkpoint`, `checkpoint_id`, `counts`,
`history`, `history_retention`, `final_metrics`, `elapsed_seconds`, `memory_admission`,
`environment`, `sampler`, plus `version`/`kind`. Status is completed, cancelled,
failed or unavailable. Stops name epoch/update/example/time budgets, cancellation,
missing supervision, execution or evaluation error. Environment records Python,
Torch, platform, device, dtype, thread count and no optimizer-resume capability.
This is a self-contained JSON artifact, **not itself a registered RunRecord**.
Direct `train-stored` admission/initialization/import/process failures still use
worker/CLI errors. Explicit `run-candidate` captures these as terminal attempts,
and successful worker artifacts are validated and stored without their duplicate
bundle. Exact references plus a captured actual manifest/protocol reconstruct the
bundle from retained immutable revisions. No queued/running/detached receipt is
persisted by this synchronous wrapper.

## RAM and fitted checkpoint

`resolve_run` combines the existing model/Adam/activation/workspace/evidence
heuristic with typed preparation and declared bounded metric history. Its legacy
batch estimator is given a ceil-inverse sample count so it covers the full actual
minibatch, not 80% of it. Metric history charges `max_points × (8192 + 512 ×
train_and_validation_rows × heads)` bytes. The estimate compares to the user-level
fraction of available RAM before Torch allocation; preparation independently
checks expansion before padding/copies. This is conservative admission guidance,
not an allocator/OOM guarantee, and constrained mode may be explicitly disabled.
There is no artificial 64-feature architecture limit.

The separate `task-predictor` checkpoint has integer version 1 and exactly
`version`, `kind`, `identity`, `fitted`, `parameter_ids`, `weights`. It is not legacy
`checkpoint_version:1`. Pure validation recomputes the expected train-only fitted
state from the frozen bundle and checks identity, stable parameter paths/IDs, exact
key sets, shapes and finite float32 values. This validates compatibility, not that
an arbitrary caller's checkpoint was trained honestly. No optimizer continuation
is claimed. Failed/cancelled attempts can still yield a valid last-finite predictor.

`restore_task_predictor` accepts new observations as a list of maps from stable
field IDs to values. It applies saved imputation/standardization, fixed-vocabulary
one-hot, explicit padding and derived length/mask without refitting. Unknown
categories, missing required fields, invalid shapes, overlong sequences and
nonfinite inference outputs fail explicitly. Outputs are tensors keyed by stable
Output IDs. Classification remains logits; regression stays in declared transformed
target space, with target transforms recorded, not automatically inverted.

`generate_task_predictor` validates first, then emits standalone Python with the
same executable graph/preprocessing lowering, fitted states, strict weights,
identity, task and target-field/vocabulary metadata. It requires only Torch plus
stdlib and contains no raw training observations or app imports. Importing the
script initializes then immediately restores parameters; it never retrains.
`predict_records` uses the same semantics as restored inference. Full result bundles,
unlike predictor source, include inline dataset bytes: treat them as potentially
sensitive research data, not as safe-to-publish merely because they have hashes.

## Entry points and evidence

Python: `train_task`, `task_bundle`, `training_protocol`, `check_task_checkpoint`,
`restore_task_predictor`, `generate_task_predictor`.
Worker: `{"kind":"task-v1","bundle":...,"constrained":true,"memory_limit_percent":10}`.
Qt: `TaskTrainingJob`, using the existing bounded JSONL transport and real signals.
CLI: `train-stored` and `export-task-trained`; see [vpw-cli.md](vpw-cli.md).
No editor task-selection controls have been added yet.

Tests independently derive two weighted MSE/MAE head gradients and two scalar Adam
steps; verify finite exact budgets, all-mask skips, cancellation, injected corrupt
steps, empty validation, seed isolation and bounded curves; fit indexed and spatial
CNN classification; execute real worker/CLI cancellation and native Qt signals;
and run standalone fitted source in an unrelated directory without importing app.
Branching two-input Tanh/ReLU/Concat paths, ragged padding/mask and one-hot also
retain bitwise native/export agreement. Existing A/B and GUI gates remain required.

[PyTorch Adam](https://docs.pytorch.org/docs/2.8/generated/torch.optim.Adam.html)
supports the explicit algorithm/options used here; the independent scalar oracle
checks this implementation, not just agreement between two uses of the same API.
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
motivates separate captured streams/environment and scoped replay claims. No
cross-version/platform determinism promise is made from these bounded local tests.

Next: GUI selection/history and notebook controls, using the implemented
independent objects, bounded selection and real-kernel portable notebook APIs.
Then proceed to D/E; this component does not satisfy those full exit gates.
