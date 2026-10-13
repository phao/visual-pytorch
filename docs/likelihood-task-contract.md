# Target-free Gaussian likelihood tasks

TaskSpec 2 adds target-free negative-log-likelihood heads and diagnostic outputs.
TaskSpec 1 retains supervised-only meaning. Concrete execution uses TaskBundle5;
temporal-v2 probability/causal families use TaskBundle6. Both use TaskResult2.
TaskSpec2 needs container29, Experiment7/Run7 need container30 and causal graphs
need container31. Policy checkpoints
keep their existing version-2 receipts; identity names `signed-additive-v2`.
Older bundles cannot carry TaskSpec 2.

## Closed grammar

The existing `id`, `name`, `inputs`, `heads`, optional `description` and computed
`revision` remain. Required `diagnostic_outputs` lists unique stable Output IDs.
Each output must have exactly one loss head or diagnostic declaration, never both.
MSE/MAE/cross-entropy heads retain existing grammar. A likelihood head requires:

```json
{"id":"density","port":"out_sequence_evidence","loss":"negative-log-likelihood","weight":1.0}
```

Optional `mask` and `sample_weight` are stable dataset field IDs. `field` and
`transforms` are forbidden. Admission requires proven `log-evidence` output
representation, not labels, arbitrary scalars, logits or posterior probabilities.
Direct GaussianFilter evidence outputs and their Output boundaries carry proof;
references that erase representation do not. Preparation creates no target
columns. Fitted metadata is `{"kind":"target-free-log-evidence"}`. Observations
and lengths are explicitly bound model inputs. Diagnostics are not optimized.

## Reduction

For log evidence $\ell_i$, loss mask $m_i\in\{0,1\}$ and fixed nonnegative weight
$w_i$, compute $N=-\sum_i m_iw_i\ell_i$, $D=\sum_i m_iw_i$, $L=N/D$.
If $D=0$, the head is explicitly unavailable. Population aggregation sums batch
numerators/denominators, never batch means. Total objective sums positive head
weights times available head means without weight re-normalization.

`sequence_evidence` `[B,1]` averages sequence log densities. `step_evidence`
`[B,T]` averages included time-element log densities: these are different
objectives. Explicitly supply a loss mask to exclude padding from the latter's
denominator. Zero padded evidence, emission missingness and prefix lengths do
not implicitly become loss masks.

Continuous densities can exceed one, so NLL numerators, means and combined
objectives may be finite and negative. Only likelihood semantics permit signed
losses; MSE/MAE/CE, denominators, weights and counters retain nonnegative checks.
Pure result admission verifies captured populations, reductions, quotients and
checkpoint identities without importing Torch.

## Replay and boundary

Every forward checks all output shapes/dtypes/devices/arithmetic. Only explicitly
declared diagnostics with proven `log-probabilities` may contain structural
negative infinity. NaN and positive infinity are invalid; evidence and optimized
outputs remain finite. Metrics/results remain finite JSON. Restored and
standalone predictors carry identical diagnostic rules and no target fields.
Worker requests are `task-v5` or `task-v6`; CLI selects the actual bundle
version. Signed report curves include negative losses within linear bounds.

Runtime likelihood carries exact Family12/Spec11 or Family13/Spec12 temporal-v2
proof. Experiment7 and Run7 admit this task; older versions explicitly refuse it.
Both actual saved and rerun notebooks reproduce Gaussian and full causal MS-VAR
checkpoints/metrics/predictors. See [neural MS-VAR](neural-msvar-contract.md).
Resets, execution/missing-history semantics and further report contracts remain;
no production workbench redesign is implied.

Reproduce with `PYTHONPATH=.:tests python -m unittest tests.test_likelihood_task -v`.
Tests cover real training, independent signed reductions/gradients, worker replay,
native/standalone inference, result admission, curves and archived-version fences.
