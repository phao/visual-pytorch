# Inspectable architecture construction recipes

`app.construction_recipes` is a pure authoring service. It returns native graph
documents, with no workspace access, execution, datasets, labels, tasks or
training configuration. Importing and building recipes require only Python's
standard library and the application's pure graph services.

- `catalogue()` returns independent, alphabetically ordered JSON declarations.
- `show(name)` returns a declaration with classification, limitations, a closed
  option schema, visible defaults, bounds, and model ID suffixes in edit order.
- `build(name, *, model_id, options=None)` returns an explicitly ordered list of
  `{"op": "create", "document": ...}` edits. Dependencies precede consumers;
  the root ID is exactly `model_id`. Every document passes `check_document` and
  `validate_model` against the preceding dependency closure before return.

These edits use the existing ordinary creation format. The caller owns collision
checks, read-only planning, transactional admission, and application to an
explicitly selected workspace. The service does not inspect or overwrite existing
work. CLI and GUI integration are separate work.

## Catalogue

| Name | Classification | Options and defaults |
| --- | --- | --- |
| `branching-mlp` | Exact deterministic model | `left=2`, `right=3`, `hidden=8`, `outputs=1` |
| `shared-encoders` | Parameter component | `D=3`, `hidden=8` |
| `recurrent-cell` | Exact deterministic model | `D=3`, `hidden=8`, `T=12`, `mask=false`, `reset=false` |
| `finite-window-regime-estimator` | Approximate supervised estimator | `window=4`, `D=3`, `K=5`, `hidden=16` |
| `neural-msvar` | Exact conditional probabilistic model | `K=5`, `p=3`, `D=3`, `hidden=32`, `T=12`, `controlled=false` |

The branching MLP applies a separate Linear/Tanh encoder to each independent
input, concatenates left then right, and applies a linear readout. The shared
encoder component has the same ordered embeddings, but its second Linear block
declares `share_with="left_encoder"`. Native validation proves the shared layer
identity and counts its parameters once; matching initial values would not establish
sharing.

The recurrence creates `model_id + "/cell"` first. Its transition maps the
concatenation of `sample` and `state` through Linear/Tanh to `next_state`. The
parent takes explicit `sequence`, `initial_state`, and valid prefix `lengths`,
returning the state sequence and final state. Prefix lengths use
`exact-int64-v1`. Optional `mask` and `reset` Inputs use `exact-bool-v1`: mask skips
steps, while reset restores the supplied initial state before an active step.
No state values, lengths or masks are synthesized.

The finite-window estimator flattens an explicitly supplied `[window,D]` tensor,
then applies Linear/Tanh/Linear/Softmax to produce a K-way supervised estimate.
Its name and declaration explicitly identify an approximation requiring genuine
external regime targets. It is not an HMM filter or a sequence likelihood, and
the service creates neither targets nor a dataset. The caller controls how the
observation window is obtained and which target it represents.

Neural MS-VAR reuses `app.neural_msvar.drawing` or `controlled_drawing` directly.
It includes a causal lag window, nonlinear transition network, state-wise affine
VAR means, initial state law, learned diagonal variances and GaussianFilter
evidence/filter outputs. Native validation recomputes all shapes and counts for
the requested K, p, D, hidden width and T. The parameter count is

\[
(pD+1)h + (h+1)K^2 + (pD+1)KD + K + KD.
\]

The ordinary variant uses prefix lengths and softplus variances with a 0.0001
positive floor. `controlled=true` selects literal learned logvariances transformed
by exp with zero floor and explicit execution/reset/emission masks and reset
histories. Exactness refers to filtering and likelihood under this conditional
model, not missing-history marginalization, identification or stationarity.
Autoregressive order p must be at least one. Setting p=0 raises an explanatory
error directing callers to the separately available exact p=0 HMM; this recipe
does not rename that model as an autoregressive construction.

## Strict options and allocation bounds

Omitted options use the defaults disclosed by `show`. Unknown keys, non-object
options, numeric strings, floats, booleans in integer slots and integers in boolean
slots raise `GraphError`. Integer dimensions lie in `[1,4096]`; booleans must be
actual JSON booleans. The explicit model ID must be a nonempty string of at most
200 characters without ASCII control characters.

Before creating graph documents or calling the MS-VAR constructors, each recipe
bounds parameter products, unroll/working tensor elements and block counts.
Limits are 1,000,000 parameters, 1,000,000 work elements, and 64 blocks per
document. MS-VAR's conservative work bound includes filter/lag work and controlled
reset histories. Counts and activation work are checked again after native graph
validation. These are authoring bounds, not runtime admission or memory guarantees.

Declarations, options and returned edits are independent JSON values. Repeated
builds with the same explicit ID/options return identical documents. No executable
expression, user code, external file or numerical runtime is accepted.

## Focused validation

`tests/test_construction_recipes.py` checks every native graph, creation reference
order, actual sharing, prefix adapters, option types and product bounds, MS-VAR
shape/count changes, approximation disclosure, JSON determinism and fresh no-site
imports. It performs no numerical fits or GUI operations.

```sh
taskset -c 4,6 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -B -m unittest discover -s tests -p test_construction_recipes.py -v
```
