# A complete padded CNN → recurrent experiment

This example gives you a small network that can actually be drawn, retained,
trained and exported. Each observation is a variable-length sequence of 4×5
monochrome images. A convolutional neural network (CNN) extracts frame features;
a recurrent cell combines them with the supplied previous state. Execution
lengths, step masks, reset flags and supervised targets are separate objects.

The generator is deliberately small:64 observations, four synthetic subjects,
eight observation timestamps and two replicates per subject/time. It does not
import Torch or train. The target generator independently evaluates the stated
CNN/state equations with Python arithmetic, so the data is not produced by
calling the application's forward implementation.

## Create and inspect

From the product repository, create a NEW workspace:

```sh
python examples/cnn_recurrent_experiment.py NEW.vtw --split temporal
```

Use `--split group` for the separate subject-held-out experiment. The command
prints exact family/specialization/experiment references, materializes both
the root drawing and referenced cell, and refuses to overwrite an existing file.
It captures a60-update CPU/Adam protocol; `--updates` changes that explicit bound.
`--time` chooses the explicit padded dataset length (default7, bounded1–32 for
this generator). Those example bounds are not global architecture limits.

Open NEW.vtw in Visual PyTorch. The root Recurrent box references the CNN cell;
double-click the reference to inspect its actual Conv2d, pooling, flattening,
Concat, Linear and activation layers. The scientific panel exposes the exact
experiment and a compatible candidate. Preview does not train. Run that retained
candidate when you want the actual fit. CLI uses the same services:

```sh
python vpw_cli.py NEW.vtw preview-experiment EXPERIMENT_ID EXPERIMENT_SHA
python vpw_cli.py NEW.vtw run-candidate --experiment EXPERIMENT_ID EXPERIMENT_SHA \
  --coordinates COORDINATES.json --run-id YOUR_NEW_RUN_ID --name 'CNN recurrent trial'
python vpw_cli.py NEW.vtw export-run-trained RUN_ID RUN_SHA
```

Use the exact references printed by the generator. COORDINATES.json is the
candidate's `coordinates` object returned by preview, not an invented model/data
selection. Saved predictor export does not retrain or rewrite the workspace.
Result-report versus rerun notebooks retain their existing explicit behavior.

## What the drawing computes

Let $X_t\in\mathbb R^{1\times4\times5}$ be one image and $h_0\in\mathbb R^2$
the supplied initial computational state. The batch axis is implicit here; each
row supplies its own sequence/state/controls. The CNN has $K$ output channels
(default2), one1×1 cross-correlation per channel, and a learnable bias. Global
average pooling then leaves $K$ frame features:

$$
z_{t,c}=w_c\left(\frac1{20}\sum_{i=1}^{4}\sum_{j=1}^{5}X_{t,1,i,j}\right)+a_c,
\qquad c=1,\ldots,K.
$$

This is exactly Conv2d(kernel1, bias on) → AdaptiveAvgPool2d(1,1) → Flatten.
Writing the pixel mean explicitly makes it possible to check the whole CNN
without reusing the application's convolution/pooling implementation.

The recurrent update uses $W\in\mathbb R^{2\times(K+2)}$, $b\in\mathbb R^2$
and an elementwise activation $\phi$ (default tanh; the family also admits ReLU).
Let $A_t$ mean that $t$ lies inside the explicit valid-length prefix and its step
mask is true. Let $R_t$ be the explicit reset flag. The state used by an active
step is $q_t=h_0$ when $R_t$ is true, and $q_t=h_{t-1}$ otherwise. Then:

$$
h_t=\begin{cases}
\phi\bigl(W[z_t;q_t]+b\bigr),&A_t,\\
h_{t-1},&\text{otherwise}.
\end{cases}
$$

The Concat order is frame features first, previous/reset state second. A reset
on an inactive step is ignored. The returned trajectory contains $h_t$ on active
steps and an explicit zero vector on inactive steps; the final output is $h_T$.
Consequently a zero-length observation returns exactly its supplied initial
state. There is no hidden state carry between observations or inference calls.

These are deterministic computational states, not latent statistical regimes.
Given weights, inputs and controls, the forward is determined. The trainable
parameter count is $K+K+2(K+2)+2=4K+6$; default $K=2$ gives14 parameters in four
tensors. The same CNN/cell weights are used at every time step, not copied and
optimized independently for each frame.

## Constructor versus actual time

Family8 has a root baseline-time symbol (default5), a CNN-channel symbol and an
activation choice. Required child constructor bindings explicitly propagate
channels/activation into the referenced cell. The retained drawing's T=5 is a
constructor baseline; the dataset uses an explicitly padded T=7. Runtime-v2
declares only leading T for frames, step mask and reset. Frame C/H/W and the
two-component state remain fixed. Changing actual T never recreates weights.

The same family can specialize to other channel counts/activation choices, but
changing a learned width is a new constructor, not a runtime shape override.
The family's finite integer domains and required child bindings are checked
before allocation. No general symbolic algebra or implicit dimension inference
is required to use the concrete example.

## Padding, execution controls and supervision

Raw field `frames` has shape `[-1,1,4,5]`. TaskSpec1 explicitly pads it to7 with
zero image frames, and a separate derived-length input measures the raw sequence.
Other inputs bind the supplied initial state, exact binary mask and exact binary
reset arrays. Model inputs have shapes `[B,7,1,4,5]`, `[B,2]`, `[B,1]`, `[B,7]`
and `[B,7]`. Outputs are trajectory `[B,7,2]` and final state `[B,2]`.

The loss masks are explicit dataset fields too. The example generator chooses
trajectory supervision on active steps and final-state supervision when at least
one step is active. That choice belongs to this dataset/task: the model does not
infer supervision from its execution lengths or masks. Training sums trajectory
mean-squared error (MSE) plus0.2 times final-state MSE, each using its declared
effective supervised elements. Entirely unsupervised batches skip optimization;
an absent denominator is unavailable, not silently a perfect zero loss.

Saved inference reuses the fitted pad/length adapters. Raw sequences longer than
the captured padding length refuse rather than truncate. A bare runtime-family
call can use another admissible T, but that does not silently change a captured
task's data pipeline or supervised target shapes.

## Two different scientific split questions

`temporal` sorts distinct scalar observation timestamps and keeps all equal-time
subjects/replicates together. Four timestamps are training, two validation and
two test:32/16/16 rows. Roles are globally chronological. It does not promise
subject-disjoint generalization.

`group` holds out whole subjects using the explicit group seed. Two subjects are
training, one validation and one test:also32/16/16 rows. Every role includes all
eight times, so this is not a chronological split. It asks about held-out subjects.
Neither policy is chronological splitting independently within each subject.
That additional interpretation requires a separately declared policy, not a
reinterpretation of the existing group/time methods.

The time field refers to whole observation blocks. It does not by itself certify
that raw windows from a real data source cannot overlap. Declare appropriate
availability times/grouping/gaps when adapting the example to real sequences.
These synthetic fits and one initialization are demonstrations, not confidence
intervals or evidence of general deployment performance. The distinction between
unseen-group evaluation and ordered-time evaluation is also explained in the
[scikit-learn grouped-data guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [TimeSeriesSplit reference](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
The application does not use scikit-learn or claim its single holdout is that
library's rolling cross-validation algorithm.

## What the verification establishes

Independent tests check the entire image→feature→state path, both outputs, frame/
initial-state/parameter gradients, six scoped channel/activation constructors,
multiple actual times, inactive/NaN padding, exact reset semantics and14 unique
default parameters. Real local/isolated-worker fits match checkpoint/history/
metrics; final train loss falls below one-fifth of the first post-update measured
loss in both split demonstrations. CLI save/export, fresh application-free
prediction and genuine saved/rerun kernels preserve exact predictions/checkpoints.

The generator has known teacher parameters, but fitting does not prove their
unique recovery. Scaling a CNN feature and inversely scaling its corresponding
Linear column preserves the same forward, for example. We verify computation,
gradients and predictive fit, not identifiability of this factorization.
Source:examples/cnn_recurrent_experiment.py; tests:test_cnn_recurrent_example.py
and test_cnn_recurrent_execution.py. Current acceptance/source/integration and
remaining original D/E gates are recorded separately in development/campaign
evidence; this example alone is not a full-stage completion certificate.
