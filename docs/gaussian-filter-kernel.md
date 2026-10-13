# Internal diagonal Gaussian filtering primitive: not app admission

`app/gaussian_filter.py` is an independently tested numerical foundation on the
MS-VAR task branch. It is not a registered graph operator, model owner, task/loss,
GUI option, notebook route or CLI template. Discovery correctly continues to
report latent likelihood unsupported. The supervised regime demo remains a
different, approximate task. No full Neural MS-VAR completion is claimed here.

Importing the module loads only the standard library; calling it imports PyTorch.
Its explicit eager CPU float32/float64 inputs are observations `[B,T,D]`, normalized
log initial beliefs `[B,K]`, normalized log transition rows `[B,T,K,K]`, means and
positive variances `[B,T,K,D]`, exact integer lengths `[B]`, and optional boolean
emission masks `[B,T]`. There is no implicit broadcasting or target/state label.

Initial belief precedes the first transition; `P[j,k]` means previous j→current k.
Missing whole emissions still transition, contributing exactly zero evidence.
Padding does not transition and carries the previous belief. Empty sequences
have neutral likelihood and preserve the initial belief. Inactive values may
be NaN but are sanitized before arithmetic; active emitted entries must be finite,
including parameters of unreachable states. Variances are strictly positive.

Return fields are filtered log probabilities, final log belief, per-step log
evidence and sequence log-likelihood. Full diagonal Gaussian normalization is
included; NLL may be negative. Structural −∞ probability groups have safe zero
derivatives, and reachable-emission centering preserves beliefs under extreme
common tails. This is filtering, not smoothing. Overflow refuses, not clips.
The separate `variances_from_raw(..., positive_floor=...)` helper performs
softplus plus a declared representable positive floor; density itself has none.

Admission caps T4096 and a conservative numerical-slot estimate at one million.
This is not a process-RSS or complete autograd-object memory bound. Integrating
the component into constrained execution must account for its actual buffers,
tape, sequence work and parameter owners, not reuse an MLP-only estimate.

Main independently reviewed the worker source and tests and ran56 checks0.759s;
post-integration91 focused checks0.446s. They cover every K2 prefix through T8
against independent path enumeration, asymmetric/time-dependent transitions,
full density constants, Jacobian-adjusted time-varying variance cases, state
permutation, float32/reference comparison, invalid shapes/dtypes/normalization,
zero lengths, missingness, NaN padding and actual float64 gradcheck of observations,
initial/transition logits, means and raw variances. These are bounded primitive
checks, not a complete model fit or new-schema/native/export conformance gate.

Next close stable tensor/constant sources and representation/shape contracts,
target-free signed likelihood objective/reduction, masks/initial law and causal
lag construction, native/generated/worker/checkpoint/notebook routes, then
K5,p3,D3 neural transitions/affine VAR emissions and the full report CLI contracts.
Architecture/Dataset/Loss/Optimizer remain separate workbench TYPES; do not replace
the Experiment UI or install a generic shared canvas under this kernel checkpoint.
