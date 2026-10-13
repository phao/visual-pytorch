"""Internal CPU Gaussian HMM primitive; importing this module does not import Torch.

This is not an application model/operator, MSVAR feature, smoother, loss adapter,
or discovery registration. Callers provide every transition and emission tensor.
The supported execution contract is eager CPU float32 or float64 with autograd.
"""

from dataclasses import dataclass
import math


MAX_WORK_ELEMENTS = 1_000_000
MAX_STEPS = 4096
_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class GaussianFilterResult:
    """Tensor fields in input dtype/device; padded beliefs carry the last belief."""

    filtered_log_probs: object  # [B,T,K]
    final_log_probs: object  # [B,K]
    step_log_evidence: object  # [B,T], exactly zero for padding/missing emissions
    sequence_log_likelihood: object  # [B], sum of step_log_evidence


def _cpu_tensor(torch, value, name, *, dtype=None):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Torch tensor")
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError(f"{name} must be a strided CPU tensor")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")


def _finite(torch, value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite at evaluated entries")


def _safe_logsumexp(torch, value, dim):
    """All -inf groups have value -inf and zero derivatives, not 0/0 gradients."""
    reachable = torch.isfinite(value).any(dim=dim, keepdim=True)
    safe = torch.where(reachable, value, torch.zeros_like(value))
    reduced = torch.logsumexp(safe, dim=dim)
    return torch.where(reachable.squeeze(dim), reduced,
                       torch.full_like(reduced, -math.inf))


def _normalized_logs(torch, value, name, tolerance):
    valid = torch.isfinite(value) | torch.isneginf(value)
    if not bool(valid.all()):
        raise ValueError(f"{name} permits finite log probabilities and -inf only")
    normalizer = _safe_logsumexp(torch, value, -1)
    if not bool((torch.isfinite(normalizer) & (normalizer.abs() <= tolerance)).all()):
        raise ValueError(f"{name} must be normalized (logsumexp within {tolerance} of 0)")


def variances_from_raw(raw_variances, *, positive_floor):
    """Explicit softplus(raw_variances) + positive_floor, separate from density.

    raw_variances must be finite strided CPU float32/64. The caller must declare
    a finite positive Python int/float floor representable as positive in that
    dtype. There is no default floor. Overflow is refused, never clipped.
    """
    import torch

    _cpu_tensor(torch, raw_variances, "raw_variances")
    if raw_variances.dtype not in (torch.float32, torch.float64):
        raise ValueError("raw_variances must be float32 or float64")
    if type(positive_floor) not in (int, float):
        raise TypeError("positive_floor must be an explicit Python int or float")
    try:
        floor = float(positive_floor)
    except OverflowError as exc:
        raise ValueError("positive_floor must be finite and positive") from exc
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("positive_floor must be finite and positive")
    represented = raw_variances.new_tensor(floor)
    if not bool(torch.isfinite(represented) & (represented > 0)):
        raise ValueError("positive_floor must be representable and positive in input dtype")
    if raw_variances.numel() > MAX_WORK_ELEMENTS:
        raise ValueError("raw_variances exceeds the work budget")
    _finite(torch, raw_variances, "raw_variances")
    result = torch.nn.functional.softplus(raw_variances) + represented
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("variance transform exceeds input dtype range")
    return result


def filter_diagonal_gaussian_hmm(
    observations, *, log_initial, log_transition, means, positive_variances,
    lengths, emission_mask=None, max_work_elements=MAX_WORK_ELEMENTS,
):
    """Differentiate a log-space forward filter with fully explicit tensors.

    Shapes: observations [B,T,D], log_initial [B,K], log_transition [B,T,K,K],
    means and positive_variances [B,T,K,D], lengths [B]. All real tensors must
    share CPU float32/64 dtype; lengths is CPU int32/int64 (no float rounding).
    Optional emission_mask is CPU bool [B,T], default True. No broadcasting.

    log_initial describes S_(-1), BEFORE the first transition. Transition entry
    [b,t,j,k] is Pr(S_t=k | S_(t-1)=j) in log units: previous -> next rows.
    Initial beliefs and active transition rows must satisfy |logsumexp| <=
    2e-6 (float32) or 1e-12 (float64), with -inf allowed for structural zeros.
    Inputs are validated, never clipped or implicitly transformed. Each computed
    active belief is normalized to remove floating point reduction error.

    Active times are exactly [0,lengths[b]). Missing whole emissions (False)
    still transition, with exactly zero evidence. Unused observations, means,
    variances and padded transitions may be NaN; they are replaced BEFORE
    densities/reductions. Active emitted parameters, including unreachable
    states, must be finite, with strictly positive variances. Density includes
    -D/2 log(2*pi) and uses variances, not standard deviations or log variances.
    There is no hidden variance floor. No reset, execution, loss or label masks.

    Padded output beliefs carry the last active belief (initial if length=0).
    Empty T/B are supported; neutral scores are zero and initial beliefs remain
    unchanged when length=0. Emissions are centered over reachable states before
    adding prior log weights, preserving beliefs under enormous common tails.
    Finite inputs whose evaluated arithmetic exceeds dtype range are refused
    with FloatingPointError; this is not a guarantee of finite gradients for
    arbitrarily ill-conditioned finite inputs. Structural -inf groups themselves
    have safe zero derivatives. No smoothing or supervised proxy is performed.

    Admission precedes expanded allocations. T <= 4096, and the conservative
    numerical-slot estimate max(B,1) * (max(T,1) * (8*K*K + 16*K*D + 12*K +
    4*D + 8) + 4*K) must fit max_work_elements, an exact Python integer in
    [1,1_000_000]. It may only tighten the hard bound. This bounds tensor work,
    not process RSS, Torch runtime memory or Python autograd-object overhead.
    """
    import torch

    _cpu_tensor(torch, observations, "observations")
    dtype = observations.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("observations must be float32 or float64")
    if observations.ndim != 3:
        raise ValueError("observations must have shape [B,T,D]")
    batch, steps, dimensions = observations.shape
    _cpu_tensor(torch, log_initial, "log_initial", dtype=dtype)
    if log_initial.ndim != 2 or log_initial.shape[0] != batch:
        raise ValueError("log_initial must have shape [B,K]")
    states = log_initial.shape[1]
    if states < 1 or dimensions < 1:
        raise ValueError("K and D must be positive")
    for name, value, shape in (
        ("log_transition", log_transition, (batch, steps, states, states)),
        ("means", means, (batch, steps, states, dimensions)),
        ("positive_variances", positive_variances, (batch, steps, states, dimensions)),
    ):
        _cpu_tensor(torch, value, name, dtype=dtype)
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
    _cpu_tensor(torch, lengths, "lengths")
    if lengths.dtype not in (torch.int32, torch.int64) or tuple(lengths.shape) != (batch,):
        raise ValueError("lengths must have shape [B] and exact int32/int64 dtype")
    if emission_mask is not None:
        _cpu_tensor(torch, emission_mask, "emission_mask", dtype=torch.bool)
        if tuple(emission_mask.shape) != (batch, steps):
            raise ValueError("emission_mask must have shape [B,T]")
    if type(max_work_elements) is not int or not 1 <= max_work_elements <= MAX_WORK_ELEMENTS:
        raise ValueError(f"max_work_elements must be an integer in [1,{MAX_WORK_ELEMENTS}]")
    work = max(batch, 1) * (
        max(steps, 1) * (8 * states * states + 16 * states * dimensions
                         + 12 * states + 4 * dimensions + 8) + 4 * states
    )
    if steps > MAX_STEPS or work > max_work_elements:
        raise ValueError(f"filter work budget exceeded: T={steps}, numerical slots={work}")
    if not bool(((lengths >= 0) & (lengths <= steps)).all()):
        raise ValueError("lengths entries must lie in [0,T]")
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    _normalized_logs(torch, log_initial, "log_initial", tolerance)

    active = torch.arange(steps, device=observations.device)[None, :] < lengths[:, None]
    emitted = active if emission_mask is None else active & emission_mask
    # Sanitization precedes even validation reductions, not just final masking.
    transitions = torch.where(active[:, :, None, None], log_transition,
                              log_transition.new_tensor(-math.log(states)))
    y = torch.where(emitted[:, :, None], observations, torch.zeros_like(observations))
    mu = torch.where(emitted[:, :, None, None], means, torch.zeros_like(means))
    variance = torch.where(emitted[:, :, None, None], positive_variances,
                           torch.ones_like(positive_variances))
    _normalized_logs(torch, transitions, "active log_transition rows", tolerance)
    _finite(torch, y, "observations")
    _finite(torch, mu, "means")
    _finite(torch, variance, "positive_variances")
    if not bool((variance > 0).all()):
        raise ValueError("positive_variances must be strictly positive at emitted entries")

    belief = log_initial
    filtered, evidences = [], []
    for t in range(steps):
        # Inactive rows use a harmless uniform belief so invalid intermediate
        # arithmetic cannot occur while their diagnostic belief is carried.
        before = torch.where(active[:, t, None], belief,
                             belief.new_tensor(-math.log(states)))
        joint = before[:, :, None] + transitions[:, t]
        finite_terms = torch.isfinite(before[:, :, None]) & torch.isfinite(transitions[:, t])
        if bool((finite_terms & ~torch.isfinite(joint)).any()):
            raise FloatingPointError("transition log weights exceed input dtype range")
        prediction = _safe_logsumexp(torch, joint, 1)
        residual = (y[:, t, None, :] - mu[:, t]) / variance[:, t].sqrt()
        density = -0.5 * (_LOG_TWO_PI + variance[:, t].log() + residual.square()).sum(-1)
        if not bool(torch.isfinite(density).all()):
            raise FloatingPointError("Gaussian log density exceeds input dtype range")
        density = torch.where(emitted[:, t, None], density, torch.zeros_like(density))
        reachable = torch.isfinite(prediction)
        offset = torch.where(reachable, density, torch.full_like(density, -math.inf)).amax(-1)
        # Never subtract enormous offsets in unreachable entries either.
        centered = torch.where(reachable, density, offset[:, None]) - offset[:, None]
        if not bool(torch.isfinite(centered).all()):
            raise FloatingPointError("emission log-density difference exceeds input dtype range")
        weights = prediction + centered
        if bool((reachable & ~torch.isfinite(weights)).any()):
            raise FloatingPointError("emission log weights exceed input dtype range")
        normalizer = _safe_logsumexp(torch, weights, -1)
        posterior = weights - normalizer[:, None]
        if bool((reachable & ~torch.isfinite(posterior)).any()):
            raise FloatingPointError("filtered log probabilities exceed input dtype range")
        evidence = torch.where(emitted[:, t], offset + normalizer, torch.zeros_like(offset))
        if not bool(torch.isfinite(evidence).all()):
            raise FloatingPointError("step log evidence exceeds input dtype range")
        belief = torch.where(active[:, t, None], posterior, belief)
        filtered.append(belief)
        evidences.append(evidence)
    filtered_tensor = torch.stack(filtered, 1) if steps else log_initial[:, None, :].expand(batch, 0, states)
    if steps:
        step_tensor = torch.stack(evidences, 1)
    else:
        # Empty slices connect the neutral score to all real inputs with zero
        # derivatives, without reading possibly NaN padding or structural -inf.
        neutral = (observations[:, :0].sum((1, 2)) + log_initial[:, :0].sum(1)
                   + log_transition[:, :0].sum((1, 2, 3)) + means[:, :0].sum((1, 2, 3))
                   + positive_variances[:, :0].sum((1, 2, 3)))
        step_tensor = neutral[:, None].expand(batch, 0)
    sequence = step_tensor.sum(1)
    if not bool(torch.isfinite(sequence).all()):
        raise FloatingPointError("sequence log likelihood exceeds input dtype range")
    return GaussianFilterResult(filtered_tensor, belief, step_tensor, sequence)
