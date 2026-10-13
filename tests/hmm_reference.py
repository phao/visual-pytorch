"""Independent, bounded diagonal-Gaussian HMM oracle (standard library only).

This is a test reference, not an application filter, a fitted MSVAR model,
an autoregressive window builder, a smoother, or a state-label/BCE loss.
Every prefix is evaluated by enumerating its state paths, without a forward
filter recursion. Python floats supply binary64 arithmetic on supported CPython.
"""

from dataclasses import dataclass
from itertools import product
import math


MAX_STATES = 4
MAX_STEPS = 8
MAX_DIMENSIONS = 16
MAX_ENUMERATED_PATHS = 100_000
DEFAULT_PROBABILITY_TOLERANCE = 1e-12
_NEG_INF = -math.inf
_LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class HMMReference:
    """Tuple row t uses only observations through t; it is not smoothed."""

    prefix_log_likelihoods: tuple
    filtered_log_probabilities: tuple
    filtered_probabilities: tuple
    enumerated_paths: int

    @property
    def log_likelihood(self):
        return self.prefix_log_likelihoods[-1] if self.prefix_log_likelihoods else 0.0

    @property
    def negative_log_likelihood(self):
        return -self.log_likelihood


def _sequence(value, name, size=None):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    if size is not None and len(value) != size:
        raise ValueError(f"{name} must have length {size}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside binary64 range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _distribution(value, size, name, tolerance):
    row = tuple(_finite(x, name) for x in _sequence(value, name, size))
    if any(x < 0.0 or x > 1.0 for x in row):
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    if abs(math.fsum(row) - 1.0) > tolerance:
        raise ValueError(f"{name} probabilities must sum to 1 within {tolerance}")
    # Accepted input bytes/values are never repaired by dividing by the sum.
    return row


def _matrix(value, states, name, tolerance):
    return tuple(
        _distribution(row, states, f"{name}[{j}]", tolerance)
        for j, row in enumerate(_sequence(value, name, states))
    )


def _vectors(value, states, dimensions, name):
    return tuple(
        tuple(_finite(x, f"{name}[{k}]") for x in _sequence(row, name, dimensions))
        for k, row in enumerate(_sequence(value, name, states))
    )


def _log_probability(value):
    return math.log(value) if value else _NEG_INF


def _logsumexp(values):
    values = tuple(values)
    maximum = max(values, default=_NEG_INF)
    if maximum == _NEG_INF:
        return _NEG_INF
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))


def diagonal_gaussian_log_density(observation, mean, variances):
    """Full log density, with variances (not standard deviations or logvars).

    Inputs are equal-length nonempty vectors, bounded by MAX_DIMENSIONS.
    There is no variance floor or clipping. Nonfinite active values and positive
    variances outside representable density arithmetic are explicitly refused.
    """
    variances = _sequence(variances, "variances")
    dimensions = len(variances)
    if not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError(f"dimension must be between 1 and {MAX_DIMENSIONS}")
    observation = _sequence(observation, "observation", dimensions)
    mean = _sequence(mean, "mean", dimensions)
    terms = []
    for y, mu, variance in zip(observation, mean, variances):
        y = _finite(y, "observation")
        mu = _finite(mu, "mean")
        variance = _finite(variance, "variance")
        if variance <= 0.0:
            raise ValueError("variance must be strictly positive")
        # Divide before squaring: squaring a tiny residual first can underflow.
        residual = (y - mu) / math.sqrt(variance)
        term = _LOG_TWO_PI + math.log(variance) + residual * residual
        if not math.isfinite(term):
            raise FloatingPointError("Gaussian log density exceeds binary64 range")
        terms.append(term)
    try:
        result = -0.5 * math.fsum(terms)
    except OverflowError as exc:
        raise FloatingPointError("Gaussian log density exceeds binary64 range") from exc
    if not math.isfinite(result):
        raise FloatingPointError("Gaussian log density exceeds binary64 range")
    return result


def enumerate_gaussian_hmm(
    observations,
    *,
    initial_belief,
    variances,
    transition=None,
    transitions_by_time=None,
    means=None,
    means_by_time=None,
    length=None,
    emission_mask=None,
    probability_tolerance=DEFAULT_PROBABILITY_TOLERANCE,
    max_paths=MAX_ENUMERATED_PATHS,
):
    """Enumerate all paths for each active prefix, returning filtered marginals.

    Shapes: observations [T,D], initial_belief [K], variances [K,D]. Supply
    exactly one of transition [K,K] / transitions_by_time [T,K,K], and one of
    means [K,D] / means_by_time [T,K,D]. K<=4, stored T<=8, D<=16.
    P_t[j][k] is Pr(S_t=k | S_(t-1)=j). initial_belief describes S_(-1),
    BEFORE the first transition; the first emitting state has belief pi @ P_0.
    Time-indexed parameters are caller-supplied; no neural/VAR means are built.

    length defaults to T and selects [0,length). Padded observations and
    time-indexed parameter entries outside it are never inspected. emission_mask
    is a T-long boolean vector (default all True). An active False entry is a
    missing whole emission: transition still happens, emission log density is 0,
    and that observation/time-indexed mean is not inspected. True entries must
    be finite, including parameters of unreachable states. No coordinate masks,
    internal execution gaps, state carry, resets, or implicit missing-value rules.
    Static parameters are validated even for length=0. The empty result has
    log likelihood 0 and no posterior rows; missing-only rows still have beliefs.

    Initial belief and each active transition row must be distributions, with
    absolute sum error <= probability_tolerance (default 1e-12; at most 1e-8).
    They are not renormalized. Structural zeros yield -inf log probabilities.
    Finite values whose arithmetic overflows raise FloatingPointError rather
    than being misclassified as impossible states. Posterior probabilities can
    underflow to 0; their finite log probabilities remain available.

    Admission happens before path enumeration. max_paths is a positive integer
    <=100000, limiting sum(K**n for n in 1..length), including zero-weight paths.
    The oracle is single-threaded and has exponential cost by design.
    """
    observations = _sequence(observations, "observations")
    stored_steps = len(observations)
    if stored_steps > MAX_STEPS:
        raise ValueError(f"stored T must be <= {MAX_STEPS}")
    if length is None:
        length = stored_steps
    if type(length) is not int or not 0 <= length <= stored_steps:
        raise ValueError("length must be an integer in [0,T]")
    initial_belief = _sequence(initial_belief, "initial_belief")
    states = len(initial_belief)
    if not 1 <= states <= MAX_STATES:
        raise ValueError(f"K must be between 1 and {MAX_STATES}")
    if type(max_paths) is not int or not 1 <= max_paths <= MAX_ENUMERATED_PATHS:
        raise ValueError(f"max_paths must be an integer in [1,{MAX_ENUMERATED_PATHS}]")
    path_count = sum(states ** n for n in range(1, length + 1))
    if path_count > max_paths:
        raise ValueError(f"enumeration budget exceeded: {path_count} paths > {max_paths}")
    tolerance = _finite(probability_tolerance, "probability_tolerance")
    if not 0.0 <= tolerance <= 1e-8:
        raise ValueError("probability_tolerance must lie in [0,1e-8]")
    initial = _distribution(initial_belief, states, "initial_belief", tolerance)
    if (transition is None) == (transitions_by_time is None):
        raise ValueError("supply exactly one transition representation")
    if (means is None) == (means_by_time is None):
        raise ValueError("supply exactly one means representation")
    variances = _sequence(variances, "variances", states)
    dimensions = len(_sequence(variances[0], "variances[0]"))
    if not 1 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError(f"dimension must be between 1 and {MAX_DIMENSIONS}")
    variances = _vectors(variances, states, dimensions, "variances")
    if any(x <= 0.0 for row in variances for x in row):
        raise ValueError("variances must be strictly positive")
    if transition is not None:
        constant = _matrix(transition, states, "transition", tolerance)
        matrices = (constant,) * length
    else:
        _sequence(transitions_by_time, "transitions_by_time", stored_steps)
        matrices = tuple(
            _matrix(transitions_by_time[t], states, f"transitions_by_time[{t}]", tolerance)
            for t in range(length)
        )
    if emission_mask is None:
        emission_mask = (True,) * stored_steps
    _sequence(emission_mask, "emission_mask", stored_steps)
    if any(type(flag) is not bool for flag in emission_mask):
        raise ValueError("emission_mask entries must be booleans")
    if means is not None:
        constant_means = _vectors(means, states, dimensions, "means")
    else:
        _sequence(means_by_time, "means_by_time", stored_steps)
    emissions = []
    for t in range(length):
        if not emission_mask[t]:
            emissions.append((0.0,) * states)
            continue
        step_means = constant_means if means is not None else _vectors(
            means_by_time[t], states, dimensions, f"means_by_time[{t}]"
        )
        emissions.append(tuple(
            diagonal_gaussian_log_density(observations[t], step_means[k], variances[k])
            for k in range(states)
        ))
    if not length:
        return HMMReference((), (), (), 0)
    log_matrices = tuple(
        tuple(tuple(_log_probability(x) for x in row) for row in matrix)
        for matrix in matrices
    )
    # Marginalize S_(-1) once, then explicitly enumerate S_0..S_t per prefix.
    first_state_logs = tuple(
        _logsumexp(_log_probability(initial[j]) + log_matrices[0][j][k]
                   for j in range(states))
        for k in range(states)
    )
    # Remove the common emission offset before adding small path-prior terms.
    # This preserves, e.g., a nonuniform belief under identical extreme tails.
    emission_offsets = tuple(max(row) for row in emissions)
    centered_emissions = tuple(
        tuple(x - offset for x in row)
        for row, offset in zip(emissions, emission_offsets)
    )
    if any(not math.isfinite(x) for row in centered_emissions for x in row):
        raise FloatingPointError("emission log-density difference exceeds binary64 range")
    likelihoods, log_posteriors, posteriors = [], [], []
    for prefix_length in range(1, length + 1):
        groups = [[] for _ in range(states)]
        for path in product(range(states), repeat=prefix_length):
            terms = [first_state_logs[path[0]]]
            terms.extend(log_matrices[t][path[t - 1]][path[t]]
                         for t in range(1, prefix_length))
            if _NEG_INF in terms:
                continue
            terms.extend(centered_emissions[t][path[t]] for t in range(prefix_length))
            try:
                score = math.fsum(terms)
            except OverflowError as exc:
                raise FloatingPointError("path log weight exceeds binary64 range") from exc
            if not math.isfinite(score):
                raise FloatingPointError("path log weight exceeds binary64 range")
            groups[path[-1]].append(score)
        maximum = max((x for group in groups for x in group), default=_NEG_INF)
        if maximum == _NEG_INF:
            raise FloatingPointError("no reachable path; posterior is undefined")
        # Normalize centered weights so huge common offsets do not destroy
        # posterior normalization through subtracting two large log likelihoods.
        centered = tuple(_logsumexp(x - maximum for x in group) for group in groups)
        normalizer = _logsumexp(centered)
        log_posterior = tuple(x - normalizer for x in centered)
        try:
            likelihood = math.fsum((*emission_offsets[:prefix_length], maximum, normalizer))
        except OverflowError as exc:
            raise FloatingPointError("prefix log likelihood exceeds binary64 range") from exc
        if not math.isfinite(likelihood):
            raise FloatingPointError("prefix log likelihood exceeds binary64 range")
        likelihoods.append(likelihood)
        log_posteriors.append(log_posterior)
        posteriors.append(tuple(math.exp(x) for x in log_posterior))
    return HMMReference(tuple(likelihoods), tuple(log_posteriors), tuple(posteriors), path_count)
