"""Allocation-free CPU run estimates and Linux available-RAM observation.

This is an admission heuristic, not an allocator limit or an OOM guarantee.
No Torch import, tensor construction, cgroup write, or swap accounting occurs.
"""
import math
from pathlib import Path

DEFAULT_MEMORY_LIMIT_PERCENT = 10.0
MIB = 1024 ** 2


def memory_percent(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 100:
        raise ValueError('Constrained RAM percentage must be greater than 0 and at most 100.')
    return float(value)


def available_memory(meminfo='/proc/meminfo', membership='/proc/self/cgroup', root='/sys/fs/cgroup'):
    """MemAvailable, capped by observable cgroup-v2 ancestor headroom (bytes)."""
    try:
        line = next(line for line in Path(meminfo).read_text().splitlines() if line.startswith('MemAvailable:'))
        fields = line.split()
        if len(fields) != 3 or fields[2] != 'kB':
            return None
        available = int(fields[1]) * 1024
        if available < 0:
            return None
    except (OSError, ValueError, StopIteration):
        return None
    try:
        entry = next(line.split(':', 2)[2] for line in Path(membership).read_text().splitlines() if line.startswith('0::'))
        base = Path(root).resolve()
        current = (base / entry.lstrip('/')).resolve()
        # A namespace may hide ancestors. Never escape the mounted cgroup tree.
        if not current.is_relative_to(base):
            current = base
        while True:
            maximum = current / 'memory.max'
            if maximum.exists():
                limit = maximum.read_text().strip()
                if limit != 'max':
                    used = int((current / 'memory.current').read_text().strip())
                    available = min(available, max(0, int(limit) - used))
            if current == base:
                break
            current = current.parent
    except StopIteration:
        pass  # Not a unified cgroup-v2 hierarchy.
    except (OSError, ValueError):
        return None  # A known limit could not be inspected: fail closed.
    return available


def estimate_memory(contract, *, samples=None, training=False, kind='auto'):
    """Conservative, versioned estimate for the current float32 CPU implementation.

    Repeated reference occurrences are already expanded in the contract statistics.
    Counting view/alias outputs too deliberately overestimates activation storage.
    Python/JSON evidence and data-copy headroom are included, not just tensor bytes.
    """
    p, ni, no = contract['parameters'], contract['inputs'], contract['outputs']
    batch = max(1, int(samples * .8)) if training and samples is not None else 1
    operation = 'training' if training else 'data' if samples is not None else 'model'
    components = {'runtime_headroom': (256 if training else 8 if samples is not None else 64) * MIB}
    if contract.get('buffer_elements', 0):
        components['literal_buffer_storage'] = 8 * contract['buffer_elements']
    if training:
        # 24 B/parameter: weights, gradients, Adam moments and scratch;
        # 104 B/parameter: results/state-list/JSON/host copies and serialization.
        components['parameters_optimizer_evidence'] = 128 * p
        components['activations_backward'] = 16 * batch * contract['activation_elements']
        components['convolution_workspace'] = 32 * batch * contract.get('convolution_workspace_elements', 0)
        components['dataset_and_copies'] = 128 * (samples or 0) * (ni + no)
        components['affine_report'] = 128 * (ni * no + no) if contract['matrix_affine'] else 0
    elif samples is not None:
        affine = contract['affine'] if kind == 'auto' else kind == 'affine'
        components['dataset_and_copies'] = 128 * samples * (ni + no)
        components['synthetic_teacher'] = 128 * (ni * no + no if affine else 4 * ni + 5 * no + 4)
    else:
        components['parameters_construction'] = 8 * p
        components['activations_forward'] = 8 * contract['activation_elements']
        components['convolution_workspace'] = 8 * contract.get('convolution_workspace_elements', 0)
    return {'version': 2, 'operation': operation, 'batch': batch,
            'parameter_bytes': 4 * p, 'components': components,
            'estimated_bytes': sum(components.values())}


def format_bytes(value):
    if value is None:
        return 'unknown'
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'):
        if value < 1024 or unit == 'PiB':
            return f'{value:.2f} {unit}'
        value /= 1024
