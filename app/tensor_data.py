"""Recorded row-major structured regression adapter and small spatial teacher."""
import math


def adapter_for(contract):
    from .core import GraphError
    if len(contract['input_groups']) != 1:
        raise GraphError('The first structured adapter needs exactly one Input.')
    return {'version': 1, 'kind': 'declared-row-major',
            'sample_shape': contract['input_shape'], 'axes': contract['input_axes']}


def check_adapter(data, contract):
    """Old flat datasets remain supported; recorded adapters must not be guessed."""
    from .core import GraphError
    from .evidence import digest
    if 'adapter' in data:
        try:
            matches = digest(data['adapter']) == digest(adapter_for(contract))
        except (TypeError, ValueError, RecursionError):
            matches = False
        if not matches:
            raise GraphError('Structured dataset adapter does not match the input shape/axes. Replace the dataset or restore its declared layout; rebinding columns cannot reorder tensor values.')


def spatial_eligible(contract):
    shape = contract['input_shape']
    return (len(contract['input_groups']) == 1 and len(shape) == 3 and
            contract['input_axes'] in (None, ['C', 'H', 'W']) and shape[1] >= 3 and shape[2] >= 2 and
            len(contract['output_shape']) == 1)


def spatial_teacher(contract, rng):
    """Conv[3,2] -> Tanh -> global spatial mean -> Linear, independently in Python."""
    from .core import GraphError
    if not spatial_eligible(contract):
        raise GraphError('Spatial teacher needs one [C,H,W] Input with H ≥ 3, W ≥ 2 and a vector output. Use explicit adapters for another layout.')
    c, h, w = contract['input_shape']
    no = contract['outputs']
    weights = [[[[rng.uniform(-.6, .6) for _ in range(2)] for _ in range(3)]
                for _ in range(c)] for _ in range(2)]
    bias = [rng.uniform(-.2, .2) for _ in range(2)]
    readout = [[rng.uniform(-1., 1.) for _ in range(2)] for _ in range(no)]
    offset = [rng.uniform(-.2, .2) for _ in range(no)]
    truth = {'kind': 'spatial', 'weight': weights, 'bias': bias, 'readout': readout, 'offset': offset,
             'kernel_size': [3, 2], 'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1],
             'groups': 1, 'equation': 'y = Linear(mean_spatial(tanh(Conv2d(x))))',
             'representability': 'Not guaranteed for arbitrary student architectures.'}
    def target(row):
        pooled = []
        for oc in range(2):
            total = 0.
            for ih in range(h - 2):
                for iw in range(w - 1):
                    value = bias[oc]
                    for ic in range(c):
                        for kh in range(3):
                            for kw in range(2):
                                value += weights[oc][ic][kh][kw] * row[(ic * h + ih + kh) * w + iw + kw]
                    total += math.tanh(value)
            pooled.append(total / ((h - 2) * (w - 1)))
        return [sum(a * v for a, v in zip(coeff, pooled)) + b for coeff, b in zip(readout, offset)]
    return truth, target
