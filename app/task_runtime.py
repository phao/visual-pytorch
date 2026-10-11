"""Numerical-worker task adapters and additive masked reductions.

Torch is imported only by tensor functions. A mean of batch means is never used
to summarize a population. Accumulator records stay ordinary finite JSON.
"""
import math

from .core import GraphError


def tensor_batch(prepared, task, role, indices):
    """Explicit float32 inputs/targets, int64 class indices and stable port IDs."""
    import torch
    if role not in prepared['populations']:
        raise GraphError('Unknown task population.')
    pop = prepared['populations'][role]
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(pop['observation_ids']) for i in indices):
        raise GraphError('Batch selection must be a list of valid population indices.')
    contract = prepared['contract']; n = len(indices)
    outputs = {g['id']: g for g in contract['output_groups']}
    def select(column, dtype, shape):
        return torch.tensor([column[i] for i in indices], dtype=dtype, device='cpu').reshape(n, *shape)
    inputs = {g['id']: select(pop['inputs'][g['id']], torch.float32, g['shape']) for g in contract['input_groups']}
    targets, masks, weights = {}, {}, {}
    for head in task['heads']:
        hid = head['id']; shape = outputs[head['port']]['shape']
        if head['loss'] == 'cross-entropy':
            shape = shape[1:]
        targets[hid] = select(pop['targets'][hid], torch.int64 if head['loss'] == 'cross-entropy' else torch.float32, shape)
        # Per-record scalar masks/weights are intentionally [B], not [B,1].
        # The reduction broadcasts these only across that record's elements.
        masks[hid] = torch.tensor([pop['masks'][hid][i] for i in indices], dtype=torch.float32, device='cpu')
        weights[hid] = torch.tensor([pop['sample_weights'][hid][i] for i in indices], dtype=torch.float32, device='cpu')
    return {'observation_ids': [pop['observation_ids'][i] for i in indices],
            'inputs': inputs, 'targets': targets, 'masks': masks, 'sample_weights': weights}


def forward_ports(model, inputs, contract):
    """Bridge stable IDs to the captured model interface; never derive from labels."""
    import torch
    groups = contract['input_groups']
    if set(inputs) != {g['id'] for g in groups}:
        raise GraphError('Supply exactly the stable model input port IDs.')
    result = model({g['name']: inputs[g['id']] for g in groups})
    groups = contract['output_groups']
    if len(groups) == 1:
        if not isinstance(result, torch.Tensor):
            raise GraphError('The captured single-output contract requires a tensor.')
        return {groups[0]['id']: result}
    if not isinstance(result, dict) or set(result) != {g['name'] for g in groups}:
        raise GraphError('Returned outputs do not match the captured named contract.')
    return {g['id']: result[g['name']] for g in groups}


def _population_weight(value, loss, kind):
    import torch
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.requires_grad or
        value.dtype not in (torch.float32, torch.float64, torch.int64, torch.bool)):
        raise GraphError(kind + ' must be a fixed CPU tensor.')
    if list(value.shape) == [loss.shape[0]]:
        value = value.reshape(loss.shape[0], *([1] * (loss.ndim - 1)))
    elif value.shape != loss.shape:
        raise GraphError(kind + ' must be per-record scalar or exactly loss-shaped; implicit cross-record broadcasting is forbidden.')
    if not bool(torch.isfinite(value).all()):
        raise GraphError(kind + ' must be finite.')
    if kind == 'Mask':
        if not bool(((value == 0) | (value == 1)).all()):
            raise GraphError('Mask must contain exactly 0/1 values.')
    elif not bool((value >= 0).all()):
        raise GraphError('Sample weights must be nonnegative.')
    return value.to(dtype=torch.float64).expand_as(loss)


def head_terms(prediction, target, mask, sample_weight, kind):
    """Numerator/denominator and differentiable mean for one concrete head.

    MSE/MAE loss elements are scalar tensor components; cross-entropy elements
    are records or spatial sites, with class axis 1. Weights multiply elements,
    not heads. Float64 sums reduce accumulation error; autograd still returns
    gradients in the prediction dtype. Empty weighted populations are explicit.
    """
    import torch
    from torch.nn import functional as F
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor) or
        prediction.device.type != 'cpu' or target.device.type != 'cpu' or prediction.ndim < 1 or
        prediction.dtype not in (torch.float32, torch.float64) or target.requires_grad or
        not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all())):
        raise GraphError('Loss needs finite CPU predictions and fixed targets.')
    if kind in ('mse', 'mae'):
        if target.shape != prediction.shape or target.dtype != prediction.dtype:
            raise GraphError('Regression target must exactly match prediction shape/dtype; broadcasting is forbidden.')
        diff = prediction - target
        elements = diff.square() if kind == 'mse' else diff.abs()
    elif kind == 'cross-entropy':
        if (prediction.ndim < 2 or prediction.shape[1] < 2 or target.dtype != torch.int64 or
            list(target.shape) != [prediction.shape[0], *prediction.shape[2:]] or
            not bool(((target >= 0) & (target < prediction.shape[1])).all())):
            raise GraphError('Cross-entropy needs logits [B,C,...], C≥2, and fixed-vocabulary int64 targets [B,...].')
        elements = F.cross_entropy(prediction, target, reduction='none')
    else:
        raise GraphError('Unsupported loss kind.')
    if not bool(torch.isfinite(elements).all()):
        raise GraphError('Loss arithmetic produced nonfinite elements.')
    effective = _population_weight(mask, elements, 'Mask') * _population_weight(sample_weight, elements, 'Sample weights')
    numerator = (elements.to(torch.float64) * effective).sum()
    denominator = effective.sum()
    if not bool(torch.isfinite(numerator)) or not bool(torch.isfinite(denominator)):
        raise GraphError('Weighted reduction is nonfinite.')
    positive = bool(denominator > 0)
    result = {'kind': kind, 'numerator': numerator, 'denominator': denominator,
              'mean': numerator / denominator if positive else None,
              'records': prediction.shape[0],
              'elements': elements.numel(), 'included_elements': int((effective > 0).sum()),
              'status': 'available' if positive else 'unavailable'}
    if kind == 'cross-entropy':
        result['correct_numerator'] = ((prediction.argmax(dim=1) == target).to(torch.float64) * effective).sum()
    return result


class MetricAccumulator:
    """Additive evidence for ONE fixed head, checkpoint and population.

    It cannot merge curves from different checkpoints/roles. State is local to
    evaluation; this is not an optimizer, sampler or a mutable run record.
    """
    def __init__(self, kind, checkpoint, population):
        if kind not in ('mse', 'mae', 'cross-entropy') or not isinstance(checkpoint, str) or not checkpoint or not isinstance(population, str) or not population:
            raise GraphError('Metric needs loss kind, checkpoint ID and population ID.')
        self.kind = kind; self.checkpoint = checkpoint; self.population = population
        self.numerators = []; self.denominators = []; self.correct = []
        self.elements = 0; self.included = 0; self.observations = []

    def add(self, terms, observation_ids, *, checkpoint, population):
        if terms['kind'] != self.kind or checkpoint != self.checkpoint or population != self.population:
            raise GraphError('Cannot combine different losses, checkpoints or populations.')
        if not isinstance(observation_ids, list) or len(observation_ids) != terms['records'] or any(not isinstance(r, str) or not r for r in observation_ids):
            raise GraphError('Metric records require actual observation IDs.')
        # Repeated sampler entries remain visible and counted, not silently
        # deduplicated. Report unique counts separately from evaluated records.
        numerator = float(terms['numerator'].detach())
        denominator = float(terms['denominator'].detach())
        if not math.isfinite(numerator) or numerator < 0 or not math.isfinite(denominator) or denominator < 0:
            raise GraphError('Metric terms must be finite nonnegative values.')
        correct = float(terms['correct_numerator'].detach()) if self.kind == 'cross-entropy' else None
        if correct is not None and (not math.isfinite(correct) or not 0 <= correct <= denominator):
            raise GraphError('Correct weighted count is inconsistent.')
        self.numerators.append(numerator); self.denominators.append(denominator)
        if correct is not None: self.correct.append(correct)
        self.elements += terms['elements']; self.included += terms['included_elements']
        self.observations.extend(observation_ids)

    def report(self):
        numerator = math.fsum(self.numerators); denominator = math.fsum(self.denominators)
        report = {'kind': self.kind, 'checkpoint_id': self.checkpoint, 'population': self.population,
                  'numerator': numerator, 'denominator': denominator,
                  'value': numerator / denominator if denominator > 0 else None,
                  'status': 'available' if denominator > 0 else 'unavailable',
                  'elements': self.elements, 'included_elements': self.included,
                  'evaluated_records': len(self.observations), 'unique_observations': len(set(self.observations)),
                  'observation_ids': list(self.observations), 'reduction': 'sum-weighted-elements / sum-effective-weights',
                  'accumulation_dtype': 'float64'}
        if self.kind == 'cross-entropy':
            report['accuracy_numerator'] = math.fsum(self.correct)
            report['accuracy'] = report['accuracy_numerator'] / denominator if denominator > 0 else None
        return report
