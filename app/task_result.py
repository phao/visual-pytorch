"""Pure admission of terminal task evidence against its exact frozen bundle.

Checks representation, internal arithmetic and captured population meaning. This
does not prove execution origin or recompute neural predictions without Torch.
"""
import math
import re
import struct

from .datasets import _flat, _integer, _keys, _require, _version
from .evidence import digest
from .task_execution import check_task_checkpoint, run_identity


def _finite(value, minimum=0):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _close(a, b):
    return _finite(a) and _finite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _population(prepared, task, role, hid):
    head = next(h for h in task['heads'] if h['id'] == hid)
    shape = next(g['shape'] for g in prepared['contract']['output_groups'] if g['id'] == head['port'])
    width = math.prod(shape[1:] if head['loss'] == 'cross-entropy' else shape)
    pop = prepared['populations'][role]; effective = []
    def weights(value):
        numbers = _flat(value)
        if not isinstance(value, list):
            numbers *= width
        _require(len(numbers) == width, 'Metric weight shape disagrees with task.')
        return [struct.unpack('f', struct.pack('f', float(v)))[0] for v in numbers]
    for mask, weight in zip(pop['masks'][hid], pop['sample_weights'][hid]):
        effective.extend(m * w for m, w in zip(weights(mask), weights(weight)))
    return pop['observation_ids'], width, effective


def _metric(metric, prepared, task, role, head, checkpoint):
    fields = ('kind', 'checkpoint_id', 'population', 'numerator', 'denominator', 'value', 'status', 'elements',
              'included_elements', 'evaluated_records', 'unique_observations', 'observation_ids', 'reduction', 'accumulation_dtype')
    _keys(metric, fields + (('accuracy_numerator', 'accuracy') if head['loss'] == 'cross-entropy' else ()), context='Stored head metric')
    _require(metric['kind'] == head['loss'] and metric['checkpoint_id'] == checkpoint and metric['population'] == role and
             metric['reduction'] == 'sum-weighted-elements / sum-effective-weights' and metric['accumulation_dtype'] == 'float64',
             'Metric loss/checkpoint/population/reduction mismatch.')
    ids, width, effective = _population(prepared, task, role, head['id'])
    _require(metric['observation_ids'] == ids, 'Metric observation IDs do not match captured population.')
    counters = {'evaluated_records': len(ids), 'unique_observations': len(set(ids)), 'elements': len(ids) * width,
                'included_elements': sum(v > 0 for v in effective)}
    _require(all(type(metric[k]) is int and metric[k] == v for k, v in counters.items()), 'Metric population/element counts mismatch.')
    denominator = math.fsum(effective)
    _require(_finite(metric['numerator']) and _close(metric['denominator'], denominator), 'Metric denominator/finite numerator mismatch.')
    if denominator > 0:
        _require(metric['status'] == 'available' and metric['denominator'] > 0 and _close(metric['value'], metric['numerator'] / metric['denominator']), 'Metric mean is not its additive weighted quotient.')
    else:
        _require(metric['status'] == 'unavailable' and metric['value'] is None and metric['numerator'] == 0 and metric['denominator'] == 0, 'Empty metric must be explicitly unavailable.')
    if head['loss'] == 'cross-entropy':
        correct = metric['accuracy_numerator']
        _require(_finite(correct) and correct <= denominator and (metric['accuracy'] is None if denominator == 0 else
                 _close(metric['accuracy'], correct / denominator)), 'Weighted classification accuracy mismatch.')


def _metrics(metrics, prepared, task, roles, checkpoint):
    _require(isinstance(metrics, dict) and set(metrics) == set(roles), 'Metric roles are inconsistent with checkpoint/terminal status.')
    for role, population in metrics.items():
        _keys(population, ('heads', 'objective', 'objective_status', 'objective_rule'), context='Population objective')
        _require(isinstance(population['heads'], dict) and set(population['heads']) == {h['id'] for h in task['heads']}, 'Stored head IDs mismatch.')
        active = []
        for head in task['heads']:
            metric = population['heads'][head['id']]
            _metric(metric, prepared, task, role, head, checkpoint)
            if metric['status'] == 'available':
                active.append(head['weight'] * metric['value'])
        _require(population['objective_rule'] == 'sum-head-weight-times-available-head-mean; no weight renormalization', 'Unknown objective reduction rule.')
        _require((population['objective_status'] == 'available' and _close(population['objective'], math.fsum(active))) if active else
                 (population['objective_status'] == 'unavailable' and population['objective'] is None), 'Population objective is inconsistent with declared heads.')


def validate_result(bundle, result):
    """Validate worker artifact with bundle removed (references are stored instead)."""
    _keys(result, ('version', 'kind', 'status', 'stop_reason', 'failure', 'evaluation_error', 'identity', 'checkpoint',
                  'checkpoint_id', 'counts', 'history', 'final_metrics', 'elapsed_seconds', 'history_retention',
                  'memory_admission', 'environment', 'sampler'), context='Stored task result')
    _version(result, 'Stored task result')
    _require(result['kind'] == 'task-training-result' and result['status'] in ('completed', 'failed', 'cancelled', 'unavailable'), 'Unknown task result state.')
    _require(digest(result['identity']) == digest(run_identity(bundle)) and result['checkpoint_id'] == digest(result['checkpoint']), 'Result execution/checkpoint identity mismatch.')
    prepared = check_task_checkpoint(bundle, result['checkpoint'], constrained=False)
    counts = result['counts']
    _keys(counts, ('completed_updates', 'attempted_batches', 'examples_processed', 'updated_examples', 'skipped_batches', 'completed_epochs'), context='Actual counters')
    _require(all(_integer(v) for v in counts.values()), 'Actual counters must be nonnegative integers.')
    updates = counts['completed_updates']; attempts = counts['attempted_batches']; seen = counts['examples_processed']
    size = min(bundle['protocol']['batch_size'], len(prepared['populations']['train']['observation_ids']))
    _require(updates + counts['skipped_batches'] <= attempts and updates <= counts['updated_examples'] <= seen <= attempts * size and
             attempts <= seen and counts['completed_epochs'] * len(prepared['populations']['train']['observation_ids']) <= seen,
             'Actual updates/examples/epochs are internally inconsistent.')
    budget = bundle['protocol']['budget']
    for key, counter in (('updates', updates), ('examples', seen), ('epochs', counts['completed_epochs'])):
        _require(budget[key] is None or counter <= budget[key], 'Actual work exceeded the declared counter budget.')
    _require(_finite(result['elapsed_seconds']), 'Elapsed time must be finite/nonnegative.')
    for key in ('failure', 'evaluation_error'):
        _require(result[key] is None or isinstance(result[key], str), 'Execution diagnostics must be string/null.')
    if result['status'] == 'completed':
        reasons = {'updates_budget': ('updates', updates), 'examples_budget': ('examples', seen),
                   'epochs_budget': ('epochs', counts['completed_epochs']), 'time_budget': ('seconds', result['elapsed_seconds'])}
        _require(result['stop_reason'] in reasons, 'Completed execution needs an observed budget boundary.')
        key, actual = reasons[result['stop_reason']]
        _require(budget[key] is not None and actual >= budget[key] and result['failure'] is None and
                 result['evaluation_error'] is None, 'Completed status contradicts budget/failure evidence.')
    elif result['status'] == 'cancelled':
        _require(result['stop_reason'] == 'cancelled' and result['failure'] is None, 'Cancelled stop reason mismatch.')
    elif result['status'] == 'unavailable':
        _require(result['stop_reason'] == 'no_training_supervision' and updates == 0 and result['failure'] is None, 'Unavailable supervision state mismatch.')
    else:
        _require(result['stop_reason'] in ('execution_error', 'evaluation_error') and isinstance(result['failure'], str), 'Failed execution needs explicit failure reason.')
    retention = result['history_retention']; history = result['history']
    _keys(retention, ('kind', 'max_points', 'observed_points', 'discarded_points'), context='Curve retention evidence')
    _require(digest({k: retention[k] for k in ('kind', 'max_points')}) == digest(bundle['protocol']['history']) and
             isinstance(history, list) and len(history) <= retention['max_points'] and
             _integer(retention['observed_points']) and _integer(retention['discarded_points']) and
             retention['observed_points'] == len(history) + retention['discarded_points'], 'Curve retention count/policy mismatch.')
    _require(retention['observed_points'] <= updates + 1, 'Observed curve count exceeds possible checkpoint observations.')
    previous = (-1, -1, -1)
    for point in history:
        _keys(point, ('update', 'attempted_batches', 'examples_processed', 'checkpoint_id', 'timing', 'mode', 'metrics'), context='Curve point')
        values = (point['update'], point['attempted_batches'], point['examples_processed'])
        _require(all(_integer(v) for v in values) and all(a <= b for a, b in zip(previous, values)) and
                 all(a <= b for a, b in zip(values, (updates, attempts, seen))) and _hash(point['checkpoint_id']) and
                 point['mode'] == 'eval' and point['timing'] == ('initial' if point['update'] == 0 else 'post-update'),
                 'Curve counters/checkpoint/timing mismatch.')
        previous = values
        _metrics(point['metrics'], prepared, bundle['task'], ('train', 'validation'), point['checkpoint_id'])
    if result['evaluation_error'] is None:
        _require(history and history[-1]['checkpoint_id'] == result['checkpoint_id'] and history[-1]['update'] == updates and
                 history[-1]['attempted_batches'] == attempts and history[-1]['examples_processed'] == seen, 'Final curve does not describe the final checkpoint/work.')
    final = result['final_metrics']
    roles = ('train', 'validation', 'test') if result['status'] == 'completed' else ('train', 'validation')
    if result['evaluation_error'] is not None and final == {}:
        pass
    else:
        _metrics(final, prepared, bundle['task'], roles, result['checkpoint_id'])
        if history and history[-1]['checkpoint_id'] == result['checkpoint_id']:
            _require(digest({r: final[r] for r in ('train', 'validation')}) == digest(history[-1]['metrics']), 'Final metrics silently substituted another curve population.')
    environment = result['environment']
    _keys(environment, ('python', 'torch', 'platform', 'device', 'dtype', 'numerical_threads', 'optimizer_resume'), context='Execution environment')
    _require(all(isinstance(environment[k], str) and environment[k] for k in ('python', 'torch', 'platform')) and
             environment['device'] == 'cpu' and environment['dtype'] == 'float32' and
             type(environment['numerical_threads']) is int and environment['numerical_threads'] == 1 and
             environment['optimizer_resume'] is False, 'Environment/capability mismatch.')
    _require(digest(result['sampler']) == digest({'kind': 'finite-epoch-python-mt19937-v1', 'replacement': False,
        'seed': bundle['protocol']['seeds']['shuffle'], 'partial_batch': 'retained'}), 'Sampler metadata mismatch.')
    memory = result['memory_admission']
    _keys(memory, ('version', 'operation', 'batch', 'parameter_bytes', 'components', 'estimated_bytes',
                  'actual_batch_size', 'limit_percent', 'available_bytes'), context='RAM evidence')
    _require(type(memory['version']) is int and memory['version'] == 2 and memory['operation'] == 'training' and
             type(memory['actual_batch_size']) is int and memory['actual_batch_size'] == size and
             _integer(memory['batch'], size) and type(memory['parameter_bytes']) is int and
             memory['parameter_bytes'] == 4 * prepared['contract']['parameters'] and
             isinstance(memory['components'], dict) and memory['components'] and all(_integer(v) for v in memory['components'].values()) and
             _integer(memory['estimated_bytes']) and memory['estimated_bytes'] == sum(memory['components'].values()) and
             _finite(memory['limit_percent']) and 0 < memory['limit_percent'] <= 100 and
             (memory['available_bytes'] is None or _integer(memory['available_bytes'])), 'RAM evidence is inconsistent.')
    from .memory import estimate_memory
    expected = estimate_memory(prepared['contract'], samples=math.ceil(size / .8), training=True)
    expected['components']['typed_preparation'] = prepared['preparation_estimate']['estimated_bytes']
    point_rows = sum(len(prepared['populations'][r]['observation_ids']) for r in ('train', 'validation'))
    expected['components']['metric_history'] = bundle['protocol']['history']['max_points'] * (8192 + 512 * point_rows * len(bundle['task']['heads']))
    if bundle['version']==3:
        from .temporal_shapes import admission as temporal_admission
        model=bundle['model']
        extra=temporal_admission(model['models'],model['root'],prepared['contract'],size,constrained=False,
                                 memory_limit_percent=memory['limit_percent'],gradients=True)
        expected['components']['recurrent_unrolls']=extra['memory']['components']['recurrent_unrolls']
    _require(digest(memory['components']) == digest(expected['components']) and memory['batch'] == expected['batch'],
             'RAM component estimates do not match the recorded runtime policy.')
