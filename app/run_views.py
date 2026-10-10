"""Read-only presentation of validated exact saved attempts, never execution.

Metric/run pages bound response row counts, not validation or inline-file costs.
Normal registry reads still validate the complete retained dependency/evidence
closure in pure Python, including reconstructing fitted preprocessing.
"""
import copy
from .core import validate_model
from .datasets import _integer, _require
from .evidence import digest
from .experiment_objects import reference

ROLES = ('train', 'validation', 'test')


def _page(offset, limit, maximum):
    _require(_integer(offset) and _integer(limit, 1, maximum),
             'Evidence page needs exact nonnegative offset and limit in 1..' + str(maximum) + '.')


def _project(workspace, ref, record, attempt, offset, limit):
    selected = record['attempts'][attempt - 1]
    result = selected['result']; selection = record['selection']; resolved = record['resolved']
    expref = selection['experiment']
    experiment = workspace.get_spec('experiment', expref['id'], expref['revision'])
    pairing = next(p for p in experiment['data_tasks'] if p['id'] == selection['coordinates']['data_task'])
    refs = {'model': selection['coordinates']['model'], 'dataset': pairing['dataset'], 'task': pairing['task']}
    docs = {kind: workspace.get_spec(kind, value['id'], value['revision']) for kind, value in refs.items()}
    objects = {kind: {'reference': value, 'name': docs[kind]['name']} for kind, value in refs.items()}
    objects['model']['parameters'] = validate_model(docs['model']['models'], docs['model']['root'])['parameters']
    objects['dataset']['observations'] = len(docs['dataset']['records'])
    manifest = resolved['manifest'] if resolved else None
    split = None if manifest is None else {
        'identity': digest(manifest), 'method': manifest['method'],
        'roles': {role: {'records': len(manifest['roles'][role]),
                         'observation_ids_hash': digest(manifest['roles'][role])} for role in ROLES},
        'excluded': len(manifest['excluded']), 'excluded_ids_hash': digest(manifest['excluded']),
        'metadata': {k: v for k, v in manifest.items()
                     if k not in ('roles', 'excluded', 'dataset_revision')}}
    execution = None
    if result is not None:
        execution = {k: result[k] for k in ('status', 'stop_reason', 'failure', 'evaluation_error',
            'checkpoint_id', 'counts', 'elapsed_seconds', 'history_retention', 'environment')}
        execution['fitted_state_hash'] = digest(result['checkpoint']['fitted'])
        execution['memory_admission'] = {k: result['memory_admission'][k] for k in
                                        ('estimated_bytes', 'actual_batch_size', 'available_bytes', 'limit_percent')}
    populations = []; coordinates = []
    for role in ROLES:
        pop = result['final_metrics'].get(role) if result is not None else None
        populations.append({'population': role, 'objective': pop['objective'] if pop else None,
            'status': pop['objective_status'] if pop else 'not-retained',
            'rule': pop['objective_rule'] if pop else None})
        if pop:
            coordinates.extend((role, head) for head in docs['task']['heads'])
    entries = []
    for role, head in coordinates[offset:offset + limit]:
        field = next(f for f in docs['dataset']['fields'] if f['id'] == head['field'])
        metric = result['final_metrics'][role]['heads'][head['id']]
        entries.append({'population': role,
            'head': {k: head.get(k) for k in ('id', 'port', 'field', 'loss', 'weight', 'transforms',
                                            'mask', 'sample_weight')} | {'source_unit': field.get('unit')},
            'metric': {k: v for k, v in metric.items() if k != 'observation_ids'} |
                      {'observation_ids_hash': digest(metric['observation_ids'])}})
    total = len(coordinates); end = offset + len(entries)
    return copy.deepcopy({'version': 1, 'scope': 'read-saved-evidence-only', 'reference': ref,
        'name': record['name'], 'attempt_count': len(record['attempts']),
        'attempt': {k: selected[k] for k in ('number', 'started_at', 'finished_at', 'stage', 'status', 'error')},
        'selection': selection, 'objects': objects, 'split': split,
        'protocol': resolved['protocol'] if resolved else None, 'execution': execution,
        'populations': populations,
        'metrics': {'offset': offset, 'limit': limit, 'total': total, 'rows': entries,
                    'next_offset': end if end < total else None}})


def run_view(workspace, ref, *, attempt=None, offset=0, limit=25):
    reference(ref, 'run'); _page(offset, limit, 25)
    record = workspace.get_spec('run', ref['id'], ref['revision'])
    selected = len(record['attempts']) if attempt is None else attempt
    _require(_integer(selected, 1, len(record['attempts'])),
             'Attempt must name a retained attempt in this exact run revision.')
    return _project(workspace, ref, record, selected, offset, limit)


def report_view(workspace, ref, *, offset=0, limit=5):
    reference(ref, 'report'); _page(offset, limit, 10)
    recipe = workspace.get_spec('report', ref['id'], ref['revision'])
    chosen = recipe['runs'][offset:offset + limit]
    runs = [run_view(workspace, r) for r in chosen]
    total = len(recipe['runs']); end = offset + len(runs)
    return copy.deepcopy({'version': 1, 'scope': 'read-saved-evidence-only',
        'reference': ref, 'name': recipe['name'], 'question': recipe['question'],
        'mode': recipe['mode'], 'analysis': recipe['analysis'], 'sections': recipe['sections'],
        'execution_behavior': 'read-saved-evidence-only' if recipe['mode'] == 'result-report'
                              else 'explicit-execution-required',
        'experiment': recipe['experiment'],
        'runs': {'offset': offset, 'limit': limit, 'total': total, 'rows': runs,
                 'next_offset': end if end < total else None}})
