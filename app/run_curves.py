"""Exact retained-attempt head curves, read-only and independent of plotting libs.

Pages bound head and point counts, not complete registry validation/inline costs
or arbitrary metadata bytes. Full retained head history sets stable page axes.
"""
import copy
import math
from .authoring import workspace_id
from .datasets import _integer, _require
from .experiment_objects import reference


def curve_view(workspace, ref, *, attempt=None, head=None, offset=0, limit=100,
               head_offset=0, head_limit=25):
    reference(ref, 'run')
    _require(_integer(offset) and _integer(limit, 1, 100) and
             _integer(head_offset) and _integer(head_limit, 1, 25),
             'Curve pages need exact nonnegative offsets; point limit 1..100, head limit 1..25.')
    record = workspace.get_spec('run', ref['id'], ref['revision'])
    number = len(record['attempts']) if attempt is None else attempt
    _require(_integer(number, 1, len(record['attempts'])),
             'Select a retained attempt in this exact run revision.')
    selected = record['attempts'][number - 1]
    experiment_ref = record['selection']['experiment']
    experiment = workspace.get_spec('experiment', experiment_ref['id'], experiment_ref['revision'])
    pair = next(p for p in experiment['data_tasks'] if p['id'] == record['selection']['coordinates']['data_task'])
    task = workspace.get_spec('task', pair['task']['id'], pair['task']['revision'])
    chosen = task['heads'][0] if head is None else next((h for h in task['heads'] if h['id'] == head), None)
    _require(chosen is not None, 'Curve head must name an exact declared task head.')
    dataset = workspace.get_spec('dataset', pair['dataset']['id'], pair['dataset']['revision'])
    field = next(f for f in dataset['fields'] if f['id'] == chosen['field'])
    result = selected['result']
    history = result['history'] if result is not None else []
    # Native validation already checked this saved history. No re-evaluation,
    # fitting, split regeneration or inference occurs here.
    values = [point['metrics'][role]['heads'][chosen['id']]['value']
              for point in history for role in ('train', 'validation')]
    maximum = max((value for value in values if value is not None), default=0)
    padded = maximum * 1.05
    ymax = (padded if math.isfinite(padded) else maximum) if maximum > 0 else 1.
    rows = []
    for ordinal, point in enumerate(history[offset:offset + limit], offset):
        rows.append({k: point[k] for k in ('update', 'attempted_batches', 'examples_processed',
                                         'checkpoint_id', 'timing', 'mode')} | {
            'ordinal': ordinal,
            'metrics': {role: {k: v for k, v in point['metrics'][role]['heads'][chosen['id']].items()
                              if k != 'observation_ids'} for role in ('train', 'validation')}})
    heads = [{'id': h['id'], 'loss': h['loss']} for h in task['heads'][head_offset:head_offset + head_limit]]
    end = offset + len(rows); head_end = head_offset + len(heads)
    return copy.deepcopy({
        'version': 1, 'scope': 'read-saved-curve-only', 'workspace_id': workspace_id(workspace),
        'reference': ref, 'attempt': {k: selected[k] for k in ('number', 'status', 'stage', 'error')},
        'head': chosen, 'dataset': pair['dataset'], 'task': pair['task'],
        'source_unit': field.get('unit'), 'loss_space': 'transformed-target-space',
        'execution': 'no-fitting-or-training', 'scale': 'linear',
        'x_range': [0, max(1, history[-1]['update'] if history else 0)], 'y_range': [0, ymax],
        'retention': result['history_retention'] if result is not None else None,
        'final_checkpoint': result['checkpoint_id'] if result is not None else None,
        'heads': {'offset': head_offset, 'limit': head_limit, 'total': len(task['heads']),
                  'rows': heads, 'next_offset': head_end if head_end < len(task['heads']) else None},
        'points': {'offset': offset, 'limit': limit, 'total': len(history), 'rows': rows,
                   'next_offset': end if end < len(history) else None}})
