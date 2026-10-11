"""Exact candidate projections shared by stored and owned-draft previews.

Complete validation/dependency/split work stays native. These projections bound
successful wire bytes, not storage/manifest generation or numerical admission.
"""
from . import science_browser
from .datasets import _require
from .experiment_objects import preview_experiment, reference


def project(source, identity):
    result = {**identity, 'experiment': source['experiment'],
              'offset': source['offset'], 'limit': source['limit'],
              'total_combinations_exact': str(source['total_combinations']),
              'candidates': [], 'next_offset': None, 'execution': 'no-fitting-or-training'}
    for candidate in source['candidates']:
        manifest = candidate['manifest']
        row = {'id': candidate['id'], 'coordinates': candidate['coordinates'], 'status': candidate['status'],
               'reason': None if candidate['reason'] is None else science_browser._text(candidate['reason'], limit=128),
               'references': candidate['references'],
               'counts': None if manifest is None else {**{role:len(ids) for role,ids in manifest['roles'].items()},
                                                        'excluded':len(manifest['excluded'])}}
        if 'parameters_exact' in candidate: row['parameters_exact'] = candidate['parameters_exact']
        end = source['offset'] + len(result['candidates']) + 1
        cursor = end if end < source['total_combinations'] else None
        proposed = {**result, 'candidates': [*result['candidates'], row], 'next_offset': cursor}
        if len(science_browser.encoded(proposed)) > science_browser.MAX_BYTES:
            break
        result['candidates'].append(row)
        result['next_offset'] = cursor
    _require(result['candidates'] or source['offset'] >= source['total_combinations'],
             'One preview candidate cannot fit its projection. Inspect the exact experiment explicitly.')
    return science_browser._bounded(result)


def stored(workspace, ref, *, offset=0, limit=25):
    """Read one exact retained selection, never a head fallback or training."""
    from .authoring import workspace_id
    science_browser._page(offset, limit)
    reference(ref, 'experiment')
    experiment = workspace.get_spec('experiment', ref['id'], ref['revision'])
    source = preview_experiment(experiment, workspace.get_spec, offset=offset, limit=limit)
    return project(source, {'version': 1, 'scope': 'stored-experiment-preview',
                            'workspace_id': workspace_id(workspace), 'byte_limit': science_browser.MAX_BYTES})
