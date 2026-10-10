"""Independent concrete experiments, terminal attempts and report selections.

Pure contracts/preview: no Qt, Torch, transform fitting or global RNG use.
Execution is an explicit separate call through the real numerical worker.
"""
import copy
import math
import re
import sys
from datetime import datetime
from pathlib import Path

from .core import GraphError, timestamp, validate_model
from .datasets import (_identifier, _integer, _keys, _require, _version,
                       check_manifest, make_split, validate_task)
from .evidence import digest
from .task_execution import resolve_run, task_bundle, validate_protocol

SECTIONS = ('question', 'protocol', 'models', 'datasets', 'training', 'evaluation',
            'failures', 'environment', 'checkpoints')


def _text(value):
    return isinstance(value, str) and 0 < len(value) <= 20000 and '\0' not in value


def reference(ref, kind):
    _keys(ref, ('kind', 'id', 'revision'), context='Exact reference')
    _require(ref['kind'] == kind and _identifier(ref['id']) and isinstance(ref['revision'], str) and
             re.fullmatch('[0-9a-f]{64}', ref['revision']) is not None, 'Exact reference needs kind, stable ID and SHA-256.')
    return ref


def _distinct(values, what):
    _require(len({digest(v) for v in values}) == len(values), 'Duplicate ' + what + '.')


def _header(doc, required):
    _keys(doc, ('version', 'id', 'name', *required), ('revision',), 'Independent scientific object')
    _version(doc, 'Independent scientific object')
    _require(_identifier(doc['id']) and _identifier(doc['name']), 'Scientific object needs stable ID and name.')


def split_protocol(protocol):
    _keys(protocol, ('method',), ('seed', 'fractions', 'field', 'roles', 'excluded'), 'Split protocol')
    method = protocol['method']
    _require(method in ('row', 'group', 'temporal', 'external'), 'Unsupported split method.')
    required = {'row': {'method', 'seed', 'fractions'}, 'group': {'method', 'seed', 'fractions', 'field'},
                'temporal': {'method', 'fractions', 'field'}, 'external': {'method', 'roles', 'excluded'}}[method]
    _require(set(protocol) == required, 'Split protocol fields do not match its method.')
    if method in ('row', 'group'):
        _require(_integer(protocol['seed'], 0, 2**31 - 1), 'Split seed must be integer in 0..2^31-1.')
    if method in ('group', 'temporal'):
        _require(_identifier(protocol['field']), 'Split grouping/time field needs stable ID.')
    if method != 'external':
        fractions = protocol['fractions']
        _require(isinstance(fractions, list) and len(fractions) == 3 and all(
            type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in fractions) and
            fractions[0] > 0 and fractions[2] > 0 and math.isclose(math.fsum(fractions), 1., abs_tol=1e-12, rel_tol=0),
            'Split fractions are [train,validation,test], sum 1, with positive train/test.')
    else:
        _keys(protocol['roles'], ('train', 'validation', 'test'), context='External split roles')
        roles = list(protocol['roles'].values()) + [protocol['excluded']]
        _require(all(isinstance(ids, list) and all(_identifier(r) for r in ids) for ids in roles), 'External split needs ID arrays.')
        ids = [rid for role in roles for rid in role]
        _distinct(ids, 'observation in external split')
        _require(protocol['roles']['train'] and protocol['roles']['test'], 'External split needs train and test observations.')


def validate_experiment(doc):
    _header(doc, ('question', 'models', 'data_tasks', 'splits', 'initialization_seeds', 'protocols', 'selection'))
    _require(_text(doc['question']), 'Experiment needs an explicit question.')
    for key in ('models', 'data_tasks', 'splits', 'initialization_seeds', 'protocols'):
        _require(isinstance(doc[key], list) and doc[key], 'Experiment selection axes must be nonempty arrays.')
    for model in doc['models']:
        reference(model, 'model')
    _distinct(doc['models'], 'model revision')
    for pair in doc['data_tasks']:
        _keys(pair, ('id', 'dataset', 'task'), context='Dataset/task pairing')
        _require(_identifier(pair['id']), 'Data/task pairing needs stable selection ID.')
        reference(pair['dataset'], 'dataset'); reference(pair['task'], 'task')
    _distinct([p['id'] for p in doc['data_tasks']], 'dataset/task pairing ID')
    for split in doc['splits']:
        _keys(split, ('id', 'protocol'), context='Experiment split')
        _require(_identifier(split['id']), 'Split selection needs stable ID.')
        split_protocol(split['protocol'])
    _distinct([s['id'] for s in doc['splits']], 'split selection ID')
    _require(all(_integer(seed, 0, 2**31 - 1) for seed in doc['initialization_seeds']), 'Initialization seeds must be integer in 0..2^31-1.')
    _distinct(doc['initialization_seeds'], 'initialization seed')
    for protocol in doc['protocols']:
        _keys(protocol, ('id', 'protocol'), context='Training configuration selection')
        _require(_identifier(protocol['id']), 'Training configuration needs stable ID.')
        validate_protocol(protocol['protocol'])
    _distinct([p['id'] for p in doc['protocols']], 'training configuration ID')
    selection = doc['selection']
    _keys(selection, ('kind', 'pairs', 'exclusions'), context='Comparison selection')
    _require(selection['kind'] in ('crossed', 'paired') and isinstance(selection['pairs'], list) and
             isinstance(selection['exclusions'], list), 'Selection must explicitly be crossed or paired.')
    _require(selection['pairs'] == [] if selection['kind'] == 'crossed' else bool(selection['pairs']), 'Crossed uses no pair whitelist; paired needs explicit pairs.')
    models = {digest(m) for m in doc['models']}; data = {d['id'] for d in doc['data_tasks']}
    for entries, excluded in ((selection['pairs'], False), (selection['exclusions'], True)):
        for entry in entries:
            _keys(entry, ('model', 'data_task', 'reason') if excluded else ('model', 'data_task'), context='Comparison pairing/exclusion')
            reference(entry['model'], 'model')
            _require(digest(entry['model']) in models and entry['data_task'] in data, 'Pair/exclusion must name selected model and data/task.')
            if excluded:
                _require(_text(entry['reason']), 'Exclusion needs a visible reason.')
        _distinct([{'model': e['model'], 'data_task': e['data_task']} for e in entries], 'comparison pair')


def validate_report(doc):
    _header(doc, ('question', 'mode', 'runs', 'experiment', 'sections', 'analysis'))
    _require(_text(doc['question']) and doc['mode'] in ('result-report', 'rerun'), 'Report needs question and explicit execution mode.')
    _require(isinstance(doc['runs'], list), 'Report runs must be an array of exact references.')
    for ref in doc['runs']:
        reference(ref, 'run')
    _distinct(doc['runs'], 'report run revision')
    _distinct([r['id'] for r in doc['runs']], 'logical run ID in run-level report')
    if doc['mode'] == 'result-report':
        _require(bool(doc['runs']) and doc['experiment'] is None, 'Result report reads exact existing runs; no implicit experiment execution.')
    else:
        _require(doc['runs'] == [], 'Rerun recipe selects experiment execution, not a mixed result selection.')
        reference(doc['experiment'], 'experiment')
    _require(isinstance(doc['sections'], list) and doc['sections'] and all(s in SECTIONS for s in doc['sections']), 'Unsupported report section.')
    _distinct(doc['sections'], 'report section')
    _require(doc['analysis'] == {'kind': 'descriptive', 'inferential_unit': 'run'},
             'Only explicit descriptive run-level analysis is implemented; no guessed confidence interval.')


def _utc(value):
    _require(isinstance(value, str), 'Attempt timestamp must be a UTC string.')
    try:
        stamp = datetime.fromisoformat(value)
        _require(stamp.utcoffset() is not None and stamp.utcoffset().total_seconds() == 0, 'Attempt timestamp must be explicit UTC.')
        return stamp
    except (ValueError, OverflowError) as error:
        raise GraphError('Invalid attempt timestamp.') from error


def validate_run(doc):
    _header(doc, ('selection', 'resolved', 'attempts'))
    _keys(doc['selection'], ('experiment', 'coordinates', 'candidate_id'), context='Run selection')
    reference(doc['selection']['experiment'], 'experiment')
    coords = doc['selection']['coordinates']
    _keys(coords, ('model', 'data_task', 'split', 'initialization', 'protocol'), context='Resolved selection coordinates')
    reference(coords['model'], 'model')
    _require(all(_identifier(coords[k]) for k in ('data_task', 'split', 'protocol')) and
             _integer(coords['initialization'], 0, 2**31 - 1), 'Invalid selection coordinate.')
    _require(doc['selection']['candidate_id'] == digest(doc['selection']['experiment'] | {'coordinates': coords}), 'Candidate identity mismatch.')
    if doc['resolved'] is not None:
        _keys(doc['resolved'], ('version', 'references', 'manifest', 'protocol'), context='Frozen resolved execution')
        _version(doc['resolved'], 'Frozen resolved execution')
        _keys(doc['resolved']['references'], ('model', 'dataset', 'task'), context='Frozen execution references')
        for kind, ref in doc['resolved']['references'].items():
            reference(ref, kind)
        validate_protocol(doc['resolved']['protocol'])
    _require(isinstance(doc['attempts'], list) and doc['attempts'], 'RunRecord needs actual terminal attempts.')
    previous = None
    for number, attempt in enumerate(doc['attempts'], 1):
        _keys(attempt, ('number', 'started_at', 'finished_at', 'stage', 'status', 'error', 'result'), context='Terminal attempt')
        _require(type(attempt['number']) is int and attempt['number'] == number, 'Attempt numbers must be ordered consecutive integers.')
        started = _utc(attempt['started_at']); finished = _utc(attempt['finished_at'])
        _require(finished >= started and (previous is None or started >= previous), 'Attempt timestamp ordering is inconsistent.')
        previous = finished
        _require(attempt['stage'] in ('compatibility', 'admission', 'worker', 'execution') and
                 attempt['status'] in ('completed', 'failed', 'cancelled', 'unavailable'), 'Unsupported terminal attempt state/stage.')
        if attempt['error'] is not None:
            _keys(attempt['error'], ('type', 'message'), context='Attempt error')
            _require(_identifier(attempt['error']['type']) and _text(attempt['error']['message']), 'Attempt error needs type and message.')
        if attempt['result'] is None:
            _require(attempt['stage'] != 'execution' and attempt['status'] in ('failed', 'unavailable') and
                     attempt['error'] is not None, 'Checkpoint-free attempts must have explicit failure/unavailable reason.')
            _require((doc['resolved'] is None) == (attempt['stage'] == 'compatibility'), 'Failure stage contradicts captured execution resolution.')
        else:
            _require(attempt['stage'] == 'execution' and isinstance(attempt['result'], dict) and
                     attempt['result'].get('status') == attempt['status'], 'Task result must match execution attempt status.')
            _require('bundle' not in attempt['result'], 'RunRecord stores exact revisions, not duplicate raw dataset bundles.')
            _require(doc['resolved'] is not None and (attempt['error'] is not None if attempt['status'] == 'failed' else attempt['error'] is None),
                     'Execution attempt error/resolution contradicts its terminal state.')


def object_reference(kind, doc):
    return {'kind': kind, 'id': doc['id'], 'revision': doc['revision']}


def _get(resolve, ref):
    result = resolve(ref['kind'], ref['id'], ref['revision'])
    _require(result['id'] == ref['id'] and result['revision'] == ref['revision'], 'Resolver substituted a different immutable revision.')
    return result


def dependencies(kind, doc):
    if kind == 'experiment':
        return doc['models'] + [ref for p in doc['data_tasks'] for ref in (p['dataset'], p['task'])]
    if kind == 'run':
        return [doc['selection']['experiment']]
    if kind == 'report':
        return doc['runs'] if doc['mode'] == 'result-report' else [doc['experiment']]
    return []


def candidate(experiment, coordinates, resolve, *, saved_manifest=None):
    """Resolve one coordinate directly, without allocating the full Cartesian set."""
    validate_experiment(experiment)
    _keys(coordinates, ('model', 'data_task', 'split', 'initialization', 'protocol'), context='Selection coordinates')
    _require(any(digest(m) == digest(coordinates['model']) for m in experiment['models']), 'Model not selected by experiment.')
    pairs = {d['id']: d for d in experiment['data_tasks']}
    splits = {s['id']: s for s in experiment['splits']}; protocols = {p['id']: p for p in experiment['protocols']}
    _require(coordinates['data_task'] in pairs and coordinates['split'] in splits and coordinates['protocol'] in protocols and
             type(coordinates['initialization']) is int and coordinates['initialization'] in experiment['initialization_seeds'], 'Coordinate is outside declared experiment selection.')
    ref = object_reference('experiment', experiment)
    result = {'id': digest(ref | {'coordinates': coordinates}), 'experiment': ref, 'coordinates': copy.deepcopy(coordinates),
              'status': 'compatible', 'reason': None, 'references': {}, 'manifest': None, 'protocol': None}
    pair = pairs[coordinates['data_task']]
    result['references'] = {'model': copy.deepcopy(coordinates['model']), 'dataset': copy.deepcopy(pair['dataset']), 'task': copy.deepcopy(pair['task'])}
    selection = experiment['selection']; pair_key = {'model': coordinates['model'], 'data_task': coordinates['data_task']}
    exclusion = next((e['reason'] for e in selection['exclusions'] if digest({k: e[k] for k in ('model', 'data_task')}) == digest(pair_key)), None)
    if exclusion is not None or (selection['kind'] == 'paired' and pair_key not in selection['pairs']):
        result.update(status='excluded', reason=exclusion or 'Not in the declared comparison pairing.')
        return result
    try:
        model = _get(resolve, coordinates['model']); dataset = _get(resolve, pair['dataset']); task = _get(resolve, pair['task'])
        contract = validate_model(model['models'], model['root'])
        validate_task(task, dataset, contract)
        _require(contract['parameters'] > 0, 'Training selection has no trainable parameters.')
        if saved_manifest is None:
            result['manifest'] = make_split(dataset, splits[coordinates['split']]['protocol'])
        else:
            expected = copy.deepcopy(splits[coordinates['split']]['protocol'])
            if expected['method'] in ('row', 'group'):
                expected['algorithm'] = 'python-mt19937-fisher-yates-v1'
            if expected['method'] in ('group', 'temporal'):
                expected['allocation_unit'] = 'distinct-field-values'
            _require(saved_manifest['method'] == expected['method'] and digest(saved_manifest['protocol']) == digest(expected),
                     'Captured split protocol is not the selected experiment protocol.')
            result['manifest'] = check_manifest(dataset, saved_manifest)
        protocol = copy.deepcopy(protocols[coordinates['protocol']]['protocol'])
        protocol['seeds']['initialization'] = coordinates['initialization']
        result['protocol'] = validate_protocol(protocol)
    except (GraphError, KeyError, ValueError) as error:
        result.update(status='incompatible', reason=str(error))
    return result


def preview_experiment(experiment, resolve, *, offset=0, limit=100):
    """Bounded explicit page; incompatibles/exclusions are NOT filtered out."""
    validate_experiment(experiment)
    _require(_integer(offset) and _integer(limit, 1, 1000), 'Preview offset/limit must be integers; limit 1..1000.')
    axes = (experiment['models'], [d['id'] for d in experiment['data_tasks']], [s['id'] for s in experiment['splits']],
            experiment['initialization_seeds'], [p['id'] for p in experiment['protocols']])
    total = math.prod(map(len, axes)); rows = []
    # Mixed-radix indexing: large offsets do not enumerate skipped combinations.
    for index in range(offset, min(total, offset + limit)):
        value = index; coordinate = []
        for axis in reversed(axes):
            value, digit = divmod(value, len(axis)); coordinate.append(axis[digit])
        coordinate.reverse()
        rows.append(candidate(experiment, dict(zip(('model', 'data_task', 'split', 'initialization', 'protocol'), coordinate)), resolve))
    return {'version': 1, 'experiment': object_reference('experiment', experiment), 'offset': offset, 'limit': limit,
            'total_combinations': total, 'candidates': rows, 'next_offset': offset + len(rows) if offset + len(rows) < total else None,
            'scope': 'bounded-selection-preview; no-fitting-or-training'}


def candidate_bundle(experiment, coordinates, resolve, *, saved_manifest=None):
    selected = candidate(experiment, coordinates, resolve, saved_manifest=saved_manifest)
    _require(selected['status'] == 'compatible', selected['reason'] or 'Selection is not executable.')
    return task_bundle(*[_get(resolve, selected['references'][kind]) for kind in ('model', 'dataset', 'task')],
                       selected['manifest'], selected['protocol'])


def validate_dependencies(kind, doc, resolve):
    for ref in dependencies(kind, doc):
        _get(resolve, ref)
    if kind == 'run':
        experiment = _get(resolve, doc['selection']['experiment'])
        resolved = doc['resolved']
        selected = candidate(experiment, doc['selection']['coordinates'], resolve,
                             saved_manifest=resolved['manifest'] if resolved is not None else None)
        _require(selected['id'] == doc['selection']['candidate_id'], 'Run selection identity mismatch.')
        if resolved is not None:
            _require(selected['status'] == 'compatible' and digest(resolved['references']) == digest(selected['references']) and
                     digest(resolved['protocol']) == digest(selected['protocol']), 'Run captured execution does not match the selected revisions/configuration.')
        for attempt in doc['attempts']:
            if attempt['result'] is not None:
                from .task_result import validate_result
                validate_result(candidate_bundle(experiment, doc['selection']['coordinates'], resolve,
                                saved_manifest=resolved['manifest']), attempt['result'])
            else:
                _require((selected['status'] != 'compatible') == (attempt['stage'] == 'compatibility'), 'Attempt failure stage contradicts selection compatibility.')


def execute_candidate(workspace, experiment_ref, coordinates, run_id, name, *, python=sys.executable,
                      retry=False, progress=None, cancel=None, memory_limit_percent=10., timeout=None,
                      saved_manifest=None):
    """Explicit synchronous execution + terminal storage; never silently retries.

    Caller performs atomic Save. No pending/detached receipt is claimed here.
    """
    reference(experiment_ref, 'experiment')
    _require(_identifier(run_id) and _identifier(name) and type(retry) is bool, 'Run ID/name/retry must be explicit and valid.')
    experiment = _get(workspace.get_spec, experiment_ref)
    selection = {'experiment': copy.deepcopy(experiment_ref), 'coordinates': copy.deepcopy(coordinates),
                 'candidate_id': digest(experiment_ref | {'coordinates': coordinates})}
    existing = [r for r in workspace.list_specs('run', True) if r['id'] == run_id]
    attempts = []
    resolved = None
    if existing:
        _require(retry, 'Run ID already exists; explicit retry is required.')
        records = [workspace.get_spec('run', run_id, r['revision']) for r in existing]
        previous = max(records, key=lambda r: len(r['attempts']))
        _require(digest(previous['selection']) == digest(selection), 'Retry cannot substitute another experiment/selection.')
        attempts = copy.deepcopy(previous['attempts'])
        resolved = copy.deepcopy(previous['resolved'])
    if resolved is not None and saved_manifest is not None:
        _require(digest(saved_manifest) == digest(resolved['manifest']), 'Retry cannot substitute the captured split manifest.')
    selected = candidate(experiment, coordinates, workspace.get_spec,
                         saved_manifest=resolved['manifest'] if resolved is not None else saved_manifest)
    if resolved is None and selected['status'] == 'compatible':
        resolved = {'version': 1, 'references': copy.deepcopy(selected['references']),
                    'manifest': copy.deepcopy(selected['manifest']), 'protocol': copy.deepcopy(selected['protocol'])}
    started = timestamp(); stage = 'compatibility'; result = None; error = None; status = 'unavailable'
    try:
        _require(selected['status'] == 'compatible', selected['reason'] or 'Incompatible selection.')
        stage = 'admission'; bundle = candidate_bundle(experiment, coordinates, workspace.get_spec, saved_manifest=resolved['manifest'])
        resolve_run(bundle, constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)
        stage = 'worker'
        from .transport import run_json_worker
        raw = run_json_worker([python, '-u', '-m', 'app.worker'],
            {'kind': 'task-v1', 'bundle': bundle, 'constrained': workspace.constrained_mode,
             'memory_limit_percent': memory_limit_percent}, on_progress=progress, cancel=cancel,
            cwd=str(Path(__file__).resolve().parent.parent), timeout=timeout)
        _require(digest(raw.get('bundle')) == digest(bundle), 'Worker substituted a different frozen execution bundle.')
        result = {k: v for k, v in raw.items() if k != 'bundle'}
        from .task_result import validate_result
        validate_result(bundle, result)
        stage = 'execution'; status = result['status']
        if status == 'failed':
            error = {'type': 'TaskExecutionError', 'message': result['failure'] or result['evaluation_error'] or 'Task execution failed.'}
    except Exception as exc:
        result = None; status = 'unavailable' if stage == 'compatibility' else 'failed'
        error = {'type': type(exc).__name__, 'message': str(exc) or type(exc).__name__}
    attempts.append({'number': len(attempts) + 1, 'started_at': started, 'finished_at': timestamp(),
                     'stage': stage, 'status': status, 'error': error, 'result': result})
    record = {'version': 1, 'id': run_id, 'name': name, 'selection': selection, 'resolved': resolved, 'attempts': attempts}
    ref = workspace.put_spec('run', record)
    return {'reference': ref, 'record': workspace.get_spec('run', run_id, ref['revision'])}


def report_data(recipe, resolve):
    """Read exact saved evidence, never execute a rerun merely by viewing it."""
    validate_report(recipe); validate_dependencies('report', recipe, resolve)
    result = {'version': 1, 'recipe': object_reference('report', recipe), 'question': recipe['question'],
              'mode': recipe['mode'], 'sections': copy.deepcopy(recipe['sections']), 'analysis': copy.deepcopy(recipe['analysis']),
              'execution_behavior': 'read-saved-evidence-only' if recipe['mode'] == 'result-report' else 'explicit-execution-required',
              'experiment': copy.deepcopy(recipe['experiment']), 'runs': [], 'notebook_status': 'not-generated'}
    for ref in recipe['runs']:
        run = _get(resolve, ref); latest = run['attempts'][-1]; evidence = latest['result']
        result['runs'].append({'reference': copy.deepcopy(ref), 'name': run['name'], 'selection': copy.deepcopy(run['selection']),
            'resolved': copy.deepcopy(run['resolved']), 'latest_attempt': latest['number'], 'status': latest['status'],
            'stage': latest['stage'], 'error': copy.deepcopy(latest['error']),
            'checkpoint_id': evidence['checkpoint_id'] if evidence else None,
            'counts': copy.deepcopy(evidence['counts']) if evidence else None,
            'metrics': copy.deepcopy(evidence['final_metrics']) if evidence else {},
            'environment': copy.deepcopy(evidence['environment']) if evidence else None,
            'attempt_history': [{k: copy.deepcopy(a[k]) for k in ('number', 'started_at', 'finished_at', 'stage', 'status', 'error')} for a in run['attempts']]})
    return result
