"""Optional closed example-pack v1: inspect saved evidence; replay only explicitly.

No numerical/config imports, alternate database loader, Save or history capture.
Version 1 deliberately supports only the existing unlabelled neural MS-VAR pack.
"""
import copy
import hashlib
import json
import math
import re
import uuid
from contextlib import contextmanager
from pathlib import Path

from .core import GraphError, Workspace
from .construction_recipes import build, show
from .evidence import digest, graph_signature
from .experiment_objects import candidate_bundle, execute_candidate, reference


MAX_UPDATES = 24
MAX_TIMEOUT = 60
PRODUCER_COMMIT = 'e086c5157d0a5fa221c53d7f3d896e34bbdf4f3a'
ASSUMPTIONS = {
    'conditioning': 'explicit-observed-history-and-valid-prefix-lengths',
    'emissions': 'state-specific-diagonal-gaussian',
    'variance': 'softplus-plus-0.0001',
    'inference': 'exact-conditional-filter-and-likelihood',
    'estimation': 'finite-budget-parameter-estimate',
    'regime_targets': 'absent',
    'limitations': 'no-identifiability-stationarity-or-calibration-claim',
}


def _require(condition, where, message):
    if not condition:
        raise GraphError(where + ': ' + message)


def _keys(value, keys, where):
    _require(type(value) is dict, where, 'expected a closed JSON object')
    _require(all(type(key) is str for key in value), where, 'JSON object keys must be strings')
    missing, extra = set(keys) - value.keys(), value.keys() - set(keys)
    _require(not missing and not extra, where,
             'closed keys; missing=' + repr(sorted(missing)) + ', unknown=' + repr(sorted(extra)))


def _at(where, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        raise GraphError(where + ': ' + str(error)) from error


def _hash(value, where):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
             where, 'expected a lowercase SHA256')


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _manifest(value):
    if isinstance(value, (str, Path)):
        def pairs(items):
            result = {}
            for key, item in items:
                _require(key not in result, '$', 'duplicate JSON key: ' + key)
                result[key] = item
            return result
        def nonfinite(item):
            raise GraphError('$: nonfinite JSON number: ' + item)
        def read():
            with Path(value).open('rb') as stream:
                raw = stream.read(1024 * 1024 + 1)
            _require(len(raw) <= 1024 * 1024, '$', 'manifest exceeds 1 MiB')
            return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                              parse_constant=nonfinite)
        value = _at('$', read)
    # Also reject overflowed exponents, non-JSON values and surrogate escapes.
    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise GraphError('$: expected finite UTF-8 JSON: ' + str(error)) from error
    _require(len(encoded) <= 1024 * 1024, '$', 'manifest exceeds 1 MiB')
    _keys(value, ('version', 'id', 'producer_source_commit', 'workspace_sha256',
                 'architectures', 'references', 'verification'), '$')
    pack = _at('$', copy.deepcopy, value)
    _require(type(pack['version']) is int and pack['version'] == 1, '$.version',
             'only integer version 1 is supported')
    _require(pack['id'] == 'neural-msvar-example-v1', '$.id', 'unsupported example pack')
    _require(pack['producer_source_commit'] == PRODUCER_COMMIT, '$.producer_source_commit',
             'this version supports the original committed neural MS-VAR example producer')
    _hash(pack['workspace_sha256'], '$.workspace_sha256')
    architectures = pack['architectures']
    _require(type(architectures) is list and len(architectures) == 1, '$.architectures',
             'this pack requires exactly one architecture')
    arch = architectures[0]
    root = '$.architectures[0]'
    _keys(arch, ('id', 'intended_use', 'assumptions', 'recipe'), root)
    _require(arch['id'] == 'neural-msvar', root + '.id', 'expected stable ID neural-msvar')
    _require(type(arch['intended_use']) is str and arch['intended_use'] in
             ('exact-model', 'approximate-estimator', 'parameter-component'),
             root + '.intended_use', 'unknown exact/approximate/component classification')
    _keys(arch['assumptions'], ASSUMPTIONS, root + '.assumptions')
    for key, expected in ASSUMPTIONS.items():
        _require(arch['assumptions'][key] == expected, root + '.assumptions.' + key,
                 'this pack requires ' + expected)
    recipe = arch['recipe']
    _keys(recipe, ('name', 'version', 'bindings'), root + '.recipe')
    _require(recipe['name'] == 'neural-msvar' and type(recipe['version']) is int and
             recipe['version'] == 1, root + '.recipe', 'only neural-msvar recipe version 1 is supported')
    declaration = show(recipe['name'])
    _require(arch['intended_use'] == declaration['kind'], root + '.intended_use',
             'recipe is exact-model; fitted parameters remain finite-budget estimates')
    properties = declaration['options']['properties']
    _keys(recipe['bindings'], properties, root + '.recipe.bindings')
    options = {}
    for key, spec in properties.items():
        where = root + '.recipe.bindings.' + key
        atom = recipe['bindings'][key]
        _keys(atom, ('type', 'value'), where)
        _require(atom['type'] == spec['type'] and type(atom['value']) is
                 (bool if spec['type'] == 'boolean' else int), where,
                 'expected typed ' + spec['type'] + ' without coercion')
        options[key] = atom['value']
    _require(options['controlled'] is False, root + '.recipe.bindings.controlled',
             'the committed example uses prefix lengths without extra controls')
    # Closed recipe service performs dimension/product admission before building.
    document = _at(root + '.recipe', build, recipe['name'], model_id=arch['id'],
                   options=options)[0]['document']
    refs = pack['references']
    _keys(refs, ('dataset', 'task', 'family', 'specialization', 'model', 'experiment', 'run'),
          '$.references')
    for kind, ref in refs.items():
        _at('$.references.' + kind, reference, ref, kind)
    verification = pack['verification']
    _keys(verification, ('kind', 'coordinates', 'budget', 'expected'), '$.verification')
    _require(verification['kind'] == 'saved-candidate-replay-v1', '$.verification.kind',
             'unsupported verification recipe; no commands or executable text accepted')
    budget = verification['budget']
    _keys(budget, ('updates', 'max_updates'), '$.verification.budget')
    _require(type(budget['max_updates']) is int and budget['max_updates'] == MAX_UPDATES and
             type(budget['updates']) is int and budget['updates'] == 12,
             '$.verification.budget', 'only the existing 12 updates are supported; hard ceiling is 24')
    expected = verification['expected']
    _keys(expected, ('checkpoint_sha256', 'metrics_sha256', 'completed_updates'),
          '$.verification.expected')
    for key in ('checkpoint_sha256', 'metrics_sha256'):
        _hash(expected[key], '$.verification.expected.' + key)
    _require(type(expected['completed_updates']) is int and expected['completed_updates'] == 12,
             '$.verification.expected.completed_updates', 'expected exactly 12 actual updates')
    return pack, document, options


@contextmanager
def _workspace(value):
    owned = not isinstance(value, Workspace)
    workspace = _at('workspace', Workspace.load, value) if owned else value
    try:
        _require(workspace.path is not None, 'workspace.path', 'a saved .vtw source is required')
        yield workspace
    finally:
        if owned:
            workspace.db.close()
            if workspace.history.directory:
                workspace.history.directory.cleanup()


def _inspect(workspace, pack, document, options):
    actual_hash = _at('$.workspace_sha256', _file_hash, workspace.path)
    _require(actual_hash == pack['workspace_sha256'], '$.workspace_sha256',
             'source file SHA256 mismatch')
    refs = pack['references']
    docs = {kind: _at('$.references.' + kind, workspace.get_spec,
                     kind, ref['id'], ref['revision']) for kind, ref in refs.items()}
    run = docs['run']
    _require(run['selection']['experiment'] == refs['experiment'], '$.references.experiment',
             'reference differs from the saved run selection')
    _require(run['resolved'] is not None and run['resolved']['references'] ==
             {k: refs[k] for k in ('dataset', 'task', 'family', 'specialization', 'model')},
             '$.references', 'references do not describe the saved run execution')
    verify = pack['verification']
    _require(digest(verify['coordinates']) == digest(run['selection']['coordinates']), '$.verification.coordinates',
             'coordinates differ from the actual saved candidate')
    # get_spec('run') already validates run dependencies/result through canonical
    # candidate services. Reconstruct its exact frozen bundle, with its saved split.
    bundle = _at('$.verification', candidate_bundle, docs['experiment'], verify['coordinates'],
                 workspace.get_spec, saved_manifest=run['resolved']['manifest'])
    _require(bundle['protocol']['budget'] ==
             {'epochs': None, 'updates': verify['budget']['updates'], 'examples': None, 'seconds': None},
             '$.verification.budget', 'saved protocol must be updates-only and match the declared budget')
    mid = pack['architectures'][0]['id']
    models = workspace.models()
    _require(mid in models, '$.architectures[0].id', 'architecture is absent from the editable workspace')
    family, model = docs['family'], docs['model']
    _require(family['root'] == mid and set(family['models']) == {mid} and
             family['symbols'] == family['assignments'] == family['repeats'] == [],
             '$.references.family', 'expected the fixed, single-architecture MS-VAR family')
    _require(family['runtime_policy'] ==
             {'version': 2, 'kind': 'runtime-temporal-v2', 'inputs': {'observations': [0]}},
             '$.references.family', 'only observed time is variable; history and state dimensions are fixed')
    expected_graph = graph_signature(document['graph'])
    for where, actual in (('$.architectures[0].recipe.bindings', models[mid]),
                          ('$.references.family', family['models'][mid]),
                          ('$.references.model', model['models'][model['root']])):
        _require(graph_signature(actual['graph']) == expected_graph, where,
                 'graph meaning differs from the typed recipe (including conditioning/variance/ports)')
    data, task = docs['dataset'], docs['task']
    _require({f['id']: (f['dtype'], f['shape']) for f in data['fields']} ==
             {'observations': ('float32', [options['T'], options['D']]),
              'history': ('float32', [options['p'], options['D']]),
              'lengths': ('float32', [1]), 'trajectory': ('string', [])} and
             len(data['records']) == 24, '$.references.dataset',
             'expected 24 unlabelled observed trajectories with explicit history and lengths')
    _require(task['version'] == 2 and task['inputs'] ==
             [{'port': p, 'field': p, 'transforms': []} for p in ('observations', 'history', 'lengths')] and
             task['heads'] == [{'id': 'sequence-nll', 'port': 'out_sequence_evidence',
                                'loss': 'negative-log-likelihood', 'weight': 1.}],
             '$.references.task', 'requires conditional sequence NLL with observed history; no regime targets')
    attempt = run['attempts'][-1]
    _require(attempt['status'] == 'completed' and attempt['result'] is not None,
             '$.references.run', 'a real completed saved fit is required')
    evidence = attempt['result']
    expected = verify['expected']
    _require(digest(evidence['checkpoint']) == evidence['checkpoint_id'] == expected['checkpoint_sha256'],
             '$.verification.expected.checkpoint_sha256', 'does not match the actual saved checkpoint')
    _require(digest(evidence['final_metrics']) == expected['metrics_sha256'],
             '$.verification.expected.metrics_sha256', 'does not match complete saved metrics/populations')
    _require(evidence['counts']['completed_updates'] == expected['completed_updates'],
             '$.verification.expected.completed_updates', 'differs from actual saved update count')
    _require(_at('$.workspace_sha256', _file_hash, workspace.path) == actual_hash,
             '$.workspace_sha256', 'source file changed during verification')
    summary = {'version': 1, 'status': 'checked', 'executed': False, 'pack_id': pack['id'],
               'workspace_sha256': actual_hash, 'producer_source_commit': pack['producer_source_commit'],
               'architectures': copy.deepcopy(pack['architectures']), 'references': copy.deepcopy(refs),
               'checkpoint_sha256': evidence['checkpoint_id'], 'metrics_sha256': digest(evidence['final_metrics']),
               'counts': copy.deepcopy(evidence['counts']), 'environment': copy.deepcopy(evidence['environment']),
               'claim': 'Exact conditional filtering/likelihood; finite-budget estimated parameters. '
                        'No regime-label, identifiability, stationarity or calibration claim.'}
    return summary, run


def check(workspace, manifest):
    """Read-only check of a saved Workspace (or path) and JSON dict (or path).

    Return JSON-compatible saved provenance; raise located GraphError on failure.
    This never invokes a worker, Save, history capture, config or numerical code.
    """
    pack, document, options = _manifest(manifest)
    with _workspace(workspace) as source:
        return _inspect(source, pack, document, options)[0]


def replay(workspace, manifest, *, python, timeout, memory_limit_percent=10.):
    """Explicit bounded CPU replay on a private in-memory canonical Workspace.

    Never save either workspace. Compare actual checkpoint/metrics/identity/counts
    exactly, excluding timings and machine-dependent admission/environment data.
    A changed numerical environment may produce a reported mismatch, not success.
    """
    _require(isinstance(python, (str, Path)) and bool(str(python)), 'python',
             'supply an explicit trusted Python executable, outside the manifest')
    _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= MAX_TIMEOUT,
             'timeout', 'explicit finite worker timeout must be in (0, 60] seconds')
    _require(type(memory_limit_percent) in (int, float) and math.isfinite(memory_limit_percent) and
             0 < memory_limit_percent <= 10., 'memory_limit_percent', 'must be in (0, 10]')
    pack, document, options = _manifest(manifest)
    with _workspace(workspace) as source:
        summary, original = _inspect(source, pack, document, options)
        private = Workspace()
        try:
            source.db.backup(private.db)
            private.set_constrained_mode(True)
            execution = execute_candidate(private, pack['references']['experiment'],
                pack['verification']['coordinates'], 'example-replay-' + uuid.uuid4().hex,
                'Explicit twelve-update example-pack verification', python=str(python),
                timeout=timeout, memory_limit_percent=memory_limit_percent,
                saved_manifest=original['resolved']['manifest'])
            attempt = execution['record']['attempts'][-1]
            actual, saved = attempt['result'], original['attempts'][-1]['result']
            comparisons = {key: actual is not None and digest(actual[key]) == digest(saved[key])
                           for key in ('checkpoint', 'checkpoint_id', 'final_metrics', 'identity', 'counts')}
            matched = attempt['status'] == 'completed' and all(comparisons.values())
            _require(_at('$.workspace_sha256', _file_hash, source.path) == summary['workspace_sha256'],
                     '$.workspace_sha256', 'source file changed during replay')
            return {'version': 1, 'status': 'matched' if matched else
                    'mismatch' if attempt['status'] == 'completed' else 'failed',
                    'executed': True, 'verified': matched, 'comparisons': comparisons,
                    'provenance': summary, 'attempt_status': attempt['status'],
                    'error': attempt['error'], 'result': actual,
                    'retention': 'private working copy discarded; source never saved'}
        finally:
            private.db.close()
            if private.history.directory:
                private.history.directory.cleanup()
