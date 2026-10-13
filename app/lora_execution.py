"""Explicit native LoRA comparisons; baseline evaluation is NOT a training run.

Importing this module is pure. Numerical imports happen only when evaluation or
comparisons are explicitly executed. Results are descriptive synthetic evidence,
not statistical claims, pretrained language models, or optimizer continuation.
"""
import copy
import math
import platform
import sys

from .core import preflight, validate_model
from .datasets import _integer, _require, prepare
from .evidence import digest, parameter_schema
from .experiment_objects import execute_candidate, preview_experiment, reference
from .experiment_ui import result_report
from .parameter_sources import float32_values, inventory
from .policy_graph import owner_key, plan
from .revisions import frozen_document, model_document
from .task_result import _metrics


GROUPS = (('linear', 'lora-linear-comparison', 'base-linear'),
          ('mlp', 'lora-mlp-comparison', 'mlp-base'))
OBJECTIVE_RULE = 'sum-head-weight-times-available-head-mean; no weight renormalization'


def _reference(kind, doc):
    return {'kind': kind, 'id': doc['id'], 'revision': doc['revision']}


def _native_initial(model, *, seed, constrained, memory_limit_percent):
    """Compile actual initial values with private RNG, before any optimizer."""
    from .core import compile_model
    import torch
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        native = compile_model(model['models'], model['root'], constrained,
                               memory_limit_percent=memory_limit_percent)
    native.eval()
    schema = parameter_schema(model['models'], model['root'])
    snapshot = {'version': 1, 'kind': 'lora-native-initial-snapshot',
                'model': _reference('model', model), 'model_snapshot': copy.deepcopy(model),
                'dtype': 'float32', 'device': 'cpu', 'initialization_seed': seed, 'schema': schema,
                'weights': {key: value.detach().tolist() for key, value in native.state_dict().items()},
                'parameter_policy_receipt': copy.deepcopy(native.parameter_policy_receipt)}
    snapshot['sha256'] = digest(snapshot)
    return native, snapshot


def _base_hashes(snapshot):
    """Compare actual W0 bytes by relative structural owner, not native indices."""
    import json
    result = {}
    for path, row in snapshot['schema'].items():
        parts = json.loads(row['id'])
        if parts[-2:] == ['base', 'weight']:
            result[owner_key(parts[1:])] = float32_values(snapshot['weights'][path], row['shape'])[1]
    _require(bool(result), 'Teaching comparison has no supplied base owners.')
    return result


def evaluate_baseline(model, dataset, task, manifest, *, batch_size=24, seed=7,
                      constrained=True, memory_limit_percent=10.):
    """Evaluate an exact all-frozen graph on saved populations, without training.

    The returned JSON is an evaluation artifact, not an admitted TaskCheckpoint,
    TaskBundle, TaskTrainingResult or RunRecord. Targets come from the captured
    dataset; predictions come exclusively from the actual native graph.
    """
    _require(_integer(batch_size, 1) and _integer(seed, 0, 2**31 - 1), 'Invalid evaluation batch/seed.')
    model = frozen_document('model', model)
    dataset = frozen_document('dataset', dataset); task = frozen_document('task', task)
    counts = plan(model['models'], model['root'])['resolved']['counts']
    _require(counts['trainable'] == 0, 'Baseline must have all parameter owners frozen.')
    contract = validate_model(model['models'], model['root'])
    prepared = prepare(dataset, task, contract, manifest, constrained=constrained,
                       memory_limit_percent=memory_limit_percent)
    admission = preflight(model['models'], model['root'], constrained, samples=batch_size,
                          memory_limit_percent=memory_limit_percent)
    native, initial = _native_initial(model, seed=seed, constrained=constrained,
                                      memory_limit_percent=memory_limit_percent)
    import torch
    from .task_runtime import MetricAccumulator, forward_ports, head_terms, tensor_batch
    checkpoint = initial['sha256']; metrics = {}
    with torch.no_grad():
        for role, population in prepared['populations'].items():
            accumulators = {h['id']: MetricAccumulator(h['loss'], checkpoint, role) for h in task['heads']}
            for start in range(0, len(population['observation_ids']), batch_size):
                indices = list(range(start, min(start + batch_size, len(population['observation_ids']))))
                batch = tensor_batch(prepared, task, role, indices)
                predictions = forward_ports(native, batch['inputs'], contract)
                for head in task['heads']:
                    hid = head['id']
                    terms = head_terms(predictions[head['port']], batch['targets'][hid],
                                       batch['masks'][hid], batch['sample_weights'][hid], head['loss'])
                    accumulators[hid].add(terms, batch['observation_ids'], checkpoint=checkpoint, population=role)
            heads = {hid: acc.report() for hid, acc in accumulators.items()}
            active = [h['weight'] * heads[h['id']]['value'] for h in task['heads']
                      if heads[h['id']]['status'] == 'available']
            metrics[role] = {'heads': heads, 'objective': math.fsum(active) if active else None,
                             'objective_status': 'available' if active else 'unavailable',
                             'objective_rule': OBJECTIVE_RULE}
    _metrics(metrics, prepared, task, tuple(prepared['populations']), checkpoint)
    _require(digest(initial['weights']) == digest({k: v.detach().tolist() for k, v in native.state_dict().items()}),
             'Evaluation changed initial weights.')
    return {'version': 1, 'kind': 'lora-baseline-evaluation', 'status': 'evaluated',
            'scope': 'native-evaluation-only; not-a-training-run', 'training': False,
            'completed_updates': 0, 'optimizer_created': False, 'optimizer_resume': False,
            'model': _reference('model', model), 'dataset': _reference('dataset', dataset),
            'task': _reference('task', task), 'dataset_snapshot': dataset, 'task_snapshot': task,
            'manifest': copy.deepcopy(manifest), 'manifest_sha256': digest(manifest),
            'fitted': copy.deepcopy(prepared['fitted']), 'fitted_sha256': digest(prepared['fitted']),
            'initial_snapshot': initial, 'checkpoint_id': checkpoint, 'parameter_counts': counts,
            'metrics': metrics, 'batch_size': batch_size, 'initialization_seed': seed,
            'preparation_admission': prepared['preparation_estimate'],
            'inference_admission': admission, 'environment': {'python': sys.version.split()[0],
                'torch': str(torch.__version__), 'platform': platform.platform(),
                'device': 'cpu', 'dtype': 'float32', 'numerical_threads': 1}}


def run_comparisons(workspace, experiments, *, python=sys.executable, timeout=120.,
                    memory_limit_percent=10.):
    """Run exactly four linear and two MLP candidates, retaining genuine evidence.

    Call populate first. Exact manifests/protocols and native initial W0 bytes
    are checked per comparison. Failed attempts remain real retained attempts;
    an incomplete comparison is never relabelled completed.
    """
    groups = {}; rank2 = None
    for label, sid, baseline_id in GROUPS:
        ref = experiments[sid]; reference(ref, 'experiment')
        experiment = workspace.get_spec('experiment', ref['id'], ref['revision'])
        rows = preview_experiment(experiment, workspace.get_spec, limit=25)['candidates']
        expected = {'mlp-full', 'mlp-adapter'} if label == 'mlp' else {'full-finetune', 'lora-r1', 'lora-r2', 'lora-r4'}
        _require(len(rows) == len(expected) and {r['coordinates']['model']['id'] for r in rows} == expected and
                 all(r['status'] == 'compatible' for r in rows), 'Unexpected teaching comparison candidates.')
        first = rows[0]; manifest = first['manifest']; protocol = first['protocol']
        _require(all(digest(r['manifest']) == digest(manifest) and digest(r['protocol']) == digest(protocol) and
                     r['references']['dataset'] == first['references']['dataset'] and
                     r['references']['task'] == first['references']['task'] for r in rows),
                 'Comparison must use identical frozen data/task/split/seeds/protocol.')
        dataref = first['references']['dataset']; taskref = first['references']['task']
        dataset = workspace.get_spec('dataset', dataref['id'], dataref['revision'])
        task = workspace.get_spec('task', taskref['id'], taskref['revision'])
        baseline_model = frozen_document('model', model_document(workspace.models(), baseline_id))
        workspace.put_spec('model', baseline_model)
        baseline = evaluate_baseline(baseline_model, dataset, task, manifest,
            batch_size=protocol['batch_size'], seed=protocol['seeds']['initialization'],
            constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)
        base_hashes = _base_hashes(baseline['initial_snapshot']); runs = []
        for row in rows:
            modelref = row['references']['model']; mid = modelref['id']
            model = workspace.get_spec('model', mid, modelref['revision'])
            prepared = prepare(dataset, task, validate_model(model['models'], model['root']), manifest,
                constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)
            _require(digest(prepared['fitted']) == baseline['fitted_sha256'], 'Comparison preprocessing differs.')
            _, initial = _native_initial(model, seed=protocol['seeds']['initialization'],
                constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)
            _require(_base_hashes(initial) == base_hashes, 'Native supplied W0 values differ across comparison.')
            trial = execute_candidate(workspace, ref, row['coordinates'], sid + '-' + mid + '-run',
                'Synthetic comparison · ' + model['name'], python=python, timeout=timeout,
                saved_manifest=manifest, memory_limit_percent=memory_limit_percent)
            record = trial['record']; attempt = record['attempts'][0]; result = attempt['result']
            _require(record['resolved']['manifest'] == manifest and record['resolved']['protocol'] == protocol,
                     'Worker retained a different split or protocol.')
            if result is not None:
                _require(digest(result['checkpoint']['fitted']) == baseline['fitted_sha256'] and
                         result['checkpoint']['parameter_policy_receipt'] == initial['parameter_policy_receipt'],
                         'Worker initialization/preprocessing differs from captured native snapshot.')
                roles = tuple(result['final_metrics'])
                if roles: _metrics(result['final_metrics'], prepared, task, roles, result['checkpoint_id'])
                for role in roles:
                    for head in task['heads']:
                        actual = result['final_metrics'][role]['heads'][head['id']]
                        oracle = baseline['metrics'][role]['heads'][head['id']]
                        _require(all(actual[k] == oracle[k] for k in ('observation_ids', 'denominator', 'elements',
                            'included_elements', 'reduction', 'accumulation_dtype')), 'Evaluation populations/reductions differ.')
                if mid != 'full-finetune' and mid != 'mlp-full':
                    final = {**initial, 'weights': result['checkpoint']['weights']}
                    _require(_base_hashes(final) == base_hashes, 'Adapter training changed frozen W0.')
            item = {'model': modelref, 'run': trial['reference'], 'attempt': 1, 'status': attempt['status'],
                    'error': copy.deepcopy(attempt['error']), 'initial_snapshot': initial,
                    'checkpoint_id': result['checkpoint_id'] if result else None,
                    'counts': copy.deepcopy(result['counts']) if result else None,
                    'metrics': copy.deepcopy(result['final_metrics']) if result else {}}
            runs.append(item)
            if mid == 'lora-r2': rank2 = {'run': trial['reference'], 'attempt': 1, 'status': attempt['status']}
        report = result_report(workspace, [r['run'] for r in runs], sid=sid + '-results',
                               name=experiment['name'] + ' · actual runs', question=experiment['question'])
        complete = all(r['status'] == 'completed' for r in runs)
        groups[label] = {'experiment': ref, 'baseline': baseline, 'runs': runs, 'report': report,
            'complete': complete,
            'invariants': {'dataset': dataref, 'task': taskref, 'manifest_sha256': digest(manifest),
                'fitted_sha256': baseline['fitted_sha256'], 'protocol': copy.deepcopy(protocol),
                'protocol_sha256': digest(protocol), 'native_initial_base_values_sha256': base_hashes,
                'same_populations_and_denominators': True if complete else None, 'same_supplied_native_base': True,
                'scope': 'descriptive fixed synthetic comparison; not a general superiority claim'}}
    return {'version': 1, 'kind': 'lora-comparison-evidence', 'groups': groups, 'rank2': rank2,
            'complete': all(group['complete'] for group in groups.values()),
            'scope': 'six actual worker fits plus two separate zero-update native baseline evaluations'}


def merge_selections(workspace, run, *, attempt):
    """Select base/A/B from ONE exact rank2 checkpoint using actual semantic IDs."""
    import json
    page = inventory(workspace, run, attempt=attempt); selected = {}
    for role in ('base', 'A', 'B'):
        matches = [p for p in page['parameters'] if json.loads(p['id'])[-2:] == [role, 'weight']]
        _require(len(matches) == 1, 'Rank2 merge requires one unambiguous ' + role + ' owner.')
        row = matches[0]
        selected[role] = {'run': copy.deepcopy(page['run']), 'attempt': attempt,
                          'checkpoint_id': page['checkpoint_id'],
                          'parameter': {key: row[key] for key in ('path', 'id')}}
    _require({role: next(p['shape'] for p in page['parameters'] if p['id'] == selected[role]['parameter']['id'])
              for role in selected} == {'base': [16, 16], 'A': [2, 16], 'B': [16, 2]},
             'Merge selection must be the actual rank2 teaching checkpoint.')
    return selected
