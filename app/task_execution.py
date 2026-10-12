"""Frozen CPU task execution contracts; importing this module allocates no tensors.

This is the execution component of Stage C; experiment_objects handles independent
terminal storage. Legacy identity-preprocessing checkpoints are separate.
"""
import copy
import ast
import inspect
import math
import types

from .core import GraphError, generate_code, validate_model
from .datasets import (_apply_step, _column, _flat, _integer, _keys, _number,
                       _require, _unflatten, _value, _version, prepare)
from .evidence import digest, parameter_schema
from .memory import available_memory, estimate_memory, memory_percent
from .revisions import frozen_document
from .task_runtime import forward_ports
from .runtime_tasks import record_shapes


def training_protocol(*, epochs=20, updates=None, examples=None, seconds=None,
                      batch_size=16, lr=.03, initialization_seed=7, shuffle_seed=11,
                      shuffle=True, evaluate_every=10, history_limit=200):
    """Explicit defaults are captured, not inferred from mutable app settings."""
    result = {'version': 1, 'backend': 'cpu-float32-v1', 'batch_size': batch_size,
              'budget': {'epochs': epochs, 'updates': updates, 'examples': examples, 'seconds': seconds},
              'optimizer': {'kind': 'Adam', 'lr': lr, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.},
              'seeds': {'initialization': initialization_seed, 'shuffle': shuffle_seed},
              'shuffle': shuffle, 'evaluate_every': evaluate_every, 'augmentation': 'none',
              'history': {'kind': 'latest-window', 'max_points': history_limit},
              'checkpoint_selection': 'last-successful-update', 'test_evaluation': 'final-completed-only'}
    return validate_protocol(result)


def validate_protocol(protocol):
    _keys(protocol, ('version', 'backend', 'batch_size', 'budget', 'optimizer', 'seeds',
                     'shuffle', 'evaluate_every', 'history', 'augmentation', 'checkpoint_selection', 'test_evaluation'), context='TrainingProtocol')
    _version(protocol, 'TrainingProtocol')
    _require(protocol['backend'] == 'cpu-float32-v1' and protocol['augmentation'] == 'none' and
             protocol['checkpoint_selection'] == 'last-successful-update' and
             protocol['test_evaluation'] == 'final-completed-only', 'Unsupported task execution policy.')
    _require(_integer(protocol['batch_size'], 1) and _integer(protocol['evaluate_every'], 1) and
             type(protocol['shuffle']) is bool, 'Batch/evaluation sizes must be positive integers; shuffle must be boolean.')
    _keys(protocol['history'], ('kind', 'max_points'), context='Curve retention')
    _require(protocol['history']['kind'] == 'latest-window' and _integer(protocol['history']['max_points'], 1),
             'Curve retention requires a positive explicit latest-window size.')
    budget = protocol['budget']
    _keys(budget, ('epochs', 'updates', 'examples', 'seconds'), context='Training budget')
    _require(any(v is not None for v in budget.values()), 'Declare at least one finite training budget.')
    for key in ('epochs', 'updates', 'examples'):
        _require(budget[key] is None or _integer(budget[key], 1), 'Budget counters must be positive integers or null.')
    _require(budget['seconds'] is None or (_number(budget['seconds']) and budget['seconds'] > 0), 'Time budget must be positive finite seconds or null.')
    _keys(protocol['seeds'], ('initialization', 'shuffle'), context='Training seeds')
    _require(all(_integer(v, 0, 2**31 - 1) for v in protocol['seeds'].values()), 'Independent seeds must be integers in 0..2^31-1.')
    opt = protocol['optimizer']
    _keys(opt, ('kind', 'lr', 'betas', 'eps', 'weight_decay'), context='Optimizer')
    _require(opt['kind'] == 'Adam' and _number(opt['lr']) and 0 < opt['lr'] <= 1 and
             isinstance(opt['betas'], list) and len(opt['betas']) == 2 and
             all(_number(v) and 0 <= v < 1 for v in opt['betas']) and
             _number(opt['eps']) and opt['eps'] > 0 and _number(opt['weight_decay']) and opt['weight_decay'] >= 0,
             'Adam settings must be explicit, finite and valid.')
    return copy.deepcopy(protocol)


def task_bundle(model, dataset, task, manifest, protocol, *, runtime=None):
    """Freeze the actual runnable closure/data/task, not just mutable IDs."""
    result = {'version': 1, 'model': frozen_document('model', model),
              'dataset': frozen_document('dataset', dataset), 'task': frozen_document('task', task),
              'manifest': copy.deepcopy(manifest), 'protocol': validate_protocol(protocol)}
    if runtime is not None:
        sv=runtime['specialization'].get('version')
        result.update(version=4 if sv in (7,8,9) else 3 if sv==5 else 2,runtime=copy.deepcopy(runtime))
    validate_bundle(result)
    return result


def validate_bundle(bundle):
    _require(isinstance(bundle,dict),'TaskBundle must be a closed object.')
    _keys(bundle, ('version', 'model', 'dataset', 'task', 'manifest', 'protocol')+
          (('runtime',) if bundle.get('version') in (2,3,4) else ()), context='TaskBundle')
    _require(type(bundle['version']) is int and bundle['version'] in (1,2,3,4),'TaskBundle supports integer versions1–4.')
    for kind in ('model', 'dataset', 'task'):
        _require('revision' in bundle[kind], 'Execution requires exact frozen ' + kind + ' revision.')
        frozen_document(kind, bundle[kind])
    validate_protocol(bundle['protocol'])
    from .datasets import check_manifest, validate_task
    model = bundle['model']; contract = validate_model(model['models'], model['root'])
    if bundle['version'] in (2,3,4):
        from .runtime_tasks import contract as runtime_contract
        contract=runtime_contract(model,bundle['dataset'],bundle['task'],bundle['runtime'],
                                  execution='runtime-temporal-v2' if bundle['version']==4 else
                                  'runtime-temporal-v1' if bundle['version']==3 else 'runtime-spatial-v1')
    validate_task(bundle['task'], bundle['dataset'], contract)
    check_manifest(bundle['dataset'], bundle['manifest'])
    _require(contract['parameters'] > 0, 'Task training requires trainable parameters.')
    from .policy_graph import present, plan
    if present(model['models'], model['root']):
        _require(plan(model['models'], model['root'])['resolved']['counts']['trainable'] > 0,
                 'Task training has no trainable parameter owners; inference remains allowed.')
    return contract


def run_identity(bundle):
    validate_bundle(bundle)
    result={'version': 1, **{kind: {'kind': kind, 'id': bundle[kind]['id'], 'revision': bundle[kind]['revision']}
                            for kind in ('model', 'dataset', 'task')},
            'manifest': digest(bundle['manifest']), 'protocol': digest(bundle['protocol']),
            'compiler': 'native-graph-v1', 'task_runtime': 'masked-additive-v1'}
    if bundle['version']==2:
        result.update(version=2,compiler='native-runtime-spatial-v1',runtime=digest(bundle['runtime']))
    elif bundle['version']==3:
        result.update(version=3,compiler='native-runtime-temporal-v1',runtime=digest(bundle['runtime']))
    elif bundle['version']==4:
        result.update(version=4,compiler='native-runtime-temporal-v2',runtime=digest(bundle['runtime']))
    if bundle['model']['version'] >= 6:
        from .policy_graph import plan
        result['parameter_policy'] = plan(bundle['model']['models'], bundle['model']['root'])['resolved']['identity']
        result['compiler'] += '-parameter-policy-v1'
    return result


def compile_task_model(bundle, *, constrained=True, memory_limit_percent=10.):
    """Use one compiler path for workers and restored task predictors."""
    contract = validate_bundle(bundle)
    from .core import compile_model
    spec=bundle['model'];decl=None
    if bundle['version'] in (2,3,4):decl=bundle['runtime']['specialization']['runtime']['inputs']
    policy=({'runtime_temporal_v2':decl} if bundle['version']==4 else
            {'runtime_temporal':decl} if bundle['version']==3 else {'runtime_inputs':decl})
    return compile_model(spec['models'],spec['root'],constrained,
                         memory_limit_percent=memory_limit_percent,**policy,
                         initial_shapes={g['id']:g['shape'] for g in contract['input_groups']} if decl is not None else None)


def resolve_run(bundle, *, constrained=True, memory_limit_percent=10., available_bytes=None):
    """Collation plus combined model/optimizer/evidence RAM admission, before Torch."""
    contract = validate_bundle(bundle)
    _require(type(constrained) is bool, 'Constrained mode must be boolean.')
    try:
        percentage = memory_percent(memory_limit_percent)
    except ValueError as error:
        raise GraphError(str(error)) from error
    available = available_memory() if available_bytes is None else available_bytes
    prepared = prepare(bundle['dataset'], bundle['task'], contract, bundle['manifest'],
                       constrained=constrained, memory_limit_percent=percentage, available_bytes=available)
    # Legacy estimator uses 80% of samples as its training batch. Supply a ceil
    # inverse so its activation/workspace term covers the FULL actual minibatch.
    size = min(bundle['protocol']['batch_size'], len(prepared['populations']['train']['observation_ids']))
    memory = estimate_memory(contract, samples=math.ceil(size / .8), training=True)
    memory['components']['typed_preparation'] = prepared['preparation_estimate']['estimated_bytes']
    from .policy_graph import present, plan
    model_spec = bundle['model']
    if present(model_spec['models'], model_spec['root']):
        policy = plan(model_spec['models'], model_spec['root'])['resolved']
        memory['components']['parameter_policy_initialization'] = 256 * policy['counts']['stored'] + 4096 * len(policy['owners'])
        memory['parameter_counts'] = policy['counts']
    # Curves carry exact evaluated IDs per head/checkpoint. Charge their bounded
    # retention too: an enormous requested curve must not evade run admission.
    point_rows = sum(len(prepared['populations'][r]['observation_ids']) for r in ('train', 'validation'))
    memory['components']['metric_history'] = bundle['protocol']['history']['max_points'] * (
        8192 + 512 * point_rows * len(bundle['task']['heads']))
    if bundle['version'] in (3,4):
        if bundle['version']==4:
            from .temporal_shapes_v2 import admission as temporal_admission
        else:
            from .temporal_shapes import admission as temporal_admission
        model=bundle['model']
        extra=temporal_admission(model['models'],model['root'],contract,size,constrained=False,
                                 memory_limit_percent=percentage,gradients=True)
        memory['components']['recurrent_unrolls']=extra['memory']['components']['recurrent_unrolls']
        if bundle['version']==4:
            # Whole-tree v2 owners/gates are ADDITIONAL to collation, outputs,
            # metric history and head supervision, never a replacement for them.
            memory['components']['parameters_optimizer_evidence']=max(
                memory['components']['parameters_optimizer_evidence'],extra['memory']['components']['parameters'])
            memory['components']['lstm_unrolls']=extra['memory']['components']['lstm_unrolls']
    memory['estimated_bytes'] = sum(memory['components'].values())
    memory.update(actual_batch_size=size, limit_percent=percentage, available_bytes=available)
    if constrained:
        _require(_integer(available), 'Cannot observe available RAM for task training.')
        _require(memory['estimated_bytes'] <= available * percentage / 100., 'Task training exceeds the constrained available-RAM budget.')
    return prepared, memory


def checkpoint_for(bundle, fitted, model):
    """No optimizer-resume claim; inference state includes the fitted transform."""
    spec = bundle['model']; schema = parameter_schema(spec['models'], spec['root'])
    result = {'version': 2 if hasattr(model, 'parameter_policy_receipt') else 1,
            'kind': 'task-predictor', 'identity': run_identity(bundle),
            'fitted': copy.deepcopy(fitted), 'parameter_ids': {k: v['id'] for k, v in schema.items()},
            'weights': {key: value.detach().tolist() for key, value in model.state_dict().items()}}
    if hasattr(model, 'parameter_policy_receipt'):
        result['parameter_policy_receipt'] = copy.deepcopy(model.parameter_policy_receipt)
    return result


def check_task_checkpoint(bundle, checkpoint, *, constrained=True, memory_limit_percent=10.):
    """Pure structural/identity validation; not proof of training origin."""
    from .policy_graph import present
    policy = present(bundle['model']['models'], bundle['model']['root'])
    _keys(checkpoint, ('version', 'kind', 'identity', 'fitted', 'parameter_ids', 'weights') +
          (('parameter_policy_receipt',) if policy else ()), context='Task checkpoint')
    _require(type(checkpoint['version']) is int and checkpoint['version'] == (2 if policy else 1),
             'Task checkpoint version must match its parameter policy.')
    if policy:
        from .parameter_execution import check_receipt
        check_receipt(bundle['model']['models'], bundle['model']['root'], checkpoint['parameter_policy_receipt'])
    _require(checkpoint['kind'] == 'task-predictor' and digest(checkpoint['identity']) == digest(run_identity(bundle)), 'Task checkpoint identity mismatch.')
    prepared, _ = resolve_run(bundle, constrained=constrained, memory_limit_percent=memory_limit_percent)
    _require(digest(checkpoint['fitted']) == digest(prepared['fitted']), 'Task checkpoint fitted preprocessing mismatch.')
    spec = bundle['model']; schema = parameter_schema(spec['models'], spec['root'])
    _require(checkpoint['parameter_ids'] == {k: v['id'] for k, v in schema.items()}, 'Task checkpoint stable parameter IDs mismatch.')
    _require(isinstance(checkpoint['weights'], dict) and set(checkpoint['weights']) == set(schema), 'Task checkpoint parameter keys mismatch.')
    def tensor(value, shape):
        if shape:
            _require(isinstance(value, list) and len(value) == shape[0], 'Task checkpoint tensor shape mismatch.')
            for item in value:
                tensor(item, shape[1:])
        else:
            _require(_number(value), 'Task checkpoint needs finite float32 weights.')
    for key, info in schema.items():
        tensor(checkpoint['weights'][key], info['shape'])
    return prepared


def fitted_inputs(bundle, fitted, records):
    """Apply captured states to NEW field-valued observations; never fit here."""
    fields = {f['id']: f for f in bundle['dataset']['fields']}
    _require(isinstance(records, list), 'Inference records must be a list of field-value objects.')
    result = {}
    for binding in bundle['task']['inputs']:
        fid = binding['field']; field = fields[fid]; state = fitted['inputs'][binding['port']]
        _require(all(isinstance(r, dict) and fid in r for r in records), 'Inference record is missing input field ' + fid + '.')
        values = [r[fid] for r in records]
        actual=None
        if 'runtime_policy' in bundle and binding['port'] in bundle['runtime_policy'].get('runtime_temporal',bundle['runtime_policy'].get('runtime_inputs',{})):
            actual=record_shapes(bundle,records)[binding['port']]
        if actual is not None and bundle['runtime_policy'].get('kind') in ('runtime-temporal-v1','runtime-temporal-v2') and binding['transforms']:
            actual=None  # Validate raw field shape; captured transforms supply the fitted execution shape.
        for value in values:
            _value(value,field if actual is None else {**field,'shape':actual})
        if 'derived' in binding:
            length = binding['length']
            _require(all(isinstance(v, list) and len(v) <= length for v in values), 'Derived inference input cannot truncate/miss sequences.')
            values = [[float(len(v))] if binding['derived'] == 'length' else [float(i < len(v)) for i in range(length)] for v in values]
        else:
            shape = list(field['shape'] if actual is None else actual)
            for transform, step in zip(binding['transforms'], state['steps']):
                values, shape, _ = _apply_step(values, shape, field, transform, step)
            width = math.prod(shape)
            rows = _column(values, width)
            _require(all(_number(v) or type(v) is bool for row in rows for v in row), 'Inference inputs require explicit missing-value handling.')
            values = [_unflatten([float(v) for v in row], shape or [1]) for row in rows]
        result[binding['port']] = values
    return result


def restore_task_predictor(bundle, checkpoint, *, constrained=True, memory_limit_percent=10.):
    check_task_checkpoint(bundle, checkpoint, constrained=constrained, memory_limit_percent=memory_limit_percent)
    import torch
    bundle = copy.deepcopy(bundle); checkpoint = copy.deepcopy(checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = compile_task_model(bundle,constrained=constrained,memory_limit_percent=memory_limit_percent)
    model.load_state_dict({k: torch.tensor(v, dtype=torch.float32, device='cpu') for k, v in checkpoint['weights'].items()}, strict=True)
    model.eval()
    if bundle['version'] in (2,3,4):
        from .runtime_tasks import input_policy
        bundle['runtime_policy']=input_policy(bundle)
    class Predictor:
        """Returns stable Output IDs. Regression stays in declared target space;
        classification returns LOGITS, not secretly softmaxed probabilities.
        """
        def __call__(self, records):
            return _predict_records(model, bundle, checkpoint['fitted'], records)
    return Predictor()


def _predict_records(model, input_spec, fitted, records):
    import torch
    contract=model.contract
    if 'runtime_policy' in input_spec:
        shapes=record_shapes(input_spec,records)
        contract=model.preview_shapes(shapes,batch_size=len(records),gradients=False)['contract']
    from_values = fitted_inputs(input_spec, fitted, records)
    n = len(records)
    inputs = {g['id']: torch.tensor(from_values[g['id']], dtype=torch.float32, device='cpu').reshape(n, *g['shape'])
              for g in contract['input_groups']}
    with torch.no_grad():
        result = forward_ports(model, inputs, contract)
    _require(all(bool(torch.isfinite(v).all()) for v in result.values()), 'Inference arithmetic produced nonfinite outputs.')
    return result


def _helper_source(helper):
    """Never lower changed on-disk code as if it were the loaded implementation."""
    raw=inspect.getsource(helper);tree=ast.parse(raw)
    _require(len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name==helper.__name__,
             'Predictor helper source changed since import; restart before exporting.')
    # Compile in its whole-module context: Python's optimizer can generate
    # different instructions for an otherwise identical isolated definition.
    module_source=inspect.getsource(inspect.getmodule(helper))
    compiled=compile(module_source,'<predictor-helper-module>','exec',dont_inherit=True)
    code=next((c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name==helper.__name__),None)
    def meaning(c):
        return (c.co_code,c.co_flags,getattr(c,'co_exceptiontable',b''),c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,
                c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,
                tuple(meaning(v) if isinstance(v,types.CodeType) else v for v in c.co_consts))
    _require(code is not None and meaning(code)==meaning(helper.__code__),
             'Predictor helper implementation changed since import; restart before exporting.')
    return raw


def generate_task_architecture(bundle, *, constrained=True, memory_limit_percent=10.):
    """The exact execution compiler, distinct from the constructor-baseline export."""
    contract=validate_bundle(bundle)
    if bundle['version']==1:
        return generate_code(bundle['model']['models'],bundle['model']['root'])
    from .family_export import family_source
    architecture=family_source(bundle['runtime']['family'],constrained=constrained,
                               memory_limit_percent=memory_limit_percent)['code']
    shapes={g['id']:g['shape'] for g in contract['input_groups']}
    return architecture+'\nmodel = ArchitectureFamily(bindings='+repr(bundle['runtime']['specialization']['bindings'])+', initial_shapes='+repr(shapes)+').network\n'


def generate_task_predictor(bundle, checkpoint, *, constrained=True, memory_limit_percent=10.):
    """One standalone Python source: Torch only, no raw training observations.

    Reuse the executable graph and deterministic preprocessing lowering rather
    than maintaining a visually similar but scientifically different exporter.
    """
    check_task_checkpoint(bundle, checkpoint, constrained=constrained, memory_limit_percent=memory_limit_percent)
    helpers = (_require, _integer, _number, _flat, _unflatten, _value, _column, _apply_step, record_shapes,
               fitted_inputs, forward_ports, _predict_records)
    architecture=generate_task_architecture(bundle,constrained=constrained,memory_limit_percent=memory_limit_percent)
    source = ['# Visual PyTorch trained task inference; GPL-3.0-only. No training occurs.',
              'import copy', 'import math', 'MAX_FLOAT32 = 3.4028234663852886e38','MAX_INTEGER = 2**53 - 1',
              'class GraphError(ValueError):\n    pass', architecture]
    source.extend(_helper_source(helper) for helper in helpers)
    fields = {binding['field'] for binding in bundle['task']['inputs']}
    input_spec = {'dataset': {'fields': [f for f in bundle['dataset']['fields'] if f['id'] in fields]},
                  'task': {'inputs': bundle['task']['inputs']}}
    if bundle['version'] in (2,3,4):
        from .runtime_tasks import input_policy
        input_spec['runtime_policy']=input_policy(bundle)
    source.extend([f'_input_spec = {input_spec!r}', f'_fitted = {checkpoint["fitted"]!r}',
                   f'task_specification = {bundle["task"]!r}',
                   f'target_fields = {[f for f in bundle["dataset"]["fields"] if f["id"] in {h["field"] for h in bundle["task"]["heads"]}]!r}',
                   f'_weights = {checkpoint["weights"]!r}', f'checkpoint_identity = {checkpoint["identity"]!r}',
                   "model.load_state_dict({k: torch.tensor(v, dtype=torch.float32, device='cpu') for k,v in _weights.items()}, strict=True)",
                   'model.eval()', 'def predict_records(records):',
                   '    """Field-valued rows -> stable Output IDs; logits/declared target space."""',
                   '    return _predict_records(model, _input_spec, _fitted, records)', ''])
    return '\n'.join(source)
