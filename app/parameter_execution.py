"""Internal CPU/float32 parameter initialization and freezing proof.

Not an automatic compiler/trainer hook or admitted workspace field. Call only
on a freshly constructed, memory-admitted model, before constructing an optimizer.
No forward is wrapped in no_grad: frozen weights still transmit input gradients.
"""
from .core import GraphError
from .evidence import digest
from .parameter_policy import resolve, _require


def check_receipt(models, mid, receipt):
    """Pure compatibility of initialization provenance, not authenticity of a fit."""
    import re
    from .policy_graph import plan
    from .parameter_policy import _keys
    resolved = plan(models, mid)['resolved']
    _keys(receipt, ('version', 'policy_identity', 'initial_values_identity', 'counts',
                    'addresses', 'runtime', 'training_state'))
    _keys(receipt['counts'], ('stored', 'trainable', 'frozen'))
    _require(all(type(v) is int and v >= 0 for v in receipt['counts'].values()), 'receipt counts must be nonnegative integers.')
    _require(type(receipt['version']) is int and receipt['version'] == 1 and
             receipt['policy_identity'] == resolved['identity'] and receipt['counts'] == resolved['counts'],
             'initialization receipt does not match parameter policy.')
    _require(isinstance(receipt['initial_values_identity'], str) and
             re.fullmatch('[0-9a-f]{64}', receipt['initial_values_identity']) is not None,
             'initial values identity must be a digest.')
    expected = {sid: {'path': owner['path'], 'native_alias_paths': [owner['path']]}
                for sid, owner in resolved['owners'].items()}
    _require(receipt['addresses'] == expected, 'initialization addresses do not match graph owners.')
    _keys(receipt['runtime'], ('library', 'version', 'device', 'dtype', 'random_rule'))
    runtime = receipt['runtime']
    _require(runtime['library'] == 'torch' and isinstance(runtime['version'], str) and
             1 <= len(runtime['version']) <= 100 and runtime['device'] == 'cpu' and runtime['dtype'] == 'float32' and
             runtime['random_rule'] == 'private-generator-explicit-seed-per-owner-v1' and
             receipt['training_state'] == 'initial-parameters-only; not optimizer continuation',
             'unsupported initialization execution provenance.')


def configure(module, schema, policy=None, *, aliases=None):
    """Validate/prepare every unique owner, then atomically apply the policy.

    Paths address registered native Parameters, not arbitrary attributes. Each
    unique native object must have exactly one canonical semantic owner. Native
    aliases of that object are recorded; distinct objects sharing storage refuse.
    A failure restores values and requires_grad flags. Existing gradients refuse,
    rather than silently leaving stale optimizer inputs. No optimizer is created.
    """
    resolved = resolve(schema, policy, aliases=aliases)
    import torch
    _require(isinstance(module, torch.nn.Module), 'execution requires a native module.')
    paths = dict(module.named_parameters(remove_duplicate=False))
    unique = {id(p): p for p in paths.values()}
    bound = {}
    storage = set()
    for sid, owner in resolved['owners'].items():
        path = owner['path']
        _require(path in paths, 'native owner path is missing: ' + path + '.')
        p = paths[path]
        _require(id(p) not in bound, 'one native object cannot have two semantic owners.')
        _require(list(p.shape) == owner['shape'], 'native owner shape mismatch: ' + path + '.')
        _require(type(p) is torch.nn.Parameter and p.device.type == 'cpu' and p.dtype == torch.float32 and
                 p.layout == torch.strided and p.is_contiguous() and p.is_leaf,
                 'execution currently requires contiguous leaf CPU float32 parameters.')
        _require(p.grad is None, 'configure only before gradients/optimizer use.')
        _require(bool(torch.isfinite(p).all()), 'constructor parameter values must be finite.')
        pointer = p.untyped_storage().data_ptr()
        _require(pointer not in storage, 'distinct parameter objects cannot share storage.')
        storage.add(pointer)
        bound[id(p)] = sid
    _require(set(bound) == set(unique), 'schema must cover every unique native parameter.')

    prepared, old, address_records = {}, {}, {}
    for sid, owner in resolved['owners'].items():
        p = paths[owner['path']]
        old[sid] = (p.detach().clone(), p.requires_grad)
        init = owner['initializer']; kind = init['kind']
        if kind == 'constructor':
            value = p.detach().clone()
        elif kind == 'values':
            value = torch.tensor(init['values'], dtype=torch.float32, device='cpu')
        elif kind == 'zeros':
            value = torch.zeros_like(p)
        else:
            # A private per-owner generator does not consume global RNG. The
            # explicit seed is used literally; equal seeds/shapes may match.
            generator = torch.Generator(device='cpu').manual_seed(init['seed'])
            value = torch.empty_like(p)
            if kind == 'normal':
                rounded_std = torch.tensor(init['std'], dtype=torch.float32).item()
                _require(rounded_std > 0, 'normal std must remain positive in float32.')
                value.normal_(init['mean'], init['std'], generator=generator)
            else:
                rounded = torch.tensor([init['low'], init['high']], dtype=torch.float32)
                _require(rounded[0] < rounded[1], 'uniform bounds must remain ordered in float32.')
                value.uniform_(init['low'], init['high'], generator=generator)
        _require(list(value.shape) == owner['shape'] and bool(torch.isfinite(value).all()),
                 'materialized initializer is nonfinite or has the wrong shape.')
        prepared[sid] = value
        address_records[sid] = {'path': owner['path'], 'native_alias_paths':
                               sorted(path for path, q in paths.items() if q is p)}

    # Receipt preparation can fail too; do it before mutating any parameter.
    values = {sid: prepared[sid].tolist() for sid in sorted(prepared)}
    receipt = {'version': 1, 'policy_identity': resolved['identity'],
               'initial_values_identity': digest({'dtype': 'float32', 'owners': values}),
               'counts': dict(resolved['counts']), 'addresses': address_records,
               'runtime': {'library': 'torch', 'version': str(torch.__version__),
                           'device': 'cpu', 'dtype': 'float32',
                           'random_rule': 'private-generator-explicit-seed-per-owner-v1'},
               'training_state': 'initial-parameters-only; not optimizer continuation'}
    try:
        with torch.no_grad():
            for sid, owner in resolved['owners'].items():
                p = paths[owner['path']]
                p.copy_(prepared[sid]); p.requires_grad_(owner['trainable'])
    except Exception as error:
        with torch.no_grad():
            for sid, owner in resolved['owners'].items():
                p = paths[owner['path']]; value, flag = old[sid]
                p.copy_(value); p.requires_grad_(flag)
        raise GraphError('Parameter policy: atomic application failed.') from error
    return receipt


def trainable_parameters(module):
    """Unique selected native owners; no optimizer or training step is created."""
    parameters = [p for p in module.parameters() if p.requires_grad]
    _require(bool(parameters), 'no trainable parameter owners; inference remains allowed.')
    return parameters
