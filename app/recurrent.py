"""Explicit-state eager recurrence over one compiled transition graph.

This execution primitive is not yet a persisted diagram operator. Importing it,
checking a specification and estimating RAM do not import Torch or allocate it.
"""
import copy
import math
from pathlib import Path

from .core import GraphError, feature_count, validate_model
from .evidence import identity, parameter_schema, digest
from .memory import available_memory, memory_percent, format_bytes, MIB


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise GraphError(where + ': expected exactly ' + ', '.join(sorted(expected)) + '.')


def _ids(values, where, *, empty=False):
    if not isinstance(values, list) or (not values and not empty) or any(
            not isinstance(v, str) or not v.strip() for v in values) or len(set(values)) != len(values):
        raise GraphError(where + ': use unique, nonempty stable port IDs.')


def interface_contract(c, specification):
    """Pure interface validation over an already checked transition contract."""
    _keys(specification, ('version', 'sequence_inputs', 'states', 'sequence_outputs'), 'Recurrent specification')
    if type(specification['version']) is not int or specification['version'] != 1:
        raise GraphError('Recurrent specification version must be integer 1.')
    spec = copy.deepcopy(specification)
    _ids(spec['sequence_inputs'], 'Sequence inputs')
    if not isinstance(spec['states'], list) or not spec['states']:
        raise GraphError('Declare at least one explicit initial/final state.')
    if not isinstance(spec['sequence_outputs'], list):
        raise GraphError('Sequence outputs must be a list (possibly empty).')
    for state in spec['states']:
        _keys(state, ('input', 'output', 'port'), 'State')
        if any(not isinstance(v, str) or not v.strip() for v in state.values()):
            raise GraphError('State identifiers must be nonempty strings.')
    for output in spec['sequence_outputs']:
        _keys(output, ('output', 'port'), 'Sequence output')
        if any(not isinstance(v, str) or not v.strip() for v in output.values()):
            raise GraphError('Sequence output identifiers must be nonempty strings.')
    _ids([s['input'] for s in spec['states']], 'State inputs')
    _ids([s['output'] for s in spec['states']], 'Next-state outputs')
    _ids([s['port'] for s in spec['states']], 'State ports')
    _ids([s['port'] for s in spec['sequence_outputs']], 'Sequence output ports', empty=True)
    if set(spec['sequence_inputs']) & {s['input'] for s in spec['states']}:
        raise GraphError('A transition Input cannot be both sequence and state.')
    inputs = {g['id']: g for g in c['input_groups']}
    outputs = {g['id']: g for g in c['output_groups']}
    if set(inputs) != set(spec['sequence_inputs']) | {s['input'] for s in spec['states']}:
        raise GraphError('Bind every transition Input exactly once, by stable ID, as sequence or state.')
    for item in spec['states'] + spec['sequence_outputs']:
        if item['output'] not in outputs:
            raise GraphError('Unknown transition Output ID: ' + item['output'] + '.')
    for state in spec['states']:
        before, after = inputs[state['input']], outputs[state['output']]
        if before['shape'] != after['shape'] or before.get('axes') != after.get('axes'):
            raise GraphError('State ' + state['port'] + ': next state must preserve initial-state shape and axes.')
    return {'version': 1, 'specification': spec, 'parameters': c['parameters'],
            'inputs': inputs, 'outputs': outputs,
            'activation_elements': c['activation_elements'],
            'convolution_workspace_elements': c['convolution_workspace_elements'],
            'policies': {'state': 'explicit-in-out', 'mask': 'skip-step-zero-output',
                         'reset': 'initial-before-active-step', 'gradient': 'full-no-implicit-detach',
                         'time': 'positive-runtime-T', 'execution': 'cpu-float32-eager'}}


def recurrent_contract(models, mid, specification):
    """Validate a closed v1 state interface against the actual transition graph."""
    from .revisions import model_document
    models = model_document(models, mid)['models']
    result = interface_contract(validate_model(models, mid), specification)
    architecture = identity(models, mid)['architecture']
    result.update(transition=mid, transition_identity=architecture,
                  identity={'version': 1, 'kind': 'recurrent-transition-v1',
                            'architecture': architecture, 'interface': digest(result['specification'])},
                  parameter_schema={'transition.' + k: v for k, v in parameter_schema(models, mid).items()})
    return result


def recurrent_memory_report(contract, batch, steps, constrained=True, *,
                            memory_limit_percent=10., training=True):
    """Per-call whole-unroll admission heuristic, not an allocator/OOM guarantee."""
    feature_count(batch); feature_count(steps)
    if type(constrained) is not bool or type(training) is not bool:
        raise GraphError('Constrained and training flags must be booleans.')
    percentage = memory_percent(memory_limit_percent)
    spec = contract['specification']
    ni = sum(math.prod(contract['inputs'][i]['shape']) for i in spec['sequence_inputs'])
    ns = sum(math.prod(contract['inputs'][s['input']]['shape']) for s in spec['states'])
    no = sum(math.prod(contract['outputs'][o['output']]['shape']) for o in spec['sequence_outputs'])
    components = {
        'runtime_headroom': (256 if training else 64) * MIB,
        'parameters_optimizer_evidence' if training else 'parameters_construction':
            (128 if training else 8) * contract['parameters'],
        'sequence_state_output_copies': 16 * batch * (steps * (ni + no) + ns * (steps + 1)),
        'activations_backward' if training else 'activations_forward':
            (16 if training else 8) * batch * steps * contract['activation_elements'],
        'convolution_workspace': (32 if training else 8) * batch * steps * contract['convolution_workspace_elements'],
        'controls': 16 * batch * steps,
    }
    total = sum(components.values())
    free = available_memory()
    budget = math.floor(free * percentage / 100) if free is not None else None
    allowed = not constrained or budget is not None and total <= budget
    reason = None if allowed else ('Available RAM cannot be inspected; constrained recurrence is refused.'
                                  if budget is None else 'Recurrent unroll estimates ' + format_bytes(total) +
                                  ', exceeding the current RAM budget ' + format_bytes(budget) + '.')
    return {'version': 1, 'operation': 'recurrent-training' if training else 'recurrent-inference',
            'batch': batch, 'steps': steps, 'components': components, 'estimated_bytes': total,
            'available_bytes': free, 'budget_bytes': budget, 'limit_percent': percentage,
            'allowed': allowed, 'reason': reason}


def _recurrent_inputs(contract, sequences, initial_state, lengths, mask, reset):
    """No coercion, implicit broadcast, padding, sorting, or tensor repacking."""
    import torch
    spec = contract['specification']
    if not isinstance(sequences, dict) or set(sequences) != set(spec['sequence_inputs']):
        raise ValueError('Supply exactly the stable sequence Input IDs.')
    if not isinstance(initial_state, dict) or set(initial_state) != {s['port'] for s in spec['states']}:
        raise ValueError('Supply exactly the declared initial-state port IDs.')
    batch = steps = None
    for key, value in sequences.items():
        shape = contract['inputs'][key]['shape']
        if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32:
            raise ValueError('Sequence ' + key + ' must be a CPU float32 tensor.')
        if value.ndim != len(shape) + 2 or list(value.shape[2:]) != shape or min(value.shape[:2]) < 1:
            raise ValueError('Sequence ' + key + ' must have positive [B, T, *transition_input_shape].')
        if batch is not None and tuple(value.shape[:2]) != (batch, steps):
            raise ValueError('Every sequence must have the same B and T.')
        batch, steps = value.shape[:2]
    for state in spec['states']:
        value = initial_state[state['port']]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or
                list(value.shape) != [batch, *contract['inputs'][state['input']]['shape']]):
            raise ValueError('Initial state ' + state['port'] + ' must have declared [B, *state_shape], CPU float32.')
    if lengths is not None and (not isinstance(lengths, torch.Tensor) or lengths.device.type != 'cpu' or
                                lengths.dtype != torch.int64 or list(lengths.shape) != [batch] or
                                bool(((lengths < 0) | (lengths > steps)).any())):
        raise ValueError('Lengths must be CPU int64 [B], with values from 0 through T.')
    for name, value in (('Mask', mask), ('Reset', reset)):
        if value is not None and (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                                  value.dtype != torch.bool or list(value.shape) != [batch, steps]):
            raise ValueError(name + ' must be CPU bool [B, T]; no broadcasting or numeric coercion.')
    return batch, steps


def _recurrent_run(transition, contract, sequences, initial_state, lengths, mask, reset, batch, steps):
    """Run only active rows; out-of-place state replacement preserves BPTT."""
    import torch
    spec = contract['specification']
    state = dict(initial_state)
    collected = {o['port']: [] for o in spec['sequence_outputs']}
    reference = next(iter(sequences.values()))
    # Empty slices create differentiable zeros without reading NaN padding. This
    # also makes an entirely masked sequence objective safe to backward through.
    anchor = reference[:, :0].sum()
    for value in initial_state.values():
        anchor = anchor + value[:0].sum()
    for value in sequences.values():
        anchor = anchor + value[:, :0].sum()
    for value in transition.parameters():
        anchor = anchor + value.reshape(-1)[:0].sum()
    for time in range(steps):
        active = torch.ones(batch, dtype=torch.bool, device='cpu')
        if lengths is not None:
            active = active & (time < lengths)
        if mask is not None:
            active = active & mask[:, time]
        indices = active.nonzero(as_tuple=True)[0]
        returned = {}
        if indices.numel():
            base = dict(state)
            if reset is not None:
                for item in spec['states']:
                    port = item['port']
                    selector = (active & reset[:, time]).reshape(batch, *([1] * (state[port].ndim - 1)))
                    base[port] = torch.where(selector, initial_state[port], state[port])
            args = {contract['inputs'][key]['name']: value[:, time].index_select(0, indices)
                    for key, value in sequences.items()}
            args.update({contract['inputs'][item['input']]['name']: base[item['port']].index_select(0, indices)
                         for item in spec['states']})
            raw = transition(args)
            outputs = list(contract['outputs'].values())
            named = {outputs[0]['name']: raw} if len(outputs) == 1 else raw
            returned = {g['id']: named[g['name']] for g in outputs}
            for item in spec['states']:
                port = item['port']
                state[port] = base[port].index_copy(0, indices, returned[item['output']])
        for output in spec['sequence_outputs']:
            value = reference.new_zeros(batch, *contract['outputs'][output['output']]['shape']) + anchor
            if indices.numel():
                value = value.index_copy(0, indices, returned[output['output']])
            collected[output['port']].append(value)
    return {'sequences': {port: torch.stack(values, dim=1) for port, values in collected.items()},
            'state': {port: value + anchor for port, value in state.items()}}


def _runner_class(nn):
    """Shared native/exported class factory; Torch is supplied only on execution."""
    class RecurrentTransition(nn.Module):
        def __init__(self, transition, contract, *, constrained=True, memory_limit_percent=10.):
            super().__init__()
            self.transition = transition
            self.contract = contract
            self.constrained = constrained
            self.memory_limit_percent = memory_limit_percent

        def forward(self, sequences, initial_state, *, lengths=None, mask=None, reset=None):
            batch, steps = _recurrent_inputs(self.contract, sequences, initial_state, lengths, mask, reset)
            import torch
            if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.transition.parameters()):
                raise ValueError('Recurrent transition parameters must remain CPU float32.')
            report = recurrent_memory_report(self.contract, batch, steps, self.constrained,
                                            memory_limit_percent=self.memory_limit_percent,
                                            training=self.training or torch.is_grad_enabled())
            if not report['allowed']:
                raise ValueError(report['reason'])
            return _recurrent_run(self.transition, self.contract, sequences, initial_state,
                                  lengths, mask, reset, batch, steps)
    return RecurrentTransition


def compile_recurrent_transition(models, mid, specification, *, constrained=True, memory_limit_percent=10.):
    """Compile one private graph snapshot; runtime T never recreates parameters."""
    from .revisions import model_document
    models = model_document(models, mid)['models']
    contract = recurrent_contract(models, mid, specification)
    report = recurrent_memory_report(contract, 1, 1, constrained,
                                     memory_limit_percent=memory_limit_percent, training=False)
    if not report['allowed']:
        raise GraphError(report['reason'])
    from .core import compile_model
    import torch.nn as nn
    transition = compile_model(models, mid, constrained=False)
    return _runner_class(nn)(transition, contract, constrained=constrained, memory_limit_percent=memory_limit_percent)


def generate_recurrent_code(models, mid, specification, *, constrained=True, memory_limit_percent=10.):
    """Checked app-free eager source, with the same state/control/RAM semantics."""
    from .core import generate_code
    from .task_execution import _helper_source
    from .revisions import model_document
    models = model_document(models, mid)['models']
    contract = recurrent_contract(models, mid, specification)
    # Validate options without requiring current RAM to admit code generation.
    recurrent_memory_report(contract, 1, 1, False, memory_limit_percent=memory_limit_percent, training=False)
    if type(constrained) is not bool:
        raise GraphError('Constrained mode must be a boolean.')
    helpers = (feature_count, memory_percent, available_memory, format_bytes,
               recurrent_memory_report, _recurrent_inputs, _recurrent_run, _runner_class)
    source = ['from pathlib import Path', 'import math', 'GraphError = ValueError',
              'MAX_INTEGER = 2**53 - 1', 'MIB = 1024**2', generate_code(models, mid)]
    source.extend(_helper_source(helper) for helper in helpers)
    source.extend([f'recurrent_contract = {contract!r}', f'recurrent_identity = {contract["identity"]!r}',
                   f'model = _runner_class(nn)(model, recurrent_contract, constrained={constrained!r}, '
                   f'memory_limit_percent={float(memory_limit_percent)!r})', ''])
    return '\n\n'.join(source)
