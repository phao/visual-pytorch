"""Explicit runtime-temporal-v2: variable T with Recurrent and LSTM.

Version 1 remains closed. This version permits LSTM's three supplied/returned
roles, never runtime constructor/state widths, hidden state carry or inferred
masks/losses. Pure proof and whole-graph memory preview allocate no tensors.
"""
from .core import GraphError
from .graph import contract
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT
from .runtime_shapes import _dependency_specification, _compile_runtime_model, shape_contract
from .temporal_shapes import declaration, admission as recurrent_admission


def specification(models, mid, runtime_inputs):
    return _dependency_specification(models, mid, declaration(models, mid, runtime_inputs),
                                     temporal=True, temporal_version=2)


def admission(models, mid, c, batch_size, *, constrained=True,
              memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, gradients=True, dtype_bytes=4):
    """Charge LSTM gates/copies in addition to the existing whole recurrent tree.

    Each call is charged, even aliases; parameter owners are counted once by the
    graph. LSTMs in recurrent bodies are charged outer-T times. Conservative
    backward storage is retained even for inference, not an allocator guarantee.
    """
    report = recurrent_admission(models, mid, c, batch_size, constrained=False,
                                 memory_limit_percent=memory_limit_percent,
                                 gradients=gradients, dtype_bytes=dtype_bytes)
    if type(constrained) is not bool:
        raise GraphError('Runtime constrained policy must be boolean.')
    from .lstm_nodes import lstm_memory_report
    memo, cache = {}, {}

    def charge(key, actual):
        context = (key, id(actual))
        if context in memo:
            return memo[context]
        total = 0
        lookup = {n['id']:n for n in models[key]['graph']['nodes']}
        for item in actual['plan']:
            if item['type'] == 'LSTM':
                value = lstm_memory_report(item['lstm'], batch_size, constrained=False,
                                           memory_limit_percent=memory_limit_percent)
                total += sum(v for k,v in value['components'].items()
                             if k not in {'runtime_headroom', 'parameters_optimizer_copies'})
            elif item['type'] in {'Module', 'Recurrent'}:
                child_key = lookup[item['id']]['ref']
                child = item.get('runtime_contract')
                if child is None:
                    child = contract(models, child_key, cache=cache)
                total += (item['recurrent']['steps'] if item['type'] == 'Recurrent' else 1) * charge(child_key, child)
        memo[context] = total
        return total

    memory = report['memory']
    memory['version'] = 2
    # Match the conservative per-LSTM owner allowance even under no_grad/eval.
    # Global ownership deduplicates aliases; activation charges still count calls.
    memory['components']['parameters'] = max(memory['components']['parameters'],128*c['parameters'])
    memory['components']['lstm_unrolls'] = charge(mid, c)
    memory['estimated_bytes'] = sum(memory['components'].values())
    memory['operation'] = 'runtime-temporal-v2-training' if gradients else 'runtime-temporal-v2-inference'
    budget = memory['budget_bytes']
    report.update(constrained=constrained,
                  allowed=not constrained or budget is not None and memory['estimated_bytes'] <= budget)
    if not report['allowed']:
        raise GraphError('Constrained mode blocked runtime temporal-v2 execution: the whole current-B/T graph exceeds or cannot be checked against the current RAM budget.')
    return report


def compile_temporal_model(models, mid, runtime_inputs, *, constrained=True,
                           memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, initial_shapes=None):
    spec = specification(models, mid, runtime_inputs)
    return _compile_runtime_model(models, mid, spec, constrained=constrained,
                                  memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes)
