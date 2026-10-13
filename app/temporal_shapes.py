"""Opt-in runtime-temporal-v1: variable T, fixed frames/states and parameter owners.

This is not a spatial-policy upgrade or a retained family grammar. Root Input IDs
explicitly declare axis 0 with semantic name T; every other dimension is fixed.
Pure proof/preview imports neither Torch nor the GUI. Calls remain explicit-state
CPU float32 eager full-BPTT graphs, with admission before packing/conversion.
"""
from .core import GraphError, validate_model
from .graph import contract
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT
from .runtime_shapes import _dependency_specification, _compile_runtime_model, shape_contract, admission as graph_admission


def declaration(models, mid, runtime_inputs):
    groups = {g['id']:g for g in validate_model(models, mid)['input_groups']}
    if not isinstance(runtime_inputs, dict) or not runtime_inputs or not set(runtime_inputs) <= set(groups):
        raise GraphError('Runtime temporal inputs must map existing root Input IDs to explicit T sample axis [0].')
    result = {}
    for nid, axes in runtime_inputs.items():
        group = groups[nid]
        if not isinstance(axes, list) or len(axes) != 1 or type(axes[0]) is not int or axes[0] != 0:
            raise GraphError('Runtime-temporal-v1 only permits the exact sample-axis declaration [0]; B is not a sample axis.')
        names = group.get('axes')
        if not names or names[0] != 'T':
            raise GraphError('Runtime temporal Input ' + nid + ' needs an explicit leading semantic T axis.')
        result[nid] = [0]
    return result


def specification(models, mid, runtime_inputs):
    return _dependency_specification(models, mid, declaration(models, mid, runtime_inputs), temporal=True)


def admission(models, mid, c, batch_size, *, constrained=True,
              memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, gradients=True, dtype_bytes=4):
    """Charge the whole graph plus conservative copies/controls of every unroll.

Shared parameters may be charged repeatedly here; they are never constructed
repeatedly. Nested transition charges are multiplied by outer T, so admission
does not overlook retained inner BPTT graphs. Memoization bounds traversal by
actual contract contexts rather than expanding shared reference paths.
Fixed source buffers are charged once in the whole-graph buffer component,
not as newly allocated owners on every transition step.
"""
    if dtype_bytes != 4 or type(dtype_bytes) is not int:
        raise GraphError('Runtime-temporal-v1 admission supports CPU float32 only.')
    if type(constrained) is not bool:
        raise GraphError('Runtime constrained policy must be boolean.')
    report = graph_admission(c, batch_size, constrained=False, memory_limit_percent=memory_limit_percent,
                             gradients=gradients, dtype_bytes=dtype_bytes)
    from .recurrent import recurrent_memory_report
    memo, cache = {}, {}
    def charge(key, actual):
        context = (key, id(actual))
        if context in memo: return memo[context]
        total = 0
        lookup = {n['id']:n for n in models[key]['graph']['nodes']}
        for item in actual['plan']:
            if item['type'] not in {'Module', 'Recurrent'}: continue
            child_key = lookup[item['id']]['ref']
            if item['type'] == 'Module':
                child = item.get('runtime_contract')
                if child is None: child = contract(models, child_key, cache=cache)
                total += charge(child_key, child)
            else:
                recurrent = item['recurrent']
                value = recurrent_memory_report(recurrent['contract'], batch_size, recurrent['steps'], False,
                                                 memory_limit_percent=memory_limit_percent, training=gradients)
                # Already charge learned parameters in the whole-graph report.
                total += sum(v for k,v in value['components'].items() if k not in
                             {'runtime_headroom', 'parameters_optimizer_evidence', 'parameters_construction', 'buffers'})
                total += recurrent['steps'] * charge(child_key, contract(models, child_key, cache=cache))
        memo[context] = total
        return total
    memory = report['memory']
    memory['components']['recurrent_unrolls'] = charge(mid, c)
    memory['estimated_bytes'] = sum(memory['components'].values())
    memory['operation'] = 'runtime-temporal-training' if gradients else 'runtime-temporal-inference'
    budget = memory['budget_bytes']
    report.update(constrained=constrained, allowed=not constrained or budget is not None and memory['estimated_bytes'] <= budget)
    if not report['allowed']:
        raise GraphError('Constrained mode blocked runtime temporal execution: the whole current-B/T graph exceeds or cannot be checked against the current RAM budget.')
    return report


def compile_temporal_model(models, mid, runtime_inputs, *, constrained=True,
                           memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, initial_shapes=None):
    spec = specification(models, mid, runtime_inputs)
    return _compile_runtime_model(models, mid, spec, constrained=constrained,
                                  memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes)
