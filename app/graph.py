"""Pure shape propagation and port-aware DAG execution plans (no Torch imports)."""
import math
from .operators import TENSOR_OPERATORS, LAYER_TYPES, axes_for, describe


def parameter_owners(model):
    """Local explicit module identity, independent of wire/topological order.

    Whole Linear/Conv2d/Module/Recurrent reuse is supported. Chained aliases resolve
    to one existing owner; constructor compatibility is checked by contract.
    """
    from .core import GraphError
    nodes = {n['id']: n for n in model['graph']['nodes']}
    owners = {}
    for nid, node in nodes.items():
        if 'share_with' not in node:
            continue
        target = node['share_with']
        if node['type'] not in {'Linear', 'Conv2d', 'Module', 'Recurrent'} or not isinstance(target, str) or not target:
            raise GraphError('share_with requires a Linear, Conv2d, Module or Recurrent block and an existing local block ID.')
        if target not in nodes or nodes[target]['type'] != node['type']:
            raise GraphError('Shared parameter owner is missing or has a different block type: ' + nid + '.')
    for nid in nodes:
        visited, current = set(), nid
        while 'share_with' in nodes[current]:
            if current in visited:
                raise GraphError('Shared parameter ownership contains a cycle: ' + nid + '.')
            visited.add(current)
            current = nodes[current]['share_with']
        owners[nid] = current
    return owners


def topology(model):
    from .core import GraphError
    nodes, edges = model['graph']['nodes'], model['graph']['edges']
    lookup = {n['id']: n for n in nodes}
    if len(lookup) != len(nodes):
        raise GraphError('Node IDs must be unique.')
    incoming = {nid: [] for nid in lookup}
    outgoing = {nid: [] for nid in lookup}
    seen = set()
    for edge in edges:
        a, b = edge['from'], edge['to']
        if a not in lookup or b not in lookup:
            raise GraphError('An edge refers to a missing node.')
        key = (a, b, edge.get('from_port'))
        if key in seen:
            raise GraphError('Remove the duplicated wire.')
        seen.add(key)
        outgoing[a].append(b)
        incoming[b].append(edge)
    if not any(n['type'] == 'Input' for n in nodes) or not any(n['type'] == 'Output' for n in nodes):
        raise GraphError('Use at least one Input and one Output.')
    for n in nodes:
        ins, outs = incoming[n['id']], outgoing[n['id']]
        if n['type'] == 'Input':
            if ins:
                raise GraphError('Input cannot receive a wire.')
        elif n['type'] == 'Concat':
            sources = n.get('sources', [])
            keys = [e['from'] + '@' + e['from_port'] if 'from_port' in e else e['from'] for e in ins]
            if len(sources) < 2 or len(set(sources)) != len(sources) or set(sources) != set(keys) or len(ins) != len(sources):
                raise GraphError('Concat sources must list each incoming source exactly once, in order.')
        elif n['type'] == 'Recurrent':
            from .recurrent_nodes import sources
            keys = [e['from'] + '@' + e['from_port'] if 'from_port' in e else e['from'] for e in ins]
            expected = sources(n)
            if len(set(keys)) != len(keys) or set(keys) != set(expected):
                raise GraphError('Recurrent bindings must name exactly its unambiguous incoming wire sources.')
        elif len(ins) != 1:
            raise GraphError('Connect every block: each ordinary block has exactly one input; use Concat to join branches.')
        if n['type'] == 'Output' and outs:
            raise GraphError('Output cannot send a wire.')
        if n['type'] != 'Output' and not outs:
            raise GraphError('Connect every block to a declared Output; remove unused branches.')
    # Stable Kahn order; root input Concat's declared order retains legacy CSV order.
    roots = [n for n in nodes if n['type'] == 'Input']
    root_join = next((n for n in nodes if n['type'] == 'Concat' and
                     set(n['sources']) == {r['id'] for r in roots}), None)
    if root_join:
        roots = [lookup[nid] for nid in root_join['sources']]
    if 'input_order' in model['graph']:
        declared=model['graph']['input_order']
        if not isinstance(declared,list) or len(declared)!=len(roots) or set(declared)!={n['id'] for n in roots}:
            raise GraphError('input_order must list each Input ID exactly once.')
        roots=[lookup[nid] for nid in declared]
    pending = {nid: len(ins) for nid, ins in incoming.items()}
    queue, order = list(roots), []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for dest in outgoing[n['id']]:
            pending[dest] -= 1
            if pending[dest] == 0:
                queue.append(lookup[dest])
    if len(order) != len(nodes):
        raise GraphError('The graph contains a cycle or disconnected blocks.')
    return order


def contract(models, mid, ancestry=(), cache=None, *, input_shapes=None, _runtime_contexts=None):
    from .core import (GraphError, feature_count, input_shape, tensor_shape,
                       reshape_shape, activation_name, boundary_groups)
    if mid in ancestry:
        raise GraphError('Architecture reference cycle: ' + ' → '.join(models[k]['name'] for k in (*ancestry, mid)))
    if mid not in models:
        raise GraphError('A referenced architecture is missing.')
    cache = {} if cache is None else cache
    cache_key = mid if input_shapes is None else (mid, tuple(sorted((k, tuple(v)) for k, v in input_shapes.items())))
    if cache_key in cache:
        return cache[cache_key]
    if input_shapes is not None and len(cache) >= 64:
        raise GraphError('Runtime shape plan exceeds 64 distinct Module contexts; reduce call fan-out.')
    if input_shapes is not None:
        _runtime_contexts = set() if _runtime_contexts is None else _runtime_contexts
        if cache_key not in _runtime_contexts and len(_runtime_contexts) >= 64:
            raise GraphError('Runtime shape plan exceeds 64 distinct Module contexts; reduce call fan-out.')
        _runtime_contexts.add(cache_key)
    model = models[mid]
    order = topology(model)
    owners = parameter_owners(model)
    lookup = {n['id']: n for n in order}
    inputs = [n for n in order if n['type'] == 'Input']
    if input_shapes is not None and set(input_shapes) != {n['id'] for n in inputs}:
        raise GraphError('Runtime shape plan must bind every input ID exactly once.')
    effective = {n['id']: input_shape(n) if input_shapes is None else tensor_shape(input_shapes[n['id']]) for n in inputs}
    groups = boundary_groups(inputs, [math.prod(effective[n['id']]) for n in inputs])
    for g, n in zip(groups, inputs):
        g['shape'] = effective[n['id']]
        axes = axes_for(n, len(g['shape']))
        if axes is not None:
            g['axes'] = axes
    total_in = feature_count(sum(g['features'] for g in groups))
    initial = groups[0]['shape'] if len(groups) == 1 else [total_in]
    shapes, ports, origins, plan = {}, {}, {}, []
    params = ops = activation_elements = 0
    convolution_workspace_elements = 0
    max_width = max_input = total_in
    affine, matrix_affine = True, True
    operation_index = 0
    parameter_counts = {}

    def source(edge):
        src = edge['from']
        port = edge.get('from_port')
        if src in ports:
            port = port if port is not None else edge['to']
            if port not in ports[src]:
                raise GraphError(f'Wire from {src} must select a declared output port.')
            return {'node': src, 'port': port}, ports[src][port], origins[(src, port)]
        if port is not None:
            raise GraphError('Only Splitter, Replicator and Recurrent wires select output ports.')
        return {'node': src, 'port': None}, shapes[src]['output_shape'], origins[(src, None)]

    def source_axes(src):
        entry = shapes[src['node']]
        return entry.get('port_axes', {}).get(src['port'], entry['output_axes'])

    for n in order:
        kind, nid = n['type'], n['id']
        previous_params = params
        incoming = [e for e in model['graph']['edges'] if e['to'] == nid]
        if kind == 'Concat':
            incoming.sort(key=lambda e: n['sources'].index(e['from'] + '@' + e['from_port'] if 'from_port' in e else e['from']))
        if kind == 'Recurrent':
            from .recurrent_nodes import sources
            incoming.sort(key=lambda e: sources(n).index(e['from'] + '@' + e['from_port'] if 'from_port' in e else e['from']))
        resolved = [source(e) for e in incoming]
        before = resolved[0][1] if resolved else effective[nid]
        current = list(before)
        axes = source_axes(resolved[0][0]) if resolved else axes_for(n, len(before))
        operator = None
        item = {'id': nid, 'type': kind, 'sources': [r[0] for r in resolved]}
        origin = resolved[0][2] if resolved else (nid, None)
        if kind == 'Input':
            item['group'] = next(g for g in groups if g['id'] == nid)
        elif kind == 'Linear':
            width = feature_count(n['features'])
            current = before[:-1] + [width]
            params += before[-1] * width + (width if n.get('bias', True) else 0)
            matrix_affine = matrix_affine and len(before) == 1
            origin = (nid, None)
        elif kind == 'Activation':
            from .activations import normalize_activation
            normalize_activation(n)
            if n['activation'] == 'Softmax':
                from .activations import softmax_axis
                softmax_axis(n, len(before))
            affine = False
            origin = (nid, None)
        elif kind == 'Reshape':
            current = reshape_shape(before, n.get('shape'))
            if input_shapes is not None:
                item['runtime_reshape'] = list(current)
            axes = None
            origin = (nid, None)
        elif kind in TENSOR_OPERATORS:
            operator = describe(n, before, axes)
            current, axes = operator['output_shape'], operator['output_axes']
            if kind == 'Conv2d':
                convolution_workspace_elements += before[0] * math.prod(operator['weight_shape'][2:]) * math.prod(current[1:])
            params += operator['parameters']
            affine = affine and operator['affine']
            matrix_affine = matrix_affine and operator['matrix_affine']
            origin = (nid, None)
        elif kind == 'Module':
            child_shapes = None
            if input_shapes is not None:
                child_inputs = [v for v in topology(models[n['ref']]) if v['type'] == 'Input']
                child_shapes = ({child_inputs[0]['id']: before} if len(child_inputs) == 1 else
                                {v['id']: input_shape(v) for v in child_inputs})
            ref = contract(models, n['ref'], (*ancestry, mid), cache, input_shapes=child_shapes,
                           _runtime_contexts=_runtime_contexts)
            if input_shapes is not None:
                item['runtime_contract'] = ref
            if before != ref['input_shape']:
                raise GraphError(f"{models[n['ref']]['name']} needs {ref['inputs']} features with sample shape {ref['input_shape']}; incoming shape {before}. Use Reshape.")
            if ref['input_axes'] is not None and axes != ref['input_axes']:
                raise GraphError('Module input axes differ; use an explicit axis adapter.')
            current = ref['output_shape']
            axes = ref['output_axes']
            params += ref['parameters']
            activation_elements += ref['activation_elements']
            convolution_workspace_elements += ref['convolution_workspace_elements']
            ops += ref['operations']
            max_width = max(max_width, ref['max_width'])
            max_input = max(max_input, ref['max_input'])
            affine = affine and ref['affine']
            matrix_affine = matrix_affine and ref['matrix_affine']
            origin = (nid, None)
        elif kind == 'Recurrent':
            from .recurrent_nodes import describe as recurrent_describe
            child = contract(models, n['ref'], (*ancestry, mid), cache, _runtime_contexts=_runtime_contexts)
            bound = {}
            for edge, result in zip(incoming, resolved):
                src = result[0]
                entry = shapes[src['node']]
                src_axes = entry.get('port_axes', {}).get(src['port'], entry['output_axes'])
                key = edge['from'] + '@' + edge['from_port'] if 'from_port' in edge else edge['from']
                bound[key] = (result[1], src_axes)
            recurrent, outputs, port_axes = recurrent_describe(n, child, bound)
            item['recurrent'] = recurrent
            ports[nid] = outputs
            current = [feature_count(sum(math.prod(s) for s in outputs.values()))]
            axes = None
            params += child['parameters']
            activation_elements += recurrent['steps'] * child['activation_elements']
            convolution_workspace_elements += recurrent['steps'] * child['convolution_workspace_elements']
            ops += recurrent['steps'] * child['operations']
            max_width = max(max_width, child['max_width'])
            max_input = max(max_input, child['max_input'])
            affine = matrix_affine = False
            for row in n['sequence_outputs']:
                origins[(nid, row['port'])] = (nid, 'sequence/' + row['output'])
            for row in n['states']:
                origins[(nid, row['port'])] = (nid, 'state/' + row['input'])
        elif kind == 'Concat':
            item['flatten'] = n.get('flatten', True)
            if item['flatten']:
                current = [sum(math.prod(r[1]) for r in resolved)]
                axes = None
            else:
                if any(r[1][:-1] != before[:-1] for r in resolved):
                    raise GraphError('Last-axis Concat requires matching other sample dimensions.')
                current = before[:-1] + [sum(r[1][-1] for r in resolved)]
                if any(source_axes(r[0]) != axes for r in resolved):
                    axes = None
            origin = (nid, None)
        elif kind in {'Splitter', 'Replicator'}:
            ids = [p['output'] for p in n.get('parts', [])] if kind == 'Splitter' else n.get('outputs', [])
            if len(ids) < 2 or len(ids) != len(set(ids)):
                raise GraphError('Router ports must have at least two unique IDs.')
            sizes = [feature_count(p['features']) for p in n['parts']] if kind == 'Splitter' else [before[-1]] * len(ids)
            if kind == 'Splitter' and sum(sizes) != before[-1]:
                raise GraphError(f'Splitter sizes must sum to {before[-1]} last-axis features.')
            ports[nid] = {pid: before[:-1] + [w] for pid, w in zip(ids, sizes)}
            item.update(ports=ids, sizes=sizes)
            for pid in ids:
                origins[(nid, pid)] = (nid, pid) if kind == 'Splitter' else origin
        elif kind == 'Output':
            if 'shape' in n and tensor_shape(n['shape']) != current:
                raise GraphError(f'Output {nid} declares {n["shape"]}, but receives {current}. Add a Reshape.')
        else:
            raise GraphError(f'Unsupported block: {kind}.')
        feature_count(math.prod(current))
        activation_elements += math.prod(current)
        shapes[nid] = {'input': before[-1], 'output': current[-1],
                       'input_shape': list(before), 'output_shape': list(current),
                       'output_axes': axes}
        if kind == 'Recurrent':
            shapes[nid].update(ports=outputs, port_axes=port_axes, recurrent=recurrent)
        if operator is not None:
            shapes[nid]['operator'] = operator
        origins[(nid, None)] = origin
        max_width = max(max_width, math.prod(current))
        if kind in LAYER_TYPES:
            item['layer'] = f'layer_{operation_index}'
            operation_index += 1
            if kind not in {'Module', 'Recurrent'}:
                ops += 1
        elif kind in {'Concat', 'Splitter', 'Replicator'}:
            ops += 1
        plan.append(item)
        parameter_counts[nid] = params - previous_params

    layers = {item['id']: item['layer'] for item in plan if 'layer' in item}
    for item in plan:
        nid, owner = item['id'], owners[item['id']]
        if nid == owner:
            continue
        node, canonical = lookup[nid], lookup[owner]
        a, b = shapes[nid], shapes[owner]
        kind = node['type']
        if kind == 'Linear':
            same = (a['input'], a['output'], node.get('bias', True)) == (b['input'], b['output'], canonical.get('bias', True))
        elif kind == 'Conv2d':
            same = a['operator']['kwargs'] == b['operator']['kwargs']
        elif kind == 'Recurrent':
            from .recurrent_nodes import specification
            same = node['ref'] == canonical['ref'] and specification(node) == specification(canonical)
        else:
            same = node['ref'] == canonical['ref']
        if not same:
            raise GraphError('Shared block ' + nid + ' must have the same constructor/reference as owner ' + owner + '.')
        params -= parameter_counts[nid]
        item['layer'] = layers[owner]
        item['parameter_owner'] = owner
        shapes[nid]['parameter_owner'] = owner

    terminals = [n for n in order if n['type'] == 'Output']
    out_groups = boundary_groups(terminals, [math.prod(shapes[n['id']]['output_shape']) for n in terminals])
    for g in out_groups:
        g['shape'] = shapes[g['id']]['output_shape']
        axes = shapes[g['id']]['output_axes']
        if axes is not None:
            g['axes'] = axes
    output_items = [p for p in plan if p['type'] == 'Output']
    shared_source = output_items[0]['sources'][0]['node']
    router = lookup[shared_source]
    direct_router = len(terminals) > 1 and router['type'] in {'Splitter', 'Replicator'} and all(
        p['sources'][0]['node'] == shared_source for p in output_items)
    mode = 'tensor' if len(terminals) == 1 else 'branches'
    tensor_source = terminals[0]['id'] if len(terminals) == 1 else None
    if direct_router:
        router_item = next(p for p in plan if p['id'] == shared_source)
        # Compatibility: legacy boundary routers return the pre-router tensor for training/references.
        declared = router_item['ports']
        by_port = {p['sources'][0]['port']: p['id'] for p in output_items}
        if set(by_port) == set(declared) and len(by_port) == len(output_items):
            out_groups.sort(key=lambda g: declared.index(next(p['sources'][0]['port'] for p in output_items if p['id'] == g['id'])))
            tensor_source = router_item['sources'][0]['node'] if router_item['sources'][0]['port'] is None else None
            mode = 'split' if router['type'] == 'Splitter' else 'replicate'
    if tensor_source is not None:
        final_shape = shapes[tensor_source]['output_shape']
        axis, start = 0, 0
        for g in out_groups:
            g.update(start=start if mode != 'replicate' else 0,
                     stop=start + g['features'] if mode != 'replicate' else math.prod(final_shape),
                     axis_start=axis if mode == 'split' else 0,
                     axis_stop=axis + g['shape'][-1] if mode == 'split' else g['shape'][-1])
            axis += g['shape'][-1]
            start += g['features']
    else:
        packed, start = {}, 0
        for g in out_groups:
            origin = origins[(g['id'], None)]
            if origin not in packed:
                packed[origin] = (start, start + g['features'])
                start += g['features']
            g['start'], g['stop'] = packed[origin]
        final_shape = [feature_count(start)]
        mode = 'branches'
    # Effective matrix composition is valid only for one serial vector path.
    root_join=any(n['type']=='Concat' and set(n['sources'])=={r['id'] for r in inputs} for n in order)
    matrix_affine = affine and matrix_affine and (len(inputs)==1 or root_join) and len(initial)==1 and not any(
        len([e for e in model['graph']['edges'] if e['from'] == n['id']]) > 1 and
        n['type'] not in {'Splitter', 'Replicator'} for n in order)
    matrix_affine = matrix_affine and not any(n['type'] == 'Concat' and
        any(src not in lookup or lookup[src]['type'] != 'Input' for src in n['sources']) for n in order)
    matrix_affine = matrix_affine and not any(n['type'] in {'Splitter','Replicator'} and
        any(lookup[e['to']]['type'] != 'Output' for e in model['graph']['edges'] if e['from'] == n['id']) for n in order)
    result = {'inputs': total_in, 'outputs': math.prod(final_shape), 'parameters': params,
              'affine': affine, 'matrix_affine': matrix_affine, 'input_shape': initial,
              'input_axes': groups[0].get('axes') if len(groups) == 1 else None,
              'output_axes': shapes[tensor_source]['output_axes'] if tensor_source is not None else None,
              'output_shape': final_shape, 'max_width': max_width, 'max_input': max_input,
              'operations': ops, 'shapes': shapes, 'order': [n['id'] for n in order],
              'activation_elements': activation_elements,
              'convolution_workspace_elements': convolution_workspace_elements,
              'input_groups': groups, 'output_groups': out_groups, 'output_mode': mode,
              'plan': plan, 'tensor_source': tensor_source}
    cache[cache_key] = result
    return result
