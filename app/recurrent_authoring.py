"""Guarded, non-numerical authoring of an explicit recurrent diagram branch."""
import copy
import uuid
from .core import GraphError, MAX_INTEGER, validate_model
from .evidence import digest, pin_input_order
from .revisions import model_document
from .recurrent import interface_contract


def prepare(models, mid, ref):
    if not isinstance(mid, str) or not isinstance(ref, str) or mid not in models or ref not in models:
        raise GraphError('Choose an existing architecture and transition.')
    child = model_document(models, ref)
    if mid in child['models']:
        raise GraphError('This transition would create an architecture reference cycle.')
    contract = validate_model(models, ref)
    return {'id': mid, 'ref': ref, 'root_graph': digest(models[mid]['graph']),
            'transition': digest(child), 'name': models[ref]['name'],
            'inputs': contract['input_groups'], 'outputs': contract['output_groups']}


def build(models, token, options):
    """No writes: reject stale graph/closure and invalid choices before allocation."""
    if not isinstance(token, dict) or set(token) != {'id', 'ref', 'root_graph', 'transition'}:
        raise GraphError('Recurrent authoring requires an exact graph/transition token.')
    info = prepare(models, token['id'], token['ref'])
    if any(info[key] != value for key, value in token.items()):
        raise GraphError('Architecture or transition changed; reopen recurrent authoring.')
    if not isinstance(options, dict) or set(options) != {'version', 'steps', 'inputs', 'sequence_outputs', 'controls', 'label'}:
        raise GraphError('Recurrent choices have missing or unsupported fields.')
    if type(options['version']) is not int or options['version'] != 1:
        raise GraphError('Recurrent authoring supports version 1.')
    steps = options['steps']
    if type(steps) is not int or not 1 <= steps <= MAX_INTEGER:
        raise GraphError('Choose a positive exact integer T.')
    label = options['label']
    if not isinstance(label, str) or not label.strip() or len(label) > 60:
        raise GraphError('Choose a recurrent block name of 1–60 characters.')
    inputs, outputs = info['inputs'], {g['id']: g for g in info['outputs']}
    choices = options['inputs']
    if not isinstance(choices, list) or len(choices) != len(inputs):
        raise GraphError('Assign every transition Input explicitly.')
    assigned = {}
    for row in choices:
        if not isinstance(row, dict) or set(row) != {'input', 'role', 'output'} or not isinstance(row['role'], str) or row['role'] not in {'sequence', 'state'}:
            raise GraphError('Each Input requires an explicit sequence or initial-state role.')
        if not isinstance(row['input'], str) or row['input'] in assigned:
            raise GraphError('Transition Input identities must be unique.')
        if row['role'] == 'sequence' and row['output'] is not None:
            raise GraphError('A sequence Input does not declare a next-state Output.')
        assigned[row['input']] = row
    if set(assigned) != {g['id'] for g in inputs}:
        raise GraphError('Input choices must match this transition exactly.')
    for field, allowed in (('sequence_outputs', set(outputs)), ('controls', {'lengths', 'mask', 'reset'})):
        value = options[field]
        if not isinstance(value, list) or any(not isinstance(v, str) or v not in allowed for v in value) or len(set(value)) != len(value):
            raise GraphError('Choose distinct declared ' + field + '.')
    if 'lengths' in options['controls'] and steps > 2**24:
        raise GraphError('Float32 lengths controls require T at most 2**24.')
    specification = {'version': 1, 'sequence_inputs': [g['id'] for g in inputs if assigned[g['id']]['role'] == 'sequence'],
                     'states': [{'input': g['id'], 'output': assigned[g['id']]['output'], 'port': 'state/' + g['id']}
                                for g in inputs if assigned[g['id']]['role'] == 'state'],
                     'sequence_outputs': [{'output': oid, 'port': 'sequence/' + oid} for oid in options['sequence_outputs']]}
    child = interface_contract(validate_model(models, token['ref']), specification)
    doc = copy.deepcopy(models[token['id']])
    extra = 1 + len(inputs) + len(options['controls']) + len(specification['states']) + len(specification['sequence_outputs'])
    if len(doc['graph']['nodes']) + extra > 64:
        raise GraphError('This prototype supports at most 64 blocks.')
    for group in inputs:
        dims = ([steps] if assigned[group['id']]['role'] == 'sequence' else []) + group['shape']
        if len(dims) > 8:
            raise GraphError('A sequence needs room for T within eight sample dimensions.')
        product = 1
        for dim in dims:
            product *= dim
        if product > MAX_INTEGER:
            raise GraphError('Sequence dimensions exceed the exact graph size domain.')
    graph = doc['graph'];pin_input_order(graph)
    used = {n['id'] for n in graph['nodes']}
    names = {kind: {(n.get('label') or n['id']).strip() for n in graph['nodes'] if n['type'] == kind}
             for kind in ('Input', 'Output')}
    def boundary_name(kind, value):
        value = value[:60];candidate = value;index = 2
        while candidate in names[kind]:
            suffix = ' ' + str(index);candidate = value[:60-len(suffix)] + suffix;index += 1
        names[kind].add(candidate);return candidate
    def identifier():
        while True:
            value = 'n_' + uuid.uuid4().hex
            if value not in used:
                used.add(value);return value
    x = min((n.get('x', 0) for n in graph['nodes']), default=90)
    y = max((n.get('y', 0) + n.get('height', 100) for n in graph['nodes']), default=0) + 160
    recurrent = {'id': identifier(), 'type': 'Recurrent', 'op_version': 1, 'label': label.strip(),
                 'ref': token['ref'], 'sequence_inputs': [], 'states': [],
                 'sequence_outputs': specification['sequence_outputs'], 'controls': {}, 'x': x + 250, 'y': y}
    nodes, edges = [], []
    for group in inputs:
        row = assigned[group['id']];sequence = row['role'] == 'sequence'
        source = {'id': identifier(), 'type': 'Input', 'shape': ([steps] if sequence else []) + group['shape'],
                  'label': boundary_name('Input', ('Sequence ' if sequence else 'Initial ') + group['name']), 'x': x, 'y': y + len(nodes)*120}
        if group.get('axes') is not None:
            source['axes'] = (['T'] if sequence else []) + group['axes']
        elif sequence:
            source['axes'] = ['T', *('D' + str(i) for i in range(len(group['shape']))) ]
        nodes.append(source);edges.append({'from': source['id'], 'to': recurrent['id']})
        if sequence:recurrent['sequence_inputs'].append({'input': group['id'], 'source': source['id']})
        else:recurrent['states'].append({'input': group['id'], 'output': row['output'], 'port': 'state/' + group['id'], 'source': source['id']})
    for role in ('lengths', 'mask', 'reset'):
        if role not in options['controls']:continue
        source = {'id': identifier(), 'type': 'Input', 'shape': [1] if role == 'lengths' else [steps],
                  'label': boundary_name('Input', role.capitalize() + ' control'), 'x': x, 'y': y + len(nodes)*120}
        nodes.append(source);edges.append({'from': source['id'], 'to': recurrent['id']})
        recurrent['controls'][role] = {'source': source['id'], 'adapter': 'exact-int64-v1' if role == 'lengths' else 'exact-bool-v1'}
    nodes.append(recurrent)
    for i, row in enumerate([*recurrent['sequence_outputs'], *recurrent['states']]):
        sequence = row in recurrent['sequence_outputs']
        name = outputs[row['output']]['name'] if sequence else child['inputs'][row['input']]['name']
        output = {'id': identifier(), 'type': 'Output', 'label': boundary_name('Output', ('Sequence ' if sequence else 'Final ') + name),
                  'x': x + 500, 'y': y + i*120}
        nodes.append(output);edges.append({'from': recurrent['id'], 'from_port': row['port'], 'to': output['id']})
    # Validate the new branch independently: unrelated invalid editable drafts
    # remain repairable and are not silently replaced by this operation.
    branch = copy.deepcopy(doc);branch['graph'] = {'nodes': nodes, 'edges': edges}
    validate_model({**models, doc['id']: branch}, doc['id'])
    graph['nodes'].extend(nodes);graph['edges'].extend(edges)
    graph['input_order'].extend(n['id'] for n in nodes if n['type'] == 'Input')
    doc.update(dataset=None, result=None)
    return doc, recurrent['id']
