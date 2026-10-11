"""Guarded allocation-free LSTM branch creation and stable in-place edits."""
import copy
import uuid
from .core import GraphError, MAX_INTEGER, REFERENCE_TYPES, validate_model
from .evidence import digest, pin_input_order
from .lstm_nodes import normalized, sources, PORTS

CREATE_TOKEN = ('id', 'root_graph', 'root_model')
EDIT_TOKEN = (*CREATE_TOKEN, 'node')
CONSTRUCTOR = ('hidden_size', 'num_layers', 'bidirectional', 'proj_size', 'bias')
CREATE_OPTIONS = ('version', 'label', 'steps', 'input_size', *CONSTRUCTOR, 'lengths')
EDIT_OPTIONS = ('version', 'label', *CONSTRUCTOR, 'inputs', 'lengths', 'share_with')


def _context(models, mid):
    if not isinstance(mid, str) or mid not in models:
        raise GraphError('Choose an existing architecture.')
    # Capture even unfinished reference closures without demanding that unrelated
    # drafts execute. Includes reference names/graphs, never attached data/results.
    closure = {}
    pending = [mid]
    while pending:
        current = pending.pop()
        if current in closure:
            continue
        if current not in models:
            closure[current] = {'missing': True}
            continue
        doc = models[current]
        closure[current] = {key: copy.deepcopy(doc[key]) for key in ('id', 'name', 'graph')}
        pending.extend(n['ref'] for n in doc['graph']['nodes'] if n['type'] in REFERENCE_TYPES)
    return {'id': mid, 'root_graph': digest(models[mid]['graph']), 'root_model': digest(closure)}


def prepare(models, mid):
    return {**_context(models, mid), 'options': {
        'version': 1, 'label': 'LSTM', 'steps': 4, 'input_size': 3,
        'hidden_size': 8, 'num_layers': 1, 'bidirectional': False,
        'proj_size': 0, 'bias': True, 'lengths': False}}


def _guard(models, token, fields):
    if not isinstance(token, dict) or set(token) != set(fields):
        raise GraphError('LSTM authoring requires an exact architecture token.')
    actual = _context(models, token['id'])
    if any(actual[k] != token[k] for k in CREATE_TOKEN):
        raise GraphError('Architecture or reference changed; reopen LSTM authoring.')


def _options(options, fields):
    if not isinstance(options, dict) or set(options) != set(fields):
        raise GraphError('LSTM choices have missing or unsupported fields.')
    if type(options['version']) is not int or options['version'] != 1:
        raise GraphError('LSTM authoring supports integer version 1.')
    label = options['label']
    if not isinstance(label, str) or not label.strip() or len(label) > 60:
        raise GraphError('Choose an LSTM block name of 1–60 characters.')
    return label.strip()


def build(models, token, options):
    _guard(models, token, CREATE_TOKEN)
    label = _options(options, CREATE_OPTIONS)
    for field in ('steps', 'input_size'):
        if type(options[field]) is not int or not 1 <= options[field] <= MAX_INTEGER:
            raise GraphError('LSTM ' + field + ' must be a positive exact integer.')
    if type(options['lengths']) is not bool:
        raise GraphError('LSTM supplied lengths must be boolean.')
    if options['lengths'] and options['steps'] > 2**24:
        raise GraphError('LSTM supplied float32 lengths require T <= 2**24.')
    node = normalized({'id': 'candidate', 'type': 'LSTM', 'label': label,
                       **{k: options[k] for k in CONSTRUCTOR},
                       'inputs': {k: k for k in PORTS}})
    doc = copy.deepcopy(models[token['id']]); graph = doc['graph']
    if len(graph['nodes']) + 7 + int(options['lengths']) > 64:
        raise GraphError('This prototype supports at most 64 blocks.')
    pin_input_order(graph)
    used = {n['id'] for n in graph['nodes']}
    names = {kind: {n.get('label') or n['id'] for n in graph['nodes'] if n['type'] == kind}
             for kind in ('Input', 'Output')}
    def identifier():
        while True:
            value = 'n_' + uuid.uuid4().hex
            if value not in used:
                used.add(value)
                return value
    def name(kind, text):
        text = text[:60]; value = text; index = 2
        while value in names[kind]:
            suffix = ' ' + str(index); value = text[:60-len(suffix)] + suffix; index += 1
        names[kind].add(value)
        return value
    ld = node['num_layers'] * (2 if node['bidirectional'] else 1)
    q = node['proj_size'] or node['hidden_size']
    x = min((n.get('x', 0) for n in graph['nodes']), default=90)
    y = max((n.get('y', 0) + n.get('height', 100) for n in graph['nodes']), default=0) + 160
    node.update(id=identifier(), x=x+250, y=y)
    nodes, edges = [], []
    shapes = {'sequence': [options['steps'], options['input_size']],
              'hidden': [ld, q], 'cell': [ld, node['hidden_size']]}
    axes = {'sequence': ['T', 'F'], 'hidden': ['LD', 'Hout'], 'cell': ['LD', 'Hcell']}
    for role in PORTS:
        source = {'id': identifier(), 'type': 'Input', 'label': name('Input', label+' '+('sequence' if role=='sequence' else 'initial '+role)),
                  'shape': shapes[role], 'axes': axes[role], 'x': x, 'y': y+len(nodes)*120}
        node['inputs'][role] = source['id']; nodes.append(source)
        edges.append({'from': source['id'], 'to': node['id']})
    if options['lengths']:
        source = {'id': identifier(), 'type': 'Input', 'label': name('Input', label+' lengths'),
                  'shape': [1], 'x': x, 'y': y+len(nodes)*120}
        node['lengths'] = {'source': source['id'], 'adapter': 'exact-int64-v1'}
        nodes.append(source); edges.append({'from': source['id'], 'to': node['id']})
    nodes.append(node)
    for index, role in enumerate(PORTS):
        output = {'id': identifier(), 'type': 'Output', 'label': name('Output', label+' '+('sequence' if role=='sequence' else 'final '+role)),
                  'x': x+500, 'y': y+index*120}
        nodes.append(output); edges.append({'from': node['id'], 'from_port': role, 'to': output['id']})
    branch = copy.deepcopy(doc); branch['graph'] = {'nodes': nodes, 'edges': edges}
    validate_model({**models, doc['id']: branch}, doc['id'])
    graph['nodes'].extend(nodes); graph['edges'].extend(edges)
    graph['input_order'].extend(n['id'] for n in nodes if n['type'] == 'Input')
    doc.update(dataset=None, result=None)
    return doc, node['id']


def prepare_edit(models, mid, nid):
    context = _context(models, mid)
    node = next((n for n in models[mid]['graph']['nodes'] if n['id'] == nid), None)
    if node is None or node['type'] != 'LSTM':
        raise GraphError('Choose an existing LSTM block.')
    n = normalized(node)
    options = {'version': 1, 'label': n.get('label') or nid,
               **{k: copy.deepcopy(n[k]) for k in CONSTRUCTOR},
               'inputs': copy.deepcopy(n['inputs']), 'lengths': copy.deepcopy(n['lengths']),
               'share_with': node.get('share_with')}
    contract = None
    try:
        contract = validate_model(models, mid)
    except GraphError:
        pass  # Unknown catalog dimensions are not a successful shape proof.
    catalog = []
    for source in models[mid]['graph']['nodes']:
        if source['id'] == nid or source['type'] == 'Output':
            continue
        shape = (contract or {}).get('shapes', {}).get(source['id'], {})
        ports = (PORTS if source['type'] == 'LSTM' else
                 [r['port'] for r in source['sequence_outputs']+source['states']] if source['type'] == 'Recurrent' else
                 [r['output'] for r in source['parts']] if source['type'] == 'Splitter' else
                 source['outputs'] if source['type'] == 'Replicator' else [None])
        for port in ports:
            key = source['id'] + ('@'+port if port is not None else '')
            catalog.append({'key': key, 'name': (source.get('label') or source['id']) + (' · '+port if port else ''),
                            'shape': shape.get('ports', {}).get(port) if port else shape.get('output_shape')})
    catalog.sort(key=lambda item: (item['name'].casefold(), item['key']))
    return {**context, 'node': nid, 'options': options, 'sources': catalog}


def build_edit(models, token, options):
    _guard(models, token, EDIT_TOKEN)
    label = _options(options, EDIT_OPTIONS)
    info = prepare_edit(models, token['id'], token['node'])
    doc = copy.deepcopy(models[token['id']]); graph = doc['graph']
    node = next(n for n in graph['nodes'] if n['id'] == token['node'])
    node.update({k: copy.deepcopy(options[k]) for k in (*CONSTRUCTOR, 'inputs', 'lengths')}, label=label)
    owner = options['share_with']
    if owner is None:
        node.pop('share_with', None)
    elif not isinstance(owner, str) or not owner.strip():
        raise GraphError('LSTM parameter owner must be a stable block ID or null.')
    else:
        node['share_with'] = owner
    normalized(node)
    desired = sources(node); available = {s['key'] for s in info['sources']}
    if any(key not in available for key in desired):
        raise GraphError('Choose existing source sockets; LSTM cannot feed itself.')
    incoming = {e['from'] + ('@'+e['from_port'] if e.get('from_port') else ''): e
                for e in graph['edges'] if e['to'] == node['id']}
    graph['edges'] = [e for e in graph['edges'] if e['to'] != node['id'] or
                      e['from'] + ('@'+e['from_port'] if e.get('from_port') else '') in desired]
    for key in desired:
        if key not in incoming:
            parts = key.split('@', 1); edge = {'from': parts[0], 'to': node['id']}
            if len(parts) == 2:
                edge['from_port'] = parts[1]
            graph['edges'].append(edge)
    from .recurrent_reconfiguration import _affected_contract
    _affected_contract(models, doc, node['id'])
    if doc != models[doc['id']]:
        doc.update(dataset=None, result=None)
    return doc, node['id']
