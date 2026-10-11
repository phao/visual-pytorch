"""Explicit in-place recurrent interface edits; no Torch, allocation or writes."""
import copy

from .core import GraphError, validate_model, REFERENCE_TYPES
from .evidence import digest
from .revisions import model_document
from .recurrent_authoring import prepare
from .recurrent_nodes import normalized, specification, sources
from .recurrent import interface_contract

TOKEN = ('id', 'ref', 'node', 'root_graph', 'root_model', 'transition')
OPTIONS = ('version', 'label', 'sequence_inputs', 'states', 'sequence_outputs', 'controls', 'share_with')


def prepare_edit(models, mid, nid, ref=None):
    if not isinstance(mid, str) or mid not in models or not isinstance(nid, str):
        raise GraphError('Choose an existing recurrent block.')
    node = next((n for n in models[mid]['graph']['nodes'] if n['id'] == nid), None)
    if node is None or node['type'] != 'Recurrent':
        raise GraphError('Choose an existing recurrent block.')
    info = prepare(models, mid, node['ref'] if ref is None else ref)
    options = {key: copy.deepcopy(node.get(key)) for key in OPTIONS if key != 'version'}
    options.update(version=1, label=node.get('label') or nid, controls=copy.deepcopy(node.get('controls', {})))
    refs = {n['ref'] for n in models[mid]['graph']['nodes'] if n['type'] in REFERENCE_TYPES}
    closure = {'id': mid, 'name': models[mid]['name'],
               'references': {ref: model_document(models, ref) for ref in sorted(refs)}}
    return {**info, 'node': nid, 'root_model': digest(closure), 'options': options}


def _affected_contract(models, doc, nid):
    """Prove edited paths and sharing peers without repairing unrelated drafts.

    Synthetic sinks exist only in this private proof graph. Original blocks,
    names, input order and wires are never added/removed by the proof itself.
    """
    from .graph import parameter_owners
    owners = parameter_owners(doc)
    graph = doc['graph'];lookup = {n['id']: n for n in graph['nodes']}
    affected = {n for n in lookup if owners[n] == owners[nid]}
    while True:
        expanded = affected | {e['to'] for e in graph['edges'] if e['from'] in affected}
        if expanded == affected:
            break
        affected = expanded
    needed = set(affected)
    while True:
        expanded = needed | {e['from'] for e in graph['edges'] if e['to'] in needed}
        expanded |= {owners[n] for n in needed if n in owners}
        if expanded == needed:
            break
        needed = expanded
    proof = copy.deepcopy(doc)
    proof['graph'] = {'nodes': [copy.deepcopy(n) for n in graph['nodes'] if n['id'] in needed],
                      'edges': [copy.deepcopy(e) for e in graph['edges'] if e['to'] in needed]}
    if 'input_order' in graph:
        proof['graph']['input_order'] = [i for i in graph['input_order'] if i in needed]
    outgoing = {e['from'] for e in proof['graph']['edges']}
    for node in list(proof['graph']['nodes']):
        if node['type'] == 'Output' or node['id'] in outgoing:
            continue
        if node['type'] == 'Recurrent':
            ports = [r['port'] for r in node['sequence_outputs'] + node['states']]
        elif node['type'] == 'LSTM':
            from .lstm_nodes import PORTS
            ports = PORTS
        elif node['type'] == 'Splitter':
            ports = [r['output'] for r in node['parts']]
        elif node['type'] == 'Replicator':
            ports = node['outputs']
        else:
            ports = [None]
        for port in ports:
            sink = '__recurrent_edit_check_' + str(len(proof['graph']['nodes']))
            while sink in lookup:
                sink += '_'
            proof['graph']['nodes'].append({'id': sink, 'type': 'Output', 'label': sink})
            edge = {'from': node['id'], 'to': sink}
            if port is not None:
                edge['from_port'] = port
            proof['graph']['edges'].append(edge)
    validate_model({**models, doc['id']: proof}, doc['id'])


def build_edit(models, token, options):
    """Accept valid affected paths, preserving non-target nodes/outgoing wires.

    Incoming wires follow the explicitly supplied source bindings. Removing a
    connected return port is refused, never converted into a guessed bypass.
    Unrelated invalid drafts remain editable, not silently repaired.
    """
    if not isinstance(token, dict) or set(token) != set(TOKEN):
        raise GraphError('Recurrent editing requires an exact graph/closure token.')
    info = prepare_edit(models, token['id'], token['node'], token['ref'])
    if any(info[key] != value for key, value in token.items()):
        raise GraphError('Architecture or transition changed; reopen recurrent editing.')
    if not isinstance(options, dict) or set(options) != set(OPTIONS):
        raise GraphError('Recurrent interface choices have missing or unsupported fields.')
    if type(options['version']) is not int or options['version'] != 1:
        raise GraphError('Recurrent editing supports integer version 1.')
    label = options['label']
    if not isinstance(label, str) or not label.strip() or len(label) > 60:
        raise GraphError('Choose a recurrent block name of 1–60 characters.')
    owner = options['share_with']
    if owner is not None and (not isinstance(owner, str) or not owner.strip()):
        raise GraphError('Parameter owner must be a stable block ID or null.')
    doc = copy.deepcopy(models[token['id']])
    graph = doc['graph']
    node = next(n for n in graph['nodes'] if n['id'] == token['node'])
    node.update({key: copy.deepcopy(options[key]) for key in OPTIONS if key not in {'version', 'share_with'}})
    node.update(ref=token['ref'], label=label.strip())
    if owner is None:
        node.pop('share_with', None)
    else:
        node['share_with'] = owner
    normalized(node)
    interface_contract(validate_model(models, token['ref']), specification(node))
    ports = {row['port'] for row in node['sequence_outputs'] + node['states']}
    for edge in graph['edges']:
        if edge['from'] == node['id'] and edge.get('from_port') not in ports:
            raise GraphError('Cannot remove a connected return port; disconnect its wires explicitly first.')
    incoming = {edge['from'] + ('@' + edge['from_port'] if edge.get('from_port') else ''): edge
                for edge in graph['edges'] if edge['to'] == node['id']}
    desired = sources(node)
    # Keep unchanged wire bytes and their original positions. Append only truly
    # new physical sources, in declared sequence/state/control order.
    graph['edges'] = [edge for edge in graph['edges'] if edge['to'] != node['id'] or
                      edge['from'] + ('@' + edge['from_port'] if edge.get('from_port') else '') in desired]
    for source in desired:
        if source in incoming:
            continue
        parts = source.split('@', 1)
        edge = {'from': parts[0], 'to': node['id']}
        if len(parts) == 2:
            edge['from_port'] = parts[1]
        graph['edges'].append(edge)
    _affected_contract(models, doc, node['id'])
    # No-op acceptance does not erase data, create history, or dirty the file.
    if doc == models[doc['id']]:
        return doc, node['id']
    doc.update(dataset=None, result=None)
    return doc, node['id']
