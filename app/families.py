"""Concrete-first family specialization, without Torch, Qt, eval or a solver.

An external family is a checked template closure plus scoped integer assignments
and independent or explicitly tied finite chain repeats. Runtime recurrence is
not approximated by finite copies; sharing is explicit local module identity.
"""
import copy

from .core import GraphError, MAX_INTEGER, check_document, validate_model
from .evidence import digest, identity, pin_input_order
from .operators import LAYER_TYPES, COMMON, FIELDS as OPERATOR_FIELDS


FIELDS = {
    'Input': {'features', 'shape'}, 'Output': {'shape'},
    'Linear': {'features'}, 'Reshape': {'shape'},
    'Conv2d': {'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups'},
    'AvgPool2d': {'kernel_size', 'stride', 'padding'},
    'MaxPool2d': {'kernel_size', 'stride', 'padding', 'dilation'},
    'AdaptiveAvgPool2d': {'output_size'}, 'Flatten': {'start_dim', 'end_dim'},
    'Permute': {'dims'}, 'Activation': {'dim'},
}
ARRAY_FIELDS = {'shape', 'kernel_size', 'stride', 'padding', 'dilation', 'output_size', 'dims'}
TEMPLATE_FIELDS = dict(OPERATOR_FIELDS, Input={'features', 'shape', 'axes'},
    Output={'shape'}, Linear={'features', 'bias'}, Reshape={'shape'},
    Activation={'activation', 'dim'}, Module={'ref'}, Concat={'sources', 'flatten'},
    Splitter={'parts'}, Replicator={'outputs'})


def require(condition, message):
    if not condition:
        raise GraphError(message)


def keys(value, required, optional=(), where='Family'):
    require(isinstance(value, dict) and set(required) <= set(value) <= set(required) | set(optional),
            where + ': missing or unsupported fields.')


def identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 100 and value == value.strip()


def integer(value, where):
    require(type(value) is int and abs(value) <= MAX_INTEGER,
            where + ': expected an exact editor-range integer (not a boolean).')
    return value


def expression(value, symbols, scope, where, *, depth=0, budget=None):
    """Check a typed JSON AST and collect stable symbol IDs in first-use order."""
    budget = [256] if budget is None else budget
    budget[0] -= 1
    require(depth <= 24 and budget[0] >= 0, where + ': expression exceeds depth/node limit.')
    if type(value) is int:
        integer(value, where)
        return []
    require(isinstance(value, dict), where + ': use integer or typed expression, not Python text.')
    if 'symbol' in value:
        keys(value, ('symbol',), where=where)
        sid = value['symbol']
        require(isinstance(sid, str) and sid in symbols, where + ': unknown symbol ID.')
        require(symbols[sid]['scope'] == scope, where + ': symbol is outside this model scope.')
        return [sid]
    keys(value, ('op', 'args'), where=where)
    op, args = value['op'], value['args']
    require(isinstance(op, str) and op in {'add', 'mul', 'sub', 'div', 'floordiv'},
            where + ': unsupported integer expression operation.')
    require(isinstance(args, list) and (2 <= len(args) <= 16 if op in {'add', 'mul'} else len(args) == 2),
            where + ': invalid expression arity.')
    result = []
    for arg in args:
        for sid in expression(arg, symbols, scope, where, depth=depth + 1, budget=budget):
            if sid not in result:
                result.append(sid)
    return result


def evaluate(value, bindings, where):
    """Called only after grammar checks; every intermediate stays exactly bounded."""
    if type(value) is int:
        return value
    if 'symbol' in value:
        return bindings[value['symbol']]
    args = [evaluate(v, bindings, where) for v in value['args']]
    op, result = value['op'], args[0]
    for arg in args[1:]:
        if op == 'add':
            result += arg
        elif op == 'mul':
            result *= arg
        elif op == 'sub':
            result -= arg
        else:
            require(arg > 0, where + ': division requires a positive denominator.')
            require(op != 'div' or result % arg == 0, where + ': exact division has a remainder.')
            result //= arg
        integer(result, where + ' intermediate')
    return result


def _chain(model, repeat):
    """One unary chain with one incoming and one outgoing wire, no hidden branches."""
    where = 'Repeat ' + repeat['id']
    nodes, edges = model['graph']['nodes'], model['graph']['edges']
    lookup = {n['id']: n for n in nodes}
    chain = repeat['nodes']
    require(isinstance(chain, list) and 1 <= len(chain) <= 62 and
            all(isinstance(n, str) for n in chain) and len(set(chain)) == len(chain),
            where + ': nodes must be an ordered nonempty unique chain.')
    require(all(n in lookup and lookup[n]['type'] in LAYER_TYPES for n in chain),
            where + ': repeat only unary operation blocks, not graph boundaries/routers.')
    selected = set(chain)
    internal = [e for e in edges if e['from'] in selected and e['to'] in selected]
    expected = [(a, b) for a, b in zip(chain, chain[1:])]
    require(len(internal) == len(expected) and all(set(e) == {'from', 'to'} for e in internal) and
            {(e['from'], e['to']) for e in internal} == set(expected),
            where + ': body must have precisely the declared consecutive chain wires.')
    incoming = [e for e in edges if e['to'] in selected and e['from'] not in selected]
    outgoing = [e for e in edges if e['from'] in selected and e['to'] not in selected]
    require(len(incoming) == len(outgoing) == 1 and incoming[0]['to'] == chain[0] and
            outgoing[0]['from'] == chain[-1] and 'from_port' not in outgoing[0],
            where + ': body needs one ingress and one egress, with no branch crossings.')
    return lookup, incoming[0], outgoing[0]


def validate_family(doc):
    """Validate the family grammar; unresolved assignments are legitimate drafts."""
    keys(doc, ('version', 'id', 'name', 'root', 'models', 'symbols', 'assignments', 'repeats'))
    require(type(doc['version']) is int and doc['version'] in (1, 2), 'Family supports integer version 1 or 2.')
    require(identifier(doc['id']) and identifier(doc['name']), 'Family needs stable ID and name.')
    models = doc['models']
    require(isinstance(models, dict) and 1 <= len(models) <= 64 and
            isinstance(doc['root'], str) and doc['root'] in models, 'Family needs a root template closure.')
    for mid, model in models.items():
        require(identifier(mid), 'Family model needs a stable ID.')
        keys(model, ('id', 'name', 'graph'), where='Family model ' + mid)
        require(model['id'] == mid, 'Family model key/ID mismatch.')
        keys(model['graph'], ('nodes', 'edges'), ('input_order',), 'Family graph ' + mid)
        try:
            check_document(model)
            for node in model['graph']['nodes']:
                require(set(node) <= COMMON | TEMPLATE_FIELDS[node['type']],
                        'Block ' + node['id'] + ': unsupported family template fields.')
                require(doc['version'] == 2 or 'share_with' not in node,
                        'Explicit shared templates require family version 2.')
                require('op_version' not in node or (type(node['op_version']) is int and node['op_version'] == 1),
                        'Block ' + node['id'] + ': unsupported operator version.')
                require(node['type'] != 'Activation' or node['activation'] == 'Softmax' or 'dim' not in node,
                        'Block ' + node['id'] + ': dim applies only to Softmax.')
                if node['type'] == 'Splitter':
                    for part in node['parts']:
                        keys(part, ('output', 'features'), where='Family Splitter part')
            for edge in model['graph']['edges']:
                keys(edge, ('from', 'to'), ('from_port',), 'Family wire')
            validate_model(models, mid)
        except (GraphError, KeyError, TypeError, AttributeError) as error:
            raise GraphError('Family template ' + mid + ': ' + str(error)) from error
    reached = set()
    def visit(mid):
        if mid in reached:
            return
        reached.add(mid)
        for node in models[mid]['graph']['nodes']:
            if node['type'] == 'Module':
                visit(node['ref'])
    visit(doc['root'])
    require(reached == set(models), 'Family closure contains unrelated models.')
    require(isinstance(doc['symbols'], list) and len(doc['symbols']) <= 128, 'Family allows at most 128 symbols.')
    symbols, names = {}, set()
    for symbol in doc['symbols']:
        keys(symbol, ('id', 'scope', 'name'), ('default', 'minimum', 'maximum'), 'Family symbol')
        sid, scope, name = symbol['id'], symbol['scope'], symbol['name']
        require(identifier(sid) and sid not in symbols and isinstance(scope, str) and scope in models and
                identifier(name) and (scope, name) not in names, 'Symbol ID must be unique; name unique within model scope.')
        low = integer(symbol.get('minimum', 1), 'Symbol ' + sid + ' minimum')
        high = integer(symbol.get('maximum', MAX_INTEGER), 'Symbol ' + sid + ' maximum')
        require(low <= high, 'Symbol ' + sid + ': minimum exceeds maximum.')
        if 'default' in symbol:
            default = integer(symbol['default'], 'Symbol ' + sid + ' default')
            require(low <= default <= high, 'Symbol ' + sid + ': default outside bounds.')
        symbols[sid] = symbol
        names.add((scope, name))
    require(isinstance(doc['assignments'], list) and len(doc['assignments']) <= 512,
            'Family allows at most 512 field assignments.')
    used, targets = set(), set()
    for assignment in doc['assignments']:
        keys(assignment, ('model', 'node', 'field', 'value'), where='Family assignment')
        mid, nid, field = (assignment[k] for k in ('model', 'node', 'field'))
        require(isinstance(mid, str) and mid in models and isinstance(nid, str) and isinstance(field, str),
                'Assignment needs model/node/field identities.')
        node = next((n for n in models[mid]['graph']['nodes'] if n['id'] == nid), None)
        where = mid + '/' + nid + '.' + field
        require(node is not None and field in FIELDS.get(node['type'], set()) and field in node,
                where + ': target must be an existing supported integer field.')
        require(not (node['type'] == 'Input' and field == 'features' and 'shape' in node),
                where + ': explicit shape takes precedence; assign shape instead of ignored features.')
        target = (mid, nid, field)
        require(target not in targets, where + ': duplicate assignment target.')
        targets.add(target)
        value = assignment['value']
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list):
            require(field in ARRAY_FIELDS and 1 <= len(value) <= 8, where + ': unsupported field array.')
        for part in values:
            used.update(expression(part, symbols, mid, where))
    require(isinstance(doc['repeats'], list) and len(doc['repeats']) <= 32, 'Family allows at most 32 chain repeats.')
    repeat_ids, repeated = set(), set()
    for repeat in doc['repeats']:
        keys(repeat, ('id', 'model', 'nodes', 'count', 'mode'), where='Family repeat')
        require(identifier(repeat['id']) and repeat['id'] not in repeat_ids, 'Repeat ID must be unique.')
        require(repeat['mode'] == 'independent' or (doc['version'] == 2 and repeat['mode'] == 'tied'),
                'Repeat ' + repeat['id'] + ': tied mode requires family version 2; runtime recurrence/unknown modes are not copies.')
        mid = repeat['model']
        require(isinstance(mid, str) and mid in models, 'Repeat needs an existing model scope.')
        _chain(models[mid], repeat)
        members = {(mid, nid) for nid in repeat['nodes']}
        require(not repeated & members, 'Repeat bodies cannot overlap or nest in version 1.')
        repeated.update(members)
        repeat_ids.add(repeat['id'])
        used.update(expression(repeat['count'], symbols, mid, 'Repeat ' + repeat['id'] + '.count'))
    require(used == set(symbols), 'Every family symbol must have an assignment or repeat-count dependent.')
    return symbols


def _expanded(models, repeats, bindings):
    """Bound expansion BEFORE materializing copies; never silently clamp a depth."""
    counts, totals = {}, {mid: len(m['graph']['nodes']) for mid, m in models.items()}
    for repeat in repeats:
        count = evaluate(repeat['count'], bindings, 'Repeat ' + repeat['id'] + '.count')
        require(count >= 1, 'Repeat ' + repeat['id'] + ': count must be positive (zero is not an identity assumption).')
        counts[repeat['id']] = count
        totals[repeat['model']] += (count - 1) * len(repeat['nodes'])
    require(all(n <= 64 for n in totals.values()), 'Resolved repeat exceeds current 64-block concrete graph limit; reduce depth.')
    occurrences = []
    for repeat in repeats:
        mid, rid = repeat['model'], repeat['id']
        model, count = models[mid], counts[rid]
        lookup, incoming, outgoing = _chain(model, repeat)
        body = set(repeat['nodes'])
        clones, wires = [], []
        for index in range(count):
            ids = {nid: rid + '/' + str(index) + '/' + nid for nid in repeat['nodes']}
            require(not (set(ids.values()) & set(lookup)), 'Repeat ' + rid + ': occurrence ID collides with existing block.')
            for nid in repeat['nodes']:
                clone = copy.deepcopy(lookup[nid])
                clone['id'] = ids[nid]
                if 'share_with' in clone and clone['share_with'] in body:
                    clone['share_with'] = rid + '/' + str(0 if repeat['mode'] == 'tied' else index) + '/' + clone['share_with']
                elif repeat['mode'] == 'tied' and index and clone['type'] in {'Linear', 'Conv2d', 'Module'} and 'share_with' not in clone:
                    clone['share_with'] = rid + '/0/' + nid
                clones.append(clone)
                occurrences.append({'model': mid, 'repeat': rid, 'index': index,
                                    'template_node': nid, 'node': ids[nid], 'mode': repeat['mode']})
            wires.extend({'from': ids[a], 'to': ids[b]} for a, b in zip(repeat['nodes'], repeat['nodes'][1:]))
            if index:
                wires.append({'from': rid + '/' + str(index - 1) + '/' + repeat['nodes'][-1],
                              'to': ids[repeat['nodes'][0]]})
        first, last = clones[0]['id'], clones[-1]['id']
        old_source, new_source = outgoing['from'], last
        # A downstream Concat identifies its source explicitly; keep its order.
        dest = lookup[outgoing['to']]
        if dest['type'] == 'Concat':
            dest['sources'] = [new_source if s == old_source else s for s in dest['sources']]
        ingress, egress = dict(incoming, to=first), dict(outgoing, **{'from': last})
        model['graph']['edges'] = [e for e in model['graph']['edges'] if e['from'] not in body and e['to'] not in body] + [ingress] + wires + [egress]
        insertion = min(i for i, n in enumerate(model['graph']['nodes']) if n['id'] in body)
        retained = [n for n in model['graph']['nodes'] if n['id'] not in body]
        for node in retained:
            if node.get('share_with') in body:
                node['share_with'] = rid + '/0/' + node['share_with']
        model['graph']['nodes'] = retained[:insertion] + clones + retained[insertion:]
    return occurrences


def _initial_layout(model):
    """Readable layered seed positions, not the configurable browser relayout.

Only expanded models or templates lacking coordinates use this seed. It does
not enter computational identity; the existing Layout menu remains available.
"""
    from .graph import topology
    depths, rows = {}, {}
    for node in topology(model):
        parents = [e['from'] for e in model['graph']['edges'] if e['to'] == node['id']]
        depth = max((depths[p] + 1 for p in parents), default=0)
        row = rows.get(depth, 0)
        rows[depth] = row + 1
        depths[node['id']] = depth
        node.update(x=90 + 260 * depth, y=80 + 160 * row)


def specialize(doc, supplied=None):
    """Return a verified concrete closure plus binding/provenance receipt, or raise."""
    symbols = validate_family(doc)
    supplied = {} if supplied is None else supplied
    require(isinstance(supplied, dict) and set(supplied) <= set(symbols), 'Bindings must map known stable symbol IDs to integers.')
    bindings = {}
    for sid, symbol in symbols.items():
        require(sid in supplied or 'default' in symbol, 'Unresolved symbol ' + sid + ': bind ' + symbol['scope'] + '/' + symbol['name'] + '.')
        value = integer(supplied.get(sid, symbol.get('default')), 'Binding ' + sid)
        require(symbol.get('minimum', 1) <= value <= symbol.get('maximum', MAX_INTEGER), 'Binding ' + sid + ': outside declared bounds.')
        bindings[sid] = value
    models = copy.deepcopy(doc['models'])
    for assignment in doc['assignments']:
        mid, nid, field = (assignment[k] for k in ('model', 'node', 'field'))
        node = next(n for n in models[mid]['graph']['nodes'] if n['id'] == nid)
        value, where = assignment['value'], mid + '/' + nid + '.' + field
        node[field] = ([evaluate(v, bindings, where) for v in value] if isinstance(value, list)
                       else evaluate(value, bindings, where))
    occurrences = _expanded(models, doc['repeats'], bindings)
    expanded_models = {r['model'] for r in doc['repeats']}
    for mid, model in models.items():
        pin_input_order(model['graph'])
        try:
            if mid in expanded_models or any('x' not in n or 'y' not in n for n in model['graph']['nodes']):
                _initial_layout(model)
            check_document(model)
            validate_model(models, mid)
        except GraphError as error:
            fields = [a['node'] + '.' + a['field'] for a in doc['assignments'] if a['model'] == mid]
            origin = ' (assigned fields: ' + ', '.join(fields) + ')' if fields else ''
            raise GraphError('Resolved model ' + mid + origin + ': ' + str(error)) from error
    family_hash = digest(doc)
    specialization_hash = digest({'version': 1, 'family': family_hash, 'bindings': bindings})
    remap = {mid: 'family-' + digest({'specialization': specialization_hash, 'model': mid}) for mid in models}
    resolved = {}
    for mid, model in models.items():
        model['id'] = remap[mid]
        for node in model['graph']['nodes']:
            if node['type'] == 'Module':
                node['ref'] = remap[node['ref']]
        resolved[model['id']] = model
    root = remap[doc['root']]
    return {'version': 1, 'family': {'id': doc['id'], 'revision': family_hash},
            'specialization': specialization_hash, 'bindings': bindings, 'model_ids': remap,
            'root': root, 'models': resolved, 'occurrences': occurrences,
            'contract': validate_model(resolved, root), 'architecture_hash': identity(resolved, root)['architecture']}


def preview(doc, supplied=None):
    """Keep unresolved/invalid selections visible; malformed grammar still raises."""
    symbols = validate_family(doc)
    supplied = {} if supplied is None else supplied
    require(isinstance(supplied, dict) and set(supplied) <= set(symbols), 'Bindings must map known stable symbol IDs to integers.')
    # Validate supplied values even if another symbol is still unresolved.
    for sid, value in supplied.items():
        integer(value, 'Binding ' + sid)
        if not symbols[sid].get('minimum', 1) <= value <= symbols[sid].get('maximum', MAX_INTEGER):
            return {'status': 'incompatible', 'reason': 'Binding ' + sid + ': outside declared bounds.'}
    missing = [{'id': sid, 'scope': s['scope'], 'name': s['name']}
               for sid, s in symbols.items() if sid not in supplied and 'default' not in s]
    if missing:
        return {'status': 'conditional', 'missing': missing, 'reason': 'Bind required architecture symbols before execution.'}
    try:
        return {'status': 'verified', 'resolved': specialize(doc, supplied)}
    except GraphError as error:
        return {'status': 'incompatible', 'reason': str(error)}
