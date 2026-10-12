"""Concrete-first family specialization, without Torch, Qt, eval or a solver.

An external family is a checked template closure plus scoped integer assignments
and explicit scalar constructor choices and finite chain repeats. Runtime recurrence is
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
    'LSTM': {'hidden_size', 'num_layers', 'proj_size'},
}
ARRAY_FIELDS = {'shape', 'kernel_size', 'stride', 'padding', 'dilation', 'output_size', 'dims'}
CHOICE_FIELDS = {
    'Linear': {'bias': 'boolean'}, 'Conv2d': {'bias': 'boolean', 'padding_mode': 'enum'},
    'Activation': {'activation': 'enum'}, 'Concat': {'flatten': 'boolean'},
    'AvgPool2d': {'count_include_pad': 'boolean', 'ceil_mode': 'boolean'},
    'MaxPool2d': {'ceil_mode': 'boolean'},
    'LSTM': {'bias': 'boolean', 'bidirectional': 'boolean'},
}
TEMPLATE_FIELDS = dict(OPERATOR_FIELDS, Input={'features', 'shape', 'axes'},
    Output={'shape'}, Linear={'features', 'bias'}, Reshape={'shape'},
    Activation={'activation', 'dim'}, Module={'ref'}, Concat={'sources', 'flatten'},
    Splitter={'parts'}, Replicator={'outputs'},
    Recurrent={'ref', 'sequence_inputs', 'states', 'sequence_outputs', 'controls'},
    LSTM={'inputs', 'hidden_size', 'num_layers', 'proj_size', 'bias', 'bidirectional', 'lengths'})


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


def binding_atom(value, where, *, typed=False):
    """Closed JSON scalar, not Python text, floats, containers or bool-as-int."""
    if typed and (type(value) is bool or identifier(value)):
        return value
    return integer(value, where)


def symbol_type(symbol):
    return symbol.get('type', 'integer')


def binding_value(symbol, value, where):
    kind = symbol_type(symbol)
    require(type(value) is {'integer': int, 'boolean': bool, 'enum': str}[kind],
            where + ': expected ' + kind + ' binding (no coercion).')
    if kind == 'integer':
        integer(value, where)
        require(symbol.get('minimum', 1) <= value <= symbol.get('maximum', MAX_INTEGER),
                where + ': outside declared bounds.')
    elif kind == 'enum':
        require(identifier(value), where + ': enum value must be a bounded trimmed string.')
        require(value in symbol['values'], where + ': outside declared enum choices.')
    return value


def constructor_expression(value, kind, symbols, scope, where):
    if isinstance(value, dict):
        keys(value, ('symbol',), where=where)
        sid = value['symbol']
        require(isinstance(sid, str) and sid in symbols and symbols[sid]['scope'] == scope,
                where + ': unknown or out-of-scope constructor symbol.')
        require(symbol_type(symbols[sid]) == kind, where + ': constructor symbol type mismatch.')
        return [sid]
    require(type(value) is (bool if kind == 'boolean' else str),
            where + ': expected ' + kind + ' literal or symbol reference.')
    if kind == 'enum': require(identifier(value), where + ': invalid constructor string literal.')
    return []


def _condition(value, symbols, scope, where, *, depth, budget):
    """Family7 closed boolean constructor predicate, never tensor control flow."""
    budget[0] -= 1
    require(depth <= 24 and budget[0] >= 0, where + ': expression exceeds depth/node limit.')
    if type(value) is bool:
        return []
    require(isinstance(value, dict), where + ': condition needs a boolean or typed predicate.')
    if 'symbol' in value:
        return constructor_expression(value, 'boolean', symbols, scope, where)
    keys(value, ('cmp', 'args'), where=where)
    require(isinstance(value['cmp'], str) and value['cmp'] in ('eq', 'ne', 'gt', 'ge', 'lt', 'le') and
            isinstance(value['args'], list) and len(value['args']) == 2,
            where + ': condition needs an exact two-integer comparison.')
    return [sid for part in value['args'] for sid in
            expression(part, symbols, scope, where, depth=depth+1, budget=budget, choices=True)]


def expression(value, symbols, scope, where, *, depth=0, budget=None, choices=False):
    """Check a typed JSON AST and collect stable symbol IDs in first-use order."""
    budget = [256] if budget is None else budget
    budget[0] -= 1
    require(depth <= 24 and budget[0] >= 0, where + ': expression exceeds depth/node limit.')
    if type(value) is int:
        integer(value, where)
        return []
    require(isinstance(value, dict), where + ': use integer or typed expression, not Python text.')
    if 'select' in value:
        require(choices, where + ': constructor shape selection requires family version 7.')
        keys(value, ('select', 'then', 'else'), where=where)
        return (_condition(value['select'], symbols, scope, where, depth=depth+1, budget=budget) +
                expression(value['then'], symbols, scope, where, depth=depth+1, budget=budget, choices=True) +
                expression(value['else'], symbols, scope, where, depth=depth+1, budget=budget, choices=True))
    if 'symbol' in value:
        keys(value, ('symbol',), where=where)
        sid = value['symbol']
        require(isinstance(sid, str) and sid in symbols, where + ': unknown symbol ID.')
        require(symbols[sid]['scope'] == scope, where + ': symbol is outside this model scope.')
        require(symbol_type(symbols[sid]) == 'integer', where + ': arithmetic requires integer symbols.')
        return [sid]
    keys(value, ('op', 'args'), where=where)
    op, args = value['op'], value['args']
    require(isinstance(op, str) and op in {'add', 'mul', 'sub', 'div', 'floordiv'},
            where + ': unsupported integer expression operation.')
    require(isinstance(args, list) and (2 <= len(args) <= 16 if op in {'add', 'mul'} else len(args) == 2),
            where + ': invalid expression arity.')
    result = []
    for arg in args:
        for sid in expression(arg, symbols, scope, where, depth=depth + 1, budget=budget, choices=choices):
            if sid not in result:
                result.append(sid)
    return result


def evaluate(value, bindings, where):
    """Called only after grammar checks; every intermediate stays exactly bounded."""
    if type(value) is int:
        return value
    if 'select' in value:
        test = value['select']
        if type(test) is bool:
            selected = test
        elif 'symbol' in test:
            selected = bindings[test['symbol']]
        else:
            a, b = [evaluate(v, bindings, where) for v in test['args']]
            selected = {'eq': a == b, 'ne': a != b, 'gt': a > b,
                        'ge': a >= b, 'lt': a < b, 'le': a <= b}[test['cmp']]
        # Check both branches' grammar, but evaluate only the chosen branch.
        return evaluate(value['then'] if selected else value['else'], bindings, where)
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
    require(all(n in lookup and lookup[n]['type'] in LAYER_TYPES - {'Recurrent', 'LSTM'} for n in chain),
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
    keys(doc, ('version', 'id', 'name', 'root', 'models', 'symbols', 'assignments', 'repeats') +
         (('runtime_policy',) if doc.get('version') in (6, 7, 8, 9, 10) else ()),
         (('constructors',) if type(doc.get('version')) is int and doc['version'] >= 4 else ()) +
         (('runtime_inputs',) if doc.get('version') == 5 else ()))
    require(type(doc['version']) is int and doc['version'] in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'Family supports integer version 1 through 10.')
    require(identifier(doc['id']) and identifier(doc['name']), 'Family needs stable ID and name.')
    models = doc['models']
    require(isinstance(models, dict) and 1 <= len(models) <= 64 and
            isinstance(doc['root'], str) and doc['root'] in models, 'Family needs a root template closure.')
    if doc['version'] >= 10:
        require(any(isinstance(m, dict) and isinstance(m.get('graph'), dict) and 'parameter_policy' in m['graph']
                    for m in models.values()), 'Family10 requires an explicit parameter policy; use Family9 for old semantics.')
    for mid, model in models.items():
        require(identifier(mid), 'Family model needs a stable ID.')
        keys(model, ('id', 'name', 'graph'), where='Family model ' + mid)
        require(model['id'] == mid, 'Family model key/ID mismatch.')
        keys(model['graph'], ('nodes', 'edges'), ('input_order',) + (('parameter_policy',) if doc['version'] >= 10 else ()), 'Family graph ' + mid)
        try:
            check_document(model)
            for node in model['graph']['nodes']:
                require(node['type'] not in {'Add', 'Scale'} or doc['version'] >= 9, 'Add/Scale templates require explicit Family9.')
                require(node['type'] != 'LSTM' or doc['version'] in (7,8,9,10), 'Family versions 1–6 do not support LSTM templates; use explicit family7/8/9/10.')
                require(node['type'] != 'Recurrent' or doc['version'] in (6,7,8,9,10),
                        'Persisted recurrent nodes require a later recurrent family grammar, not family1-5.')
                require(set(node) <= COMMON | TEMPLATE_FIELDS[node['type']],
                        'Block ' + node['id'] + ': unsupported family template fields.')
                require(doc['version'] >= 2 or 'share_with' not in node,
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
            if node['type'] in {'Module', 'Recurrent'}:
                visit(node['ref'])
    visit(doc['root'])
    require(reached == set(models), 'Family closure contains unrelated models.')
    if 'runtime_inputs' in doc:
        from .runtime_shapes import declaration
        declaration(models, doc['root'], doc['runtime_inputs'])
    if doc['version'] in (6,7) and doc['runtime_policy'] is not None:
        require(not any(n['type'] == 'LSTM' for m in models.values() for n in m['graph']['nodes']),
                'LSTM families require a null fixed-T policy; runtime-temporal-v1 does not support LSTM.')
        value = doc['runtime_policy']
        keys(value, ('version', 'kind', 'inputs'), where='Family temporal policy')
        require(type(value['version']) is int and value['version'] == 1 and value['kind'] == 'runtime-temporal-v1',
                'Family temporal policy requires integer version1 and kind runtime-temporal-v1; LSTM needs a later runtime grammar.')
        from .temporal_shapes import declaration
        declaration(models, doc['root'], value['inputs'])
    if doc['version'] in (8,9,10) and doc['runtime_policy'] is not None:
        value = doc['runtime_policy']
        keys(value, ('version', 'kind', 'inputs'), where='Family temporal-v2 policy')
        require(type(value['version']) is int and value['version'] == 2 and value['kind'] == 'runtime-temporal-v2',
                'Family8 requires explicit version2/kind runtime-temporal-v2, or null fixed policy.')
        from .temporal_shapes_v2 import declaration
        declaration(models, doc['root'], value['inputs'])
    require(isinstance(doc['symbols'], list) and len(doc['symbols']) <= 128, 'Family allows at most 128 symbols.')
    symbols, names = {}, set()
    for symbol in doc['symbols']:
        keys(symbol, ('id', 'scope', 'name'), ('default', 'minimum', 'maximum', 'type', 'values') if doc['version'] >= 3
             else ('default', 'minimum', 'maximum'), 'Family symbol')
        sid, scope, name = symbol['id'], symbol['scope'], symbol['name']
        require(identifier(sid) and sid not in symbols and isinstance(scope, str) and scope in models and
                identifier(name) and (scope, name) not in names, 'Symbol ID must be unique; name unique within model scope.')
        kind = symbol_type(symbol)
        require(isinstance(kind, str) and kind in ('integer', 'boolean', 'enum'), 'Symbol ' + sid + ': unsupported scalar type.')
        if kind == 'integer':
            require('values' not in symbol, 'Integer symbol cannot have enum values.')
            low = integer(symbol.get('minimum', 1), 'Symbol ' + sid + ' minimum')
            high = integer(symbol.get('maximum', MAX_INTEGER), 'Symbol ' + sid + ' maximum')
            require(low <= high, 'Symbol ' + sid + ': minimum exceeds maximum.')
        else:
            require('minimum' not in symbol and 'maximum' not in symbol, 'Noninteger symbols cannot have arithmetic bounds.')
            if kind == 'boolean': require('values' not in symbol, 'Boolean domain is exactly false/true; no enum values.')
            else:
                values = symbol.get('values')
                require(isinstance(values, list) and 1 <= len(values) <= 64 and
                        all(identifier(v) for v in values) and len(set(values)) == len(values),
                        'Enum symbol needs 1..64 distinct bounded string choices.')
        if 'default' in symbol:
            binding_value(symbol, symbol['default'], 'Symbol ' + sid + ' default')
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
        choice = CHOICE_FIELDS.get(node['type'], {}).get(field) if node is not None and doc['version'] >= 3 else None
        require(node is not None and (choice or field in FIELDS.get(node['type'], set())) and field in node,
                where + ': target must be an existing supported ' + ('constructor' if doc['version'] >= 3 else 'integer') + ' field.')
        require(not (node['type'] == 'Input' and field == 'features' and 'shape' in node),
                where + ': explicit shape takes precedence; assign shape instead of ignored features.')
        target = (mid, nid, field)
        require(target not in targets, where + ': duplicate assignment target.')
        targets.add(target)
        value = assignment['value']
        if choice:
            used.update(constructor_expression(value, choice, symbols, mid, where))
            continue
        values = value if isinstance(value, list) else [value]
        if isinstance(value, list):
            require(field in ARRAY_FIELDS and 1 <= len(value) <= 8, where + ': unsupported field array.')
        for part in values:
            used.update(expression(part, symbols, mid, where, choices=doc['version'] >= 7))
    require(isinstance(doc['repeats'], list) and len(doc['repeats']) <= 32, 'Family allows at most 32 chain repeats.')
    repeat_ids, repeated = set(), set()
    for repeat in doc['repeats']:
        keys(repeat, ('id', 'model', 'nodes', 'count', 'mode'), where='Family repeat')
        require(identifier(repeat['id']) and repeat['id'] not in repeat_ids, 'Repeat ID must be unique.')
        require(repeat['mode'] == 'independent' or (doc['version'] >= 2 and repeat['mode'] == 'tied'),
                'Repeat ' + repeat['id'] + ': tied mode requires family version 2; runtime recurrence/unknown modes are not copies.')
        mid = repeat['model']
        require(isinstance(mid, str) and mid in models, 'Repeat needs an existing model scope.')
        _chain(models[mid], repeat)
        members = {(mid, nid) for nid in repeat['nodes']}
        require(not repeated & members, 'Repeat bodies cannot overlap or nest in version 1.')
        repeated.update(members)
        repeat_ids.add(repeat['id'])
        used.update(expression(repeat['count'], symbols, mid, 'Repeat ' + repeat['id'] + '.count', choices=doc['version'] >= 7))
    constructors = doc.get('constructors', [])
    require(isinstance(constructors, list) and len(constructors) <= 512,
            'Family allows at most 512 scoped constructor bindings.')
    constructor_targets = set()
    for call in constructors:
        keys(call, ('model', 'node', 'symbol', 'value'), where='Scoped constructor binding')
        mid, nid, sid = (call[k] for k in ('model', 'node', 'symbol'))
        require(isinstance(mid, str) and mid in models and isinstance(nid, str) and
                isinstance(sid, str) and sid in symbols, 'Constructor needs existing model/node/symbol identities.')
        node = next((n for n in models[mid]['graph']['nodes'] if n['id'] == nid), None)
        where = mid + '/' + nid + ' constructor ' + sid
        require(node is not None and node['type'] in {'Module', 'Recurrent'} and symbols[sid]['scope'] == node['ref'],
                where + ': target must belong to this reference\u2019s direct child scope.')
        target = (mid, nid, sid)
        require(target not in constructor_targets, where + ': duplicate constructor target.')
        constructor_targets.add(target)
        kind = symbol_type(symbols[sid])
        used.update(expression(call['value'], symbols, mid, where, choices=doc['version'] >= 7) if kind == 'integer' else
                    constructor_expression(call['value'], kind, symbols, mid, where))
    require(used == set(symbols), 'Every family symbol must have an assignment, repeat-count or constructor dependent.')
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
        if 'parameter_policy' in model['graph']:
            # Family10 template-local declarations apply to every occurrence,
            # not an implicit selection of occurrence zero. IDs remain atoms.
            from .policy_graph import owner_key
            import json
            translated = {}
            for selector, entry in model['graph']['parameter_policy']['parameters'].items():
                parts = json.loads(selector)
                variants = [[rid + '/' + str(i) + '/' + parts[0], *parts[1:]] for i in range(count)] if parts[0] in body else [parts]
                for variant in variants:
                    key = owner_key(variant)
                    require(key not in translated or translated[key] == entry, 'Conflicting repeated parameter policy.')
                    translated[key] = copy.deepcopy(entry)
            model['graph']['parameter_policy']['parameters'] = translated
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
        if dest['type'] in {'Concat', 'Add'}:
            dest['sources'] = [new_source if s == old_source else s for s in dest['sources']]
        elif dest['type'] == 'Recurrent':
            rows = dest['sequence_inputs'] + dest['states'] + list(dest.get('controls', {}).values())
            for row in rows:
                if row['source'] == old_source: row['source'] = new_source
        elif dest['type'] == 'LSTM':
            dest['inputs'] = {role: new_source if source == old_source else source
                              for role, source in dest['inputs'].items()}
            if dest.get('lengths') and dest['lengths']['source'] == old_source:
                dest['lengths']['source'] = new_source
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


class _UnresolvedConstructor(GraphError):
    def __init__(self, missing):
        self.missing = missing
        super().__init__('Bind required architecture symbols at ' + missing[0]['path'] +
                         ': ' + ', '.join(m['id'] for m in missing) + '.')


def _scoped_specialize(doc, symbols, supplied):
    """Resolve model specs by local constructor context, NOT parameter instances.

    A memoized child spec may be called by many independent Module objects.
    Explicit share_with still controls PyTorch parameter ownership. No tensors
    are allocated; bound the number of distinct contexts before copying graphs.
    """
    bindings = {sid: binding_value(s, supplied.get(sid, s.get('default')), 'Binding ' + sid)
                for sid, s in symbols.items() if sid in supplied or 'default' in s}
    family_hash = digest(doc)
    specialization_hash = digest({'version': 2, 'family': family_hash, 'bindings': bindings})
    scoped = {mid: {sid: s for sid, s in symbols.items() if s['scope'] == mid} for mid in doc['models']}
    memo, resolved, instances, occurrences = {}, {}, [], []
    model_ids = {mid: [] for mid in doc['models']}

    def instantiate(mid, overrides, path):
        local = {sid: overrides[sid] if sid in overrides else bindings[sid]
                 for sid in scoped[mid] if sid in overrides or sid in bindings}
        missing = [{'id': sid, 'scope': mid, 'name': s['name'], 'path': path}
                   for sid, s in scoped[mid].items() if sid not in local]
        if missing: raise _UnresolvedConstructor(missing)
        for sid, value in local.items(): binding_value(scoped[mid][sid], value, path + ' constructor ' + sid)
        key = digest({'model': mid, 'bindings': local})
        if key in memo: return memo[key]
        require(len(memo) < 64, path + ': resolved family exceeds 64 distinct constructor contexts; reduce variant fan-out.')
        rid = 'family-' + digest({'specialization': specialization_hash, 'context': key})
        memo[key] = rid
        model = copy.deepcopy(doc['models'][mid])
        for a in doc['assignments']:
            if a['model'] != mid: continue
            node = next(n for n in model['graph']['nodes'] if n['id'] == a['node'])
            value, where = a['value'], path + '/' + a['node'] + '.' + a['field']
            if a['field'] in CHOICE_FIELDS.get(node['type'], {}):
                node[a['field']] = local[value['symbol']] if isinstance(value, dict) else value
            else:
                node[a['field']] = ([evaluate(v, local, where) for v in value] if isinstance(value, list)
                                    else evaluate(value, local, where))
        repeats = [r for r in doc['repeats'] if r['model'] == mid]
        expanded = _expanded({mid: model}, repeats, local)
        origins = {o['node']: o['template_node'] for o in expanded}
        for node in model['graph']['nodes']:
            if node['type'] not in {'Module', 'Recurrent'}: continue
            origin = origins.get(node['id'], node['id'])
            child_path = path + '/' + node['id']
            overrides = {}
            for call in doc.get('constructors', []):
                if call['model'] != mid or call['node'] != origin: continue
                sid, value = call['symbol'], call['value']
                overrides[sid] = (evaluate(value, local, child_path + ' constructor ' + sid)
                                  if symbol_type(symbols[sid]) == 'integer' else
                                  local[value['symbol']] if isinstance(value, dict) else value)
            node['ref'] = instantiate(node['ref'], overrides, child_path)
        pin_input_order(model['graph'])
        if repeats or any('x' not in n or 'y' not in n for n in model['graph']['nodes']): _initial_layout(model)
        model['id'] = rid
        resolved[rid] = model
        try:
            check_document(model)
            validate_model(resolved, rid)
        except GraphError as error:
            raise GraphError('Resolved constructor ' + path + ': ' + str(error)) from error
        model_ids[mid].append(rid)
        instances.append({'model': rid, 'template': mid, 'bindings': local})
        occurrences.extend(dict(o, model=rid, template=mid) for o in expanded)
        return rid

    root = instantiate(doc['root'], {}, doc['root'])
    return {'version': 2, 'family': {'id': doc['id'], 'revision': family_hash},
            'specialization': specialization_hash, 'bindings': bindings, 'model_ids': model_ids,
            'instances': instances, 'root': root, 'models': resolved, 'occurrences': occurrences,
            'contract': validate_model(resolved, root), 'architecture_hash': identity(resolved, root)['architecture']}


def specialize(doc, supplied=None):
    """Return a verified concrete closure plus binding/provenance receipt, or raise."""
    symbols = validate_family(doc)
    supplied = {} if supplied is None else supplied
    require(isinstance(supplied, dict) and set(supplied) <= set(symbols), 'Bindings must map known stable symbol IDs to declared scalar values.')
    if doc['version'] >= 4:
        result = _scoped_specialize(doc, symbols, supplied)
        if doc['version'] == 5:
            from .runtime_shapes import specification
            spec = specification(result['models'], result['root'], doc['runtime_inputs']) if 'runtime_inputs' in doc else None
            result.update(version=3, runtime=None if spec is None else {
                'version':1,'inputs':copy.deepcopy(spec['runtime_inputs']),'specification_sha256':digest(spec)},
                runtime_specification=spec)
        elif doc['version'] in (6,7):
            from .temporal_shapes import specification
            policy = doc['runtime_policy']
            spec = specification(result['models'], result['root'], policy['inputs']) if policy is not None else None
            result.update(version=5 if doc['version']==7 else 4, runtime=None if spec is None else {
                'version':2, 'kind':'runtime-temporal-v1', 'inputs':copy.deepcopy(spec['runtime_inputs']),
                'specification_sha256':digest(spec)}, runtime_specification=spec)
        elif doc['version'] in (8,9,10):
            from .temporal_shapes_v2 import specification
            policy = doc['runtime_policy']
            spec = specification(result['models'], result['root'], policy['inputs']) if policy is not None else None
            result.update(version=6, runtime=None if spec is None else {
                'version':3, 'kind':'runtime-temporal-v2', 'inputs':copy.deepcopy(spec['runtime_inputs']),
                'specification_sha256':digest(spec)}, runtime_specification=spec)
        if doc['version'] >= 10:
            from .policy_graph import plan
            result['parameter_policy'] = plan(result['models'], result['root'])['resolved']
        return result
    bindings = {}
    for sid, symbol in symbols.items():
        require(sid in supplied or 'default' in symbol, 'Unresolved symbol ' + sid + ': bind ' + symbol['scope'] + '/' + symbol['name'] + '.')
        value = binding_value(symbol, supplied.get(sid, symbol.get('default')), 'Binding ' + sid)
        bindings[sid] = value
    models = copy.deepcopy(doc['models'])
    for assignment in doc['assignments']:
        mid, nid, field = (assignment[k] for k in ('model', 'node', 'field'))
        node = next(n for n in models[mid]['graph']['nodes'] if n['id'] == nid)
        value, where = assignment['value'], mid + '/' + nid + '.' + field
        if field in CHOICE_FIELDS.get(node['type'], {}):
            node[field] = bindings[value['symbol']] if isinstance(value, dict) else value
        else:
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
    require(isinstance(supplied, dict) and set(supplied) <= set(symbols), 'Bindings must map known stable symbol IDs to declared scalar values.')
    # Validate supplied values even if another symbol is still unresolved.
    for sid, value in supplied.items():
        binding_atom(value, 'Binding ' + sid, typed=doc['version'] >= 3)
        require(type(value) is {'integer': int, 'boolean': bool, 'enum': str}[symbol_type(symbols[sid])],
                'Binding ' + sid + ': expected ' + symbol_type(symbols[sid]) + ' binding (no coercion).')
        try: binding_value(symbols[sid], value, 'Binding ' + sid)
        except GraphError as error: return {'status': 'incompatible', 'reason': str(error)}
    missing = [{'id': sid, 'scope': s['scope'], 'name': s['name']}
               for sid, s in symbols.items() if sid not in supplied and 'default' not in s]
    if missing and doc['version'] < 4:
        return {'status': 'conditional', 'missing': missing, 'reason': 'Bind required architecture symbols before execution.'}
    try:
        return {'status': 'verified', 'resolved': specialize(doc, supplied)}
    except _UnresolvedConstructor as error:
        return {'status': 'conditional', 'missing': error.missing, 'reason': str(error)}
    except GraphError as error:
        return {'status': 'incompatible', 'reason': str(error)}
