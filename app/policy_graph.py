"""Versioned graph-scoped parameter policies; pure discovery and execution plan.

Selectors are canonical JSON component arrays relative to their architecture.
Old node metadata such as trainable:false remains unrelated and ignored.
"""
import copy
import json
from .core import GraphError
from .parameter_policy import _keys, _require, initializer, resolve


def owner_key(parts):
    return json.dumps(parts, separators=(',', ':'), ensure_ascii=True)


def present(models, mid, seen=None):
    seen = set() if seen is None else seen
    if mid in seen: return False
    seen.add(mid)
    graph = models[mid]['graph']
    return 'parameter_policy' in graph or any(present(models, n['ref'], seen)
        for n in graph['nodes'] if n['type'] in {'Module', 'Recurrent'})


def declaration(value):
    _keys(value, ('version', 'parameters'))
    _require(type(value['version']) is int and value['version'] == 1,
             'graph policy supports integer version 1.')
    _require(isinstance(value['parameters'], dict), 'parameters must be an object.')
    for selector, entry in value['parameters'].items():
        try: parts = json.loads(selector)
        except (ValueError, TypeError): raise GraphError('Parameter policy: selector must be a JSON component array.')
        _require(isinstance(parts, list) and 2 <= len(parts) <= 64 and
                 all(isinstance(p, str) and p for p in parts) and owner_key(parts) == selector,
                 'selector must be a canonical nonempty JSON component array, relative to the architecture.')
        _keys(entry, (), ('trainable', 'initializer'))
        if 'trainable' in entry: _require(type(entry['trainable']) is bool, 'trainable must be boolean.')
        if 'initializer' in entry: _require(isinstance(entry['initializer'], dict), 'initializer must be an object.')
    return copy.deepcopy(value)


def plan(models, mid):
    """Resolve explicit fields across scopes/aliases, then apply defaults once.

    A parent's freeze selection can combine with a child's initializer. Two
    explicit unequal declarations for the same field/owner refuse; order never
    chooses a winner. No arrays, random draws or numerical imports occur.
    """
    from .evidence import parameter_schema
    from .parameter_policy import graph_aliases
    schema = parameter_schema(models, mid)
    aliases = graph_aliases(models, mid)
    owners = {entry['id']: entry for entry in schema.values()}
    merged = {}
    def visit(current, components):
        graph = models[current]['graph']
        if 'parameter_policy' in graph:
            local = declaration(graph['parameter_policy'])
            for selector, entry in local['parameters'].items():
                key = owner_key(components + json.loads(selector))
                target = aliases.get(key, key)
                _require(target in owners, 'unknown graph parameter selector: ' + selector + '.')
                fields = copy.deepcopy(entry)
                if 'initializer' in fields: fields['initializer'] = initializer(fields['initializer'], owners[target]['shape'])
                previous = merged.setdefault(target, {})
                for field, value in fields.items():
                    _require(field not in previous or previous[field] == value,
                             'conflicting explicit declarations for shared owner: ' + target + '.')
                    previous[field] = value
        for node in graph['nodes']:
            if node['type'] in {'Module', 'Recurrent'}: visit(node['ref'], components + [node['id']])
    if present(models, mid): visit(mid, [mid])
    resolved = resolve(schema, {'version': 1, 'parameters': merged}, aliases=aliases)
    return {'schema': schema, 'policy': {'version': 1, 'parameters': merged}, 'resolved': resolved}
