"""Typed owned-family cards; bounded projections, never partial retained specs.

Expressions are finite JSON AST values, never evaluated Python or inferred from
data. Draft edits may be incomplete; explicit check/retention enforce the grammar.
"""
import copy

from . import dataset_drafts, science_browser
from .datasets import _integer, _keys, _require
from .core import GraphError
from .experiment_drafts import _locate
from .families import preview as resolve_preview

PROPERTIES = {
    'symbols': ('id', 'scope', 'name', 'default', 'minimum', 'maximum'),
    'assignments': ('model', 'node', 'field', 'value'),
    'repeats': ('id', 'model', 'nodes', 'count', 'mode'),
    'templates': ('id', 'name', 'graph'),
}
ROOTS = ('name', 'root')
MAX_TEXT = 2000


def _header(header, scope, **fields):
    return {'version': 1, 'scope': scope, 'session': header['session'], 'sequence': header['sequence'],
            'workspace_id': header['workspace_id'], 'id': header['id'],
            'byte_limit': science_browser.MAX_BYTES, **fields}


def _collection(document, collection):
    _require(isinstance(collection, str) and collection in PROPERTIES, 'Choose symbols, assignments, repeats or templates.')
    value, path = _locate(document, ('models' if collection == 'templates' else collection,), [])
    _require(isinstance(value, dict if collection == 'templates' else list), 'Family collection has the wrong container; repair the staged value explicitly.')
    return value, path


def _property(value, path, key):
    _require(isinstance(value, dict), 'Family card needs an object; repair its staged value explicitly.')
    current, absolute = _locate(value, (key,), path, optional=True)
    return dataset_drafts._descriptor(current, absolute, present=absolute is not None, key=key, limit=MAX_TEXT)


def page(document, header, collection, *, offset=0, limit=6):
    _require(_integer(offset) and _integer(limit, 1, 6), 'Family cards need an exact nonnegative offset and 1..6 entries.')
    values, path = _collection(document, collection)
    result = _header(header, 'owned-family-page', collection=collection, offset=offset, limit=limit,
                     total=len(values), path=path, rows=[], next_offset=None, text_limit=MAX_TEXT,
                     roots=[_property(document, [], k) for k in ROOTS])
    for ordinal in range(offset, min(len(values), offset + limit)):
        key = list(values)[ordinal] if isinstance(values, dict) else ordinal
        row = {'ordinal': ordinal, 'template': key if collection == 'templates' else None,
               'path': path + [ordinal],
               'properties': [_property(values[key], path + [ordinal], k) for k in PROPERTIES[collection]]}
        proposed = {**result, 'rows': [*result['rows'], row],
                    'next_offset': ordinal + 1 if ordinal + 1 < len(values) else None}
        if len(science_browser.encoded(proposed)) > science_browser.MAX_BYTES: break
        result = proposed
    _require(result['rows'] or offset >= len(values), 'Family card cannot fit its bounded projection; inspect the staged value.')
    return science_browser._bounded(result)


def _target(document, target, key):
    _require(isinstance(target, dict), 'Choose a family property target.')
    if target.get('kind') == 'document':
        _keys(target, ('kind',), context='Family document target')
        _require(key in ROOTS, 'Choose family name or root; stable document ID cannot be edited.')
        return document, [], key
    _keys(target, ('kind', 'collection', 'ordinal'), context='Family card target')
    _require(target['kind'] == 'entry', 'Choose a family document or ordered card.')
    values, path = _collection(document, target['collection']); ordinal = target['ordinal']
    _require(_integer(ordinal) and ordinal < len(values), 'Family card ordinal no longer exists.')
    _require(isinstance(key, str) and key in PROPERTIES[target['collection']] and
             not (target['collection'] == 'templates' and key == 'id'), 'Choose an editable family property; template identity is stable.')
    member = list(values)[ordinal] if isinstance(values, dict) else ordinal
    _require(isinstance(values[member], dict), 'Family card needs an object.')
    return values[member], path + [ordinal], key


def edit(document, operation):
    from .authoring_sessions import _operation
    _require(isinstance(operation, dict), 'Family edit needs a typed operation.')
    if operation.get('op') == 'property':
        _keys(operation, ('op', 'target', 'key', 'text'), context='Family property edit')
        value, path, key = _target(document, operation['target'], operation['key'])
        text = operation['text']
        if text is None:
            if key not in value: return document
            return _operation(document, {'op': 'remove', 'path': path + [list(value).index(key)]})
        _require(isinstance(text, str) and len(text) <= MAX_TEXT, 'Use staged JSON editing for values over 2000 characters.')
        from .revisions import _parse
        parsed = _parse(text)
        return _operation(document, {'op': 'put', 'path': path, 'key': key, 'value': parsed})
    kind = operation.get('op')
    _require(kind in ('insert', 'remove', 'move'), 'Choose a property, insert, remove or move family edit.')
    _keys(operation, ('op', 'collection', 'index', 'value') if kind == 'insert' else
          ('op', 'collection', 'from', 'to') if kind == 'move' else ('op', 'collection', 'index'),
          context='Family structural edit')
    _require(operation['collection'] != 'templates', 'Seed template identity/closure from a drawing or explicitly stage its models.')
    _, path = _collection(document, operation['collection'])
    if kind == 'insert':
        return _operation(document, {'op': 'insert', 'path': path, 'index': operation['index'], 'value': operation['value']})
    if kind == 'move':
        return _operation(document, {'op': 'move', 'path': path, 'from': operation['from'], 'to': operation['to']})
    _require(_integer(operation['index']), 'Remove needs an exact nonnegative family card index.')
    return _operation(document, {'op': 'remove', 'path': path + [operation['index']]})


def preview(document, header, bindings=None):
    source = resolve_preview(document, bindings)
    result = _header(header, 'owned-family-preview', status=source['status'])
    if source['status'] != 'verified':
        result['reason'] = source['reason']
        if 'missing' in source:
            result.update(missing=[], missing_total=len(source['missing']), missing_next_offset=None)
            for row in source['missing'][:25]:
                proposed = {**result, 'missing': [*result['missing'], row]}
                if len(science_browser.encoded(proposed)) > science_browser.MAX_BYTES: break
                result = proposed
            if len(result['missing']) < result['missing_total']:
                result['missing_next_offset'] = len(result['missing'])
    else:
        resolved = source['resolved']; contract = resolved['contract']
        result.update(family=resolved['family'], specialization=resolved['specialization'],
                      resolved_bindings=dataset_drafts._descriptor(resolved['bindings'], None, limit=MAX_TEXT),
                      binding_count=len(resolved['bindings']), architecture_hash=resolved['architecture_hash'],
                      root=resolved['root'], architectures=len(resolved['models']),
                      blocks=sum(len(m['graph']['nodes']) for m in resolved['models'].values()),
                      parameters=contract['parameters'], input_shape=contract['input_shape'],
                      output_shape=contract['output_shape'], occurrences=len(resolved['occurrences']))
    return science_browser._bounded(result)
