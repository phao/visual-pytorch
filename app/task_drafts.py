"""Literal owned task card/transform pages, not partial persisted TaskSpecs.

Ordinals retain scientific order and are valid only at the exact session
sequence. Large properties remain explicit staged-value targets, not clipped
JSON. Full validation/storage costs remain native.
"""
from . import dataset_drafts, science_browser
from .datasets import _integer, _keys, _require

COLLECTIONS = ('inputs', 'heads')
PROPERTIES = {
    'inputs': ('port', 'field', 'derived', 'length'),
    'heads': ('id', 'port', 'field', 'loss', 'weight', 'mask', 'sample_weight'),
}
TRANSFORM_PROPERTIES = ('kind', 'scale', 'offset', 'length', 'value')
MAX_CARDS = 6


def _collection(document, collection):
    _require(isinstance(collection, str) and collection in COLLECTIONS,
             'Choose ordered input bindings or loss heads.')
    _require(isinstance(document.get(collection), list),
             'Typed task editing needs an ordered binding array.')
    return document[collection], [list(document).index(collection)]


def _cursor(offset, limit):
    _require(_integer(offset) and _integer(limit, 1, MAX_CARDS),
             'Typed task pages need an exact nonnegative offset and 1..6 cards.')


def _base(header, collection, offset, limit, total, scope):
    return {'version': 1, 'scope': scope, 'session': header['session'],
            'sequence': header['sequence'], 'workspace_id': header['workspace_id'],
            'id': header['id'], 'collection': collection, 'offset': offset,
            'limit': limit, 'total': total, 'rows': [], 'next_offset': None,
            'byte_limit': science_browser.MAX_BYTES,
            'text_limit': dataset_drafts.MAX_TEXT}


def _append(result, item):
    end = result['offset'] + len(result['rows']) + 1
    cursor = end if end < result['total'] else None
    prospective = {**result, 'rows': [*result['rows'], item], 'next_offset': cursor}
    if len(science_browser.encoded(prospective)) > science_browser.MAX_BYTES:
        return False
    result['rows'].append(item)
    result['next_offset'] = cursor
    return True


def _finish(result):
    _require(result['rows'] or result['offset'] >= result['total'],
             'Task card cannot fit its bounded projection. Inspect the raw draft value explicitly.')
    return science_browser._bounded(result)


def page(document, header, collection, *, offset=0, limit=MAX_CARDS):
    _cursor(offset, limit)
    bindings, path = _collection(document, collection)
    result = _base(header, collection, offset, limit, len(bindings), 'owned-task-page')
    result['path'] = path
    result['description'] = dataset_drafts._property(document, 'description', [])
    for index in range(offset, min(len(bindings), offset + limit)):
        binding = bindings[index]; target = path + [index]
        _require(isinstance(binding, dict) and isinstance(binding.get('transforms'), list),
                 'Typed task cards need binding objects with ordered transforms. Use raw draft inspection to repair them.')
        item = {'ordinal': index, 'path': target,
                'properties': [dataset_drafts._property(binding, key, target)
                               for key in PROPERTIES[collection]],
                'transforms': {'path': target + [list(binding).index('transforms')],
                               'total': len(binding['transforms'])}}
        if not _append(result, item):
            break
    return _finish(result)


def transforms(document, header, collection, binding, *, offset=0, limit=MAX_CARDS):
    _cursor(offset, limit)
    bindings, path = _collection(document, collection)
    _require(_integer(binding) and binding < len(bindings), 'Choose an existing exact binding ordinal.')
    owner = bindings[binding]
    _require(isinstance(owner, dict) and isinstance(owner.get('transforms'), list),
             'Typed transform editing needs an ordered transform array.')
    values = owner['transforms']; target = path + [binding, list(owner).index('transforms')]
    result = _base(header, collection, offset, limit, len(values), 'owned-task-transform-page')
    result['path'] = target
    result['binding'] = binding
    for index in range(offset, min(len(values), offset + limit)):
        step = values[index]; step_path = target + [index]
        _require(isinstance(step, dict), 'Typed transforms need objects. Use raw draft inspection to repair them.')
        item = {'ordinal': index, 'path': step_path,
                'properties': [dataset_drafts._property(step, key, step_path)
                               for key in TRANSFORM_PROPERTIES]}
        if not _append(result, item):
            break
    return _finish(result)


def operations(values):
    """Keep literal numbers out of JavaScript Number; normal ordinal edits stay ordered."""
    _require(isinstance(values, list) and 1 <= len(values) <= 25,
             'Typed task edit needs 1..25 ordered operations.')
    result = []
    for value in values:
        _require(isinstance(value, dict) and isinstance(value.get('op'), str),
                 'Choose a typed task edit operation.')
        kind = value['op']
        if kind in ('put-json', 'replace-json', 'insert-json'):
            required = ('op', 'path', 'text') + (('key',) if kind == 'put-json'
                        else ('index',) if kind == 'insert-json' else ())
            _keys(value, required, context='Literal task edit')
            result.append({**{key: item for key, item in value.items() if key != 'text'},
                           'op': kind.removesuffix('-json'),
                           'value': dataset_drafts._text(value['text'])})
        else:
            _require(kind in ('remove', 'move'), 'Typed task values use literal JSON text.')
            result.append(value)
    return result


def edit(document, operation):
    """Resolve named properties against each current draft, not stale dict ordinals."""
    from .authoring_sessions import _operation
    if isinstance(operation, dict) and operation.get('op') == 'property':
        _keys(operation, ('op', 'target', 'key', 'text'), context='Task property edit')
        target = operation['target']; _require(isinstance(target, dict), 'Choose a task property target.')
        kind = target.get('kind'); path = []
        if kind == 'document':
            _keys(target, ('kind',), context='Task document target'); allowed = ('name', 'description')
        else:
            _require(kind in ('binding', 'transform'), 'Choose document, binding or transform properties.')
            _keys(target, ('kind', 'collection', 'binding') + (('transform',) if kind == 'transform' else ()),
                  context='Task member target')
            bindings, path = _collection(document, target['collection'])
            index = target['binding']; _require(_integer(index) and index < len(bindings), 'Task binding ordinal is missing.')
            path += [index]; member = bindings[index]
            _require(isinstance(member, dict), 'Task binding must be an object.')
            allowed = PROPERTIES[target['collection']]
            if kind == 'transform':
                values = member.get('transforms'); index = target['transform']
                _require(isinstance(values, list) and _integer(index) and index < len(values),
                         'Task transform ordinal is missing.')
                path += [list(member).index('transforms'), index]; allowed = TRANSFORM_PROPERTIES
        key = operation['key']; _require(isinstance(key, str) and key in allowed, 'Choose a supported typed task property.')
        if operation['text'] is None:
            # Locate the current dictionary key after earlier removals/puts.
            from .authoring_sessions import _at
            member = _at(document, path); _require(isinstance(member, dict), 'Task property needs an object.')
            if key not in member: return document
            return _operation(document, {'op':'remove', 'path':path+[list(member).index(key)]})
        return _operation(document, {'op':'put', 'path':path, 'key':key,
                                     'value':dataset_drafts._text(operation['text'])})
    return _operation(document, operations([operation])[0])
