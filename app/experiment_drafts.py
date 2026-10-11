"""Bounded owned experiment controls; projections are not persisted specs.

Scientific axes keep their order. Named fields resolve against each current
draft; local choice browsing may sort metadata without reordering stored axes.
Full native base/validation/dependency/manifest costs are separate from bounds.
"""
import copy
from . import dataset_drafts, science_browser
from .datasets import _identifier, _integer, _keys, _require
from .experiment_objects import reference
from .core import GraphError

COLLECTIONS = ('models', 'data_tasks', 'splits', 'initialization_seeds',
               'protocols', 'pairs', 'exclusions')
ROOTS = {'name': ('name',), 'question': ('question',),
         'selection.kind': ('selection', 'kind')}
PROPERTIES = {
    'models': {'reference': ()},
    'data_tasks': {k: (k,) for k in ('id', 'dataset', 'task')},
    'splits': {'id': ('id',), **{k: ('protocol', k) for k in ('method', 'seed', 'field', 'excluded')},
               **{'fractions.'+str(i): ('protocol', 'fractions', i) for i in range(3)},
               **{'roles.'+r: ('protocol', 'roles', r) for r in ('train', 'validation', 'test')}},
    'initialization_seeds': {'value': ()},
    'protocols': {'id': ('id',), **{k: ('protocol', k) for k in ('batch_size', 'evaluate_every', 'shuffle')},
                  'history.max_points': ('protocol', 'history', 'max_points'),
                  **{'budget.'+k: ('protocol', 'budget', k) for k in ('epochs', 'updates', 'examples', 'seconds')},
                  **{'optimizer.'+k: ('protocol', 'optimizer', k) for k in ('lr', 'eps', 'weight_decay')},
                  **{'optimizer.betas.'+str(i): ('protocol', 'optimizer', 'betas', i) for i in range(2)},
                  **{'seeds.'+k: ('protocol', 'seeds', k) for k in ('initialization', 'shuffle')}},
    'pairs': {k: (k,) for k in ('model', 'data_task')},
    'exclusions': {k: (k,) for k in ('model', 'data_task', 'reason')},
}
REFERENCE_KEYS = {'reference': 'model', 'model': 'model', 'dataset': 'dataset', 'task': 'task'}


def _collection(document, collection):
    _require(isinstance(collection, str) and collection in COLLECTIONS,
             'Choose an ordered experiment selection collection.')
    keys = ('selection', collection) if collection in ('pairs', 'exclusions') else (collection,)
    value, path = _locate(document, keys, [])
    _require(isinstance(value, list), 'Typed experiment selection needs an array. Repair the raw draft explicitly.')
    return value, path


def _locate(value, keys, path, *, optional=False):
    path = list(path)
    for key in keys:
        if isinstance(key, str):
            if not isinstance(value, dict) or key not in value:
                if optional: return None, None
                raise GraphError('Typed experiment property is missing; repair the raw draft explicitly.')
            path.append(list(value).index(key)); value = value[key]
        else:
            if not isinstance(value, list) or key >= len(value):
                if optional: return None, None
                raise GraphError('Typed experiment ordered property is missing; repair the raw draft explicitly.')
            path.append(key); value = value[key]
    return value, path


def _property(value, keys, path, key):
    current, absolute = _locate(value, keys, path, optional=True)
    # Complete bounded exact references must not become clipped selector values.
    limit = 8192 if key in REFERENCE_KEYS else dataset_drafts.MAX_TEXT
    return dataset_drafts._descriptor(current, absolute, present=absolute is not None, key=key, limit=limit)


def _header(header, scope, **fields):
    return {'version': 1, 'scope': scope, 'session': header['session'],
            'sequence': header['sequence'], 'workspace_id': header['workspace_id'],
            'id': header['id'], 'byte_limit': science_browser.MAX_BYTES, **fields}


def _append(page, row):
    end = page['offset'] + len(page['rows']) + 1
    next_offset = end if end < page['total'] else None
    proposed = {**page, 'rows': [*page['rows'], row], 'next_offset': next_offset}
    if len(science_browser.encoded(proposed)) > science_browser.MAX_BYTES: return False
    page['rows'].append(row); page['next_offset'] = next_offset
    return True


def _finish(page):
    _require(page['rows'] or page['offset'] >= page['total'],
             'Experiment entry cannot fit its projection. Inspect the raw value explicitly.')
    return science_browser._bounded(page)


def page(document, header, collection, *, offset=0, limit=6):
    _require(_integer(offset) and _integer(limit, 1, 6),
             'Experiment cards need an exact nonnegative offset and 1..6 entries.')
    values, path = _collection(document, collection)
    result = _header(header, 'owned-experiment-page', collection=collection, offset=offset,
                     limit=limit, total=len(values), path=path, rows=[], next_offset=None,
                     text_limit=dataset_drafts.MAX_TEXT, reference_text_limit=8192,
                     roots=[_property(document, keys, [], key) for key, keys in ROOTS.items()])
    for ordinal in range(offset, min(len(values), offset+limit)):
        row = {'ordinal': ordinal, 'path': path+[ordinal],
               'properties': [_property(values[ordinal], keys, path+[ordinal], key)
                              for key, keys in PROPERTIES[collection].items()]}
        if not _append(result, row): break
    return _finish(result)


def _target(document, target, key):
    _require(isinstance(target, dict), 'Choose an experiment property target.')
    if target.get('kind') == 'document':
        _keys(target, ('kind',), context='Experiment document target')
        _require(isinstance(key, str) and key in ROOTS, 'Choose a supported experiment document property.')
        return document, [], ROOTS[key]
    _keys(target, ('kind', 'collection', 'ordinal'), context='Experiment entry target')
    _require(target['kind'] == 'entry', 'Choose a document or ordered experiment entry.')
    values, path = _collection(document, target['collection']); ordinal = target['ordinal']
    _require(_integer(ordinal) and ordinal < len(values), 'Experiment selection ordinal is missing.')
    properties = PROPERTIES[target['collection']]
    _require(isinstance(key, str) and key in properties, 'Choose a supported experiment entry property.')
    return values[ordinal], path+[ordinal], properties[key]


def locate(document, header, target, key):
    value, path, keys = _target(document, target, key)
    return _header(header, 'owned-experiment-value', target=copy.deepcopy(target), key=key,
                   property=_property(value, keys, path, key))


def edit(document, operation):
    from .authoring_sessions import _operation
    from .task_drafts import operations
    if isinstance(operation, dict) and operation.get('op') == 'property':
        _keys(operation, ('op', 'target', 'key', 'text'), context='Experiment property edit')
        value, path, keys = _target(document, operation['target'], operation['key'])
        if operation['text'] is None:
            _require(bool(keys), 'An ordered entry cannot be removed as a named property; use structural Remove.')
            parent, pp = _locate(value, keys[:-1], path)
            key = keys[-1]
            _require(isinstance(key, str) and isinstance(parent, dict), 'Only named object properties can be unset.')
            if key not in parent: return document
            return _operation(document, {'op': 'remove', 'path': pp+[list(parent).index(key)]})
        parsed = dataset_drafts._text(operation['text'])
        if not keys: return _operation(document, {'op': 'replace', 'path': path, 'value': parsed})
        parent, pp = _locate(value, keys[:-1], path); key = keys[-1]
        if isinstance(key, str):
            _require(isinstance(parent, dict), 'Experiment property needs an object.')
            return _operation(document, {'op': 'put', 'path': pp, 'key': key, 'value': parsed})
        _require(isinstance(parent, list) and key < len(parent), 'Experiment ordered property is missing.')
        return _operation(document, {'op': 'replace', 'path': pp+[key], 'value': parsed})
    return _operation(document, operations([operation])[0])


def choices(document, header, collection, *, offset=0, limit=25, query='', pin=None):
    science_browser._page(offset, limit)
    _require(collection in ('models', 'data_tasks'), 'Local choices are selected models or dataset/task pairings.')
    _require(isinstance(query, str) and len(query)<=200 and '\0' not in query,
             'Local choice search needs at most 200 characters and no NUL.')
    if pin is not None:
        if collection == 'models': reference(pin, 'model')
        else: _require(_identifier(pin), 'Pinned pairing requires a stable selection ID.')
    values, _ = _collection(document, collection); items = []
    for ordinal, value in enumerate(values):
        if collection == 'models':
            try: reference(value, 'model')
            except (GraphError, TypeError, KeyError): chosen=None
            else: chosen=copy.deepcopy(value)
        else:
            chosen=value.get('id') if isinstance(value, dict) else None
            if not _identifier(chosen): chosen=None
        label=(chosen['id'] if collection=='models' else chosen) if chosen is not None else 'Incomplete selection '+str(ordinal+1)
        row={'ordinal':ordinal, 'value':chosen, 'label':label, 'valid':chosen is not None}
        if chosen is not None and collection=='models': row['label']+=' · '+chosen['revision'][:12]
        items.append(row)
    matching=[r for r in items if pin is not None and r['value']==pin]
    rows=sorted([r for r in items if query.casefold() in r['label'].casefold() or
                 (isinstance(r['value'],dict) and query.casefold() in r['value']['revision'])],
                key=lambda r:(r['label'].casefold(),r['ordinal']))
    result=_header(header,'owned-experiment-choices',collection=collection,offset=offset,limit=limit,
                   query=query,total=len(rows),rows=[],next_offset=None,pin=copy.deepcopy(pin),
                   pinned=matching[0] if matching else None,missing_pin=copy.deepcopy(pin) if pin is not None and not matching else None,
                   ambiguous_pin=len(matching)>1)
    for row in rows[offset:offset+limit]:
        if not _append(result,row):break
    return _finish(result)


def preview(workspace, draft, header, *, offset=0, limit=25):
    from .authoring import experiment_preview
    from .candidate_views import project
    source=experiment_preview(workspace,draft,offset=offset,limit=limit)['preview']
    return project(source,_header(header,'owned-experiment-preview'))
