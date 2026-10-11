"""Ordered, lazy family binding selections in version-2 experiments.

No tensors, fitting, SQL writes, inferred bindings or product enumeration. A
coordinate is an explicit choice ID plus fixed/axis bindings; defaults belong
to the exact family revision and are included in its specialization receipt.
"""
import copy
import math

from .datasets import _identifier, _keys, _require
from .families import binding_atom, validate_family, preview


def family_choice(value):
    return isinstance(value, dict) and 'family' in value


def validate_choice(value, *, version=2):
    from .experiment_objects import reference
    if not family_choice(value):
        reference(value, 'model')
        return
    _keys(value, ('id', 'family', 'bindings', 'axes'),('execution',) if version>=4 else (),context='Family model choice')
    if 'execution' in value:
        policies=('concrete-v1','runtime-spatial-v1')+(('runtime-temporal-v1',) if version>=5 else ())
        _require(value['execution'] in policies,'Choose an explicit execution policy supported by this experiment version; temporal execution requires version5.')
    _require(_identifier(value['id']), 'Family choice needs a stable selection ID.')
    reference(value['family'], 'family')
    _require(isinstance(value['bindings'], dict) and len(value['bindings']) <= 128,
             'Family choice needs fixed stable-symbol-ID bindings.')
    for sid, val in value['bindings'].items():
        _require(_identifier(sid), 'Fixed binding needs a stable symbol ID.')
        binding_atom(val, 'Fixed binding ' + sid, typed=version >= 3)
    _require(isinstance(value['axes'], list) and len(value['axes']) <= 128,
             'Family choice needs an ordered array of at most 128 binding axes.')
    seen = set(value['bindings'])
    for axis in value['axes']:
        _keys(axis, ('symbol', 'values'), context='Family binding axis')
        sid, values = axis['symbol'], axis['values']
        _require(_identifier(sid) and sid not in seen,
                 'Binding axes need distinct symbol IDs, disjoint from fixed bindings.')
        seen.add(sid)
        _require(isinstance(values, list) and 1 <= len(values) <= 1000,
                 'Each family axis needs 1..1000 explicit ordered integers.')
        for val in values: binding_atom(val, 'Axis ' + sid, typed=version >= 3)
        _require(len({(type(v), v) for v in values}) == len(values), 'Duplicate value in family axis ' + sid + '.')


def dependencies(choices):
    return [value['family'] if family_choice(value) else value for value in choices]


def validate_dependencies(choices, resolve, *, version=2):
    for value in choices:
        if not family_choice(value): continue
        ref = value['family']
        source = resolve(ref['kind'], ref['id'], ref['revision'])
        from .family_records import definition
        symbols = validate_family(definition(source))
        _require(version >= 3 or source['version'] < 3, 'Family version 3 selections require experiment version 3.')
        selected = set(value['bindings']) | {a['symbol'] for a in value['axes']}
        _require(selected <= set(symbols), 'Family choice names an unknown stable symbol ID.')


def count(value):
    return math.prod(len(a['values']) for a in value['axes']) if family_choice(value) else 1


def total(choices):
    return sum(count(value) for value in choices)


def at(choices, index):
    """Skip choices/Cartesian prefixes arithmetically, not by compilation."""
    _require(type(index) is int and index >= 0, 'Model selection index must be an exact nonnegative integer.')
    for choice in choices:
        size = count(choice)
        if index >= size:
            index -= size
            continue
        if not family_choice(choice): return copy.deepcopy(choice)
        bindings = copy.deepcopy(choice['bindings'])
        digits = []
        for axis in reversed(choice['axes']):
            index, digit = divmod(index, len(axis['values']))
            digits.append((axis['symbol'], axis['values'][digit]))
        bindings.update(reversed(digits))
        return {'id': choice['id'], 'bindings': bindings}
    raise IndexError('Model selection index is outside the declared axes.')


def selected(choices, coordinate, *, version=2):
    """Strict membership, including no invented, omitted or rounded bindings."""
    if isinstance(coordinate, dict) and coordinate.get('kind') == 'model':
        from .experiment_objects import reference
        reference(coordinate, 'model')
        _require(any(not family_choice(c) and c == coordinate for c in choices),
                 'Model not selected by experiment.')
        return coordinate
    _keys(coordinate, ('id', 'bindings'), context='Family model coordinate')
    _require(_identifier(coordinate['id']) and isinstance(coordinate['bindings'], dict),
             'Family coordinate needs a choice ID and explicit bindings.')
    choice = next((c for c in choices if family_choice(c) and c['id'] == coordinate['id']), None)
    _require(choice is not None, 'Family choice not selected by experiment.')
    expected = set(choice['bindings']) | {a['symbol'] for a in choice['axes']}
    _require(set(coordinate['bindings']) == expected, 'Coordinate must specify exactly the fixed and axis bindings.')
    for sid, val in coordinate['bindings'].items(): binding_atom(val, 'Coordinate binding ' + sid, typed=version >= 3)
    _require(all(type(coordinate['bindings'][k]) is type(v) and coordinate['bindings'][k] == v for k, v in choice['bindings'].items()),
             'Coordinate changed a fixed family binding.')
    _require(all(any(type(coordinate['bindings'][a['symbol']]) is type(v) and coordinate['bindings'][a['symbol']] == v
                     for v in a['values']) for a in choice['axes']),
             'Coordinate binding is outside the declared family axis.')
    return choice


def resolve_model(choices, coordinate, resolve, *, version=2):
    """Return pure frozen provenance; conditional points remain non-executable."""
    choice = selected(choices, coordinate, version=version)
    from .experiment_objects import _get
    if not family_choice(choice):
        model = _get(resolve, coordinate)
        return {'status': 'verified', 'model': model, 'references': {'model': copy.deepcopy(coordinate)}}
    ref = choice['family']
    source = _get(resolve, ref)
    from .family_records import definition, resolution_records
    result = preview(definition(source), coordinate['bindings'])
    if result['status'] != 'verified':
        return {k: v for k, v in result.items() if k != 'missing'} | {'references': {'family': copy.deepcopy(ref)}}
    records = resolution_records(source, result['resolved'])
    runtime=None
    if choice.get('execution') in ('runtime-spatial-v1','runtime-temporal-v1'):
        from .runtime_tasks import proof
        try:runtime=proof(source,records['specialization'],records['model'],execution=choice['execution'])
        except ValueError as error:
            return {'status':'incompatible','reason':str(error),'references':{'family':copy.deepcopy(ref)}}
    return {'status': 'verified', 'model': records['model'], **({'runtime':runtime} if runtime is not None else {}),
            'references': {k: records[k + '_ref'] for k in ('model', 'family', 'specialization')}}


def label(coordinate):
    if coordinate.get('kind') == 'model': return coordinate['id'] + ' @ ' + coordinate['revision'][:12]
    return coordinate['id'] + ' · ' + ', '.join(k + '=' + str(v) for k, v in coordinate['bindings'].items())
