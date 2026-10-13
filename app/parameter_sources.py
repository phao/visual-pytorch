"""Exact retained-run tensor sources; pure reads, never initialization or resume.

This is the source half of retained-base binding. Returned descriptors are not
yet admitted parameter-policy or independent `.vtw` specification syntax.
"""
import hashlib
import json
import math
import struct

from .datasets import _integer, _keys, _require
from .evidence import parameter_schema
from .experiment_objects import candidate_bundle, reference
from .parameter_policy import initializer
from .revisions import _hash, frozen_document
from .task_result import validate_result

VALUE_FORMAT = 'ieee754-binary32-le; shape-domain-v1'


def float32_values(values, shape):
    """Return detached float32-roundtripped JSON and a shape/domain-bound digest."""
    _require(isinstance(shape, list) and 1 <= len(shape) <= 8 and
             all(_integer(d, 1, 2**53 - 1) for d in shape),
             'Retained parameter source shape requires 1–8 positive integer dimensions.')
    values = initializer({'kind': 'values', 'dtype': 'float32', 'values': values}, shape)['values']
    shape_bytes = json.dumps(shape, separators=(',', ':'), ensure_ascii=True).encode('ascii')
    h = hashlib.sha256(b'Visual PyTorch float32 tensor v1\0' + shape_bytes + b'\0')
    def convert(value, dimensions):
        if dimensions: return [convert(part, dimensions[1:]) for part in value]
        packed = struct.pack('<f', value)
        rounded = struct.unpack('<f', packed)[0]
        _require(math.isfinite(rounded), 'Retained source must round to finite float32.')
        h.update(packed)
        return rounded
    rounded = convert(values, shape)
    return rounded, h.hexdigest()


def _attempt(workspace, run, attempt):
    reference(run, 'run')
    record = workspace.get_spec('run', run['id'], run['revision'])
    _require(_integer(attempt, 1, len(record['attempts'])),
             'Parameter source requires an explicit retained attempt in this exact run revision.')
    selected = record['attempts'][attempt - 1]
    result = selected['result']
    _require(result is not None,
             'Selected attempt has no parameter checkpoint; select another attempt explicitly.')
    expref = record['selection']['experiment']
    experiment = workspace.get_spec('experiment', expref['id'], expref['revision'])
    bundle = candidate_bundle(experiment, record['selection']['coordinates'], workspace.get_spec,
                              saved_manifest=record['resolved']['manifest'])
    validate_result(bundle, result)
    model = frozen_document('model', bundle['model'])
    modelref = {'kind': 'model', 'id': model['id'], 'revision': model['revision']}
    # Derived family models must have been retained by normal run admission.
    _require(workspace.get_spec('model', modelref['id'], modelref['revision']) == model,
             'Parameter source model does not match its retained exact model revision.')
    schema = parameter_schema(model['models'], model['root'])
    return selected, result, modelref, schema


def inventory(workspace, run, *, attempt):
    """Explicit attempt owner list, no arrays returned and no implicit latest."""
    selected, result, model, schema = _attempt(workspace, run, attempt)
    return {'version': 1, 'kind': 'retained-parameter-inventory', 'run': dict(run),
            'attempt': attempt, 'checkpoint_id': result['checkpoint_id'], 'model': model,
            'status': selected['status'], 'completed_updates': result['counts']['completed_updates'],
            'parameters': [{'path': path, 'id': entry['id'], 'shape': entry['shape'], 'dtype': 'float32'}
                           for path, entry in sorted(schema.items())], 'optimizer_resume': False}


def read(workspace, selection):
    """Require exact run/attempt/checkpoint/path/owner ID; never guess a mapping."""
    _keys(selection, ('run', 'attempt', 'checkpoint_id', 'parameter'), context='Retained parameter selection')
    _keys(selection['parameter'], ('path', 'id'), context='Retained source parameter')
    _require(_hash(selection['checkpoint_id']), 'Parameter source requires an exact checkpoint SHA-256.')
    selected, result, model, schema = _attempt(workspace, selection['run'], selection['attempt'])
    _require(selection['checkpoint_id'] == result['checkpoint_id'], 'Parameter source checkpoint identity mismatch.')
    param = selection['parameter']; path = param['path']
    _require(isinstance(path, str) and path in schema and param['id'] == schema[path]['id'],
             'Parameter source path and semantic owner identity must match the exact checkpoint.')
    shape = schema[path]['shape']
    values, value_hash = float32_values(result['checkpoint']['weights'][path], shape)
    descriptor = {'version': 1, 'kind': 'retained-run-parameter', 'run': dict(selection['run']),
                  'attempt': selection['attempt'], 'checkpoint_id': result['checkpoint_id'],
                  'model': model, 'parameter': dict(param), 'shape': list(shape), 'dtype': 'float32',
                  'values_sha256': value_hash, 'value_format': VALUE_FORMAT,
                  'status': selected['status'], 'completed_updates': result['counts']['completed_updates'],
                  'optimizer_resume': False}
    return {'source': descriptor, 'values': values}
