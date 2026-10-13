"""Portable, exact-base LoRA inference packages. Export itself is numerical-free.

This is raw architecture inference, NOT fitted task preprocessing, training
continuation, a trusted signature, or an adapter usable with an arbitrary base.
"""
import hashlib
import json
import os
import stat
import struct
from pathlib import Path

from .core import validate_model
from .datasets import _keys, _require
from .evidence import digest
from .parameter_derivation import preview, merged_values
from .parameter_sources import read, float32_values, VALUE_FORMAT
from .parameter_binding import validate_source

FILES = {'base': 'base.f32', 'A': 'A.f32', 'B': 'B.f32'}
MAX_MANIFEST = 1_000_000


def _read_file(path, limit, exact=False):
    """Nonblocking regular-file admission and a bounded descriptor-based read."""
    _require(not path.is_symlink(), 'LoRA package file must not be a symlink.')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode), 'LoRA package requires a regular file.')
        _require(info.st_size == limit if exact else info.st_size <= limit,
                 'LoRA package file size exceeds or differs from its contract.')
        raw = stream.read(limit + 1)
        _require(len(raw) == limit if exact else len(raw) <= limit,
                 'LoRA package file changed size or exceeds its contract.')
        return raw


def _binary(values):
    if isinstance(values, list): return b''.join(_binary(v) for v in values)
    return struct.pack('<f', values)


def _matrix(raw, shape):
    rows, columns = shape
    _require(len(raw) == 4 * rows * columns, 'LoRA tensor byte count differs from declared shape.')
    flat = struct.unpack('<' + 'f' * (rows * columns), raw)
    return [list(flat[i * columns:(i + 1) * columns]) for i in range(rows)]


def prepare(workspace, selections, *, alpha=4.):
    """Validate exact retained sources and topology; no workspace edits/allocations."""
    candidate = preview(workspace, selections, alpha=alpha)
    sources = {key: read(workspace, selection) for key, selection in selections.items()}
    ref = candidate['derivation']['sources']['base']['model']
    model = workspace.get_spec('model', ref['id'], ref['revision'])
    contract = validate_model(model['models'], model['root'])
    payloads = {FILES[key]: _binary(value['values']) for key, value in sources.items()}
    manifest = {'version': 1, 'kind': 'visual-pytorch-lora-inference-package',
        'scope': 'raw-architecture-input; exact-base-included; no-task-preprocessing',
        'optimizer_resume': False, 'operation': candidate['derivation']['operation'],
        'sources': {key: value['source'] for key, value in sources.items()},
        'ports': {role: [{k: row[k] for k in ('id', 'shape', 'axes') if k in row}
                        for row in contract[role + '_groups']] for role in ('input', 'output')},
        'tensors': {key: {'file': FILES[key], 'bytes': len(payloads[FILES[key]]),
                         'sha256': hashlib.sha256(payloads[FILES[key]]).hexdigest()}
                    for key in FILES},
        'merged_values_sha256': candidate['derivation']['destination']['values_sha256'],
        'value_format': VALUE_FORMAT}
    return {'manifest': manifest, 'manifest_sha256': digest(manifest), 'payloads': payloads}


def predictor_source(manifest):
    """Self-contained frozen PyTorch predictor; validates bytes BEFORE allocation."""
    # Generated source must not depend on pre-save dictionary insertion order.
    manifest = json.loads(json.dumps(manifest, sort_keys=True, allow_nan=False))
    expected = digest(manifest)
    shapes = {key: value['shape'] for key, value in manifest['sources'].items()}
    ports = manifest['ports']
    return f'''# Visual PyTorch exact-base LoRA inference package, version 1.
# Raw architecture input only. No fitted task transforms or optimizer state.
import hashlib
import json
import math
import os
import stat
import struct
from pathlib import Path

EXPECTED_MANIFEST = {expected!r}
SHAPES = {shapes!r}
FILES = {FILES!r}
PORTS = {ports!r}

def _read_file(path, limit, exact=False):
    if path.is_symlink(): raise ValueError('LoRA package file must not be a symlink.')
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode): raise ValueError('LoRA package requires a regular file.')
        if not (info.st_size == limit if exact else info.st_size <= limit):
            raise ValueError('LoRA package file size differs from its contract.')
        raw = stream.read(limit + 1)
        if not (len(raw) == limit if exact else len(raw) <= limit):
            raise ValueError('LoRA package file changed size or exceeds its contract.')
        return raw

def load():
    folder = Path(__file__).resolve().parent
    manifest = json.loads(_read_file(folder / 'manifest.json', {MAX_MANIFEST}))
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    if identity != EXPECTED_MANIFEST:
        raise ValueError('LoRA manifest identity mismatch.')
    values = {{}}
    for key, name in FILES.items():
        path = folder / name
        count = math.prod(SHAPES[key])
        raw = _read_file(path, count * 4, exact=True)
        if hashlib.sha256(raw).hexdigest() != manifest['tensors'][key]['sha256']:
            raise ValueError('Exact-base/adapter tensor identity mismatch.')
        values[key] = struct.unpack('<' + 'f' * count, raw)
        if not all(math.isfinite(v) for v in values[key]):
            raise ValueError('Nonfinite tensor in LoRA package.')
    import torch
    tensors = {{key: torch.tensor(value, dtype=torch.float32).reshape(SHAPES[key])
               for key, value in values.items()}}
    scale = manifest['operation']['scale']
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for key, tensor in tensors.items(): self.register_buffer(key, tensor)
        def forward(self, inputs):
            port = PORTS['input'][0]
            if not isinstance(inputs, dict) or set(inputs) != {{port['id']}}:
                raise ValueError('Use the exact stable input port ID.')
            x = inputs[port['id']]
            if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.device.type != 'cpu':
                raise ValueError('LoRA v1 requires CPU float32 input.')
            if list(x.shape[1:]) != port['shape'] or x.ndim != len(port['shape']) + 1:
                raise ValueError('LoRA input sample shape differs from the frozen contract.')
            y = torch.nn.functional.linear(x, self.base) + scale * torch.nn.functional.linear(
                torch.nn.functional.linear(x, self.A), self.B)
            return {{PORTS['output'][0]['id']: y}}
    return Predictor().eval()
'''


def validate_package(folder, *, expected_manifest=None):
    """Pure integrity read. Hashes provide identity, not cryptographic authorship."""
    folder = Path(folder)
    manifest = json.loads(_read_file(folder / 'manifest.json', MAX_MANIFEST))
    return _validate_contents(manifest, lambda name, limit, exact=False:
                             _read_file(folder / name, limit, exact), expected_manifest)


def _validate_contents(manifest, read_file, expected_manifest):
    _keys(manifest, ('version', 'kind', 'scope', 'optimizer_resume', 'operation', 'sources',
                     'ports', 'tensors', 'merged_values_sha256', 'value_format'), context='LoRA package')
    _require(type(manifest['version']) is int and manifest['version'] == 1 and
        manifest['kind'] == 'visual-pytorch-lora-inference-package' and
        manifest['scope'] == 'raw-architecture-input; exact-base-included; no-task-preprocessing' and
        manifest['optimizer_resume'] is False and manifest['value_format'] == VALUE_FORMAT,
        'Unsupported LoRA inference package semantics.')
    identity = digest(manifest)
    _require(expected_manifest is None or identity == expected_manifest, 'LoRA manifest identity mismatch.')
    _keys(manifest['sources'], tuple(FILES), context='LoRA package sources')
    _keys(manifest['tensors'], tuple(FILES), context='LoRA package tensors')
    _keys(manifest['ports'], ('input', 'output'), context='LoRA package ports')
    values = {}
    for key, source in manifest['sources'].items():
        validate_source(source)
        _require(len(source['shape']) == 2, 'LoRA package tensors must be matrices.')
        rows, columns = source['shape']
        _require(rows * columns <= 1_000_000, 'LoRA package tensor exceeds local limit.')
        entry = manifest['tensors'][key]
        _keys(entry, ('file', 'bytes', 'sha256'), context='LoRA tensor file')
        _require(entry['file'] == FILES[key] and type(entry['bytes']) is int and
                 entry['bytes'] == 4 * rows * columns, 'LoRA tensor file/shape mismatch.')
        raw = read_file(entry['file'], entry['bytes'], True)
        _require(hashlib.sha256(raw).hexdigest() == entry['sha256'], 'LoRA tensor file hash mismatch.')
        values[key], value_hash = float32_values(_matrix(raw, source['shape']), source['shape'])
        _require(value_hash == source['values_sha256'], 'LoRA tensor differs from its exact retained source.')
    op = manifest['operation']
    _keys(op, ('kind', 'alpha', 'rank', 'scale', 'arithmetic'), context='LoRA package operation')
    from .parameter_derivation import ARITHMETIC
    _require(op['kind'] == 'lora-merge' and op['arithmetic'] == ARITHMETIC and
        op['rank'] == len(values['A']) and type(op['rank']) is int and
        type(op['alpha']) in (int, float) and type(op['scale']) in (int, float) and
        op['scale'] == op['alpha'] / op['rank'], 'Invalid LoRA package operation.')
    base = manifest['sources']['base']
    _require(all(all(source[field] == base[field] for field in ('run', 'attempt', 'checkpoint_id', 'model'))
        for source in manifest['sources'].values()) and
        len({source['parameter']['id'] for source in manifest['sources'].values()}) == 3,
        'LoRA package must identify three distinct owners in one exact retained checkpoint.')
    for role in ('input', 'output'):
        groups = manifest['ports'][role]
        _require(isinstance(groups, list) and len(groups) == 1, 'LoRA v1 package requires one input and output port.')
        port = groups[0]; _keys(port, ('id', 'shape'), ('axes',), 'LoRA package port')
        shape = port['shape']
        _require(isinstance(port['id'], str) and port['id'] and isinstance(shape, list) and shape and
            all(type(d) is int and d > 0 for d in shape), 'Invalid LoRA package port contract.')
    ins, outs = manifest['ports']['input'][0]['shape'], manifest['ports']['output'][0]['shape']
    _require(ins[:-1] == outs[:-1] and ins[-1] == base['shape'][1] and outs[-1] == base['shape'][0],
             'LoRA package ports differ from matrix dimensions.')
    merged = merged_values(*(values[key] for key in FILES), alpha=op['alpha'])
    _, value_hash = float32_values(merged, base['shape'])
    _require(value_hash == manifest['merged_values_sha256'], 'LoRA package merged witness mismatch.')
    _require(read_file('predictor.py', MAX_MANIFEST) == predictor_source(manifest).encode('utf-8'),
             'LoRA predictor source identity mismatch.')
    return {'manifest': manifest, 'manifest_sha256': identity, 'values': values,
            'scope': 'integrity-checked; not-executed; not-authenticated'}


def write_new_package(prepared, output):
    """Exclusive directory creation; failures retain partial files, never overwrite."""
    manifest = prepared['manifest']
    _require(digest(manifest) == prepared['manifest_sha256'], 'Prepared LoRA manifest changed; nothing exported.')
    _require(set(prepared['payloads']) == set(FILES.values()), 'LoRA package requires exact base AND adapter files.')
    for key, entry in manifest['tensors'].items():
        raw = prepared['payloads'][FILES[key]]
        _require(len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'],
                 'Prepared tensor changed; nothing exported.')
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    source_raw = predictor_source(manifest).encode('utf-8')
    _require(len(manifest_raw) <= MAX_MANIFEST and len(source_raw) <= MAX_MANIFEST,
             'LoRA metadata/source exceeds local size limit; nothing exported.')
    files = {**prepared['payloads'], 'manifest.json': manifest_raw, 'predictor.py': source_raw}
    def read_prepared(name, limit, exact=False):
        raw = files[name]
        _require(len(raw) == limit if exact else len(raw) <= limit, 'Prepared package file size invalid.')
        return raw
    _validate_contents(manifest, read_prepared, prepared['manifest_sha256'])
    folder = Path(output).expanduser().absolute()
    folder.mkdir(exist_ok=False)
    for name, raw in files.items():
        with (folder / name).open('xb') as stream: stream.write(raw)
    validate_package(folder, expected_manifest=prepared['manifest_sha256'])
    return {'path': str(folder), 'manifest_sha256': prepared['manifest_sha256'],
            'status': 'exact-base-and-adapter-exported; not-executed', 'optimizer_resume': False}
