"""Checked standalone concrete-graph temporal source; generation allocates nothing."""
import hashlib

from .core import GraphError
from .evidence import digest
from .family_export import runtime_sources, _ENTRY
from .memory import memory_percent
from .revisions import model_document
from .temporal_shapes import specification


def temporal_source(models, mid, runtime_inputs, *, constrained=True, memory_limit_percent=10.):
    if type(constrained) is not bool:
        raise GraphError('Temporal export constrained policy must be boolean.')
    percentage = memory_percent(memory_limit_percent)
    frozen = model_document(models, mid)
    spec = specification(frozen['models'], mid, runtime_inputs)
    sources = runtime_sources()
    metadata = {'version':1, 'kind':'visual-pytorch-checked-temporal-source',
                'policy':'runtime-temporal-v1', 'model_sha256':digest(frozen),
                'runtime_inputs':spec['runtime_inputs'], 'specification_sha256':digest(spec),
                'runtime_sha256':digest(sources), 'runtime_files':sorted(sources),
                'execution':{'constrained':constrained,'memory_limit_percent':percentage}}
    # Reuse the verified isolated package loader, not another runtime interpreter.
    bootstrap, _ = _ENTRY.split('\ndef family_definition():', 1)
    code = ('# Visual PyTorch checked runtime-temporal-v1 source; GPL-3.0-only.\n'
            '# Import creates no parameters. Call build_model() and supply explicit named tensors.\n'
            'import copy\nimport hashlib\nimport json\nimport sys\nimport threading\nimport types\nimport uuid\n\n'
            '_MODEL = ' + repr(frozen) + '\n_EXPORT = ' + repr(metadata) +
            '\n_SOURCES = ' + repr(sources) + '\n' + bootstrap + _TEMPORAL_ENTRY)
    return {'version':1, 'kind':metadata['kind'], 'factory_name':'build_model',
            'scope':'generate-source-only; no-model-allocation-or-training', 'provenance':metadata,
            'code':code, 'code_sha256':hashlib.sha256(code.encode('utf-8')).hexdigest(),
            'code_bytes':len(code.encode('utf-8'))}


_TEMPORAL_ENTRY = '''
def export_metadata():
    return copy.deepcopy(_EXPORT)

def model_definition():
    if _digest(_MODEL) != _EXPORT['model_sha256']:
        raise ValueError('Captured temporal model changed; regenerate the export.')
    return copy.deepcopy(_MODEL)

def _checked():
    definition = model_definition()
    runtime = _runtime()
    spec = runtime.temporal_shapes.specification(definition['models'], definition['root'], _EXPORT['runtime_inputs'])
    if _digest(spec) != _EXPORT['specification_sha256']:
        raise ValueError('Captured temporal policy/proof changed; regenerate the export.')
    return runtime, definition, spec

def runtime_specification():
    return _checked()[2]

def build_model(*, constrained=None, memory_limit_percent=None, initial_shapes=None):
    runtime, definition, spec = _checked()
    policy = _EXPORT['execution']
    return runtime.temporal_shapes.compile_temporal_model(
        definition['models'], definition['root'], spec['runtime_inputs'],
        constrained=policy['constrained'] if constrained is None else constrained,
        memory_limit_percent=policy['memory_limit_percent'] if memory_limit_percent is None else memory_limit_percent,
        initial_shapes=initial_shapes)
'''
