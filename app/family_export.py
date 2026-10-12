"""Checked family source export; no Torch, Qt, tensor allocation or training.

The standalone file captures the actual allocation-free resolver/compiler closure,
not a second approximate interpretation. Private runtime names do not shadow app.
"""
import ast
import copy
import hashlib
from pathlib import Path

from .core import VERSION
from .evidence import digest
from .families import require, validate_family
from .family_records import definition
from .experiment_objects import reference
from .memory import memory_percent

CORE = ('GraphError', 'feature_count', 'tensor_shape', 'input_shape', 'reshape_shape',
        'ordered_nodes', 'operation_nodes', 'boundary_groups', 'validate_model',
        'safety_report', 'preflight', 'check_document', '_pack_inputs', '_graph_values',
        '_graph_tensor', '_recurrent_graph_forward', '_lstm_graph_forward', 'compile_model')
EVIDENCE = ('digest', 'legacy_input_order', 'pin_input_order', 'graph_signature',
            'identity', 'parameter_schema')


def _selected(path, names, header):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    require(set(names) <= set(nodes), 'Family export runtime closure is incomplete.')
    return header + '\n\n' + '\n\n'.join(ast.unparse(nodes[n]) for n in names) + '\n'


def runtime_sources():
    folder = Path(__file__).resolve().parent
    sources = {n: (folder / (n + '.py')).read_text(encoding='utf-8')
               for n in ('memory', 'operators', 'activations', 'graph', 'families', 'runtime_shapes', 'recurrent', 'recurrent_nodes', 'temporal_shapes', 'lstm_nodes', 'temporal_shapes_v2')}
    sources['core'] = _selected(folder / 'core.py', CORE, '''from __future__ import annotations
import json
import math
from collections import OrderedDict
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT, memory_percent, available_memory, estimate_memory, format_bytes
from .operators import ARITHMETIC_OPERATORS, BLOCK_TYPES, TENSOR_OPERATORS, LAYER_TYPES, REFERENCE_TYPES, normalized, axes_for
from .activations import activation_name, normalize_activation, softmax_axis
MAX_INTEGER = 2**53 - 1''')
    sources['evidence'] = _selected(folder / 'evidence.py', EVIDENCE, '''import hashlib
import json
PRESENTATION = {'x', 'y', 'width', 'height', 'collapsed'}''')
    return sources


def family_source(family, *, constrained=True, memory_limit_percent=10.):
    source = definition(family); validate_family(source)
    require(type(constrained) is bool, 'Export constrained policy must be boolean.')
    percentage = memory_percent(memory_limit_percent)
    sources = runtime_sources(); runtime_hash = digest(sources)
    provenance = {'version': 1, 'family': {'kind': 'family', 'id': source['id'], 'revision': digest(source)},
                  'root_template': source['root'], 'family_grammar': source['version'],
                  'export_format': 1, 'operator_version': 1, 'container_version': VERSION,
                  'runtime_sha256': runtime_hash, 'runtime_files': sorted(sources),
                  'policy': {'constrained': constrained, 'memory_limit_percent': percentage},
                  'runtime_inputs':copy.deepcopy(source.get('runtime_inputs')),
                  **({'runtime_policy':copy.deepcopy(source['runtime_policy'])} if source['version'] in (6,7,8,9) else {}),
                  'parameter_semantics': 'typed-JSON-component-path; family/template/call/owner/weight-or-bias; exact-shape',
                  'optimizer_resume': False}
    code = ('# Visual PyTorch checked architecture family; GPL-3.0-only. See project LICENSE.\n'
            '# Import creates no model. Construct ArchitectureFamily(bindings={stable_symbol_id: value}).\n'
            'import copy\nimport hashlib\nimport json\nimport sys\nimport threading\nimport types\nimport uuid\n'
            'import torch\nfrom torch import nn\n\n' +
            '_FAMILY = ' + repr(source) + '\n_EXPORT = ' + repr(provenance) +
            '\n_SOURCES = ' + repr(sources) + '\n' +
            _ENTRY)
    return {'version': 1, 'kind': 'visual-pytorch-checked-family-source',
            'scope': 'generate-source-only; no-model-allocation-or-training',
            'class_name': 'ArchitectureFamily', 'provenance': provenance, 'code': code,
            'code_sha256': hashlib.sha256(code.encode('utf-8')).hexdigest(), 'code_bytes': len(code.encode('utf-8'))}


def stored_source(workspace, family_ref, *, memory_limit_percent=10.):
    reference(family_ref, 'family')
    family = workspace.get_spec('family', family_ref['id'], family_ref['revision'])
    return family_source(family, constrained=workspace.constrained_mode, memory_limit_percent=memory_limit_percent)


def write_new_family(source, output):
    code = source['code']
    require(type(code) is str and hashlib.sha256(code.encode('utf-8')).hexdigest() == source['code_sha256'],
            'Family source no longer matches its byte hash. Nothing exported.')
    path = Path(output).expanduser().absolute()
    with path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(code)
    return {'version': 1, 'kind': source['kind'], 'class_name': source['class_name'],
            'path': str(path), 'provenance': copy.deepcopy(source['provenance']),
            'code_sha256': source['code_sha256'], 'code_bytes': source['code_bytes'],
            'status': 'checked-family-source-exported; not-instantiated-or-trained'}


_ENTRY = '''
_capsule = None
_capsule_lock = threading.RLock()

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def _runtime():
    global _capsule
    with _capsule_lock:
        if _capsule is not None:
            return _capsule
        if _digest(_SOURCES) != _EXPORT['runtime_sha256']:
            raise ValueError('Captured family runtime bytes changed; regenerate the export.')
        name = '_visual_pytorch_family_' + uuid.uuid4().hex
        package = types.ModuleType(name); package.__path__ = []; package.__package__ = name
        created = {name: package}
        for child in _SOURCES:
            module = types.ModuleType(name + '.' + child)
            module.__package__ = name; module.__file__ = '<captured-family-runtime/' + child + '.py>'
            setattr(package, child, module); created[module.__name__] = module
        sys.modules.update(created)
        try:
            for child in ('memory', 'operators', 'activations', 'core', 'graph', 'evidence', 'recurrent', 'recurrent_nodes', 'lstm_nodes', 'families', 'runtime_shapes', 'temporal_shapes', 'temporal_shapes_v2'):
                module = getattr(package, child)
                exec(compile(_SOURCES[child], module.__file__, 'exec'), module.__dict__)
        except BaseException:
            for key, module in created.items():
                if sys.modules.get(key) is module: del sys.modules[key]
            raise
        _capsule = package
        return package

def family_definition():
    """Private copy of the immutable captured family; no implicit binding changes."""
    if _digest(_FAMILY) != _EXPORT['family']['revision']:
        raise ValueError('Captured family definition changed; regenerate the export.')
    return copy.deepcopy(_FAMILY)

def export_metadata():
    return copy.deepcopy(_EXPORT)

def preview_family(bindings=None):
    """Shape/constructor diagnostics only; no model or tensor allocation."""
    return _runtime().families.preview(family_definition(), bindings)

def _semantic_schema(runtime, resolved, family):
    schema = runtime.evidence.parameter_schema(resolved['models'], resolved['root'], prefix='network.')
    def visit(mid, prefix, components):
        contract = runtime.core.validate_model(resolved['models'], mid)
        for index, node in enumerate(runtime.core.operation_nodes(resolved['models'][mid])):
            if 'parameter_owner' in contract['shapes'][node['id']]: continue
            path = prefix + 'layer_' + str(index)
            if node['type'] == 'Module':
                visit(node['ref'], path + '.', components + ['call', node['id']])
            elif node['type'] == 'Recurrent':
                visit(node['ref'], path + '.transition.', components + ['recurrent', node['id']])
            elif node['type'] in {'Linear', 'Conv2d'}:
                for kind in ('weight', 'bias'):
                    key = path + '.' + kind
                    if key in schema:
                        schema[key]['id'] = json.dumps(components + ['owner', node['id'], 'parameter', kind],
                                                      separators=(',', ':'), ensure_ascii=True)
            elif node['type'] == 'LSTM':
                for key, value in schema.items():
                    if key.startswith(path + '.'):
                        value['id'] = json.dumps(components + ['owner', node['id'], 'parameter', key[len(path)+1:]],
                                                 separators=(',', ':'), ensure_ascii=True)
    visit(resolved['root'], 'network.', ['family', family['id'], 'template', family['root']])
    return schema

class ArchitectureFamily(nn.Module):
    """Checked finite family constructor; CPU float32, independent instances.

    bindings maps stable symbol IDs to exact integers, booleans or enum strings.
    Required child arguments may instead be supplied by parent call annotations.
    Invalid grammar/types/shapes/sharing/budgets fail BEFORE parameter allocation.
    None policy values use the explicitly captured workspace export policy.
    """
    def __init__(self, *, bindings=None, constrained=None, memory_limit_percent=None, runtime_inputs=None, initial_shapes=None, runtime_temporal=None, runtime_temporal_v2=None):
        super().__init__()
        runtime = _runtime(); source = family_definition()
        resolved = runtime.families.specialize(source, bindings)
        policy = _EXPORT['policy']
        constrained = policy['constrained'] if constrained is None else constrained
        percentage = policy['memory_limit_percent'] if memory_limit_percent is None else memory_limit_percent
        runtime_inputs = source.get('runtime_inputs') if runtime_inputs is None else runtime_inputs
        if runtime_temporal is not None and source['version'] not in (6,7):
            raise ValueError('Temporal family construction requires an explicit family6/7 grammar.')
        if source['version'] in (6,7) and runtime_temporal is None and source['runtime_policy'] is not None:
            runtime_temporal = source['runtime_policy']['inputs']
        if runtime_temporal_v2 is not None and source['version'] not in (8,9):
            raise ValueError('Runtime temporal-v2 requires an explicit family8 grammar.')
        if source['version'] in (8,9):
            expected = None if source['runtime_policy'] is None else source['runtime_policy']['inputs']
            if runtime_inputs is not None or runtime_temporal is not None or (runtime_temporal_v2 is not None and _digest(runtime_temporal_v2) != _digest(expected)):
                raise ValueError('Family8 execution must match its exact retained runtime policy; no implicit policy override.')
            runtime_temporal_v2 = copy.deepcopy(expected)
        # compile_model performs the whole-model preflight before any nn layer.
        self.network = runtime.core.compile_model(resolved['models'], resolved['root'], constrained,
                                                   memory_limit_percent=percentage, runtime_inputs=runtime_inputs,
                                                   initial_shapes=initial_shapes, runtime_temporal=runtime_temporal,
                                                   runtime_temporal_v2=runtime_temporal_v2)
        self._resolved = copy.deepcopy(resolved)
        self._semantic = _semantic_schema(runtime, resolved, source)
        self._provenance = export_metadata()
        self._provenance.update(bindings=copy.deepcopy(resolved['bindings']),
                                constructor_contexts=copy.deepcopy(resolved.get('instances', [])),
                                architecture_hash=resolved['architecture_hash'],
                                runtime_inputs=copy.deepcopy(runtime_inputs),
                                actual_policy={'constrained': constrained, 'memory_limit_percent': percentage})
        if source['version'] in (6,7):
            self._provenance['runtime_temporal'] = copy.deepcopy(runtime_temporal)
        if source['version'] in (8,9):
            self._provenance['runtime_temporal_v2'] = copy.deepcopy(runtime_temporal_v2)
            self._initial_shapes = copy.deepcopy(initial_shapes) if initial_shapes is not None else {
                g['id']:copy.deepcopy(g['shape']) for g in resolved['contract']['input_groups']}

    @classmethod
    def constructor_specification(cls):
        source = family_definition()
        return {'family': export_metadata()['family'], 'root': source['root'], 'symbols': source['symbols'],
                'constructors': source.get('constructors', []), 'runtime_inputs':source.get('runtime_inputs'),
                **({'runtime_policy':copy.deepcopy(source['runtime_policy'])} if source['version'] in (6,7,8,9) else {}),
                'policy': export_metadata()['policy']}

    @property
    def contract(self):
        return copy.deepcopy(self._resolved['contract'])

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def semantic_parameter_schema(self):
        return copy.deepcopy(self._semantic)

    def runtime_specification(self):
        if self._provenance['runtime_inputs'] is None and self._provenance.get('runtime_temporal') is None and self._provenance.get('runtime_temporal_v2') is None:
            return None
        return self.network.runtime_specification()

    def preview_shapes(self, shapes, *, batch_size=1, gradients=True, dtype_bytes=4):
        if self._provenance['runtime_inputs'] is None and self._provenance.get('runtime_temporal') is None and self._provenance.get('runtime_temporal_v2') is None:
            raise ValueError('Construct with explicit runtime_inputs before previewing runtime shapes.')
        return self.network.preview_shapes(shapes, batch_size=batch_size, gradients=gradients, dtype_bytes=dtype_bytes)

    def forward(self, x):
        return self.network(x)

    def forward_tensor(self, x):
        return self.network.forward_tensor(x)

    def intermediate_values(self, x):
        return self.network.intermediate_values(x)

    def semantic_state(self):
        """Caller-owned CPU tensor snapshot, not optimizer or training continuation."""
        return {'version': 1, 'family': export_metadata()['family'], 'runtime_sha256': _EXPORT['runtime_sha256'],
                'bindings': copy.deepcopy(self._resolved['bindings']),
                'schema': self.semantic_parameter_schema(),
                'weights': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}

    def warm_start(self, checkpoint, *, strict=False):
        """Match semantic IDs and exact shapes/dtypes, NOT layer-list positions.

        A complete snapshot from the same exact family/runtime is required. Partial
        compatible loading is explicit by default; strict requires full equality.
        Validate the complete plan before changing any target parameter.
        """
        if type(strict) is not bool:
            raise ValueError('Warm-start strict must be boolean.')
        if not isinstance(checkpoint, dict) or set(checkpoint) != {'version','family','runtime_sha256','bindings','schema','weights'}:
            raise ValueError('Warm-start requires a closed semantic_state snapshot.')
        if type(checkpoint['version']) is not int or checkpoint['version'] != 1 or checkpoint['family'] != _EXPORT['family'] or checkpoint['runtime_sha256'] != _EXPORT['runtime_sha256']:
            raise ValueError('Warm-start needs this exact family revision and captured runtime.')
        schema, weights = checkpoint['schema'], checkpoint['weights']
        if not isinstance(schema, dict) or not isinstance(weights, dict) or set(schema) != set(weights):
            raise ValueError('Warm-start schema and tensor paths must match exactly.')
        runtime = _runtime(); family = family_definition()
        source_resolution = runtime.families.specialize(family, checkpoint['bindings'])
        if _digest(source_resolution['bindings']) != _digest(checkpoint['bindings']) or _digest(_semantic_schema(runtime, source_resolution, family)) != _digest(schema):
            raise ValueError('Warm-start schema does not match its exact source constructor bindings.')
        policy = self._provenance['actual_policy']
        for resolution in (source_resolution, self._resolved):
            if self._provenance.get('runtime_temporal_v2') is not None:
                shapes = {g['id']:copy.deepcopy(g['shape']) for g in resolution['contract']['input_groups']}
                for nid, axes in self._provenance['runtime_temporal_v2'].items():
                    for axis in axes: shapes[nid][axis] = self._initial_shapes[nid][axis]
                spec = resolution['runtime_specification']
                actual = runtime.runtime_shapes.shape_contract(resolution['models'], resolution['root'], spec, shapes)
                runtime.temporal_shapes_v2.admission(resolution['models'], resolution['root'], actual, 1,
                                                     constrained=policy['constrained'],
                                                     memory_limit_percent=policy['memory_limit_percent'], gradients=False)
            else:
                runtime.core.preflight(resolution['models'], resolution['root'], policy['constrained'],
                                       memory_limit_percent=policy['memory_limit_percent'])
        source = {}
        for path, descriptor in schema.items():
            if not isinstance(path, str) or not isinstance(descriptor, dict) or set(descriptor) != {'id','shape'}:
                raise ValueError('Invalid semantic parameter descriptor.')
            sid, shape, tensor = descriptor['id'], descriptor['shape'], weights[path]
            if not isinstance(sid, str) or not sid or sid in source or not isinstance(shape, list) or not shape or any(type(v) is not int or v < 1 for v in shape):
                raise ValueError('Semantic parameter identities/shapes must be unique and exact.')
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided or list(tensor.shape) != shape or not torch.isfinite(tensor).all().item():
                raise ValueError('Semantic state tensor does not match its finite dense declared shape.')
            source[sid] = (path, descriptor, tensor)
        target = self.state_dict(); plan, missing, skipped = [], [], []
        for path, descriptor in self._semantic.items():
            sid = descriptor['id']; entry = source.get(sid)
            if entry is None: missing.append(sid); continue
            old_path, old, tensor = entry
            if old['shape'] != descriptor['shape'] or tensor.dtype != target[path].dtype:
                skipped.append({'id': sid, 'reason': 'shape-or-dtype-mismatch'}); continue
            plan.append((path, old_path, tensor))
        used = {old_path for _, old_path, _ in plan}
        unused = [descriptor['id'] for path, descriptor in schema.items() if path not in used]
        if strict and (missing or skipped or unused):
            raise ValueError('Strict warm-start has missing, incompatible or extra semantic parameters; nothing loaded.')
        with torch.no_grad():
            for path, _, tensor in plan: target[path].copy_(tensor)
        return {'version': 1, 'loaded': [{'target': path, 'source': old_path, 'id': self._semantic[path]['id']}
                                        for path, old_path, _ in plan],
                'missing': missing, 'skipped': skipped, 'unused': unused, 'optimizer_resume': False}
'''
