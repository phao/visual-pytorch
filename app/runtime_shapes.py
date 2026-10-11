"""Explicit runtime H/W contracts; planning/import allocate no tensors.

Parameter dimensions must be statically independent of runtime axes. Each eager
call propagates its actual shape through operator-owned rules and admits current
CPU memory before packing inputs or executing layers. No weights are rebuilt.
"""
import copy
import math
from collections import OrderedDict

from .core import GraphError, MAX_INTEGER, input_shape, tensor_shape, validate_model
from .graph import contract, topology
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT, available_memory, memory_percent, format_bytes, MIB


def declaration(models, mid, runtime_inputs):
    """Root Input IDs -> nonempty sample-axis subset of [1,2], excluding B/C."""
    baseline = validate_model(models, mid)
    groups = {g['id']: g for g in baseline['input_groups']}
    if not isinstance(runtime_inputs, dict) or not runtime_inputs or not set(runtime_inputs) <= set(groups):
        raise GraphError('Runtime inputs must map existing root Input IDs to explicit H/W sample axes.')
    result = {}
    for nid, axes in runtime_inputs.items():
        g = groups[nid]
        if len(g['shape']) != 3 or g.get('axes', ['C','H','W']) != ['C','H','W']:
            raise GraphError('Runtime spatial Input ' + nid + ' needs sample shape/axes [C,H,W]; use an explicit adapter.')
        if (not isinstance(axes, list) or not 1 <= len(axes) <= 2 or
                any(type(v) is not int or v not in (1,2) for v in axes) or len(set(axes)) != len(axes)):
            raise GraphError('Runtime spatial axes must be unique integers 1 (H) and/or 2 (W); batch and channels are not constructor symbols.')
        result[nid] = sorted(axes)
    return result


def specification(models, mid, runtime_inputs):
    """Static parameter-dependency proof, not a successful example-batch guess."""
    declared = declaration(models, mid, runtime_inputs)
    visited = set()
    def closure(key):
        if key in visited: return
        if len(visited) >= 64: raise GraphError('Runtime closure exceeds 64 model specifications.')
        visited.add(key)
        from .core import check_document
        check_document(models[key])
        for n in models[key]['graph']['nodes']:
            if n['type'] == 'Module': closure(n['ref'])
    closure(mid)
    baseline = validate_model(models, mid)
    guards, contexts, memo, baseline_cache = [], set(), {}, {}
    def merged(parts): return frozenset().union(*parts)
    def walk(key, inputs, path):
        context = (key, tuple(sorted((nid, tuple(tuple(sorted(d)) for d in deps)) for nid,deps in inputs.items())))
        if context in memo: return memo[context]
        if context not in contexts and len(contexts) >= 64:
            raise GraphError(path + ': runtime dependency plan exceeds 64 distinct Module contexts.')
        contexts.add(context)
        c = contract(models, key, cache=baseline_cache); lookup = {n['id']:n for n in topology(models[key])}; values = {}
        for item in c['plan']:
            nid, kind = item['id'], item['type']; node = lookup[nid]; location = path + '/' + nid
            parts = [values[s['node']][s['port']] if s['port'] is not None else values[s['node']] for s in item['sources']]
            before = parts[0] if parts else inputs[nid]; current = list(before)
            if kind == 'Linear':
                if before[-1]:
                    raise GraphError(location + ': Linear weight input width depends on runtime ' + ', '.join(sorted(before[-1])) + '. Bind image resolution or add AdaptiveAvgPool2d/spatial reduction before Flatten.')
                current[-1] = frozenset()
            elif kind == 'Conv2d':
                if before[0]:
                    raise GraphError(location + ': Conv2d input channels depend on runtime spatial axes; bind channels or correct Permute.')
                current[0] = frozenset()
                guards.append({'path':location,'rule':'positive-spatial-output; groups; NCHW'})
            elif kind in {'AvgPool2d','MaxPool2d'}:
                guards.append({'path':location,'rule':'positive-spatial-output; NCHW'})
            elif kind == 'AdaptiveAvgPool2d': current[1:] = [frozenset(),frozenset()]
            elif kind == 'Permute': current = [before[d] for d in node['dims']]
            elif kind == 'Flatten':
                kwargs = c['shapes'][nid]['operator']['kwargs']; start,end = kwargs['start_dim']-1,kwargs['end_dim']-1
                current = before[:start] + [merged(before[start:end+1])] + before[end+1:]
            elif kind == 'Reshape':
                current = [merged(before) if d == -1 else frozenset() for d in node['shape']]
                guards.append({'path':location,'rule':'reshape-product-and-divisibility'})
            elif kind == 'Concat':
                current = ([merged([merged(p) for p in parts])] if item['flatten'] else
                           [merged([p[i] for p in parts]) for i in range(len(before))])
                guards.append({'path':location,'rule':'ordered-concat-equality-on-unjoined-axes'})
            elif kind == 'Splitter':
                current = {pid:before[:-1]+[frozenset()] for pid in item['ports']}
                guards.append({'path':location,'rule':'last-axis-equals-declared-part-sum'})
            elif kind == 'Replicator': current = {pid:list(before) for pid in item['ports']}
            elif kind == 'Module':
                child = validate_model(models,node['ref'])
                if len(child['input_groups']) != 1 and merged(before):
                    raise GraphError(location + ': runtime Module calls need one tensor input; join/reduce named inputs explicitly before this boundary.')
                child_inputs = {g['id']:before if len(child['input_groups']) == 1 else [frozenset() for _ in g['shape']]
                                for g in child['input_groups']}
                current = walk(node['ref'],child_inputs,location)
            elif kind == 'Output' and 'shape' in node:
                guards.append({'path':location,'rule':'explicit-output-shape-equality'})
            values[nid] = current
        result = (values[c['tensor_source']] if c['tensor_source'] is not None else
                  [merged([merged(values[g['id']]) for g in c['output_groups']])])
        memo[context] = result
        return result
    inputs = {g['id']:[frozenset({g['id'] + '.' + ('H' if i == 1 else 'W')}) if i in declared.get(g['id'],[]) else frozenset()
                         for i in range(len(g['shape']))] for g in baseline['input_groups']}
    output = walk(mid, inputs, mid)
    return {'version':1,'scope':'explicit-eager-runtime-spatial-contract','runtime_inputs':declared,
            'status':'conditional','parameter_shapes':'statically-independent-of-runtime-axes',
            'guards':guards,'output_dependencies':[sorted(v) for v in output],
            'baseline_input_groups':copy.deepcopy(baseline['input_groups'])}


def shape_contract(models, mid, spec, shapes):
    """Pure actual-case shape validation; no tensor, RNG or mutable plan cache."""
    groups = spec['baseline_input_groups']
    if not isinstance(shapes, dict) or set(shapes) != {g['id'] for g in groups}:
        raise GraphError('Runtime shape preview must supply every root Input ID exactly once.')
    for g in groups:
        actual = tensor_shape(shapes[g['id']]); fixed = g['shape']; allowed = spec['runtime_inputs'].get(g['id'], [])
        if len(actual) != len(fixed) or any(a != b and i not in allowed for i,(a,b) in enumerate(zip(actual,fixed))):
            raise GraphError('Input ' + g['id'] + ': only declared H/W axes may vary; channels, rank and all other dimensions remain fixed.')
    try: return contract(models, mid, input_shapes=shapes)
    except GraphError as error:
        raise GraphError('Runtime spatial shape is incompatible: ' + str(error)) from error


def admission(c, batch_size, *, constrained=True, memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT,
              dtype_bytes=4, gradients=True):
    """Current-batch CPU heuristic, charged before any tensor operation."""
    if type(batch_size) is not int or not 1 <= batch_size <= MAX_INTEGER:
        raise GraphError('Runtime batch size must be an exact positive editor-range integer.')
    if type(constrained) is not bool or type(gradients) is not bool:
        raise GraphError('Runtime constrained/gradients policies must be boolean.')
    if type(dtype_bytes) is not int or dtype_bytes not in (4,8):
        raise GraphError('Runtime CPU execution supports float32 or float64 element sizes.')
    percentage = memory_percent(memory_limit_percent)
    # Include Adam/copy headroom when gradients are enabled; arbitrary user
    # optimizers/caches are outside this estimate, not a claimed allocator limit.
    components = {'runtime_headroom':(256 if gradients else 64)*MIB,
                  'parameters':(32 if gradients else 2)*dtype_bytes*c['parameters'],
                  'activations':(4 if gradients else 2)*dtype_bytes*batch_size*c['activation_elements'],
                  'convolution_workspace':(8 if gradients else 2)*dtype_bytes*batch_size*c.get('convolution_workspace_elements',0),
                  'boundary_packing':2*dtype_bytes*batch_size*(c['inputs']+c['outputs'])}
    available = available_memory(); budget = math.floor(available*percentage/100) if available is not None else None
    memory = {'version':1,'operation':'runtime-forward-with-gradients' if gradients else 'runtime-forward',
              'batch':batch_size,'dtype_bytes':dtype_bytes,'components':components,
              'estimated_bytes':sum(components.values()),'available_bytes':available,
              'budget_bytes':budget,'limit_percent':percentage}
    allowed = not constrained or (available is not None and memory['estimated_bytes'] <= budget)
    if not allowed:
        raise GraphError('Constrained mode blocked runtime spatial execution: estimated ' + format_bytes(memory['estimated_bytes']) +
                         ' exceeds/ cannot be checked against ' + str(percentage) + '% of current available RAM (' + format_bytes(budget) +
                         '). Reduce batch/image size, change the RAM percentage, or explicitly disable constrained execution. Weights are unchanged.')
    return {'allowed':allowed,'constrained':constrained,'memory':memory}


def compile_runtime_model(models, mid, runtime_inputs, *, constrained=True,
                          memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT):
    """Construct once; reject weight-dependent axes before allocating layers."""
    spec = specification(models, mid, runtime_inputs)
    from .core import compile_model, preflight, _graph_values, _graph_tensor
    preflight(models,mid,constrained,memory_limit_percent=memory_limit_percent)
    frozen = {}
    def capture(key):
        if key in frozen: return
        frozen[key] = {k:copy.deepcopy(models[key][k]) for k in ('id','name','graph')}
        for node in frozen[key]['graph']['nodes']:
            if node['type'] == 'Module': capture(node['ref'])
    capture(mid)
    baseline = compile_model(frozen,mid,constrained=False)
    class RuntimeGraph(type(baseline)):
        def __init__(self):
            super().__init__(OrderedDict(baseline.named_children()))
            self._runtime_spec = copy.deepcopy(spec)

        def runtime_specification(self): return copy.deepcopy(self._runtime_spec)

        def preview_shapes(self, shapes, *, batch_size=1, gradients=True, dtype_bytes=4):
            c = shape_contract(frozen,mid,spec,shapes)
            report = admission(c,batch_size,constrained=constrained,memory_limit_percent=memory_limit_percent,
                               gradients=gradients,dtype_bytes=dtype_bytes)
            return {'version':1,'status':'verified-for-supplied-shapes','contract':c,'safety':report}

        def _call_contract(self,x):
            import torch
            groups = spec['baseline_input_groups']
            if isinstance(x,dict):
                if set(x) != {g['name'] for g in groups}: raise GraphError('Supply exactly the declared named input ports.')
                parts = [x[g['name']] for g in groups]
            elif len(groups) == 1: parts = [x]
            else: raise GraphError('Runtime multi-input execution requires named tensors; packed data cannot identify variable port shapes.')
            if any(not isinstance(p,torch.Tensor) or p.layout != torch.strided or p.device.type != 'cpu' or p.dtype not in (torch.float32,torch.float64) for p in parts):
                raise GraphError('Runtime inputs must be dense CPU float32/float64 tensors.')
            if any(p.ndim < 2 for p in parts) or len({p.shape[0] for p in parts}) != 1:
                raise GraphError('Runtime named inputs must have one common positive batch size B.')
            if len({p.dtype for p in parts}) != 1 or any(p.device.type != 'cpu' or p.dtype != parts[0].dtype for p in self.parameters()):
                raise GraphError('Runtime inputs and learned parameters must have the same CPU floating dtype; move/cast them explicitly.')
            shapes = {g['id']:list(p.shape[1:]) for g,p in zip(groups,parts)}
            return self.preview_shapes(shapes,batch_size=parts[0].shape[0],dtype_bytes=parts[0].element_size(),
                                       gradients=torch.is_grad_enabled())['contract']

        def intermediate_values(self,x):
            c = self._call_contract(x); return _graph_values(self,x,c)

        def forward_tensor(self,x):
            c = self._call_contract(x); return _graph_tensor(_graph_values(self,x,c),c)

        def forward(self,x):
            c = self._call_contract(x); values = _graph_values(self,x,c); groups = c['output_groups']
            return values[groups[0]['id']] if len(groups) == 1 else {g['name']:values[g['id']] for g in groups}
    return RuntimeGraph()
