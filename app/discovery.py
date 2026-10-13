"""Shared application-contract discovery, not a second parser or executor.

Examples/ports use the actual graph checker. No Torch/Qt, user configuration,
file loading, random model weights or arbitrary source evaluation is required.
Field lists document known semantics; legacy permissive extras are not features.
"""
import copy

from . import operators
from .activations import ACTIVATIONS
from .core import FORMAT, VERSION, SUPPORTED_VERSIONS, GraphError, Workspace, check_document, validate_model
from .revisions import KINDS

_RULES = {
    'CausalLagWindow': 'Explicit observed history and prefix lengths; preceding p vectors only, oldest to newest. No current/future leakage, reset, missing-history imputation or trainable owners.',
    'GaussianFilter': 'Exact diagonal-Gaussian log-space filtering; explicit previous-to-next transition rows and initial law before first transition. No regime labels, smoothing, hidden variance floor or trainable parameters. Named posterior/evidence ports.',
    'PositiveVariance': 'Elementwise softplus(raw,beta=1,threshold=20) plus explicitly declared positive float32 floor. Shape/axes preserved; no clipping or learned parameters.',
    'Constant': 'Declared immutable CPU float32 sample tensor; explicit fill or rectangular values; no incoming data, no learnable parameters. Expand as a batch view, never a dataset input.',
    'TensorParameter': 'Declared CPU float32 sample tensor; one zero-default nn.Parameter value owner. Graph policy initializes/freezes it; compatible share_with reuses the exact owner. No incoming data.',
    'Add': 'Ordered elementwise sum, at least two identical sample shapes and axes; no broadcasting, flattening or learned parameters.',
    'Scale': 'Multiply by a fixed finite coefficient, preserving shape/axes/batch; no learned parameters.',
    'Input': 'Declare sample shape (or vector features), optional unique axes; batch B is implicit.',
    'Output': 'Expose one named tensor; optional shape must equal incoming shape.',
    'Linear': 'Replace only the last axis; earlier axes share weights. Bias defaults true.',
    'Activation': 'Elementwise except Softmax, which normalizes one sample axis, never batch B.',
    'Reshape': 'Preserve sample element count and batch B; at most one inferred -1 dimension.',
    'Concat': 'Ordered source keys. flatten=true concatenates flattened samples; false joins last axis with equal leading shape/axes.',
    'Splitter': 'Partition the last sample axis into ordered named parts; earlier axes and batch B are preserved.',
    'Replicator': 'Expose the unchanged tensor at named ports; no extra parameters or loss duplication.',
    'Module': 'Reference a complete architecture; match its packed/single input shape and axes. Not arbitrary Python.',
    'Recurrent': 'Explicit referenced cell, sequence and initial-state bindings, returned state/trajectory ports and control adapters; no implicit carry.',
    'LSTM': 'Explicit sequence [T,F] and batch-leading hidden/cell; projected/bidirectional widths checked. Optional exact lengths.',
    'Conv2d': 'Sample CHW; channels divisible by groups. Explicit kernel/stride/padding/dilation. CPU float32, zeros padding.',
    'AvgPool2d': 'Sample CHW; no ceil mode. Padding at most half kernel; explicit count_include_pad.',
    'MaxPool2d': 'Sample CHW; no ceil mode. Padding at most half kernel; no trainable parameters.',
    'AdaptiveAvgPool2d': 'Sample CHW to explicit positive output height/width.',
    'Flatten': 'Merge a contiguous range of sample axes; batch is excluded. Merged axis names are discarded.',
    'Permute': 'Reorder each sample axis exactly once; batch is fixed first. This changes values, not just shape labels.',
}
_LEGACY_FIELDS = {
    'CausalLagWindow':{'inputs','lags'},
    'Input': {'features', 'shape', 'axes'}, 'Output': {'shape'},
    'Linear': {'features', 'bias'}, 'Activation': {'activation', 'dim'},
    'Reshape': {'shape'}, 'Module': {'ref','batch_only'}, 'Concat': {'sources', 'flatten'},
    'Constant': {'shape','axes','fill','values'}, 'TensorParameter': {'shape','axes'},
    'PositiveVariance':{'positive_floor'},
    'GaussianFilter':{'inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode'},
    'Splitter': {'parts'}, 'Replicator': {'outputs'},
}


def _header(kind, **values):
    return {'version': 1, 'kind': 'visual-pytorch-' + kind,
            'scope': 'application-contract-discovery; no-workspace-read-or-execution', **values}


def _model(mid, nodes, edges):
    return {'id': mid, 'name': 'Schema example ' + mid, 'graph': {'nodes': nodes, 'edges': edges}}


def _example(kind):
    # Construction witnesses are intentionally concrete and tiny, not opaque
    # templates. Complete referenced closures are returned for copy/inspection.
    mid = 'schema-' + kind.lower(); models = {}
    nodes = [{'id': 'x', 'type': 'Input', 'shape': [2]}]
    node = {'id': 'op', 'type': kind}
    if kind in {'Input', 'Output'}:
        nodes = [{'id': 'x', 'type': 'Input', 'shape': [2]}, {'id': 'y', 'type': 'Output'}]
        edges = [{'from': 'x', 'to': 'y'}]; target = 'x' if kind == 'Input' else 'y'
    else:
        target = 'op'
        edges = [{'from': 'x', 'to': 'op'}, {'from': 'op', 'to': 'y'}]
        if kind=='CausalLagWindow':
            nodes=[{'id':'observations','type':'Input','shape':[3,1]},
                   {'id':'history','type':'Input','shape':[2,1]}, {'id':'lengths','type':'Input','shape':[1]}]
            node.update(lags=2,inputs={role:role for role in ('observations','history','lengths')})
            edges=[{'from':role,'to':'op'} for role in ('observations','history','lengths')]+[{'from':'op','to':'y'}]
        elif kind=='GaussianFilter':
            from .gaussian_nodes import ROLES
            shapes={'observations':[3,1],'initial':[2],'transition':[3,2,2],
                    'means':[3,2,1],'variances':[2,1],'lengths':[1]}
            nodes=[{'id':role,'type':'Input','shape':shapes[role]} for role in ROLES]
            node.update(inputs={role:role for role in ROLES},initial_representation='probabilities',
                        transition_representation='probabilities',transition_mode='sequence',mean_mode='sequence',variance_mode='shared')
            edges=[{'from':role,'to':'op'} for role in ROLES]
            edges.append({'from':'op','from_port':'sequence_evidence','to':'y'})
        elif kind=='PositiveVariance':node.update(positive_floor=.01)
        elif kind in operators.SOURCE_TYPES:
            nodes = []
            node['shape'] = [2]
            if kind == 'Constant': node['fill'] = 0.
            edges = [{'from':'op','to':'y'}]
        elif kind in operators.TENSOR_OPERATORS:
            nodes[0].update(shape=[1, 6, 5], axes=['C', 'H', 'W'])
            settings = {'Conv2d': {'out_channels': 2, 'kernel_size': [3, 2]},
                'AvgPool2d': {'kernel_size': [2, 2]}, 'MaxPool2d': {'kernel_size': [2, 2]},
                'AdaptiveAvgPool2d': {'output_size': [1, 2]},
                'Flatten': {}, 'Permute': {'dims': [1, 2, 0]}}
            node.update(settings[kind]); node = operators.normalized(node)
        elif kind == 'Linear': node.update(features=3, bias=True)
        elif kind == 'Scale': node.update(coefficient=2.)
        elif kind == 'Activation': node.update(activation='Softmax', dim=0)
        elif kind == 'Reshape': node.update(shape=[1, 2])
        elif kind in {'Concat', 'Add'}:
            nodes.append({'id': 'other', 'type': 'Input', 'shape': [2] if kind == 'Add' else [3]})
            node.update(sources=['x', 'other'])
            if kind == 'Concat': node['flatten'] = True
            edges.insert(1, {'from': 'other', 'to': 'op'})
        elif kind in {'Splitter', 'Replicator'}:
            if kind == 'Splitter': node.update(parts=[{'output': 'left', 'features': 1}, {'output': 'right', 'features': 1}])
            else: node.update(outputs=['left', 'right'])
            edges = [{'from': 'x', 'to': 'op'}, {'from': 'op', 'from_port': 'left', 'to': 'y'},
                     {'from': 'op', 'from_port': 'right', 'to': 'other-y'}]
        elif kind == 'Module':
            child = _model('child', [{'id': 'x', 'type': 'Input', 'shape': [2]},
                {'id': 'linear', 'type': 'Linear', 'features': 1}, {'id': 'y', 'type': 'Output'}],
                [{'from': 'x', 'to': 'linear'}, {'from': 'linear', 'to': 'y'}])
            models['child'] = child; node.update(ref='child')
        elif kind == 'Recurrent':
            models['cell'] = _model('cell', [{'id': 'x', 'type': 'Input', 'shape': [2]},
                {'id': 'state', 'type': 'Input', 'shape': [2]},
                {'id': 'join', 'type': 'Concat', 'sources': ['x', 'state']},
                {'id': 'linear', 'type': 'Linear', 'features': 2}, {'id': 'next', 'type': 'Output'}],
                [{'from': 'x', 'to': 'join'}, {'from': 'state', 'to': 'join'},
                 {'from': 'join', 'to': 'linear'}, {'from': 'linear', 'to': 'next'}])
            nodes[0]['shape'] = [3, 2]
            nodes.append({'id': 'initial', 'type': 'Input', 'shape': [2]})
            node.update(ref='cell', sequence_inputs=[{'input': 'x', 'source': 'x'}],
                states=[{'input': 'state', 'output': 'next', 'port': 'hidden', 'source': 'initial'}],
                sequence_outputs=[{'output': 'next', 'port': 'sequence'}], controls={})
            edges = [{'from': 'x', 'to': 'op'}, {'from': 'initial', 'to': 'op'},
                     {'from': 'op', 'from_port': 'sequence', 'to': 'y'},
                     {'from': 'op', 'from_port': 'hidden', 'to': 'other-y'}]
        elif kind == 'LSTM':
            nodes[0]['shape'] = [3, 2]
            nodes.extend([{'id': 'initial-h', 'type': 'Input', 'shape': [1, 2]},
                          {'id': 'initial-c', 'type': 'Input', 'shape': [1, 2]}])
            node.update(hidden_size=2, inputs={'sequence': 'x', 'hidden': 'initial-h', 'cell': 'initial-c'})
            node = operators.normalized(node)
            edges = [{'from': key, 'to': 'op'} for key in ('x', 'initial-h', 'initial-c')]
            edges += [{'from': 'op', 'from_port': port, 'to': out}
                      for port, out in (('sequence', 'y'), ('hidden', 'other-y'), ('cell', 'cell-y'))]
        nodes.extend([node, {'id': 'y', 'type': 'Output'}])
        for key in ('other-y', 'cell-y'):
            if any(edge['to'] == key for edge in edges): nodes.append({'id': key, 'type': 'Output'})
    models[mid] = _model(mid, nodes, edges)
    for model in models.values(): check_document(model)
    contract = validate_model(models, mid)
    witness={'root':mid,'models':models,'node':target}
    if kind=='CausalLagWindow':
        witness['input_values']={'observations':[[0.],[.1],[.2]],'history':[[-.2],[-.1]],'lengths':[3.]}
    if kind=='GaussianFilter':
        witness['input_values']={'observations':[[0.],[0.],[0.]],'initial':[.5,.5],
            'transition':[[[.5,.5],[.5,.5]]]*3,'means':[[[0.],[1.]]]*3,
            'variances':[[1.],[1.]],'lengths':[3.]}
    return witness, contract


def operator_schema(kind):
    if type(kind) is not str or kind not in operators.BLOCK_TYPES:
        raise GraphError('Unsupported operator; use capabilities for the accepted registry.')
    if set(_RULES) != operators.BLOCK_TYPES:
        raise GraphError('Operator discovery metadata is incomplete; no partial capabilities claim.')
    example, _ = _example(kind)
    node = next(n for n in example['models'][example['root']]['graph']['nodes'] if n['id'] == example['node'])
    if kind == 'Recurrent':
        from .recurrent_nodes import FIELDS
        fields = FIELDS
    elif kind == 'LSTM':
        from .lstm_nodes import FIELDS
        fields = FIELDS
    elif kind == 'GaussianFilter':
        from .gaussian_nodes import FIELDS
        fields=FIELDS
    else: fields = operators.FIELDS.get(kind, _LEGACY_FIELDS.get(kind, set()))
    strict = kind in operators.TENSOR_OPERATORS | operators.SOURCE_TYPES | operators.PROBABILITY_OPERATORS | operators.ARITHMETIC_OPERATORS | {'Recurrent', 'LSTM'}
    required = {
        'CausalLagWindow':{'inputs','lags'},
        'Add': {'sources'}, 'Scale': {'coefficient'},
        'Constant': {'shape'}, 'TensorParameter': {'shape'},
        'PositiveVariance':{'positive_floor'},
        'GaussianFilter':{'inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode'},
        'Input': set(), 'Output': set(), 'Linear': {'features'}, 'Activation': {'activation'},
        'Module': {'ref'}, 'Reshape': {'shape'}, 'Concat': {'sources'}, 'Splitter': {'parts'},
        'Replicator': {'outputs'}, 'Conv2d': {'out_channels', 'kernel_size'},
        'AvgPool2d': {'kernel_size'}, 'MaxPool2d': {'kernel_size'}, 'AdaptiveAvgPool2d': {'output_size'},
        'Flatten': set(), 'Permute': {'dims'},
        'Recurrent': {'ref', 'sequence_inputs', 'states', 'sequence_outputs'}, 'LSTM': {'hidden_size', 'inputs'},
    }[kind]
    minimal = {k: v for k, v in node.items() if k in required | {'id', 'type'}}
    if kind == 'Input': minimal['shape'] = node['shape']
    if kind == 'Constant': minimal['fill'] = node['fill']
    if kind == 'Activation': minimal['activation'] = 'Softmax'
    actual_defaults = {k: v for k, v in operators.normalized(minimal).items() if k not in minimal}
    if kind == 'Linear': actual_defaults['bias'] = True
    if kind == 'Concat': actual_defaults['flatten'] = True
    return _header('operator-schema', type=kind, op_versions=[1] if strict else None,
        fields={'common': sorted(operators.COMMON), 'semantic': sorted(fields),
            'required': sorted(required | {'id', 'type'}),
            'optional': sorted((set(fields) | operators.COMMON) - required - {'id', 'type'}),
            'required_in_example': sorted(set(node) - operators.COMMON),
            'unknown_policy': 'refuse' if strict else 'legacy-permissive; extras do not establish executable semantics',
            'alternatives': ['Input: declare shape or features (shape takes precedence)'] if kind == 'Input' else ['Constant: exactly one of fill or values'] if kind == 'Constant' else [],
            'conditional': ['Softmax rejects extras, dim defaults -1; other four activations retain legacy permissive extras.'] if kind == 'Activation' else []},
        defaults=actual_defaults, defaults_scope='shared normalizer plus legacy bias/flatten defaults; required fields are not defaults',
        shape_rule=_RULES[kind], batch_axis='implicit leading B; sample-axis indices exclude B',
        ports={'incoming': 'none' if kind == 'Input' or kind in operators.SOURCE_TYPES else 'ordered sources, at least two' if kind in {'Concat', 'Add'} else
               'explicit role bindings; one edge per unique source' if kind in {'LSTM', 'Recurrent','GaussianFilter'} else 'exactly one',
               'outgoing': 'none' if kind == 'Output' else 'named ports' if kind in {'Splitter', 'Replicator', 'LSTM', 'Recurrent','GaussianFilter'} else 'one tensor with ordinary fan-out'},
        sharing={'supported': kind in {'Linear', 'Conv2d', 'Module', 'Recurrent', 'LSTM', 'TensorParameter'},
                 'binding': 'share_with resolves one compatible local canonical owner; not weight freezing'},
        activations=sorted(ACTIVATIONS) if kind == 'Activation' else [],
        example=example, example_contract=port_contract(example['models'], example['root']),
        routes={'native': 'cpu float32', 'standalone': 'checked PyTorch generation',
                'training': 'requires compatible explicit dataset/task/protocol; schema discovery does not run it'},
        documentation='docs/' + ('lstm-graph-contract.md' if kind == 'LSTM' else
            'recurrent-graph-contract.md' if kind == 'Recurrent' else 'vtw-format.md'))


def capabilities():
    # Calling each schema verifies coverage AND actual example acceptance. A
    # newly registered operator without discovery cannot silently disappear.
    schemas = [operator_schema(kind) for kind in sorted(operators.BLOCK_TYPES)]
    return _header('capabilities', container_version=VERSION, operators=[s['type'] for s in schemas],
        activations=sorted(ACTIVATIONS), tensor_dtype='float32', device='cpu',
        parameter_policy={'graph_field': 'parameter_policy', 'version': 1, 'model_revision_version': 6,
            'family_version': 10, 'specialization_version': 9,
            'selectors': 'canonical JSON component arrays relative to each architecture',
            'trainability': 'per unique owner, requires_grad and selected optimizer membership',
            'initializers': ['constructor', 'normal', 'uniform', 'values', 'zeros']},
        tensor_sources={'operators':['Constant','TensorParameter'], 'model_revision_version':7,
            'family_version':11, 'specialization_version':10, 'nullary_call':'model(batch_size=B); no dummy input',
            'constant_storage':'nonpersistent buffer reconstructed from immutable graph; charged to RAM'},
        gaussian_filter={'container_version':28,'model_revision_version':8,'family_version':12,'specialization_version':11,
            'initial':'before-first-transition','transition_orientation':'previous-to-next',
            'scope':'exact forward filtering and concrete target-free likelihood training',
            'controls':['prefix-lengths','missing-whole-emission-mask']},
        likelihood_task={'task_version':2,'bundle_versions':[5,6],'result_version':2,'container_version':29,
            'loss':'negative-log-likelihood','required_representation':'log-evidence',
            'scope':'concrete and temporal-v2 families; Experiment7/Run7/container30; real portable notebooks',
            'diagnostics':'explicit stable output IDs; no fabricated targets'},
        causal_lags={'op_version':1,'container_version':31,'model_revision_version':9,
            'family_version':13,'specialization_version':12,
            'roles':['observations','history','lengths'],'order':'oldest-to-newest; excludes current observation',
            'controls':'explicit prefix lengths; no missing-history imputation or resets'},
        unavailable={
            'TrainableParameter': 'not an accepted spelling; use TensorParameter',
            'msvar_missing_history_integration': 'not admitted; explicit observed causal history is required',
            'optimizer_resume': 'deferred Stage E, not part of this source'},
        limitations=['Shapes alone do not establish output representation or loss compatibility.',
            'Current cross-entropy expects logits; the checker does not prove an upstream tensor is logits.',
            'Schema examples are construction witnesses, not datasets, training fits or statistical benchmarks.'],
        commands=['capabilities', 'format-schema', 'operator-schema', 'parameter-policy', 'ports'])


def parameter_contract(models, mid):
    from .policy_graph import present, plan
    check_document(models[mid])
    value = plan(models, mid)
    return {'version': 1, 'kind': 'visual-pytorch-parameter-policy',
            'scope': 'read-current-graph-contract-only; no allocation, initialization or training',
            'id': mid, 'explicit': present(models, mid),
            'identity_encoding': 'json-components-v2' if present(models, mid) else 'legacy-slash-v1',
            **value['resolved']}


def port_contract(models, mid):
    if not isinstance(mid, str) or mid not in models:
        raise GraphError('Port discovery requires an existing architecture ID.')
    check_document(models[mid])
    c = validate_model(models, mid)
    interfaces = {}
    def visit(current):
        for node in models[current]['graph']['nodes']:
            if node['type'] in operators.REFERENCE_TYPES and node['ref'] not in interfaces:
                child = node['ref']; check_document(models[child]); child_contract = validate_model(models, child)
                interfaces[child] = {'id': child, 'inputs': copy.deepcopy(child_contract['input_groups']),
                                     'outputs': copy.deepcopy(child_contract['output_groups'])}
                visit(child)
    visit(mid)
    return {'version': 1, 'kind': 'visual-pytorch-port-contract', 'scope': 'read-current-graph-contract-only',
        'id': mid, 'inputs': copy.deepcopy(c['input_groups']), 'outputs': copy.deepcopy(c['output_groups']),
        'dtype': 'float32', 'batch_axis': 'implicit B', 'packing': 'contract order; never alphabetically reorder ports',
        'output_representation': 'not explicitly declared for unknown tensors; known operator semantics appear per output group when proven, never inferred from labels',
        'execution_plan': copy.deepcopy(c['plan']),
        'referenced_interfaces': [interfaces[key] for key in sorted(interfaces)],
        'nodes': [{'id': n['id'], 'type': n['type'], 'shape_contract': copy.deepcopy(c['shapes'][n['id']]),
                   'incoming': copy.deepcopy([e for e in models[mid]['graph']['edges'] if e['to'] == n['id']]),
                   'outgoing': copy.deepcopy([e for e in models[mid]['graph']['edges'] if e['from'] == n['id']]),
                   'bindings': copy.deepcopy({k: n[k] for k in ('ref', 'inputs', 'sequence_inputs', 'states',
                       'sequence_outputs', 'controls', 'lengths', 'parts', 'outputs', 'sources') if k in n})}
                  for n in models[mid]['graph']['nodes']]}


def format_schema():
    # Query the actual in-memory creator, not a second hand-maintained SQL list.
    workspace = Workspace()
    try:
        sql = [{'name': name, 'type': kind, 'sql': text} for name, kind, text in workspace.db.execute(
            "SELECT name,type,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name")]
    finally: workspace.db.close()
    return _header('format-schema', format=FORMAT, container_version=VERSION, reader_versions=sorted(SUPPORTED_VERSIONS),
        extension='.vtw', storage='SQLite', sqlite_schema=sql, independent_kinds=list(KINDS),
        revision_semantics='immutable exact kind/id/SHA256 revisions; heads are browsing selections',
        loading='working-copy migration; opening does not save or downgrade unsupported newer headers',
        validation='app.core.Workspace.load/save and shared scientific validators; this is not a JSON Schema validator',
        documentation='docs/vtw-format.md')
