"""Small faithful teaching graphs/data; no Torch, training, writes or macros."""
import copy
import math
from .datasets import _integer, _require
from .policy_graph import owner_key

DIMENSION = 16
ALPHA = 4.


def synthetic_base(diagonal=.6):
    return [[diagonal if i == j else 0. for j in range(DIMENSION)] for i in range(DIMENSION)]


def linear(mid, *, rank=None, seed=17, diagonal=.6, train_base=False):
    _require(_integer(seed, 0, 2**31 - 1) and type(train_base) is bool, 'LoRA seed/trainability invalid.')
    _require(rank is None or _integer(rank, 1, DIMENSION), 'LoRA rank must be integer1–16, not a caption.')
    nodes = [{'id': 'x', 'type': 'Input', 'shape': [DIMENSION], 'label': 'x', 'x': 60, 'y': 200},
             {'id': 'base', 'type': 'Linear', 'features': DIMENSION, 'bias': False, 'x': 300, 'y': 90 if rank else 200}]
    parameters = {owner_key(['base', 'weight']): {'trainable': train_base,
        'initializer': {'kind': 'values', 'dtype': 'float32', 'values': synthetic_base(diagonal)}}}
    pairs = [('x', 'base')]
    if rank is not None:
        _require(not train_base, 'Teaching adapter keeps its supplied base frozen.')
        nodes += [{'id': 'A', 'type': 'Linear', 'features': rank, 'bias': False, 'x': 300, 'y': 310},
                  {'id': 'B', 'type': 'Linear', 'features': DIMENSION, 'bias': False, 'x': 510, 'y': 310},
                  {'id': 'scale', 'type': 'Scale', 'coefficient': ALPHA / rank, 'op_version': 1, 'x': 720, 'y': 310},
                  {'id': 'sum', 'type': 'Add', 'sources': ['base', 'scale'], 'op_version': 1, 'x': 930, 'y': 200}]
        parameters[owner_key(['A', 'weight'])] = {'trainable': True, 'initializer': {'kind': 'normal', 'seed': seed, 'mean': 0., 'std': .05}}
        parameters[owner_key(['B', 'weight'])] = {'trainable': True, 'initializer': {'kind': 'zeros'}}
        pairs += [('x', 'A'), ('A', 'B'), ('B', 'scale'), ('base', 'sum'), ('scale', 'sum')]
    last = 'sum' if rank is not None else 'base'
    nodes += [{'id': 'y', 'type': 'Output', 'label': 'y', 'shape': [DIMENSION], 'x': 1140 if rank else 540, 'y': 200}]
    pairs.append((last, 'y'))
    return {'id': mid, 'name': ('LoRA expanded · r=' + str(rank) if rank is not None else
        'Full fine-tuning · synthetic base' if train_base else 'Base-only · supplied synthetic W₀'),
        'graph': {'nodes': nodes, 'edges': [{'from': a, 'to': b} for a, b in pairs],
                  'parameter_policy': {'version': 1, 'parameters': parameters}}}


def referenced(mid, references, *, activation=False, collapsed=False, name=None):
    nodes = [{'id': 'x', 'type': 'Input', 'shape': [DIMENSION], 'label': 'x', 'x': 60, 'y': 180}]
    for index, ref in enumerate(references):
        nodes.append({'id': 'layer-' + str(index + 1), 'type': 'Module', 'ref': ref, 'collapsed': collapsed,
                      'x': 280 + index * (420 if activation else 230), 'y': 180})
        if activation and index + 1 < len(references):
            nodes.append({'id': 'activation-' + str(index + 1), 'type': 'Activation', 'activation': 'ReLU',
                          'x': 490 + index * 420, 'y': 180})
    nodes.append({'id': 'y', 'type': 'Output', 'shape': [DIMENSION], 'label': 'y',
                  'x': nodes[-1]['x'] + 230, 'y': 180})
    return {'id': mid, 'name': name or 'LoRA reusable · same forward', 'graph': {
        'nodes': nodes, 'edges': [{'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}}


def definitions():
    collection = {doc['id']: doc for doc in [linear('base-linear'), linear('full-finetune', train_base=True)]}
    for rank in (1, 2, 4): collection['lora-r' + str(rank)] = linear('lora-r' + str(rank), rank=rank)
    collection['lora-reusable'] = referenced('lora-reusable', ['lora-r2'], collapsed=True)
    for index, diagonal in ((1, .6), (2, .8)):
        for prefix, rank, trainable in (('mlp-base', None, False), ('mlp-full', None, True), ('mlp-adapter', 2, False)):
            mid = prefix + '-layer-' + str(index)
            collection[mid] = linear(mid, rank=rank, train_base=trainable, diagonal=diagonal, seed=17 + 2 * (index - 1))
            collection[mid]['name'] = prefix.replace('-', ' ').title() + ' layer ' + str(index) + ' · synthetic'
    for prefix, label in (('mlp-base', 'MLP base-only · synthetic'), ('mlp-full', 'MLP full fine-tuning · synthetic'),
                          ('mlp-adapter', 'MLP LoRA · two adapted layers')):
        collection[prefix] = referenced(prefix, [prefix + '-layer-1', prefix + '-layer-2'], activation=True, name=label)
    return collection


def task(mid='lora-regression'):
    return {'version': 1, 'id': mid, 'name': 'Synthetic LoRA regression · no transformed target',
        'inputs': [{'port': 'x', 'field': 'x', 'transforms': []}],
        'heads': [{'id': 'prediction', 'port': 'y', 'field': 'y', 'loss': 'mse', 'weight': 1., 'transforms': []}]}


def _delta(x, phase=0):
    a = [[(j + 1) / DIMENSION for j in range(DIMENSION)], [math.sin(j + 1 + phase) / 4 for j in range(DIMENSION)]]
    b = [[.015 * math.cos((k + 1 + phase) * (r + 1)) for r in range(2)] for k in range(DIMENSION)]
    z = [sum(u * v for u, v in zip(row, x)) for row in a]
    return [2 * sum(u * v for u, v in zip(row, z)) for row in b]


def dataset(*, mlp=False, samples=96):
    _require(_integer(samples, 16, 4096) and type(mlp) is bool, 'Teaching dataset size/mode invalid.')
    records = []
    for i in range(samples):
        x = [math.sin((i + 1) * (j + 1) * .13) / 2 for j in range(DIMENSION)]
        first = [.6 * v + d for v, d in zip(x, _delta(x))]
        if mlp:
            hidden = [max(0., v) for v in first]
            y = [.8 * v + d for v, d in zip(hidden, _delta(hidden, 3))]
        else: y = first
        records.append({'id': 'sample-' + str(i), 'values': {'x': x, 'y': y}})
    return {'version': 1, 'id': 'lora-mlp-data' if mlp else 'lora-linear-data',
        'name': 'Synthetic adapted MLP observations' if mlp else 'Synthetic rank2 linear adaptation observations',
        'storage': {'kind': 'inline-records', 'version': 1},
        'fields': [{'id': key, 'dtype': 'float32', 'shape': [DIMENSION]} for key in ('x', 'y')], 'records': records}


def populate(workspace, *, updates=200):
    from .task_execution import training_protocol
    for model in definitions().values(): workspace.put(copy.deepcopy(model))
    taskref = workspace.put_spec('task', task())
    experiments = {}
    for mlp in (False, True):
        dataref = workspace.put_spec('dataset', dataset(mlp=mlp))
        models = ['mlp-full', 'mlp-adapter'] if mlp else ['full-finetune', 'lora-r1', 'lora-r2', 'lora-r4']
        sid = 'lora-mlp-comparison' if mlp else 'lora-linear-comparison'
        doc = {'version': 1, 'id': sid, 'name': sid.replace('-', ' ').title(),
            'question': 'Compare identical synthetic supplied bases/data under full versus low-rank adaptation. Base-only is separate zero-update evaluation.',
            'models': [workspace.freeze_model(mid) for mid in models],
            'data_tasks': [{'id': 'same-data', 'dataset': dataref, 'task': taskref}],
            'splits': [{'id': 'same-split', 'protocol': {'method': 'row', 'seed': 31, 'fractions': [.625, .125, .25]}}],
            'initialization_seeds': [7], 'protocols': [{'id': 'same-budget', 'protocol': training_protocol(
                epochs=None, updates=updates, batch_size=24, lr=.02, evaluate_every=20, history_limit=64)}],
            'selection': {'kind': 'crossed', 'pairs': [], 'exclusions': []}}
        experiments[sid] = workspace.put_spec('experiment', doc)
        workspace.put_spec('report', {'version': 1, 'id': sid + '-rerun', 'name': doc['name'] + ' · rerun notebook',
            'question': doc['question'], 'mode': 'rerun', 'runs': [], 'experiment': experiments[sid],
            'sections': ['question', 'protocol', 'models', 'datasets', 'training', 'evaluation', 'failures', 'checkpoints'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
    for mid in ('base-linear', 'lora-reusable', 'mlp-base'): workspace.freeze_model(mid)
    return experiments
