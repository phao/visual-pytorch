"""Pure small LoRA fixture for persisted/executed policy route checks."""
from app.policy_graph import owner_key
import math


def models(rank=2, temporal=False):
    nodes = [{'id': 'x', 'type': 'Input', 'shape': [3, 16] if temporal else [16]},
             {'id': 'base', 'type': 'Linear', 'features': 16, 'bias': False},
             {'id': 'A', 'type': 'Linear', 'features': rank, 'bias': False},
             {'id': 'B', 'type': 'Linear', 'features': 16, 'bias': False},
             {'id': 'scale', 'type': 'Scale', 'coefficient': 4. / rank, 'op_version': 1},
             {'id': 'add', 'type': 'Add', 'sources': ['base', 'scale'], 'op_version': 1},
             {'id': 'y', 'type': 'Output'}]
    if temporal: nodes[0]['axes'] = ['T', 'F']
    pairs = [('x', 'base'), ('x', 'A'), ('A', 'B'), ('B', 'scale'),
             ('base', 'add'), ('scale', 'add'), ('add', 'y')]
    parameters = {owner_key(['base', 'weight']): {'trainable': False, 'initializer':
                  {'kind': 'values', 'dtype': 'float32', 'values': [[.6 if i == j else 0. for j in range(16)] for i in range(16)]}},
                  owner_key(['A', 'weight']): {'initializer': {'kind': 'normal', 'seed': 17, 'mean': 0., 'std': .05}},
                  owner_key(['B', 'weight']): {'initializer': {'kind': 'zeros'}}}
    doc = {'id': 'lora', 'name': 'Synthetic supplied base — LoRA', 'graph': {'nodes': nodes,
           'edges': [{'from': a, 'to': b} for a, b in pairs],
           'parameter_policy': {'version': 1, 'parameters': parameters}}}
    return {'lora': doc}, 'lora'


def bundle(updates=20, temporal=False):
    from app.datasets import make_split
    from app.revisions import model_document
    from app.task_execution import task_bundle, training_protocol
    collection, mid = models(temporal=temporal)
    rows = []
    def pair(i):
        x = [math.sin((i + 1) * (j + 1) * .3) / 4 for j in range(16)]
        aggregate = sum((j + 1) / 16 * x[j] for j in range(16))
        y = [.6 * v + .03 * math.cos(k + 1) * aggregate for k, v in enumerate(x)]
        return x, y
    for row in range(24):
        parts = [pair(row * 3 + t) for t in range(3)] if temporal else [pair(row)]
        x, y = ([p[0] for p in parts], [p[1] for p in parts]) if temporal else parts[0]
        rows.append({'id': 'r' + str(row), 'values': {'x': x, 'y': y}})
    shape = [3, 16] if temporal else [16]
    dataset = {'version': 1, 'id': 'data', 'name': 'Synthetic low-rank target',
               'storage': {'kind': 'inline-records', 'version': 1},
               'fields': [{'id': key, 'dtype': 'float32', 'shape': shape,
                           **({'axes': ['T', 'F']} if temporal else {})} for key in ('x', 'y')], 'records': rows}
    task = {'version': 1, 'id': 'task', 'name': 'LoRA MSE',
            'inputs': [{'port': 'x', 'field': 'x', 'transforms': []}],
            'heads': [{'id': 'target', 'port': 'y', 'field': 'y', 'loss': 'mse', 'weight': 1., 'transforms': []}]}
    manifest = make_split(dataset, {'method': 'row', 'seed': 13, 'fractions': [.75, .125, .125]})
    return task_bundle(model_document(collection, mid), dataset, task, manifest,
                       training_protocol(epochs=None, updates=updates, batch_size=6, lr=.02, evaluate_every=5))


def family(temporal=False):
    collection, mid = models(temporal=temporal)
    return {'version': 10, 'id': 'lora-family', 'name': 'Supplied synthetic base LoRA',
            'root': mid, 'models': collection, 'symbols': [], 'assignments': [], 'repeats': [],
            'constructors': [], 'runtime_policy': {'version': 2, 'kind': 'runtime-temporal-v2',
                                                  'inputs': {'x': [0]}} if temporal else None}
