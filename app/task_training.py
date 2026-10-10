"""Task-aware isolated CPU trainer with explicit budgets and checkpoint timing."""
import copy
import math
import platform
import random
import time

from .core import compile_model
from .datasets import _require
from .evidence import digest
from .task_execution import checkpoint_for, resolve_run, run_identity
from .task_runtime import MetricAccumulator, forward_ports, head_terms, tensor_batch


def train_task(bundle, progress=None, cancel=None, *, constrained=True,
               memory_limit_percent=10., clock=time.monotonic):
    # Private data first; validation/admission happens before importing Torch.
    bundle = copy.deepcopy(bundle)
    prepared, memory = resolve_run(bundle, constrained=constrained, memory_limit_percent=memory_limit_percent)
    import torch
    torch.set_num_threads(1)
    config = bundle['protocol']; task = bundle['task']; spec = bundle['model']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seeds']['initialization'])
        model = compile_model(spec['models'], spec['root'], constrained=False)
    settings = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], betas=tuple(settings['betas']),
                                 eps=settings['eps'], weight_decay=settings['weight_decay'], foreach=False, fused=False)
    rng = random.Random(config['seeds']['shuffle'])
    heads = task['heads']; size = config['batch_size']; budget = config['budget']
    n = len(prepared['populations']['train']['observation_ids'])
    counts = {'completed_updates': 0, 'attempted_batches': 0, 'examples_processed': 0,
              'updated_examples': 0, 'skipped_batches': 0, 'completed_epochs': 0}
    started = clock(); history = []; observed_points = 0; status = 'completed'; reason = None; failure = None
    last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = checkpoint_for(bundle, prepared['fitted'], model)

    def terms_for(batch):
        predictions = forward_ports(model, batch['inputs'], prepared['contract'])
        return {h['id']: head_terms(predictions[h['port']], batch['targets'][h['id']],
                                   batch['masks'][h['id']], batch['sample_weights'][h['id']], h['loss']) for h in heads}

    def evaluate(role, cid):
        accumulators = {h['id']: MetricAccumulator(h['loss'], cid, role) for h in heads}
        total = len(prepared['populations'][role]['observation_ids'])
        model.eval()
        with torch.no_grad():
            for start in range(0, total, size):
                batch = tensor_batch(prepared, task, role, list(range(start, min(total, start + size))))
                for hid, terms in terms_for(batch).items():
                    accumulators[hid].add(terms, batch['observation_ids'], checkpoint=cid, population=role)
        metrics = {hid: acc.report() for hid, acc in accumulators.items()}
        active = [h['weight'] * metrics[h['id']]['value'] for h in heads if metrics[h['id']]['status'] == 'available']
        return {'heads': metrics, 'objective': math.fsum(active) if active else None,
                'objective_status': 'available' if active else 'unavailable',
                'objective_rule': 'sum-head-weight-times-available-head-mean; no weight renormalization'}

    def observe():
        nonlocal checkpoint, observed_points
        checkpoint = checkpoint_for(bundle, prepared['fitted'], model); cid = digest(checkpoint)
        point = {'update': counts['completed_updates'], 'attempted_batches': counts['attempted_batches'],
                 'examples_processed': counts['examples_processed'], 'checkpoint_id': cid,
                 'timing': 'initial' if counts['completed_updates'] == 0 else 'post-update', 'mode': 'eval',
                 'metrics': {role: evaluate(role, cid) for role in ('train', 'validation')}}
        history.append(point)
        observed_points += 1
        if len(history) > config['history']['max_points']:
            del history[0]
        if progress:
            progress(copy.deepcopy(point))

    def stop():
        if cancel is not None and cancel.is_set():
            return 'cancelled'
        for key, count in (('updates', 'completed_updates'), ('examples', 'examples_processed'), ('epochs', 'completed_epochs')):
            if budget[key] is not None and counts[count] >= budget[key]:
                return key + '_budget'
        if budget['seconds'] is not None and clock() - started >= budget['seconds']:
            return 'time_budget'
        return None

    try:
        while reason is None:
            reason = stop()
            if reason:
                break
            order = list(range(n))
            if config['shuffle']:
                rng.shuffle(order)
            epoch_updates = counts['completed_updates']; finished_epoch = True
            for start in range(0, n, size):
                reason = stop()
                if reason:
                    finished_epoch = False
                    break
                indices = order[start:start + size]
                if budget['examples'] is not None:
                    indices = indices[:budget['examples'] - counts['examples_processed']]
                model.train(); optimizer.zero_grad(set_to_none=True)
                batch = tensor_batch(prepared, task, 'train', indices)
                counts['attempted_batches'] += 1; counts['examples_processed'] += len(indices)
                terms = terms_for(batch)
                active = [h['weight'] * terms[h['id']]['mean'] for h in heads if terms[h['id']]['mean'] is not None]
                if not active:
                    counts['skipped_batches'] += 1
                else:
                    loss = sum(active)
                    _require(bool(torch.isfinite(loss)), 'Combined task objective is nonfinite.')
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    _require(gradients and all(bool(torch.isfinite(g).all()) for g in gradients), 'Task gradients are absent/nonfinite.')
                    optimizer.step()
                    _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()) and
                             all(bool(torch.isfinite(v).all()) for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)),
                             'Task optimizer produced nonfinite state.')
                    last_good = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    counts['completed_updates'] += 1; counts['updated_examples'] += len(indices)
                    if counts['completed_updates'] == 1 or counts['completed_updates'] % config['evaluate_every'] == 0:
                        observe()
                if len(indices) < min(size, n - start):
                    finished_epoch = False
                    break
            if finished_epoch:
                counts['completed_epochs'] += 1
                if counts['completed_updates'] == epoch_updates:
                    status = 'unavailable'; reason = 'no_training_supervision'
    except Exception as error:
        model.load_state_dict(last_good, strict=True)
        status = 'failed'; reason = 'execution_error'; failure = str(error)
    if reason == 'cancelled':
        status = 'cancelled'
    # Keep failed/cancelled evidence at the last finite checkpoint. Test is NOT
    # inspected on these partial attempts; their train/validation may be shown.
    final_metrics = {}; evaluation_error = None
    try:
        if (not history or history[-1]['update'] != counts['completed_updates'] or
                history[-1]['attempted_batches'] != counts['attempted_batches']):
            observe()
        checkpoint = checkpoint_for(bundle, prepared['fitted'], model); cid = digest(checkpoint)
        final_metrics = copy.deepcopy(history[-1]['metrics'])
        if status == 'completed':
            final_metrics['test'] = evaluate('test', cid)
    except Exception as error:
        evaluation_error = str(error)
        if status == 'completed':
            status = 'failed'; reason = 'evaluation_error'; failure = evaluation_error
        checkpoint = checkpoint_for(bundle, prepared['fitted'], model); cid = digest(checkpoint)
    return {'version': 1, 'kind': 'task-training-result', 'status': status, 'stop_reason': reason,
            'failure': failure, 'evaluation_error': evaluation_error, 'identity': run_identity(bundle),
            'bundle': bundle, 'checkpoint': checkpoint, 'checkpoint_id': cid, 'counts': counts,
            'history': history, 'final_metrics': final_metrics, 'elapsed_seconds': max(0., clock() - started),
            'history_retention': {**config['history'], 'observed_points': observed_points,
                                  'discarded_points': observed_points - len(history)},
            'memory_admission': memory, 'environment': {'python': platform.python_version(),
                'torch': torch.__version__, 'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32',
                'numerical_threads': 1, 'optimizer_resume': False},
            'sampler': {'kind': 'finite-epoch-python-mt19937-v1', 'replacement': False,
                        'seed': config['seeds']['shuffle'], 'partial_batch': 'retained'}}
