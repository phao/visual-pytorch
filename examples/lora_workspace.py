"""Generate actual LoRA definitions; --run-comparisons explicitly performs fits.

Default generation is preparation-only: no Torch, optimizer or invented merged
architecture. Complete execution writes two evaluation-only baseline artifacts,
six genuine retained runs/reports and an explicit rank2 merge when its admission
service is installed. Synthetic teachers generate targets, never predictions.
"""
import argparse
import json
import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import Workspace
from app.lora import ALPHA, populate
from app.lora_execution import merge_selections, run_comparisons


MISSING_MERGE = 'Merged fifth architecture is absent: it requires an actual retained rank2 fit and explicit merge derivation.'


def complete_merge(workspace, comparisons, *, admit=None):
    """Optional main-owned integration boundary; never fabricate its implementation."""
    rank2 = comparisons['rank2']
    if not rank2 or rank2['status'] != 'completed':
        return {'status': 'unavailable', 'warning': MISSING_MERGE}
    if admit is None:
        try:
            from app.parameter_derivation import admit
        except ModuleNotFoundError as error:
            if error.name != 'app.parameter_derivation': raise
            return {'status': 'unavailable', 'warning': MISSING_MERGE + ' parameter_derivation is not installed.'}
    selections = merge_selections(workspace, rank2['run'], attempt=rank2['attempt'])
    result = admit(workspace, selections, alpha=ALPHA, model_id='lora-merged',
                   name='LoRA merged · actual retained rank2 checkpoint')
    return {'status': 'admitted', 'selections': selections, 'derivation': result}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', help='NEW .vtw path; never overwritten')
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--run-comparisons', action='store_true')
    parser.add_argument('--torch-python', default=sys.executable)
    parser.add_argument('--worker-timeout', type=float, default=120.)
    parser.add_argument('--memory-limit-percent', type=float, default=10.)
    args = parser.parse_args(argv)
    path = Path(args.workspace).expanduser().resolve()
    if path.suffix.lower() != '.vtw' or path.exists():
        parser.error('Choose a new .vtw file; existing files are never overwritten.')
    artifacts = {key: path.with_suffix(suffix) for key, suffix in (
        ('linear', '.linear-baseline.json'), ('mlp', '.mlp-baseline.json'), ('comparison', '.comparisons.json'))}
    if args.run_comparisons and any(p.exists() for p in artifacts.values()):
        parser.error('An evaluation artifact already exists; choose a fresh output basename.')
    workspace = Workspace()
    try:
        experiments = populate(workspace, updates=args.updates)
        workspace.save(path, require_new=True)
        output = {'path': str(path), 'phase': 'preparation-only', 'experiments': experiments,
                  'runs': [], 'artifacts': {}, 'merge': {'status': 'unavailable', 'warning': MISSING_MERGE}}
        if args.run_comparisons:
            comparisons = run_comparisons(workspace, experiments, python=args.torch_python,
                timeout=args.worker_timeout, memory_limit_percent=args.memory_limit_percent)
            # Preserve genuine results if the later independent derivation
            # refuses. Such a refusal must never become a fake merged graph.
            workspace.save(path)
            output['merge'] = complete_merge(workspace, comparisons)
            comparisons['merge'] = output['merge']
            workspace.save(path)
            for label in ('linear', 'mlp'):
                with artifacts[label].open('x', encoding='utf-8') as stream:
                    json.dump(comparisons['groups'][label]['baseline'], stream, allow_nan=False, sort_keys=True, indent=2)
                    stream.write('\n')
            with artifacts['comparison'].open('x', encoding='utf-8') as stream:
                json.dump(comparisons, stream, allow_nan=False, sort_keys=True, indent=2); stream.write('\n')
            output.update(phase='comparisons-executed', comparisons_complete=comparisons['complete'],
                complete=comparisons['complete'] and output['merge']['status'] == 'admitted',
                artifacts={k: str(p) for k, p in artifacts.items()},
                runs=[r['run'] for group in comparisons['groups'].values() for r in group['runs']],
                reports={label: group['report'] for label, group in comparisons['groups'].items()})
        print(json.dumps(output, allow_nan=False))
        return output
    finally:
        workspace.db.close()
        if workspace.history.directory: workspace.history.directory.cleanup()


if __name__ == '__main__':
    main()
