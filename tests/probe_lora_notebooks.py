"""Actual kernels for an EXISTING completed LoRA example, not synthetic metrics.

Usage: python tests/probe_lora_notebooks.py EXISTING_LORA.vtw NEW_OUTPUT
No source workspace writes. Rerun explicitly executes four200-update linear fits.
"""
import json
import sys
from pathlib import Path
import torch
from app.core import Workspace
from app.experiment_objects import candidate_bundle
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.task_execution import restore_task_predictor


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    source, output = map(lambda value: Path(value).resolve(), args)
    before_file = source.read_bytes(); output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    w = Workspace.load(source)
    try:
        entries = w.list_specs('run')
        runs = [w.get_spec('run', row['id'], row['revision']) for row in entries
                if row['id'].startswith('lora-linear-comparison-')]
        assert len(runs) == 4
        expected = {run['attempts'][0]['result']['checkpoint_id']: run['attempts'][0]['result'] for run in runs}
        assert all(result['counts']['completed_updates'] == 200 for result in expected.values())
        expref = runs[0]['selection']['experiment']
        report = next(row for row in w.list_specs('report') if row['id'] == 'lora-linear-comparison-results')
        references = {'result-report': {k: report[k] for k in ('kind', 'id', 'revision')}}
        references['rerun'] = w.put_spec('report', {
            'version': 1, 'id': 'lora-actual-rerun', 'name': 'Actual LoRA linear comparison rerun',
            'question': 'Does the portable kernel reproduce the actual200-update retained comparison?',
            'mode': 'rerun', 'experiment': expref, 'runs': [],
            'sections': ['models', 'training', 'evaluation', 'checkpoints', 'failures'],
            'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
        views = {}
        for mode, ref in references.items():
            raw = w.db.serialize(); capture = create_bundle(w, ref, output / mode)
            verify_bundle(capture['path'])
            assert (Path(capture['path']) / 'runtime/app/parameter_derivation.py').is_file()
            executed = execute_bundle(capture['path'], timeout=120)
            assert executed['execution']['status'] == 'completed', executed['execution']
            root = Path(executed['path']); view = json.loads((root / 'report-data.json').read_text())
            views[mode] = view
            assert len(view['runs']) == 4
            assert view['notebook_execution']['training_requested'] == (mode == 'rerun')
            for run in view['runs']:
                assert run['checkpoint_id'] in expected, run
                assert run['metrics'] == expected[run['checkpoint_id']]['final_metrics']
            predictors = list(root.glob('predictor-*.py')); assert len(predictors) == 4
            for path in predictors:
                scope = {}; exec(path.read_text(), scope)
                revision = path.stem.removeprefix('predictor-')
                checkpoint_id = next(run['checkpoint_id'] for run in view['runs']
                                     if run['reference']['revision'] == revision)
                record = next(run for run in runs if run['attempts'][0]['result']['checkpoint_id'] == checkpoint_id)
                experiment = w.get_spec('experiment', expref['id'], expref['revision'])
                bundle = candidate_bundle(experiment, record['selection']['coordinates'], w.get_spec,
                                          saved_manifest=record['resolved']['manifest'])
                records = [row['values'] for row in bundle['dataset']['records'][-7:]]
                native = restore_task_predictor(bundle, expected[checkpoint_id]['checkpoint'], constrained=True)(records)
                predicted = scope['predict_records'](records)
                torch.testing.assert_close(predicted['y'], native['y'], rtol=0, atol=0)
            assert w.db.serialize() == raw and source.read_bytes() == before_file
            print(json.dumps({'status': 'PASS', 'mode': mode, 'real_runs': 4,
                'code_cells': executed['execution']['executed_code_cells'],
                'exact_checkpoint_and_metrics': True, 'exact_predictor_parity': True,
                'original_workspace_untouched': True}), flush=True)
        assert {r['checkpoint_id']: r['metrics'] for r in views['result-report']['runs']} == {
            r['checkpoint_id']: r['metrics'] for r in views['rerun']['runs']}
    finally:
        w.db.close()
        if w.history.directory: w.history.directory.cleanup()


if __name__ == '__main__': main()
