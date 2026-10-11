"""Actual kernel/nbconvert reports plus explicitly synthetic SVG layout probes."""
import hashlib
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from app.experiment_objects import execute_candidate, object_reference, preview_experiment
from app.notebooks import create_bundle, execute_bundle
from app.report_runtime import curve_svg
from test_experiment_objects import experiment_workspace


def populate(folder):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    w, e = experiment_workspace()
    try:
        before = w.db.serialize()
        row = preview_experiment(e, w.get_spec)['candidates'][0]
        saved = execute_candidate(w, object_reference('experiment', e), row['coordinates'],
                                 'saved-render', 'Saved render trial', timeout=30)
        failed = execute_candidate(w, object_reference('experiment', e), row['coordinates'],
                                  'failed-render', 'Failed render trial', python='/no-render-python')
        assert saved['record']['attempts'][-1]['status'] == 'completed'
        assert failed['record']['attempts'][-1]['status'] == 'failed'
        results = {}; snippets = []
        for mode in ('result-report', 'rerun'):
            ref = w.put_spec('report', {'version': 1, 'id': 'render-' + mode, 'name': 'Rendered ' + mode,
                'question': 'What do these selected bounded trials show? Literal <script> evidence.',
                'mode': mode, 'runs': [saved['reference'], failed['reference']] if mode == 'result-report' else [],
                'experiment': object_reference('experiment', e) if mode == 'rerun' else None,
                'sections': ['question', 'models', 'datasets', 'protocol', 'training', 'evaluation',
                             'failures', 'environment', 'checkpoints'],
                'analysis': {'kind': 'descriptive', 'inferential_unit': 'run'}})
            source = w.db.serialize()
            create_bundle(w, ref, folder / mode)
            assert w.db.serialize() == source
            terminal = execute_bundle(folder / mode, timeout=60)
            assert terminal['execution']['status'] == 'completed', terminal
            assert w.db.serialize() == source
            path = Path(terminal['path']); notebook = nbformat.read(path / 'executed.ipynb', as_version=4)
            nbformat.validate(notebook)
            exported, _ = HTMLExporter(template_name='basic').from_notebook_node(notebook)
            document = '<!doctype html><meta charset="utf-8"><title>Executed notebook</title><style>' + \
                'body{font:18px/1.45 sans-serif;margin:16px}.cell{max-width:100%}.input_area{overflow:auto}' + \
                'pre{font-size:14px;white-space:pre}.output_subarea{overflow-x:auto}' + \
                'table{border-collapse:collapse}th,td{padding:6px;border:1px solid #ccd;vertical-align:top}' + \
                '</style>' + exported
            html_path = folder / (mode + '.html'); html_path.write_text(document, encoding='utf-8')
            curves = sorted(path.glob('curve-*.svg'))
            assert len(curves) == (1 if mode == 'result-report' else 8)
            for plot in curves:
                meta = json.loads(ET.fromstring(plot.read_text()).find('{http://www.w3.org/2000/svg}metadata').text)
                with plot.with_suffix('.csv').open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                for population, values in meta['series'].items():
                    actual = [[int(r['update']), float(r['value']) if r['value'] else None]
                              for r in rows if r['population'] == population]
                    assert actual == values
                assert all(r['checkpoint'] and r['head'] == meta['head'] and int(r['records']) >= 0
                           and float(r['denominator']) >= 0 for r in rows)
            snippets.extend('<section data-case="real-' + mode + '">' + p.read_text() + '</section>' for p in curves)
            view = json.loads((path / 'report-data.json').read_text())
            results[mode] = {'html': str(html_path), 'terminal': terminal,
                'runs': len(view['runs']), 'statuses': [r['status'] for r in view['runs']],
                'curves': len(curves), 'requested_training': view['notebook_execution']['training_requested'],
                'notebook_cells': len(notebook.cells), 'csv_series_equal': True,
                'html_sha256': hashlib.sha256(document.encode()).hexdigest()}
        # These are hand-declared display stress values, NOT measured training evidence.
        summary = {'reference': {'kind': 'run', 'id': 'synthetic-layout-' + 'W' * 130, 'revision': 'b' * 64}}
        result = {'checkpoint_id': 'c' * 64, 'history_retention': {'observed_points': 5, 'discarded_points': 1},
            'history': [{'update': u, 'metrics': {p: {'heads': {'head': {'kind': 'mse', 'value': value}}}
                for p in ('train', 'validation')}} for u, value in [(0, 4.), (1, None), (2, 2.), (10**12, 0.)]]}
        for name, label in [('wide-latin', 'W' * 160), ('wide-unicode', '神経網' * 60)]:
            context = {'model': label, 'dataset': label, 'unit': 'm', 'transforms': [{'op': 'standardize'}]}
            svg = curve_svg(summary, result, 'head', context)
            snippets.append('<section data-case="' + name + '"><h2>Synthetic layout probe · ' + name + '</h2>' + svg + '</section>')
        gallery = '<!doctype html><meta charset="utf-8"><title>Report curve rendering probes</title><style>' + \
            'body{margin:16px;background:#fff;color:#203040;font:18px sans-serif}section{margin-bottom:30px}' + \
            '</style><h1>Actual retained curves and declared synthetic layout probes</h1>' + ''.join(snippets)
        (folder / 'curves.html').write_text(gallery, encoding='utf-8')
        (folder / 'render.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        assert before != w.db.serialize(), 'Only explicit setup may admit the saved run/recipes'
        print(json.dumps({'modes': {k: {'runs': v['runs'], 'curves': v['curves']} for k, v in results.items()}}))
    finally:
        w.db.close()


if __name__ == '__main__':
    populate(sys.argv[1])
