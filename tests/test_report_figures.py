"""Pure numerical figure presentation: scoped IDs, wrapping and exact CSV rows."""
import csv
import io
import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from app.report_runtime import curve_svg, curve_figure, _wrap_lines


def fixture():
    summary = {'reference': {'kind': 'run', 'id': 'measured-run', 'revision': 'b' * 64}}
    context = {'model': 'Model <script>', 'dataset': 'Paired measurements', 'unit': 'm',
               'transforms': [{'op': 'standardize'}]}
    result = {'history': [{'update': i, 'checkpoint_id': str(i),
        'metrics': {role: {'heads': {'head': {'kind': 'mse', 'value': value,
          'denominator': 5.25, 'evaluated_records': 3}}} for role in ('train', 'validation')}}
        for i, value in enumerate((4., None, 1., 0.))], 'checkpoint_id': 'c' * 64,
        'history_retention': {'observed_points': 21, 'discarded_points': 17}}
    return summary, result, context


class ReportFigureTests(unittest.TestCase):
    def files(self, *, repeats=1):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        path = Path(temp.name) / 'curve.csv-test.svg'
        summary, result, context = fixture()
        path.write_text(curve_svg(summary, result, 'head', context), encoding='utf-8')
        stream = io.StringIO(); writer = csv.DictWriter(stream,
            fieldnames=('update', 'checkpoint', 'population', 'head', 'kind', 'value', 'denominator', 'records'))
        writer.writeheader()
        for _ in range(repeats):
            for point in result['history']:
                for role in ('train', 'validation'):
                    metric = point['metrics'][role]['heads']['head']
                    writer.writerow({'update': point['update'], 'checkpoint': point['checkpoint_id'],
                        'population': role, 'head': 'head', 'kind': 'mse', 'value': metric['value'],
                        'denominator': metric['denominator'], 'records': metric['evaluated_records']})
        path.with_suffix('.csv').write_text(stream.getvalue(), encoding='utf-8')
        return path

    def test_different_exact_run_or_head_has_distinct_accessibility_ids(self):
        summary, result, context = fixture()
        ids = []
        for rid, head in [('one', 'head'), ('two', 'head'), ('one', 'second')]:
            summary['reference']['id'] = rid
            if head == 'second':
                for point in result['history']:
                    for role in ('train', 'validation'):
                        point['metrics'][role]['heads']['second'] = point['metrics'][role]['heads']['head']
            root = ET.fromstring(curve_svg(summary, result, head, context))
            ids.extend(e.attrib['id'] for e in root if 'id' in e.attrib)
            self.assertEqual(root.attrib['aria-labelledby'].split(), [e.attrib['id'] for e in root if 'id' in e.attrib])
        self.assertEqual(len(ids), 6); self.assertEqual(len(set(ids)), 6)

    def test_wrapping_keeps_wide_and_combining_text_without_truncation(self):
        for source in ('W' * 160, '神経網' * 80, 'e\u0301' * 100, 'a long phrase with words ' * 8):
            lines = _wrap_lines(source, 60)
            self.assertTrue(lines)
            self.assertEqual(''.join(''.join(lines).split()), ''.join(source.split()))
            import unicodedata
            self.assertTrue(all(sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in line) <= 60 for line in lines))

    def test_layout_changes_preserve_series_domains_gaps_and_linear_zero(self):
        summary, result, context = fixture()
        root = ET.fromstring(curve_svg(summary, result, 'head', context))
        meta = json.loads(root.find('{http://www.w3.org/2000/svg}metadata').text)
        self.assertEqual(meta['series']['train'], [[0, 4.], [1, None], [2, 1.], [3, 0.]])
        self.assertEqual(meta['scale'], 'linear'); self.assertEqual(meta['x_range'], [0, 3])
        self.assertEqual(meta['y_range'], [0, 4.2]); self.assertEqual(meta['context'], context)
        self.assertEqual(len([e for e in root.iter() if e.tag.endswith('polyline') and e.get('data-population') == 'train']), 2)

    def test_companion_is_literal_semantic_figure_with_exact_rows_and_context(self):
        path = self.files(); rendered = curve_figure(path)
        self.assertIn('<figure ', rendered); self.assertIn('<figcaption ', rendered)
        self.assertIn('class="vpw-curve"', rendered); self.assertIn('Exact plotted rows', rendered)
        self.assertIn('5.25', rendered); self.assertIn('denominator', rendered)
        self.assertIn('unavailable', rendered); self.assertIn('>0.0<', rendered)
        self.assertIn('standardize', rendered); self.assertIn('discarded 17', rendered)
        self.assertIn('Model &lt;script&gt;', rendered); self.assertNotIn('<script>', rendered)
        self.assertIn('curve.csv-test.csv', rendered)
        self.assertIn('overflow-wrap:normal;white-space:nowrap', rendered)
        self.assertEqual(rendered.count('<tr>'), 9)

    def test_companion_table_has_explicit_first_hundred_row_bound(self):
        rendered = curve_figure(self.files(repeats=30))
        self.assertIn('first 100 of 240', rendered)
        self.assertEqual(rendered.count('<tr>'), 101)

    def test_empty_csv_has_header_and_zero_not_invented_points(self):
        path = self.files(); path.with_suffix('.csv').write_text('update,checkpoint,population,head,kind,value,denominator,records\n')
        rendered = curve_figure(path)
        self.assertIn('first 0 of 0', rendered); self.assertEqual(rendered.count('<tr>'), 1)

    def test_missing_or_wrong_csv_and_bad_svg_fail_instead_of_faking_data(self):
        path = self.files(); path.with_suffix('.csv').unlink()
        with self.assertRaises(FileNotFoundError): curve_figure(path)
        path.with_suffix('.csv').write_text('wrong,header\n1,2\n')
        with self.assertRaisesRegex(ValueError, 'CSV'): curve_figure(path)
        path.write_text('<svg>broken', encoding='utf-8')
        with self.assertRaises(ET.ParseError): curve_figure(path)

    def test_generation_has_no_numerical_or_gui_dependency(self):
        # Dependencies may already be loaded by other suite modules; inspect a fresh host.
        import subprocess
        script = 'import sys;from app.report_runtime import curve_figure;curve_figure(sys.argv[1]);print([k for k in ("torch","numpy","nbformat","nbclient","PyQt6") if k in sys.modules])'
        child = subprocess.run(['/usr/bin/python3', '-c', script, str(self.files())], capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr); self.assertEqual(child.stdout.strip(), '[]')


if __name__ == '__main__': unittest.main()
