"""Portable concrete report runtime: no Qt, no notebook authoring dependency.

Result reports read saved evidence. Reruns explicitly use captured manifests and
the real isolated numerical worker. Outputs are written only to a fresh attempt.
"""
import copy
import csv
import html
import io
import json
import math
import platform
import sys
import textwrap
from pathlib import Path

from .core import Workspace, validate_model
from .evidence import digest
from .experiment_objects import execute_candidate, object_reference, report_data


def new_file(path, content):
    data = content.encode('utf-8') if isinstance(content, str) else content
    with Path(path).open('xb') as stream:
        stream.write(data)


def new_json(path, value):
    new_file(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def workspace_for(payload):
    workspace = Workspace()
    try:
        workspace.set_constrained_mode(payload['execution_policy']['constrained'])
        for entry in payload['documents']:
            workspace.put_spec(entry['kind'], entry['document'])
        return workspace
    except BaseException:
        workspace.db.close()
        raise


def resolve_payload(payload):
    """Validate all exact dependencies and captured preview without fitting/runs."""
    from .datasets import _keys, _require, _version
    from .experiment_objects import candidate, reference
    from .memory import memory_percent
    _keys(payload, ('version', 'recipe', 'documents', 'preview', 'execution_policy', 'generation'), context='Report payload')
    _version(payload, 'Report payload'); reference(payload['recipe'], 'report')
    _require(isinstance(payload['documents'], list), 'Report dependency closure must be an array.')
    for item in payload['documents']:
        _keys(item, ('kind', 'document'), context='Report dependency')
    policy = payload['execution_policy']
    _keys(policy, ('backend', 'constrained', 'memory_limit_percent', 'worker_timeout'), context='Report execution policy')
    _require(policy['backend'] == 'cpu-float32-v1' and type(policy['constrained']) is bool, 'Unsupported report backend/policy.')
    memory_percent(policy['memory_limit_percent'])
    _require(policy['worker_timeout'] is None or (type(policy['worker_timeout']) in (int, float) and
        math.isfinite(policy['worker_timeout']) and policy['worker_timeout'] > 0), 'Worker timeout must be positive finite seconds/null.')
    _keys(payload['generation'], ('python', 'platform'), context='Report generation environment')
    _require(all(isinstance(v, str) and v for v in payload['generation'].values()), 'Missing report generation environment.')
    workspace = workspace_for(payload)
    try:
        recipe = workspace.get_spec('report', payload['recipe']['id'], payload['recipe']['revision'])
        required = set()
        def visit(ref):
            key = (ref['kind'], ref['id'], ref['revision'])
            if key in required:
                return
            required.add(key)
            from .experiment_objects import dependencies
            doc = workspace.get_spec(*key)
            for dep in dependencies(ref['kind'], doc):
                visit(dep)
        visit(payload['recipe'])
        actual = [(e['kind'], e['document']['id'], e['document']['revision']) for e in payload['documents']]
        _require(len(actual) == len(set(actual)) and set(actual) == required, 'Report must contain exactly its dependency closure.')
        if recipe['mode'] == 'result-report':
            _require(payload['preview'] is None, 'A result report cannot request candidate execution.')
        else:
            preview = payload['preview']
            _keys(preview, ('version', 'experiment', 'offset', 'limit', 'total_combinations', 'candidates',
                            'next_offset', 'scope'), context='Captured report preview')
            experiment = workspace.get_spec('experiment', recipe['experiment']['id'], recipe['experiment']['revision'])
            # Verify axes/count/page identities without regenerating captured splits.
            # Enumerate the bounded coordinates without regenerating saved splits.
            axes = (experiment['models'], [d['id'] for d in experiment['data_tasks']], [s['id'] for s in experiment['splits']],
                    experiment['initialization_seeds'], [p['id'] for p in experiment['protocols']])
            from .datasets import _integer
            _version(preview, 'Captured report preview')
            _require(preview['experiment'] == recipe['experiment'] and
                _integer(preview['offset']) and _integer(preview['limit'], 1, 1000) and
                preview['total_combinations'] == math.prod(map(len, axes)) and isinstance(preview['candidates'], list),
                'Captured selection page is invalid.')
            offset, total = preview['offset'], preview['total_combinations']
            count = max(0, min(total - offset, preview['limit']))
            _require(len(preview['candidates']) == count and preview['next_offset'] == (offset + count if offset + count < total else None)
                and preview['scope'] == 'bounded-selection-preview; no-fitting-or-training', 'Captured selection coverage is inconsistent.')
            for index, row in enumerate(preview['candidates'], offset):
                digits = []; value = index
                for axis in reversed(axes):
                    value, digit = divmod(value, len(axis)); digits.append(axis[digit])
                coords = dict(zip(('model', 'data_task', 'split', 'initialization', 'protocol'), reversed(digits)))
                _require(row['coordinates'] == coords, 'Captured candidate coordinates silently changed.')
                fresh = candidate(experiment, coords, workspace.get_spec, saved_manifest=row['manifest'])
                _require(digest(fresh) == digest(row), 'Captured candidate/protocol/split disagrees with experiment.')
        return recipe
    finally:
        workspace.db.close()


def run_report(payload, directory):
    """Called only by an explicit notebook code cell; preserve every terminal run."""
    recipe = resolve_payload(payload); directory = Path(directory)
    workspace = workspace_for(payload)
    try:
        refs = []
        if recipe['mode'] == 'rerun':
            policy = payload['execution_policy']
            for row in payload['preview']['candidates']:
                produced = execute_candidate(workspace, recipe['experiment'], row['coordinates'],
                    'notebook-' + row['id'], 'Notebook candidate ' + row['id'][:12], python=sys.executable,
                    memory_limit_percent=policy['memory_limit_percent'], timeout=policy['worker_timeout'],
                    saved_manifest=row['manifest'])
                new_json(directory / ('run-' + row['id'] + '.json'), produced['record'])
                refs.append(produced['reference'])
                print(row['id'][:12] + ': ' + produced['record']['attempts'][-1]['status'])
            if refs:
                material = copy.deepcopy(recipe)
                material.pop('revision'); material.update(id='notebook-result-' + recipe['revision'][:24],
                    mode='result-report', runs=refs, experiment=None)
                ref = workspace.put_spec('report', material)
                material = workspace.get_spec('report', ref['id'], ref['revision'])
                view = report_data(material, workspace.get_spec)
                new_json(directory / 'resolved-report-recipe.json', material)
            else:
                view = report_data(recipe, workspace.get_spec)
        else:
            view = report_data(recipe, workspace.get_spec)
        view['notebook_execution'] = {'mode': recipe['mode'], 'recipe': object_reference('report', recipe),
            'scope': 'descriptive; conditional on selected data/splits/protocol; no confidence intervals',
            'python': sys.version.split()[0], 'platform': platform.platform(), 'training_requested': recipe['mode'] == 'rerun',
            'selected_page': ({k: payload['preview'][k] for k in ('offset', 'limit', 'total_combinations', 'next_offset')}
                              if payload['preview'] is not None else None)}
        new_json(directory / 'report-data.json', view)
        rows = metric_rows(view, workspace)
        stream = io.StringIO()
        fields = ('run', 'run_revision', 'attempt', 'status', 'model', 'dataset', 'task', 'checkpoint',
                  'population', 'head', 'loss', 'units', 'target_transforms', 'value', 'denominator', 'observations', 'parameters', 'updates')
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        new_file(directory / 'metrics.csv', stream.getvalue())
        from .core import generate_code
        from .task_execution import generate_task_predictor
        from .experiment_objects import candidate_bundle
        for entry in payload['documents']:
            if entry['kind'] == 'model':
                model = entry['document']
                new_file(directory / ('architecture-' + model['revision'] + '.py'), generate_code(model['models'], model['root']))
        for summary in view['runs']:
            record = workspace.get_spec('run', summary['reference']['id'], summary['reference']['revision'])
            evidence = record['attempts'][-1]['result']
            if evidence:
                experiment = workspace.get_spec('experiment', record['selection']['experiment']['id'],
                                                record['selection']['experiment']['revision'])
                bundle = candidate_bundle(experiment, record['selection']['coordinates'], workspace.get_spec,
                                          saved_manifest=record['resolved']['manifest'])
                new_file(directory / ('predictor-' + summary['reference']['revision'] + '.py'),
                         generate_task_predictor(bundle, evidence['checkpoint'], constrained=False))
        return view
    finally:
        workspace.db.close()


def metric_rows(view, workspace):
    rows = []
    for run in view['runs']:
        resolved = run['resolved']
        model = dataset = task = None
        if resolved:
            model, dataset, task = [workspace.get_spec(k, resolved['references'][k]['id'], resolved['references'][k]['revision'])
                                    for k in ('model', 'dataset', 'task')]
        base = {'run': run['reference']['id'], 'run_revision': run['reference']['revision'], 'attempt': run['latest_attempt'],
            'status': run['status'], 'model': model['name'] if model else '', 'dataset': dataset['name'] if dataset else '',
            'task': task['name'] if task else '', 'checkpoint': run['checkpoint_id'] or '',
            'parameters': validate_model(model['models'], model['root'])['parameters'] if model else '',
            'updates': run['counts']['completed_updates'] if run['counts'] else ''}
        if not run['metrics']:
            rows.append(base | dict(population='', head='', loss='', units='', target_transforms='', value='', denominator='', observations=''))
        for role, population in run['metrics'].items():
            for hid, metric in population['heads'].items():
                head = next(h for h in task['heads'] if h['id'] == hid)
                field = next(f for f in dataset['fields'] if f['id'] == head['field'])
                rows.append(base | {'population': role, 'head': hid, 'loss': metric['kind'],
                    'units': field.get('unit', ''), 'target_transforms': json.dumps(head['transforms'], sort_keys=True),
                    'value': metric['value'] if metric['value'] is not None else '',
                    'denominator': metric['denominator'], 'observations': metric['evaluated_records']})
    return rows


def table(headers, rows):
    escape = lambda v: html.escape(str(v), quote=True)
    return '<table><thead><tr>' + ''.join('<th>' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table>'


def section_html(section, payload, view):
    workspace = workspace_for(payload)
    try:
        runs = view['runs']
        if section == 'question':
            return '<p>' + html.escape(view['question']) + '</p><p>' + html.escape(view['notebook_execution']['scope']) + '</p>'
        if section == 'evaluation':
            rows = metric_rows(view, workspace)
            return table(('Run', 'Status', 'Population', 'Head', 'Loss', 'Source target unit', 'Target transforms', 'Value', 'Effective weight', 'Records'),
                [[r[k] for k in ('run', 'status', 'population', 'head', 'loss', 'units', 'target_transforms', 'value', 'denominator', 'observations')] for r in rows])
        if section == 'failures':
            return table(('Run', 'Attempt', 'Status', 'Stage', 'Diagnostic'),
                [[r['reference']['id'], a['number'], a['status'], a['stage'], (a['error'] or {}).get('message', '')]
                 for r in runs for a in r['attempt_history']])
        if section == 'models':
            models = [e['document'] for e in payload['documents'] if e['kind'] == 'model']
            return table(('Model', 'Exact revision', 'Parameters', 'Architecture hash'),
                [[m['name'], m['revision'], validate_model(m['models'], m['root'])['parameters'], m['architecture_hash']] for m in models])
        if section == 'datasets':
            datasets = [e['document'] for e in payload['documents'] if e['kind'] == 'dataset']
            return table(('Dataset', 'Exact revision', 'Observations', 'Fields'),
                [[d['name'], d['revision'], len(d['records']), json.dumps(d['fields'], sort_keys=True)] for d in datasets])
        if section == 'training':
            return table(('Run', 'Status', 'Actual counters', 'Checkpoint'),
                [[r['reference']['id'], r['status'], json.dumps(r['counts'], sort_keys=True), r['checkpoint_id']] for r in runs])
        if section == 'checkpoints':
            return table(('Run', 'Exact revision', 'Checkpoint', 'Selection'),
                [[r['reference']['id'], r['reference']['revision'], r['checkpoint_id'], json.dumps(r['selection'], sort_keys=True)] for r in runs])
        if section == 'environment':
            return table(('Run', 'Environment'), [[r['reference']['id'], json.dumps(r['environment'], sort_keys=True)] for r in runs])
        if section == 'protocol':
            return table(('Run', 'Captured split/protocol'), [[r['reference']['id'], json.dumps(r['resolved'], sort_keys=True)] for r in runs])
        raise ValueError('Unsupported report section.')
    finally:
        workspace.db.close()


def curve_svg(summary, result, hid, context):
    """Numerical SVG with audited coordinates, missing spans and scoped identity."""
    history = result['history']
    series = {role: [(p['update'], p['metrics'][role]['heads'][hid]['value']) for p in history]
              for role in ('train', 'validation')}
    points = [(x, y) for values in series.values() for x, y in values if y is not None]
    if not points:
        return None
    max_x = max(1, max(p['update'] for p in history))
    max_y = max(1e-12, max(y for x, y in points) * 1.05)
    kind = history[-1]['metrics']['train']['heads'][hid]['kind']
    title = context['model'] + ' / ' + hid + ' / ' + kind
    retention = result['history_retention']
    captions = [
        'Run ' + summary['reference']['id'] + '; revision ' + summary['reference']['revision'],
        'Final checkpoint ' + result['checkpoint_id'],
        'Dataset: ' + context['dataset'] + '. Weighted population mean; individual heads, not the combined objective.',
        'Source target unit: ' + (context['unit'] or 'unspecified') + '. Target transforms: ' + json.dumps(context['transforms'], sort_keys=True) + '. Loss is in transformed target space.',
        'Retained ' + str(len(history)) + ' / ' + str(retention['observed_points']) + ' observations; discarded ' + str(retention['discarded_points']) + '. Missing values break lines; no smoothing or confidence intervals.',
        'Every point evaluates one checkpoint over its named population. Test metrics are final-only and are not plotted. Exact checkpoints, denominators and counts are in the companion CSV.']
    caption_lines = [line for caption in captions for line in textwrap.wrap(caption, width=100)]
    title_lines = textwrap.wrap(title, width=82) or ['Training history']
    top = 46 + 18 * (len(title_lines) - 1); bottom = top + 224
    width, height = 760, bottom + 80 + 16 * len(caption_lines)
    escape = lambda v: html.escape(str(v), quote=True)
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="' + str(height) +
                '" viewBox="0 0 760 ' + str(height) + '" role="img" aria-labelledby="curve-title curve-desc">',
                '<title id="curve-title">' + escape(title) + '</title>',
                '<desc id="curve-desc">' + escape(' '.join(captions)) + '</desc>',
                '<metadata>' + escape(json.dumps({'version': 1, 'run': summary['reference'], 'head': hid,
                    'final_checkpoint': result['checkpoint_id'], 'context': context, 'series': series,
                    'x_range': [0, max_x], 'y_range': [0, max_y], 'scale': 'linear',
                    'retention': retention}, sort_keys=True, allow_nan=False)) + '</metadata>',
                '<rect width="760" height="' + str(height) + '" fill="#f7fafc"/>',
                '<g font-family="sans-serif" font-size="12" fill="#203040">']
    for i, line in enumerate(title_lines):
        elements.append('<text x="72" y="' + str(22 + 18 * i) + '" font-size="14">' + escape(line) + '</text>')
    for fraction in (0, .25, .5, .75, 1):
        y = bottom - 224 * fraction
        elements += ['<line x1="72" y1="' + str(y) + '" x2="732" y2="' + str(y) + '" stroke="#d5dee5"/>',
                     '<text x="64" y="' + str(y + 4) + '" text-anchor="end">' + format(max_y * fraction, '.4g') + '</text>']
    for x in sorted(set(round(max_x * fraction) for fraction in (0, .25, .5, .75, 1))):
        px = 72 + 660 * x / max_x
        elements.append('<text x="' + str(px) + '" y="' + str(bottom + 18) + '" text-anchor="middle">' + str(x) + '</text>')
    elements += ['<path d="M 72 ' + str(top) + ' V ' + str(bottom) + ' H 732" fill="none" stroke="#657585"/>',
                 '<text x="18" y="' + str((top + bottom) / 2) + '" transform="rotate(-90 18 ' + str((top + bottom) / 2) + ')" text-anchor="middle">Weighted mean ' + escape(kind) + ' (linear)</text>',
                 '<text x="402" y="' + str(bottom + 39) + '" text-anchor="middle">Successful optimizer updates (0 = initial; post-update evaluation)</text>']
    for role, color, dashed in (('train', '#087f8c', False), ('validation', '#ad4b16', True)):
        segments = [[]]
        for x, y in series[role]:
            if y is None:
                if segments[-1]: segments.append([])
                continue
            segments[-1].append((72 + 660 * x / max_x, bottom - 224 * y / max_y))
        for segment in filter(None, segments):
            xy = ' '.join(format(x, '.6f') + ',' + format(y, '.6f') for x, y in segment)
            elements.append('<polyline data-population="' + role + '" fill="none" stroke="' + color +
                '" stroke-width="2"' + (' stroke-dasharray="6 4"' if dashed else '') + ' points="' + xy + '"/>')
            for x, y in segment:
                if dashed:
                    elements.append('<rect x="' + format(x - 2.5, '.6f') + '" y="' + format(y - 2.5, '.6f') + '" width="5" height="5" fill="' + color + '"/>')
                else:
                    elements.append('<circle cx="' + format(x, '.6f') + '" cy="' + format(y, '.6f') + '" r="2.5" fill="' + color + '"/>')
    elements += ['<text x="72" y="' + str(bottom + 61) + '" fill="#087f8c">Train: solid / circles</text>',
                 '<text x="290" y="' + str(bottom + 61) + '" fill="#ad4b16">Validation: dashed / squares</text>']
    for i, line in enumerate(caption_lines):
        elements.append('<text x="20" y="' + str(bottom + 86 + 16 * i) + '" font-size="11">' + escape(line) + '</text>')
    return '\n'.join(elements + ['</g>', '</svg>'])


def curves(payload, view, directory):
    """One SVG per run/head: never mix unlike datasets/losses on one numeric axis."""
    workspace = workspace_for(payload); paths = []
    try:
        for summary in view['runs']:
            if view['mode'] == 'result-report' and payload['preview'] is not None:
                path = Path(directory) / ('run-' + summary['selection']['candidate_id'] + '.json')
                workspace.put_spec('run', json.loads(path.read_text(encoding='utf-8')))
            record = workspace.get_spec('run', summary['reference']['id'], summary['reference']['revision'])
            result = record['attempts'][-1]['result']
            if result is None:
                continue
            heads = result['history'][-1]['metrics']['train']['heads'] if result['history'] else {}
            refs = record['resolved']['references']
            model, dataset, task = [workspace.get_spec(k, refs[k]['id'], refs[k]['revision'])
                                    for k in ('model', 'dataset', 'task')]
            for hid in heads:
                head = next(h for h in task['heads'] if h['id'] == hid)
                field = next(f for f in dataset['fields'] if f['id'] == head['field'])
                svg = curve_svg(summary, result, hid, {'model': model['name'], 'dataset': dataset['name'],
                    'unit': field.get('unit', ''), 'transforms': head['transforms']})
                if svg is None:
                    continue
                path = Path(directory) / ('curve-' + digest({'run': summary['reference'], 'head': hid})[:24] + '.svg')
                new_file(path, svg); paths.append(str(path))
                stream = io.StringIO(); fields = ('update', 'checkpoint', 'population', 'head', 'kind', 'value', 'denominator', 'records')
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for point in result['history']:
                    for role in ('train', 'validation'):
                        metric = point['metrics'][role]['heads'][hid]
                        writer.writerow(dict(update=point['update'], checkpoint=point['checkpoint_id'], population=role,
                            head=hid, kind=metric['kind'], value=metric['value'], denominator=metric['denominator'], records=metric['evaluated_records']))
                new_file(path.with_suffix('.csv'), stream.getvalue())
        return paths
    finally:
        workspace.db.close()
