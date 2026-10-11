"""Validated portable notebook bundles and explicit, isolated real-kernel runs.

Optional Jupyter imports are lazy. Generation never trains or executes cells.
Export takes only exact revisions; execution never overwrites requested bytes.
"""
import hashlib
import asyncio
import html
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

from .core import GraphError, timestamp
from .evidence import digest
from .experiment_objects import dependencies, preview_experiment, reference
from .memory import memory_percent
from .report_runtime import new_file, new_json, resolve_payload

RUNTIME = ('__init__.py', 'activations.py', 'core.py', 'datasets.py', 'evidence.py', 'experiment_objects.py',
           'families.py', 'family_records.py', 'family_axes.py', 'runtime_tasks.py', 'family_export.py',
           'file_sync.py', 'graph.py', 'history.py', 'memory.py', 'operators.py', 'presentation.py', 'runtime_shapes.py',
           'report_runtime.py', 'revisions.py', 'task_execution.py', 'task_result.py',
           'task_runtime.py', 'task_training.py', 'tensor_data.py', 'transport.py', 'worker.py')
RUNTIME += ('recurrent.py', 'recurrent_nodes.py')
FORMAT = 'visual-pytorch-report-bundle'


def _require(ok, message):
    if not ok:
        raise GraphError(message)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _jupyter(module):
    try:
        return __import__(module, fromlist=['*'])
    except ImportError as error:
        raise GraphError('Optional notebook dependency missing: ' + module +
                         '. Install requirements-report.txt in the chosen environment.') from error


def payload_for(workspace, recipe_ref, *, offset=0, limit=100, allow_partial=False,
                memory_limit_percent=10., worker_timeout=None):
    reference(recipe_ref, 'report')
    _require(type(allow_partial) is bool, 'Partial-selection permission must be boolean.')
    recipe = workspace.get_spec('report', recipe_ref['id'], recipe_ref['revision'])
    documents = []; seen = set()
    def visit(ref):
        key = (ref['kind'], ref['id'], ref['revision'])
        if key in seen:
            return
        seen.add(key); doc = workspace.get_spec(*key)
        for dep in dependencies(ref['kind'], doc):
            visit(dep)
        documents.append({'kind': ref['kind'], 'document': doc})
    visit(recipe_ref)
    preview = None
    if recipe['mode'] == 'rerun':
        experiment = workspace.get_spec('experiment', recipe['experiment']['id'], recipe['experiment']['revision'])
        preview = preview_experiment(experiment, workspace.get_spec, offset=offset, limit=limit)
        entire = preview['offset'] == 0 and len(preview['candidates']) == preview['total_combinations']
        _require(entire or allow_partial, 'Selection exceeds this page. Explicitly allow a partial report or increase the bounded limit.')
        _require(bool(preview['candidates']), 'Rerun report selection page is empty.')
    else:
        _require(offset == 0 and limit == 100 and allow_partial is False, 'Result reports select exact runs, not candidate pages.')
    payload = {'version': 1, 'recipe': recipe_ref, 'documents': documents, 'preview': preview,
        'execution_policy': {'backend': 'cpu-float32-v1', 'constrained': workspace.constrained_mode,
            'memory_limit_percent': memory_percent(memory_limit_percent), 'worker_timeout': worker_timeout},
        'generation': {'python': sys.version.split()[0], 'platform': platform.platform()}}
    resolve_payload(payload)
    return payload


def build_notebook(payload, file_index):
    nbformat = _jupyter('nbformat')
    recipe = next(e['document'] for e in payload['documents'] if e['kind'] == 'report' and
                  e['document']['id'] == payload['recipe']['id'] and e['document']['revision'] == payload['recipe']['revision'])
    mode = recipe['mode']; cells = []
    def add(kind, name, source):
        constructor = nbformat.v4.new_code_cell if kind == 'code' else nbformat.v4.new_markdown_cell
        cells.append(constructor(source, id='vpw-' + digest({'section': name})[:24],
                                 metadata={'visual_pytorch': {'section': name}}))
    behavior = ('Reads saved immutable results; no training is launched.'
                if mode == 'result-report' else 'Executing the resolution cell explicitly launches the captured CPU training selection.')
    coverage = ''
    if payload['preview']:
        p = payload['preview']
        coverage = ('\n\nSelection page: ' + str(p['offset']) + '–' + str(p['offset'] + len(p['candidates'])) +
                    ' (exclusive end), of ' + str(p['total_combinations']) + ' declared combinations. Exclusions/failures remain visible.')
    add('markdown', 'introduction', '# ' + html.escape(recipe['name']) + '\n\n' + html.escape(recipe['question']) +
        '\n\n**Execution behavior: ' + behavior + '**' + coverage +
        '\n\nGeneration does not execute cells. Raw inline research data and weights are included in this bundle; review before sharing. ' +
        'Descriptions are conditional on these selected observations, splits and protocols, not general model rankings.')
    bootstrap = """from pathlib import Path
import hashlib, json, os, re, sys, uuid
from IPython.display import HTML, SVG, display
bundle_root = Path.cwd().resolve()
manifest = json.loads((bundle_root / 'manifest.json').read_text(encoding='utf-8'))
def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
assert manifest['format'] == 'visual-pytorch-report-bundle' and manifest['version'] == 1
assert canonical_hash(manifest['files']) == EXPECTED_INDEX_HASH, 'Input index changed'
for relative, info in manifest['files'].items():
    from pathlib import PurePosixPath
    part = PurePosixPath(relative)
    assert not part.is_absolute() and '..' not in part.parts and relative == part.as_posix()
    path = bundle_root / relative
    assert not path.is_symlink() and path.resolve().is_relative_to(bundle_root), 'Unsafe input path'
    raw = path.read_bytes()
    assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256'], 'Input integrity failure: ' + relative
runtime = bundle_root / 'runtime'
if 'app' in sys.modules:
    assert Path(sys.modules['app'].__file__).resolve().is_relative_to(runtime), 'Another app package is already loaded; use a fresh kernel'
sys.path.insert(0, str(runtime))
from app.report_runtime import run_report, section_html, curves, curve_figure, resolve_payload, table
payload = json.loads((bundle_root / 'payload.json').read_text(encoding='utf-8'))
resolve_payload(payload)
attempt_id = os.environ.get('VPW_NOTEBOOK_ATTEMPT') or uuid.uuid4().hex
assert re.fullmatch('[0-9a-f]{32}', attempt_id)
execution_dir = bundle_root / 'executions' / attempt_id
assert execution_dir.resolve().is_relative_to(bundle_root) and not execution_dir.is_symlink()
execution_dir.mkdir(parents=True, exist_ok=True)
print('Input closure verified; mode:', payload['documents'][-1]['document']['mode'])
"""
    bootstrap = bootstrap.replace('EXPECTED_INDEX_HASH', repr(digest(file_index)))
    add('code', 'bootstrap', bootstrap)
    if mode == 'rerun':
        add('markdown', 'selection-preview-heading',
            '## Captured selection preview\n\nThis cell only displays the saved page. Incompatible/excluded combinations are not silently removed.')
        add('code', 'selection-preview',
            "display(HTML(table(('Model revision', 'Data/task', 'Split', 'Initialization', 'Configuration', 'Status', 'Reason'),\n" +
            "    [[__import__('app.family_axes', fromlist=['label']).label(r['coordinates']['model']), r['coordinates']['data_task'], r['coordinates']['split'],\n" +
            "      r['coordinates']['initialization'], r['coordinates']['protocol'], r['status'], r['reason'] or '']\n" +
            "     for r in payload['preview']['candidates']])))")
    add('markdown', 'execution-behavior', '## Resolve report evidence\n\n' + behavior +
        ' Each completed terminal attempt is retained separately. A cell error stops execution; later tables are not claimed complete.')
    add('code', 'resolution', "report_view = run_report(payload, execution_dir)\n" +
        "display(HTML('<p>Mode: ' + report_view['notebook_execution']['mode'] + " +
        "' · retained run rows: ' + str(len(report_view['runs'])) + '</p>'))")
    for section in recipe['sections']:
        add('markdown', section + '-heading', '## ' + section.replace('_', ' ').title())
        source = 'display(HTML(section_html(' + repr(section) + ', payload, report_view)))'
        if section == 'training':
            source += "\nfor svg in curves(payload, report_view, execution_dir):\n    display(HTML(curve_figure(svg)))"
        add('code', section, source)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'visual_pytorch': {'version': 1, 'recipe': payload['recipe'], 'mode': mode,
            'input_index_hash': digest(file_index), 'execution_behavior': behavior, 'state': 'requested'}})
    nbformat.validate(notebook)
    return notebook


def create_bundle(workspace, recipe_ref, destination, **selection):
    """Create a new directory atomically; never overwrite an existing export."""
    payload = payload_for(workspace, recipe_ref, **selection)
    root = Path(destination).expanduser().absolute()
    _require(not root.exists(), 'Refusing to overwrite an existing report bundle.')
    source = Path(__file__).resolve().parent
    files = {'payload.json': _json(payload), 'LICENSE': (source.parent / 'LICENSE').read_bytes()}
    for name in RUNTIME:
        files['runtime/app/' + name] = (source / name).read_bytes()
    files['requirements.txt'] = b'torch>=2.2\nnbformat>=5.10,<6\nnbclient>=0.10,<1\njupyter-client>=8,<9\nipykernel>=6,<8\n'
    files['README.txt'] = ("""Visual PyTorch portable report prototype
Original runtime is GPL-3.0-only; see LICENSE. Dependencies retain their own licenses.
Generation does not run cells. result-report reads saved evidence; rerun trains explicitly.
Use a trusted Python 3 kernel from this directory. Install requirements.txt as needed.
The inline data/weights in payload.json may be sensitive: review before sharing.
Do not move only the ipynb: transfer this complete directory.
manifest.json verifies the input closure, not authenticity or cross-version bitwise replay.
requested.ipynb is the original document; explicit automated execution keeps a fresh
executions/<attempt>/requested.ipynb, executed.ipynb and execution.json.
Manual Jupyter execution retains numerical artifacts in a fresh attempt directory,
but only the explicit execution adapter saves the executed notebook/receipt automatically.
""" + '\nRecipe: ' + json.dumps(recipe_ref, sort_keys=True) + '\n').encode('utf-8')
    index = {p: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for p, raw in files.items()}
    notebook = build_notebook(payload, index)
    nbformat = _jupyter('nbformat'); requested = nbformat.writes(notebook).encode('utf-8')
    manifest = {'version': 1, 'format': FORMAT, 'recipe': recipe_ref, 'files': index,
                'requested_notebook': {'path': 'requested.ipynb', 'sha256': hashlib.sha256(requested).hexdigest()}}
    root.mkdir(parents=True, exist_ok=False)
    for relative, raw in files.items():
        path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); new_file(path, raw)
    new_file(root / 'requested.ipynb', requested)
    # Publish the index last. A partial I/O failure remains inspectable, not reused.
    new_json(root / 'manifest.json', manifest)
    return {'path': str(root), 'recipe': recipe_ref, 'mode': notebook.metadata.visual_pytorch.mode,
            'status': 'generated-not-executed', 'input_index_hash': digest(index),
            'requested_sha256': manifest['requested_notebook']['sha256']}


def verify_bundle(root):
    from .datasets import _keys, _integer, _version
    root = Path(root).expanduser().resolve()
    _require((root / 'manifest.json').is_file() and not (root / 'manifest.json').is_symlink(), 'Missing/unsafe bundle manifest.')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    _keys(manifest, ('version', 'format', 'recipe', 'files', 'requested_notebook'), context='Report manifest')
    _version(manifest, 'Report manifest'); reference(manifest['recipe'], 'report')
    _require(manifest['format'] == FORMAT and isinstance(manifest['files'], dict), 'Unsupported report bundle.')
    expected = {'payload.json', 'LICENSE', 'README.txt', 'requirements.txt'} | {'runtime/app/' + n for n in RUNTIME}
    family_paths = {'runtime/app/families.py', 'runtime/app/family_records.py'}
    pre_axes = expected - {'runtime/app/family_axes.py'}
    pre_family = pre_axes - family_paths
    legacy = pre_family - {'runtime/app/activations.py'}
    no_activations = pre_axes - {'runtime/app/activations.py'}
    sets = (expected, pre_axes, pre_family, legacy, no_activations,
            expected - family_paths, expected - {'runtime/app/activations.py'})
    sets=sets+tuple(v-{'runtime/app/runtime_shapes.py'} for v in sets)
    for dependency in ('runtime_tasks.py','family_export.py','recurrent.py','recurrent_nodes.py'):
        sets=sets+tuple(v-{'runtime/app/'+dependency} for v in sets)
    _require(set(manifest['files']) in sets,
             'Report source closure is incomplete or contains unexpected paths.')
    for relative, info in manifest['files'].items():
        part = PurePosixPath(relative); path = root / relative
        _keys(info, ('bytes', 'sha256'), context='Report input identity')
        _require(not part.is_absolute() and '..' not in part.parts and part.as_posix() == relative and
            _integer(info['bytes']) and isinstance(info['sha256'], str) and re.fullmatch('[0-9a-f]{64}', info['sha256']),
            'Invalid report input path/identity.')
        _require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
                 'Report input escapes bundle or uses symlinks.')
        raw = path.read_bytes()
        _require(len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256'], 'Report input integrity mismatch: ' + relative)
    if 'runtime/app/family_axes.py' not in manifest['files']:
        import ast
        for name in ('experiment_objects.py', 'report_runtime.py'):
            tree = ast.parse((root / 'runtime/app' / name).read_text(encoding='utf-8'))
            _require(not any(isinstance(n, ast.ImportFrom) and n.level == 1 and
                             (n.module == 'family_axes' or (n.module is None and any(a.name == 'family_axes' for a in n.names)))
                             for n in ast.walk(tree)), 'This report runtime requires its captured family_axes.py dependency.')
    if not family_paths <= set(manifest['files']):
        import ast
        for name in ('core.py', 'revisions.py', 'experiment_objects.py'):
            tree = ast.parse((root / 'runtime/app' / name).read_text(encoding='utf-8'))
            _require(not any(isinstance(n, ast.ImportFrom) and n.level == 1 and
                             n.module in ('families', 'family_records') for n in ast.walk(tree)),
                     'This report runtime requires its captured family specialization dependencies.')
    if 'runtime/app/activations.py' not in manifest['files']:
        # Preserve genuine pre-extraction v1 bundles, not an incomplete new
        # closure. This is a supported-dependency guard, NOT a Python sandbox.
        import ast
        for name in ('core.py', 'operators.py', 'graph.py'):
            tree = ast.parse((root / 'runtime/app' / name).read_text(encoding='utf-8'))
            _require(not any(isinstance(n, ast.ImportFrom) and n.level == 1 and
                             n.module == 'activations' for n in ast.walk(tree)),
                     'This report runtime requires its captured activations.py dependency.')
    if 'runtime/app/runtime_shapes.py' not in manifest['files']:
        import ast
        tree = ast.parse((root / 'runtime/app/core.py').read_text(encoding='utf-8'))
        _require(not any(isinstance(n, ast.ImportFrom) and n.level == 1 and
                         n.module == 'runtime_shapes' for n in ast.walk(tree)),
                 'This report runtime requires its captured runtime_shapes.py dependency.')
    for dependency in ('runtime_tasks','family_export','recurrent','recurrent_nodes'):
        if 'runtime/app/'+dependency+'.py' not in manifest['files']:
            import ast
            for relative in manifest['files']:
                if not relative.startswith('runtime/app/') or not relative.endswith('.py'):continue
                tree=ast.parse((root/relative).read_text(encoding='utf-8'))
                _require(not any(isinstance(n,ast.ImportFrom) and n.level==1 and n.module==dependency for n in ast.walk(tree)),
                         'This report runtime requires its captured '+dependency+'.py dependency.')
    requested = manifest['requested_notebook']
    _keys(requested, ('path', 'sha256'), context='Requested notebook identity')
    _require(requested['path'] == 'requested.ipynb' and isinstance(requested['sha256'], str), 'Invalid notebook path/identity.')
    path = root / requested['path']
    _require(not path.is_symlink(), 'Unsafe requested notebook path.')
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == requested['sha256'], 'Requested notebook bytes changed.')
    nbformat = _jupyter('nbformat'); notebook = nbformat.reads(raw.decode('utf-8'), as_version=4)
    nbformat.validate(notebook)
    meta = notebook.metadata.get('visual_pytorch', {})
    _require(meta.get('recipe') == manifest['recipe'] and meta.get('input_index_hash') == digest(manifest['files']) and
             meta.get('state') == 'requested', 'Notebook metadata disagrees with retained input closure.')
    payload = json.loads((root / 'payload.json').read_text(encoding='utf-8'))
    _require(payload['recipe'] == manifest['recipe'], 'Payload recipe identity mismatch.')
    recipe = resolve_payload(payload)
    _require(meta.get('mode') == recipe['mode'], 'Notebook execution mode mismatch.')
    return root, manifest, notebook


class NotebookCancelled(Exception):
    """An explicit Stop request, after actual kernel cleanup, not a cell timeout."""


def execute_bundle(directory, *, python=sys.executable, timeout=600, cancel=None, progress=None,
                   expected=None):
    """Real kernel execution only; preserve both notebook bytes and partial failure."""
    _require(type(timeout) is int and timeout > 0, 'Cell timeout must be a positive integer.')
    root, manifest, notebook = verify_bundle(directory)
    if expected is not None:
        from .notebook_execution import bundle_identity
        _require(bundle_identity(root, manifest, notebook) == expected,
                 'Notebook bundle changed after inspection. Inspect it again; nothing executed.')
    nbformat = _jupyter('nbformat'); nbclient = _jupyter('nbclient')
    kernelspec = _jupyter('jupyter_client.kernelspec'); manager = _jupyter('jupyter_client.manager')
    python = str(Path(python).expanduser().resolve())
    _require(Path(python).is_file() and os.access(python, os.X_OK), 'Kernel interpreter must be an executable file.')
    requested_raw = (root / 'requested.ipynb').read_bytes()
    _require(hashlib.sha256(requested_raw).hexdigest() == manifest['requested_notebook']['sha256'],
             'Requested notebook changed during preparation. Nothing executed.')
    attempt_id = uuid.uuid4().hex; attempt = root / 'executions' / attempt_id
    _require(not (root / 'executions').is_symlink(), 'Execution directory must not be a symlink.')
    attempt.mkdir(parents=True, exist_ok=False)
    new_file(attempt / 'requested.ipynb', requested_raw)
    started = timestamp(); start = time.monotonic(); error = None; status = 'failed'
    interrupt = None
    def emit(stage, cell_index=None, cell=None, **extra):
        if progress:
            progress({'version': 1, 'attempt_id': attempt_id, 'stage': stage,
                'cell_index': cell_index, 'cell_count': len(notebook.cells),
                'section': cell.metadata.get('visual_pytorch', {}).get('section') if cell is not None else None,
                **extra})
    def requested_stop():
        return cancel is not None and cancel.is_set()
    try:
        if requested_stop():
            raise NotebookCancelled('Stop requested before kernel launch; no cells executed.')
        with tempfile.TemporaryDirectory(prefix='vpw-kernel-') as td:
            scratch = Path(td); specdir = scratch / 'kernels' / 'visual-pytorch-report'; specdir.mkdir(parents=True)
            environment = {'VPW_NOTEBOOK_ATTEMPT': attempt_id, 'PYTHONPATH': str(root / 'runtime'),
                'IPYTHONDIR': str(scratch / 'ipython'), 'JUPYTER_RUNTIME_DIR': str(scratch / 'jupyter'),
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            new_json(specdir / 'kernel.json', {'argv': [python, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                'display_name': 'Visual PyTorch explicit report kernel', 'language': 'python', 'env': environment})
            specs = kernelspec.KernelSpecManager(kernel_dirs=[str(scratch / 'kernels')], ensure_native_kernel=False)
            # A synchronous manager gives nbclient blocking channels: IOPub
            # polling can starve its deadline task. Use the asynchronous client.
            km = manager.AsyncKernelManager(kernel_name='visual-pytorch-report', kernel_spec_manager=specs,
                                            connection_file=str(scratch / 'connection.json'))
            client = nbclient.NotebookClient(notebook, km=km, kernel_name='visual-pytorch-report', timeout=timeout,
                resources={'metadata': {'path': str(root)}}, allow_errors=False, force_raise_errors=True,
                raise_on_iopub_timeout=True, store_widget_state=False)
            def cell_start(cell, cell_index):
                if requested_stop():
                    raise NotebookCancelled('Stop requested before the next cell.')
                emit('cell-running', cell_index, cell)
            def cell_done(cell, cell_index, execute_reply):
                emit('cell-replied', cell_index, cell,
                     reply_status=execute_reply.get('content', {}).get('status'))
            client.on_cell_execute = cell_start
            client.on_cell_executed = cell_done
            async def run():
                execution = None; watcher = None; stop_admitted = False
                async def watch():
                    nonlocal stop_admitted
                    while not execution.done():
                        # Do not abandon a pending launch with an unidentified
                        # child. Cancellation is admitted once the manager's
                        # ready future resolves; startup retains its own timeout.
                        if requested_stop() and km.ready.done():
                            stop_admitted = True
                            execution.cancel()
                            return
                        await asyncio.sleep(.05)
                try:
                    emit('kernel-starting')
                    execution = asyncio.create_task(client.async_execute(cleanup_kc=True))
                    watcher = asyncio.create_task(watch())
                    await execution
                except asyncio.CancelledError as exc:
                    raise NotebookCancelled('Stop requested; interrupted cell may be incomplete.') from exc
                except nbclient.exceptions.DeadKernelError as exc:
                    # nbclient translates cancellation of its shell polling
                    # task into DeadKernelError. Only our admitted cancellation
                    # is Stop; an independently dead kernel remains a failure.
                    if not stop_admitted:
                        raise
                    raise NotebookCancelled('Stop requested; interrupted cell may be incomplete.') from exc
                finally:
                    if watcher is not None:
                        watcher.cancel()
                        await asyncio.gather(watcher, return_exceptions=True)
                    # async_setup_kernel can fail before its context-manager
                    # yield; keep an independent shutdown path for that case.
                    if km.has_kernel:
                        await km.shutdown_kernel(now=True)
                    if km.has_kernel:
                        raise GraphError('Notebook kernel shutdown could not be verified.')
            asyncio.run(run())
            status = 'completed'
    except BaseException as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
        if isinstance(exc, NotebookCancelled):
            status = 'cancelled'
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            interrupt = exc; status = 'cancelled'
    notebook.metadata['visual_pytorch']['state'] = status
    notebook.metadata['visual_pytorch']['attempt_id'] = attempt_id
    nbformat.validate(notebook)
    executed = nbformat.writes(notebook).encode('utf-8'); new_file(attempt / 'executed.ipynb', executed)
    artifacts = {}
    for path in sorted(attempt.iterdir()):
        if path.is_file() and not path.is_symlink():
            raw = path.read_bytes(); artifacts[path.name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    receipt = {'version': 1, 'kind': 'notebook-execution', 'attempt_id': attempt_id, 'status': status,
        'recipe': manifest['recipe'], 'requested_sha256': manifest['requested_notebook']['sha256'],
        'input_index_hash': digest(manifest['files']), 'started_at': started, 'finished_at': timestamp(),
        'elapsed_seconds': time.monotonic() - start, 'error': error, 'engine': 'nbclient-real-ipykernel',
        'interpreter': python, 'cell_timeout_seconds': timeout, 'executed_code_cells': sum(
            c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells),
        'artifacts': artifacts}
    new_json(attempt / 'execution.json', receipt)
    if interrupt is not None:
        raise interrupt
    return {'path': str(attempt), 'execution': receipt}
