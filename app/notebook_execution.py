"""Pure bundle identity and isolated runner. No eager Jupyter/Torch/Qt imports."""
import hashlib
import json
import os
import re
import math
import tempfile
from pathlib import Path

from .core import GraphError
from .evidence import digest
from .experiment_objects import reference


def _require(ok, message):
    if not ok:
        raise GraphError(message)


def bundle_identity(root, manifest, notebook):
    return {'version': 1, 'kind': 'visual-pytorch-notebook-selection', 'path': str(root),
            'recipe': manifest['recipe'], 'mode': notebook.metadata.visual_pytorch.mode,
            'input_index_hash': digest(manifest['files']),
            'requested_sha256': manifest['requested_notebook']['sha256']}


def inspect_bundle(directory):
    from .notebooks import verify_bundle
    return bundle_identity(*verify_bundle(directory))


def inspect_isolated(directory, python, *, cancel=None):
    """Read/validate in the selected environment; no kernel launch or host Jupyter."""
    interpreter = Path(python).expanduser().resolve()
    _require(interpreter.is_file() and os.access(interpreter, os.X_OK), 'Choose an executable report/kernel Python.')
    from .transport import run_json_worker
    result = run_json_worker([str(interpreter), '-m', 'app.notebook_worker'],
        {'operation': 'inspect', 'directory': str(Path(directory).expanduser().resolve())},
        cwd=Path(__file__).resolve().parent.parent, timeout=30, max_line_bytes=1024**2, cancel=cancel)
    selection = validate_selection(result['selection'])
    _require(selection['path'] == str(Path(directory).expanduser().resolve()), 'Inspected notebook path changed.')
    return {'selection': selection, 'python': str(interpreter)}


def validate_selection(selection):
    keys = {'version', 'kind', 'path', 'recipe', 'mode', 'input_index_hash', 'requested_sha256'}
    _require(isinstance(selection, dict) and set(selection) == keys and type(selection['version']) is int
             and selection['version'] == 1 and selection['kind'] == 'visual-pytorch-notebook-selection',
             'Invalid captured notebook selection. Inspect a bundle first.')
    reference(selection['recipe'], 'report')
    _require(selection['mode'] in ('result-report', 'rerun') and isinstance(selection['path'], str)
             and Path(selection['path']).is_absolute(), 'Invalid notebook mode/path.')
    for key in ('input_index_hash', 'requested_sha256'):
        _require(isinstance(selection[key], str) and re.fullmatch('[0-9a-f]{64}', selection[key]),
                 'Notebook selection requires exact content hashes.')
    return selection


def validate_result(result, selection):
    """Check retained receipt and artifact bytes; a process result alone is not evidence."""
    validate_selection(selection)
    _require(isinstance(result, dict) and set(result) == {'path', 'execution'}, 'Invalid notebook terminal result.')
    receipt = result['execution']
    keys = {'version', 'kind', 'attempt_id', 'status', 'recipe', 'requested_sha256', 'input_index_hash',
            'started_at', 'finished_at', 'elapsed_seconds', 'error', 'engine', 'interpreter',
            'cell_timeout_seconds', 'executed_code_cells', 'artifacts'}
    _require(isinstance(receipt, dict) and set(receipt) == keys and type(receipt['version']) is int
             and receipt['version'] == 1 and receipt['kind'] == 'notebook-execution'
             and receipt['engine'] == 'nbclient-real-ipykernel', 'Invalid notebook execution receipt.')
    _require(isinstance(receipt['attempt_id'], str) and re.fullmatch('[0-9a-f]{32}', receipt['attempt_id']),
             'Invalid notebook attempt ID.')
    for key in ('recipe', 'requested_sha256', 'input_index_hash'):
        _require(receipt[key] == selection[key], 'Notebook terminal identity changed: ' + key)
    _require(receipt['status'] in ('completed', 'failed', 'cancelled') and
             ((receipt['status'] == 'completed') == (receipt['error'] is None)), 'Inconsistent notebook status/error.')
    for key in ('executed_code_cells', 'cell_timeout_seconds'):
        _require(type(receipt[key]) is int and receipt[key] >= (1 if key == 'cell_timeout_seconds' else 0),
                 'Invalid notebook count/timeout.')
    _require(type(receipt['elapsed_seconds']) in (int, float) and math.isfinite(receipt['elapsed_seconds'])
             and receipt['elapsed_seconds'] >= 0 and isinstance(receipt['interpreter'], str)
             and Path(receipt['interpreter']).is_absolute(), 'Invalid notebook elapsed time/interpreter.')
    _require(all(isinstance(receipt[key], str) and receipt[key] for key in ('started_at', 'finished_at')),
             'Invalid notebook timestamps.')
    if receipt['error'] is not None:
        _require(isinstance(receipt['error'], dict) and set(receipt['error']) == {'type', 'message'}
                 and all(isinstance(v, str) for v in receipt['error'].values()), 'Invalid notebook diagnostic.')
    path = Path(selection['path']) / 'executions' / receipt['attempt_id']
    _require(result['path'] == str(path) and not path.is_symlink() and not path.parent.is_symlink()
             and not Path(selection['path']).is_symlink(),
             'Notebook attempt path disagrees with captured bundle.')
    _require(not (path / 'execution.json').is_symlink(), 'Unsafe notebook receipt path.')
    from .revisions import _parse
    _require(_parse((path / 'execution.json').read_text(encoding='utf-8')) == receipt,
             'Retained notebook receipt changed.')
    artifacts = receipt['artifacts']
    _require(isinstance(artifacts, dict) and {'requested.ipynb', 'executed.ipynb'} <= set(artifacts)
             and 'execution.json' not in artifacts, 'Notebook artifact index is incomplete.')
    for name, info in artifacts.items():
        _require(isinstance(name, str) and name not in ('.', '..') and '/' not in name and '\\' not in name
                 and isinstance(info, dict) and set(info) == {'bytes', 'sha256'}, 'Invalid notebook artifact identity.')
        file = path / name
        _require(file.is_file() and not file.is_symlink(), 'Missing/unsafe notebook artifact: ' + name)
        raw = file.read_bytes()
        _require(type(info['bytes']) is int and len(raw) == info['bytes']
                 and hashlib.sha256(raw).hexdigest() == info['sha256'], 'Notebook artifact bytes changed: ' + name)
    _require(artifacts['requested.ipynb']['sha256'] == selection['requested_sha256'],
             'Retained requested notebook changed.')
    notebook = _parse((path / 'executed.ipynb').read_text(encoding='utf-8'))
    meta = notebook.get('metadata', {}).get('visual_pytorch', {})
    _require(meta.get('state') == receipt['status'] and meta.get('attempt_id') == receipt['attempt_id']
             and meta.get('recipe') == receipt['recipe'] and meta.get('input_index_hash') == receipt['input_index_hash'],
             'Executed notebook metadata disagrees with receipt.')
    _require(sum(c.get('cell_type') == 'code' and c.get('execution_count') is not None
                 for c in notebook.get('cells', [])) == receipt['executed_code_cells'],
             'Executed notebook cell counts disagree with receipt.')
    return result


def run_isolated(selection, python, *, timeout=600, cancel=None, progress=None):
    validate_selection(selection)
    _require(type(timeout) is int and timeout > 0, 'Cell timeout must be a positive integer.')
    interpreter = Path(python).expanduser().resolve()
    _require(interpreter.is_file() and os.access(interpreter, os.X_OK), 'Choose an executable report/kernel Python.')
    from .transport import run_json_worker
    with tempfile.TemporaryDirectory(prefix='vpw-notebook-control-') as folder:
        stop = Path(folder) / 'stop'
        def request_stop(process):
            with stop.open('x', encoding='utf-8') as stream:
                stream.write('stop\n')
        result = run_json_worker([str(interpreter), '-m', 'app.notebook_worker'],
            {'operation': 'execute', 'selection': selection, 'timeout': timeout, 'stop': str(stop)}, cwd=Path(__file__).resolve().parent.parent,
            cancel=cancel, on_progress=progress, on_cancel=request_stop, grace_seconds=30,
            max_line_bytes=16*1024**2)
    result = validate_result(result, selection)
    _require(result['execution']['interpreter'] == str(interpreter)
             and result['execution']['cell_timeout_seconds'] == timeout, 'Notebook execution policy changed.')
    return result
