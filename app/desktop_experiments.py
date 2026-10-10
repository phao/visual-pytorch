"""Qt shell for exact experiment browsing and explicit isolated local fits."""
import json
import threading
import uuid
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal as Signal
from PyQt6.QtWidgets import QFileDialog
from .core import Workspace
from . import experiment_ui


class ExperimentJob(QThread):
    point = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, snapshot, parent, python, memory_limit_percent):
        super().__init__(parent)
        self.snapshot = snapshot; self.python = python
        self.memory_limit_percent = memory_limit_percent
        self.cancel = threading.Event(); self.mid = None
        self.recoverable_result = None

    def stop(self): self.cancel.set()

    def run(self):
        workspace = None
        try:
            from .experiment_objects import execute_candidate
            workspace = Workspace(); workspace.set_constrained_mode(self.snapshot['constrained'])
            for entry in self.snapshot['documents']:
                workspace.put_spec(entry['kind'], entry['document'], select=False)
            result = execute_candidate(workspace, self.snapshot['experiment'],
                self.snapshot['coordinates'], self.snapshot['run_id'], self.snapshot['name'],
                python=self.python, retry=self.snapshot['retry'], cancel=self.cancel,
                memory_limit_percent=self.memory_limit_percent, saved_manifest=self.snapshot['manifest'],
                progress=lambda point:self.point.emit(json.dumps({'run_id': self.snapshot['run_id'], 'point': point}, allow_nan=False)))
            self.recoverable_result = json.dumps(result, allow_nan=False)
            self.completed.emit(self.recoverable_result)
        except Exception as error:
            self.failed.emit(str(error))
        finally:
            if workspace: workspace.db.close()


READS = {'scienceIndex', 'scienceShow', 'sciencePreview', 'scienceReport'}
EDITS = {'sciencePut', 'scienceFreeze', 'scienceSelect', 'scienceResultReport', 'scienceImport'}
ACTIONS = READS | EDITS | {'scienceTrain', 'scienceGenerateNotebook'}


def dispatch(window, req):
    action = req['action']; workspace = window.workspace
    if action not in READS and window.job and window.job.isRunning():
        raise ValueError('Stop the current run before changing scientific objects or starting another action.')
    if action == 'scienceIndex': return experiment_ui.index(workspace)
    if action == 'scienceShow':
        from .experiment_objects import reference
        ref = reference(req['reference'], req['reference']['kind'])
        return workspace.get_spec(ref['kind'], ref['id'], ref['revision'])
    if action == 'sciencePreview':
        from .experiment_objects import preview_experiment, reference
        ref = reference(req['reference'], 'experiment')
        return preview_experiment(workspace.get_spec('experiment', ref['id'], ref['revision']),
            workspace.get_spec, offset=req.get('offset', 0), limit=req.get('limit', 25))
    if action == 'scienceReport': return experiment_ui.read_report(workspace, req['reference'])
    if action == 'scienceFreeze': result = workspace.freeze_model(req['id'])
    elif action == 'sciencePut': result = workspace.put_spec(req['kind'], req['document'])
    elif action == 'scienceSelect':
        from .experiment_objects import reference
        ref = reference(req['reference'], req['reference']['kind'])
        workspace.select_spec(ref['kind'], ref['id'], ref['revision']); result = ref
    elif action == 'scienceResultReport':
        result = experiment_ui.result_report(workspace, req['runs'], sid=req.get('id', uuid.uuid4().hex),
            name=req.get('name', 'Saved run evidence'), question=req['question'])
    elif action == 'scienceImport':
        path, _ = QFileDialog.getOpenFileName(window, 'Import scientific specification', '', 'JSON specification (*.json)')
        if not path: return None
        if window.workspace is not workspace or (window.job and window.job.isRunning()):
            raise ValueError('Workspace changed while selecting a file. Try again.')
        from .revisions import _parse
        result = workspace.put_spec(req['kind'], _parse(Path(path).read_text(encoding='utf-8')))
    elif action == 'scienceGenerateNotebook':
        folder = req.get('output')
        if not folder:
            folder, _ = QFileDialog.getSaveFileName(window, 'New notebook bundle directory', 'report-bundle')
        if not folder: return None
        if window.workspace is not workspace or (window.job and window.job.isRunning()):
            raise ValueError('Workspace changed while selecting a destination. Try again.')
        from .notebooks import create_bundle
        return create_bundle(workspace, req['reference'], folder,
            offset=req.get('offset', 0), limit=req.get('limit', 100), allow_partial=req.get('allow_partial', False),
            memory_limit_percent=window.state()['memory_limit_percent'])
    elif action == 'scienceTrain':
        snapshot = experiment_ui.capture(workspace, req['reference'], req['coordinates'],
            req.get('run_id', uuid.uuid4().hex), req.get('name', 'Selected experiment trial'), retry=req.get('retry', False))
        if window.job: window.job.deleteLater()
        job = ExperimentJob(snapshot, window, window.torch_python or '', window.state()['memory_limit_percent'])
        window.job = job
        job.point.connect(lambda text:window.broadcast('scienceProgress', json.loads(text)))
        job.completed.connect(lambda text: complete(window, workspace, job, text))
        job.failed.connect(lambda error:window.broadcast('trainingError', 'Experiment execution: ' + error))
        job.finished.connect(lambda:window.broadcast())
        job.start(); window.broadcast()
        return {'run_id': snapshot['run_id'], 'experiment': snapshot['experiment']}
    else: raise ValueError('Unknown scientific command.')
    window.broadcast(); window.broadcast('scienceChanged')
    return result


def complete(window, captured_workspace, job, payload):
    result = json.loads(payload)
    try:
        if window.workspace is not captured_workspace:
            raise ValueError('Workspace changed; result remains recoverable on the execution job.')
        ref = captured_workspace.put_spec('run', result['record'])
        if ref != result['reference']: raise ValueError('Run evidence identity changed on admission.')
        window.broadcast('scienceComplete', result)
        window.broadcast(); window.broadcast('scienceChanged')
    except Exception as error:
        window.broadcast('scienceRecovery', {'error': str(error), 'result': result})
