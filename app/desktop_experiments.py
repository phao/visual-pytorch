"""Qt shell for exact experiment browsing and explicit isolated local fits."""
import json
import threading
import uuid
import copy
import os
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtWidgets import QFileDialog
from .core import Workspace
from . import experiment_ui
from .job_state import job_busy


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
        self.awaiting_terminal = True
        self.terminal_delivered = False

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


class Delivery(QObject):
    """Explicit GUI-thread receiver; QThread completion is not SQL admission."""
    def __init__(self, window, workspace, job):
        super().__init__(window)
        self.window, self.workspace, self.job = window, workspace, job

    @Slot(str)
    def point(self, payload):
        if (self.window.job is self.job and self.window.workspace is self.workspace
                and not self.job.terminal_delivered):
            self.window.broadcast('scienceProgress', json.loads(payload))

    @Slot(str)
    def completed(self, payload):
        complete(self.window, self.workspace, self.job, payload)

    @Slot(str)
    def failed(self, error):
        if self.job.terminal_delivered: return
        try:
            retain_recovery(self.window, self.job, None, 'Experiment execution: ' + error)
        finally:
            self.job.terminal_delivered = True; self.job.awaiting_terminal = False
            self.window.broadcast()

    @Slot()
    def finished(self):
        if not self.job.terminal_delivered:
            self.failed('Thread finished without a terminal record.')
        self.window.broadcast()
        self.deleteLater()


def recovery_rows(window):
    return [{'id': sid, 'run_id': value['snapshot']['run_id'], 'error': value['error'],
             'exported_to': value.get('exported_to')}
            for sid, value in getattr(window, 'experiment_recoveries', {}).items()]


def retain_recovery(window, job, result, error):
    """Owned by the window, not a QThread that the next run deletes."""
    if not hasattr(window, 'experiment_recoveries'): window.experiment_recoveries = {}
    sid = uuid.uuid4().hex
    value = {'version': 1, 'format': 'visual-pytorch-experiment-recovery',
             'id': sid, 'snapshot': copy.deepcopy(job.snapshot),
             'result': copy.deepcopy(result), 'error': str(error), 'exported_to': None}
    window.experiment_recoveries[sid] = value
    window.broadcast('scienceRecovery', value)
    return value


READS = {'scienceIndex', 'scienceShow', 'sciencePreview', 'scienceReport',
         'scienceRecoveries', 'scienceRecoveryShow', 'scienceAuthorBegin', 'scienceAuthorCheck',
         'scienceTaskContext', 'scienceTaskCompatibility', 'scienceAuthorPreview',
         'scienceRunEvidence', 'scienceReportEvidence', 'scienceRunPredictor'}
EDITS = {'sciencePut', 'scienceFreeze', 'scienceSelect', 'scienceResultReport', 'scienceImport', 'scienceAuthorSave'}
ACTIONS = READS | EDITS | {'scienceTrain', 'scienceGenerateNotebook', 'scienceExportRecovery', 'scienceExportRunPredictor'}


def dispatch(window, req):
    action = req['action']; workspace = window.workspace
    if action not in READS and job_busy(window.job):
        raise ValueError('Stop the current run before changing scientific objects or starting another action.')
    if action == 'scienceRecoveries': return recovery_rows(window)
    if action in {'scienceRecoveryShow', 'scienceExportRecovery'}:
        value = getattr(window, 'experiment_recoveries', {}).get(req.get('id'))
        if value is None: raise ValueError('Unknown recovery evidence ID.')
        if action == 'scienceRecoveryShow': return copy.deepcopy(value)
        path = req.get('output')
        if not path:
            path, _ = QFileDialog.getSaveFileName(window, 'Export experiment recovery evidence',
                'experiment-recovery.json', 'JSON evidence (*.json)')
        if not path: return None
        if job_busy(window.job): raise ValueError('Stop the current run before exporting recovery evidence.')
        # Explicit local export, no overwrite; never a completed/running receipt.
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        value['exported_to'] = str(Path(path).resolve())
        window.broadcast()
        return {'path': value['exported_to'], 'status': 'recovery-evidence-exported'}
    if action == 'scienceIndex': return experiment_ui.index(workspace)
    if action in {'scienceRunPredictor', 'scienceExportRunPredictor'}:
        from .authoring import workspace_id
        from .run_export import predictor_source, write_new_predictor
        if req.get('workspace_id') != workspace_id(workspace):
            raise ValueError('Predictor selection belongs to another workspace. Select the saved run again.')
        # Capture exact validated source before Qt's nested modal event loop.
        source = predictor_source(workspace, req['reference'], attempt=req.get('attempt'),
                                  memory_limit_percent=window.state()['memory_limit_percent'])
        if action == 'scienceRunPredictor': return source
        path = req.get('output')
        if not path:
            path, _ = QFileDialog.getSaveFileName(window, 'Export retained checkpoint predictor',
                                                 'trained-predictor.py', 'Python source (*.py)')
        if not path: return None
        if window.workspace is not workspace or req['workspace_id'] != workspace_id(window.workspace) or job_busy(window.job):
            raise ValueError('Workspace or active work changed while selecting a destination. Nothing exported; try again.')
        return write_new_predictor(source, path)
    if action in {'scienceRunEvidence', 'scienceReportEvidence'}:
        from . import run_views
        if action == 'scienceRunEvidence':
            return run_views.run_view(workspace, req['reference'], attempt=req.get('attempt'),
                offset=req.get('offset', 0), limit=req.get('limit', 25))
        return run_views.report_view(workspace, req['reference'],
            offset=req.get('offset', 0), limit=req.get('limit', 5))
    if action == 'scienceAuthorPreview':
        from . import authoring
        return authoring.experiment_preview(workspace, req['draft'], offset=req.get('offset', 0), limit=req.get('limit', 25))
    if action in {'scienceTaskContext', 'scienceTaskCompatibility'}:
        from . import authoring
        if action == 'scienceTaskContext': return authoring.task_context(workspace, req['model'], req['dataset'])
        return authoring.task_compatibility(workspace, req['draft'], req['model'], req['dataset'])
    if action in {'scienceAuthorBegin', 'scienceAuthorCheck', 'scienceAuthorSave'}:
        from . import authoring
        if action == 'scienceAuthorBegin':
            return authoring.begin(workspace, req['kind'], base=req.get('reference'), sid=req.get('id'))
        if action == 'scienceAuthorCheck': return authoring.check(workspace, req['draft'])
        result = authoring.save(workspace, req['draft'])
        window.broadcast(); window.broadcast('scienceChanged')
        return result
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
        if 'workspace_id' in req:
            from .authoring import workspace_id
            if req['workspace_id'] != workspace_id(workspace):
                raise ValueError('Report selection belongs to another workspace. Select runs again.')
        result = experiment_ui.result_report(workspace, req['runs'], sid=req.get('id', uuid.uuid4().hex),
            name=req.get('name', 'Saved run evidence'), question=req['question'])
    elif action == 'scienceImport':
        path, _ = QFileDialog.getOpenFileName(window, 'Import scientific specification', '', 'JSON specification (*.json)')
        if not path: return None
        if window.workspace is not workspace or job_busy(window.job):
            raise ValueError('Workspace changed while selecting a file. Try again.')
        from .revisions import _parse
        result = workspace.put_spec(req['kind'], _parse(Path(path).read_text(encoding='utf-8')))
    elif action == 'scienceGenerateNotebook':
        folder = req.get('output')
        if not folder:
            folder, _ = QFileDialog.getSaveFileName(window, 'New notebook bundle directory', 'report-bundle')
        if not folder: return None
        if window.workspace is not workspace or job_busy(window.job):
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
        job.delivery = Delivery(window, workspace, job)
        job.point.connect(job.delivery.point, Qt.ConnectionType.QueuedConnection)
        job.completed.connect(job.delivery.completed, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(job.delivery.failed, Qt.ConnectionType.QueuedConnection)
        job.finished.connect(job.delivery.finished, Qt.ConnectionType.QueuedConnection)
        job.start(); window.broadcast()
        return {'run_id': snapshot['run_id'], 'experiment': snapshot['experiment']}
    else: raise ValueError('Unknown scientific command.')
    window.broadcast(); window.broadcast('scienceChanged')
    return result


def complete(window, captured_workspace, job, payload):
    if job.terminal_delivered: return
    result = None
    try:
        result = json.loads(payload)
        if window.workspace is not captured_workspace:
            raise ValueError('Workspace changed; result remains recoverable on the execution job.')
        if window.job is not job:
            raise ValueError('Execution job changed; late result requires explicit recovery.')
        from .revisions import frozen_document
        doc = frozen_document('run', result['record'])
        if result['reference'] != {k: v for k, v in
                (('kind', 'run'), ('id', doc['id']), ('revision', doc['revision']))}:
            raise ValueError('Run evidence identity changed before admission.')
        ref = captured_workspace.put_spec('run', result['record'])
        if ref != result['reference']: raise ValueError('Run evidence identity changed on admission.')
        window.broadcast('scienceComplete', result)
        window.broadcast(); window.broadcast('scienceChanged')
    except Exception as error:
        retain_recovery(window, job, result, error)
    finally:
        job.terminal_delivered = True; job.awaiting_terminal = False
        window.broadcast()
