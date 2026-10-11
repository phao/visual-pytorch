"""Explicit external notebook execution; terminal artifacts are not SQLite runs."""
import copy
import json
import os
import threading
import uuid
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtWidgets import QFileDialog, QMessageBox
from .authoring import workspace_id
from .identity import APP_NAME
from .job_state import job_busy
from .notebook_execution import inspect_isolated, run_isolated, validate_selection

READS = {'scienceNotebookHistory', 'scienceNotebookEvidence'}
ACTIONS = READS | {'scienceNotebookInspect', 'scienceExecuteNotebook'}


class NotebookJob(QThread):
    point = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, snapshot, parent):
        super().__init__(parent)
        self.snapshot = copy.deepcopy(snapshot); self.mid = None
        self.cancel = threading.Event(); self.awaiting_terminal = True; self.terminal_delivered = False

    def stop(self):
        self.cancel.set()

    def run(self):
        try:
            if self.snapshot.get('operation') == 'inspect':
                result = inspect_isolated(self.snapshot['directory'], self.snapshot['python'], cancel=self.cancel)
            else:
                result = run_isolated(self.snapshot['selection'], self.snapshot['python'],
                    timeout=self.snapshot['timeout'], cancel=self.cancel,
                    progress=lambda p: self.point.emit(json.dumps(p, allow_nan=False)))
            self.completed.emit(json.dumps(result, allow_nan=False))
        except Exception as error:
            self.failed.emit(str(error))


def history(window):
    return [copy.deepcopy(row) for row in getattr(window, 'notebook_executions', {}).values()]


def confirm_execution(window, selection, python, timeout):
    behavior = ('This is a RERUN notebook. Its resolution cell explicitly launches fresh CPU training fits. '
                'Captured excluded/incompatible rows remain visible.' if selection['mode'] == 'rerun' else
                'This is a SAVED-RESULT notebook. It reads retained evidence; no training fits are launched.')
    dialog = QMessageBox(window)
    dialog.setWindowTitle(APP_NAME + ' — Execute notebook')
    dialog.setIcon(QMessageBox.Icon.Question)
    dialog.setTextFormat(Qt.TextFormat.PlainText)
    dialog.setText(behavior)
    dialog.setInformativeText('Run only bundles you trust: Python cells are executable code, not a sandbox. '
        'Input hashes detect changes, not authenticity. A fresh attempt directory retains requested/executed '
        'notebooks and a receipt. Generated runs stay in the bundle, not this workspace. '
        'Stop retains partial evidence when cleanup succeeds. Review exact paths and hashes under Show Details.')
    dialog.setDetailedText('Bundle: ' + selection['path'] + '\nRecipe: ' + selection['recipe']['id'] + ' · ' + selection['recipe']['revision'] +
        '\nRequested notebook: ' + selection['requested_sha256'] + '\nInput index: ' + selection['input_index_hash'] +
        '\nKernel Python: ' + python + '\nTimeout per cell: ' + str(timeout) + ' s')
    dialog.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
    dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return dialog.exec() == QMessageBox.StandardButton.Yes.value


class Delivery(QObject):
    def __init__(self, window, workspace, job):
        super().__init__(window)
        self.window, self.workspace, self.job = window, workspace, job

    @Slot(str)
    def point(self, payload):
        if self.window.job is self.job and self.window.workspace is self.workspace and not self.job.terminal_delivered:
            self.window.broadcast('scienceNotebookProgress', {'job_id': self.job.snapshot['id'],
                'workspace_id': self.job.snapshot['workspace_id'], 'progress': json.loads(payload)})

    def terminal(self, result=None, error=None):
        if self.job.terminal_delivered:
            return
        try:
            # No SQL admission or current-head substitution. External receipts
            # survive workspace/job replacement and remain explicitly viewable.
            snapshot = self.job.snapshot
            if snapshot.get('operation') == 'inspect':
                origin = self.window.workspace is self.workspace and self.window.job is self.job
                self.window.broadcast('scienceNotebookInspected', {'job_id': snapshot['id'],
                    'workspace_id': snapshot['workspace_id'], 'request_id': snapshot['request_id'],
                    'inspection': result if origin else None,
                    'error': error if origin else 'Origin changed; inspect the bundle again.',
                    'status': 'inspected-not-executed' if result and origin else
                        'inspection-stopped-no-execution' if self.job.cancel.is_set() else 'inspection-failed-no-execution'})
                return
            row = {'id': snapshot['id'], 'workspace_id': snapshot['workspace_id'],
                   'selection': copy.deepcopy(snapshot['selection']), 'python': snapshot['python'],
                   'cell_timeout_seconds': snapshot['timeout'], 'result': result, 'error': error,
                   'status': result['execution']['status'] if result else 'runner-error-no-verified-receipt'}
            if not hasattr(self.window, 'notebook_executions'):
                self.window.notebook_executions = {}
            self.window.notebook_executions[row['id']] = row
            self.window.broadcast('scienceNotebookComplete', copy.deepcopy(row))
        finally:
            self.job.terminal_delivered = True; self.job.awaiting_terminal = False
            self.window.broadcast()

    @Slot(str)
    def completed(self, payload):
        if self.job.terminal_delivered:
            return
        try:
            result = json.loads(payload)
            if self.job.snapshot.get('operation') == 'inspect':
                selection = validate_selection(result['selection'])
                if result['python'] != self.job.snapshot['python'] or selection['path'] != self.job.snapshot['directory']:
                    raise ValueError('Notebook inspection origin changed.')
            else:
                from .notebook_execution import validate_result
                result = validate_result(result, self.job.snapshot['selection'])
                receipt = result['execution']
                if (receipt['interpreter'] != self.job.snapshot['python'] or
                        receipt['cell_timeout_seconds'] != self.job.snapshot['timeout']):
                    raise ValueError('Notebook terminal execution policy changed.')
        except Exception as error:
            self.terminal(error=str(error))
        else:
            self.terminal(result=result)

    @Slot(str)
    def failed(self, error):
        self.terminal(error=error)

    @Slot()
    def finished(self):
        if not self.job.terminal_delivered:
            self.failed('Runner thread ended without verified terminal evidence. Inspect the bundle directory.')
        self.window.broadcast(); self.deleteLater()


def dispatch(window, req):
    action = req['action']; workspace = window.workspace
    if action == 'scienceNotebookHistory':
        return history(window)
    if action == 'scienceNotebookEvidence':
        row = getattr(window, 'notebook_executions', {}).get(req.get('id'))
        if row is None:
            raise ValueError('Unknown notebook execution evidence ID.')
        if row['result'] is not None:
            from .notebook_execution import validate_result
            validate_result(row['result'], row['selection'])
        return copy.deepcopy(row)
    if req.get('workspace_id') != workspace_id(workspace):
        raise ValueError('Notebook action belongs to another workspace. Inspect the bundle again.')
    if job_busy(window.job):
        raise ValueError('Stop the current run before starting notebook work.')
    python = req.get('python') or window.torch_python
    if not python:
        raise ValueError('Choose a report/kernel Python with requirements-report.txt and PyTorch installed.')
    python = str(Path(python).expanduser().resolve())
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        raise ValueError('Choose an executable report/kernel Python.')
    def origin_guard():
        if window.workspace is not workspace or req['workspace_id'] != workspace_id(window.workspace) or job_busy(window.job):
            raise ValueError('Workspace or active work changed during the dialog. Nothing started; try again.')
    if action == 'scienceNotebookInspect':
        directory = req.get('directory')
        if not directory:
            directory = QFileDialog.getExistingDirectory(window, 'Inspect notebook bundle directory')
        if not directory:
            return None
        origin_guard()
        snapshot = {'id': uuid.uuid4().hex, 'workspace_id': req['workspace_id'], 'selection': None,
                    'python': python, 'timeout': None, 'operation': 'inspect',
                    'directory': str(Path(directory).expanduser().resolve()), 'request_id': str(req.get('request_id', ''))}
        return start(window, workspace, snapshot)
    elif action != 'scienceExecuteNotebook':
        raise ValueError('Unknown notebook action.')
    selection = copy.deepcopy(validate_selection(req['selection']))
    timeout = req.get('timeout', 600)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError('Cell timeout must be a positive integer.')
    if not confirm_execution(window, selection, python, timeout):
        return None
    origin_guard()
    snapshot = {'id': uuid.uuid4().hex, 'workspace_id': req['workspace_id'], 'selection': selection,
                'python': python, 'timeout': timeout, 'operation': 'execute'}
    return start(window, workspace, snapshot)


def start(window, workspace, snapshot):
    if window.job:
        window.job.deleteLater()
    job = NotebookJob(snapshot, window); window.job = job
    job.delivery = Delivery(window, workspace, job)
    job.point.connect(job.delivery.point, Qt.ConnectionType.QueuedConnection)
    job.completed.connect(job.delivery.completed, Qt.ConnectionType.QueuedConnection)
    job.failed.connect(job.delivery.failed, Qt.ConnectionType.QueuedConnection)
    job.finished.connect(job.delivery.finished, Qt.ConnectionType.QueuedConnection)
    job.start(); window.broadcast()
    return {'job_id': snapshot['id'], 'operation': snapshot['operation'], 'selection': snapshot['selection'], 'status': 'started-not-completed'}
