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
from .desktop_notebooks import READS as NOTEBOOK_READS, ACTIONS as NOTEBOOK_ACTIONS


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
    from .recovery_views import changed, notification
    changed(window)
    window.broadcast('scienceRecovery', notification(window, sid))
    return value


READS = {'scienceIndex', 'scienceShow', 'sciencePreview', 'sciencePreviewPage', 'scienceReport',
         'scienceDraftExperimentPage', 'scienceDraftExperimentValue', 'scienceDraftExperimentChoices', 'scienceDraftExperimentPreview',
         'scienceDraftTaskPage', 'scienceDraftTaskTransforms', 'scienceDraftTaskCompatibility',
         'scienceDraftDatasetPage','scienceDraftDatasetValue',
         'scienceDraftValueRead', 'scienceDraftValueDiscard',
         'scienceDraftDescribe', 'scienceDraftOrigin', 'scienceDraftRead', 'scienceDraftCheck', 'scienceDraftClose',
         'scienceBrowse', 'scienceSummary', 'scienceInspect',
         'scienceRecoveryPage', 'scienceRecoveryInspect',
         'scienceRecoveries', 'scienceRecoveryShow', 'scienceAuthorBegin', 'scienceAuthorCheck',
         'scienceTaskContext', 'scienceTaskContextPage', 'scienceTaskCompatibility', 'scienceAuthorPreview',
         'scienceRunEvidence', 'scienceReportEvidence', 'scienceRunPredictor'} | NOTEBOOK_READS
EDITS = {'sciencePut', 'scienceFreeze', 'scienceSelect', 'scienceResultReport', 'scienceImport', 'scienceAuthorSave', 'scienceDraftCommit'}
DRAFT_ACTIONS = {'scienceDraftOpen', 'scienceDraftDescribe', 'scienceDraftOrigin', 'scienceDraftRead', 'scienceDraftPatch',
                 'scienceDraftExperimentPage', 'scienceDraftExperimentValue', 'scienceDraftExperimentChoices', 'scienceDraftExperimentPreview', 'scienceDraftExperimentEdit',
                 'scienceDraftTaskPage', 'scienceDraftTaskTransforms', 'scienceDraftTaskCompatibility', 'scienceDraftTaskEdit',
                 'scienceDraftDatasetPage','scienceDraftDatasetEdit','scienceDraftDatasetValue',
                 'scienceDraftCheck', 'scienceDraftCommit', 'scienceDraftClose',
                 'scienceDraftValueStart', 'scienceDraftValueRead', 'scienceDraftValueSplice',
                 'scienceDraftValueFinish', 'scienceDraftValueDiscard'}
ACTIONS = READS | EDITS | DRAFT_ACTIONS | {'scienceTrain', 'scienceGenerateNotebook', 'scienceExportRecovery', 'scienceExportRunPredictor'} | NOTEBOOK_ACTIONS


def dispatch(window, req, source=None):
    action = req['action']; workspace = window.workspace
    if action not in READS and job_busy(window.job):
        raise ValueError('Stop the current run before changing scientific objects or starting another action.')
    if action in DRAFT_ACTIONS:
        from .authoring_sessions import Sessions, _json
        from .datasets import _keys
        req = _json(req)  # Limit the entire compact request, not only patch values.
        required, optional = {
            'scienceDraftOpen': (('action', 'kind'), ('reference', 'id')),
            'scienceDraftDescribe': (('action', 'session'), ()),
            'scienceDraftOrigin': (('action', 'session', 'sequence'), ()),
            'scienceDraftRead': (('action', 'session', 'sequence'), ('path', 'part', 'offset', 'limit')),
            'scienceDraftPatch': (('action', 'session', 'sequence', 'operations'), ()),
            'scienceDraftCheck': (('action', 'session', 'sequence'), ()),
            'scienceDraftCommit': (('action', 'session', 'sequence'), ()),
            'scienceDraftClose': (('action', 'session'), ()),
            'scienceDraftValueStart': (('action', 'session', 'sequence', 'path'), ('from_base',)),
            'scienceDraftValueRead': (('action', 'session', 'sequence', 'value', 'value_revision'), ('offset',)),
            'scienceDraftValueSplice': (('action', 'session', 'sequence', 'value', 'value_revision', 'offset', 'delete_bytes', 'text'), ()),
            'scienceDraftValueFinish': (('action', 'session', 'sequence', 'value', 'value_revision'), ()),
            'scienceDraftValueDiscard': (('action', 'session', 'sequence', 'value', 'value_revision'), ()),
            'scienceDraftDatasetPage': (('action','session','sequence'),('field_offset','row_offset','field_limit','row_limit')),
            'scienceDraftDatasetEdit': (('action','session','sequence','operations'),()),
            'scienceDraftDatasetValue': (('action','session','sequence','target'),()),
            'scienceDraftTaskPage': (('action','session','sequence','collection'),('offset','limit')),
            'scienceDraftTaskTransforms': (('action','session','sequence','collection','binding'),('offset','limit')),
            'scienceDraftTaskCompatibility': (('action','session','sequence','model','dataset'),()),
            'scienceDraftTaskEdit': (('action','session','sequence','operations'),()),
            'scienceDraftExperimentPage': (('action','session','sequence','collection'),('offset','limit')),
            'scienceDraftExperimentValue': (('action','session','sequence','target','key'),()),
            'scienceDraftExperimentChoices': (('action','session','sequence','collection'),('offset','limit','query','pin')),
            'scienceDraftExperimentPreview': (('action','session','sequence'),('offset','limit')),
            'scienceDraftExperimentEdit': (('action','session','sequence','operations'),()),
        }[action]
        _keys(req, required, optional, context=action+' request (including exact sequence where applicable)')
        owner = source if source is not None else window
        if not hasattr(window, 'authoring_sessions'):
            window.authoring_sessions = Sessions()
        sessions = window.authoring_sessions
        configuration = getattr(window, 'config', None)
        sessions.memory_limit_percent = configuration.values.get('memory_limit_percent', 10.) if configuration else 10.
        if action == 'scienceDraftOpen':
            result = sessions.open(workspace, owner, req['kind'], base=req.get('reference'), sid=req.get('id'))
            if source is not None and hasattr(source, 'destroyed') and not getattr(source, '_draft_cleanup_connected', False):
                source.destroyed.connect(lambda: sessions.close_owner(owner))
                source._draft_cleanup_connected = True
            return result
        token = req['session']
        if action == 'scienceDraftExperimentPage':
            return sessions.experiment_page(token, owner, req['sequence'], req['collection'],
                                            **{k:req[k] for k in ('offset','limit') if k in req})
        if action == 'scienceDraftExperimentValue':
            return sessions.experiment_value(token, owner, req['sequence'], req['target'], req['key'])
        if action == 'scienceDraftExperimentChoices':
            return sessions.experiment_choices(token, owner, req['sequence'], req['collection'],
                                               **{k:req[k] for k in ('offset','limit','query','pin') if k in req})
        if action == 'scienceDraftExperimentPreview':
            return sessions.experiment_preview(workspace, token, owner, req['sequence'],
                                               **{k:req[k] for k in ('offset','limit') if k in req})
        if action == 'scienceDraftExperimentEdit':
            return sessions.experiment_edit(workspace, token, owner, req['sequence'], req['operations'])
        if action == 'scienceDraftOrigin': return sessions.origin(workspace, token, owner, req['sequence'])
        if action == 'scienceDraftTaskPage': return sessions.task_page(token, owner, req['sequence'], req['collection'],
            **{key:req[key] for key in ('offset','limit') if key in req})
        if action == 'scienceDraftTaskTransforms': return sessions.task_transforms(token, owner, req['sequence'],
            req['collection'], req['binding'], **{key:req[key] for key in ('offset','limit') if key in req})
        if action == 'scienceDraftTaskCompatibility': return sessions.task_compatibility(workspace, token, owner,
            req['sequence'], req['model'], req['dataset'])
        if action == 'scienceDraftTaskEdit': return sessions.task_edit(workspace, token, owner,
            req['sequence'], req['operations'])
        if action == 'scienceDraftDatasetPage':return sessions.dataset_page(token,owner,req['sequence'],
            **{key:req[key] for key in ('field_offset','row_offset','field_limit','row_limit') if key in req})
        if action == 'scienceDraftDatasetEdit':return sessions.dataset_edit(workspace,token,owner,req['sequence'],req['operations'])
        if action == 'scienceDraftDatasetValue':return sessions.dataset_value(token,owner,req['sequence'],req['target'])
        if action == 'scienceDraftValueStart': return sessions.start_value(workspace, token, owner, req['sequence'], req['path'], from_base=req.get('from_base',False))
        if action == 'scienceDraftValueRead': return sessions.read_value(token, owner, req['sequence'], req['value'], req['value_revision'], offset=req.get('offset',0))
        if action == 'scienceDraftValueSplice': return sessions.splice_value(workspace, token, owner, req['sequence'], req['value'], req['value_revision'], offset=req['offset'], delete_bytes=req['delete_bytes'], text=req['text'])
        if action == 'scienceDraftValueFinish': return sessions.finish_value(workspace, token, owner, req['sequence'], req['value'], req['value_revision'])
        if action == 'scienceDraftValueDiscard': return sessions.discard_value(token, owner, req['sequence'], req['value'], req['value_revision'])
        if action == 'scienceDraftDescribe': return sessions.describe(token, owner)
        if action == 'scienceDraftRead':
            return sessions.read(token, owner, req.get('sequence'), path=req.get('path'),
                part=req.get('part', 'value'), offset=req.get('offset', 0), limit=req.get('limit', 25))
        if action == 'scienceDraftClose': return sessions.close(token, owner)
        if action == 'scienceDraftPatch': return sessions.patch(workspace, token, owner, req.get('sequence'), req['operations'])
        if action == 'scienceDraftCheck': return sessions.check(workspace, token, owner, req.get('sequence'))
        result = sessions.commit(workspace, token, owner, req.get('sequence'))
        window.broadcast(); window.broadcast('scienceChanged')
        return result
    if action in NOTEBOOK_ACTIONS:
        from .desktop_notebooks import dispatch as notebook_dispatch
        return notebook_dispatch(window, req)
    if action in {'scienceRecoveryPage', 'scienceRecoveryInspect'}:
        from . import recovery_views
        from .authoring_sessions import _json
        from .datasets import _keys
        req = _json(req)
        required = ('action', 'scope_id', 'sequence')
        if action == 'scienceRecoveryPage':
            _keys(req, required, ('offset', 'limit', 'query', 'pin'), context='Recovery page')
            return recovery_views.page(window, req['scope_id'], req['sequence'],
                                       **{k:req[k] for k in ('offset', 'limit', 'query', 'pin') if k in req})
        _keys(req, (*required, 'id'), ('path', 'part', 'offset', 'limit'), context='Recovery inspection')
        return recovery_views.inspect(window, req['scope_id'], req['sequence'], req['id'],
                                      **{k:req[k] for k in ('path', 'part', 'offset', 'limit') if k in req})
    if action == 'scienceRecoveries': return recovery_rows(window)
    if action in {'scienceRecoveryShow', 'scienceExportRecovery'}:
        value = getattr(window, 'experiment_recoveries', {}).get(req.get('id'))
        if value is None: raise ValueError('Unknown recovery evidence ID.')
        if action == 'scienceRecoveryShow': return copy.deepcopy(value)
        guarded = 'scope_id' in req or 'sequence' in req
        if guarded:
            from .recovery_views import _identity
            _identity(window, req.get('scope_id'), req.get('sequence'))
        path = req.get('output')
        if not path:
            path, _ = QFileDialog.getSaveFileName(window, 'Export experiment recovery evidence',
                'experiment-recovery.json', 'JSON evidence (*.json)')
        if not path: return None
        if job_busy(window.job): raise ValueError('Stop the current run before exporting recovery evidence.')
        if guarded:
            _identity(window, req.get('scope_id'), req.get('sequence'))
        # Explicit local export, no overwrite; never a completed/running receipt.
        payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        with Path(path).open('x', encoding='utf-8') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        value['exported_to'] = str(Path(path).resolve())
        from .recovery_views import changed
        changed(window)
        window.broadcast()
        return {'path': value['exported_to'], 'status': 'recovery-evidence-exported'}
    if action == 'scienceIndex': return experiment_ui.index(workspace)
    if action in {'scienceBrowse', 'scienceSummary', 'scienceInspect'}:
        from . import science_browser
        from .authoring import workspace_id
        if req.get('workspace_id') != workspace_id(workspace):
            raise ValueError('Scientific selection belongs to another workspace. Refresh its choices.')
        if action == 'scienceBrowse':
            return science_browser.index_page(workspace, req['kind'], offset=req.get('offset', 0),
                limit=req.get('limit', 25), query=req.get('query', ''), preferred=req.get('preferred'),
                heads_only=req.get('heads_only', False), allow_missing_pin=req.get('allow_missing_pin', False))
        if action == 'scienceSummary': return science_browser.summary(workspace, req['reference'])
        return science_browser.inspect_page(workspace, req['reference'], path=req.get('path'),
            part=req.get('part', 'value'), offset=req.get('offset', 0), limit=req.get('limit', 25))
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
    if action == 'scienceTaskContextPage':
        from . import task_context
        from .authoring import workspace_id
        from .authoring_sessions import _json
        from .datasets import _keys
        req=_json(req)
        _keys(req,('action','workspace_id','model','dataset','collection'),('query','offset','limit','pin'),context='Task context page')
        if req['workspace_id']!=workspace_id(workspace):raise ValueError('Task suggestions belong to another workspace. Refresh context choices.')
        return task_context.page(workspace,req['model'],req['dataset'],req['collection'],
            **{key:req[key] for key in ('query','offset','limit','pin') if key in req})
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
    if action == 'sciencePreviewPage':
        from . import candidate_views
        from .authoring import workspace_id
        from .authoring_sessions import _json
        from .datasets import _keys
        req = _json(req)
        _keys(req, ('action','workspace_id','reference'), ('offset','limit'), context='Stored candidate preview')
        if req['workspace_id'] != workspace_id(workspace):
            raise ValueError('Candidate selection belongs to another workspace. Refresh its choices.')
        return candidate_views.stored(workspace, req['reference'],
                                      **{key:req[key] for key in ('offset','limit') if key in req})
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
        if 'workspace_id' in req:
            from .authoring import workspace_id
            if req['workspace_id'] != workspace_id(workspace):
                raise ValueError('Candidate selection belongs to another workspace. Refresh its choices.')
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
        from .recovery_views import completion
        from .authoring import workspace_id
        notice = completion(workspace_id(captured_workspace), result['reference'], doc)
        ref = captured_workspace.put_spec('run', result['record'])
        if ref != result['reference']: raise ValueError('Run evidence identity changed on admission.')
        window.broadcast('scienceComplete', notice)
        window.broadcast(); window.broadcast('scienceChanged')
    except Exception as error:
        retain_recovery(window, job, result, error)
    finally:
        job.terminal_delivered = True; job.awaiting_terminal = False
        window.broadcast()
