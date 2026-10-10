"""Native Qt admission/rebind/status contract; private DB, no WebEngine or Torch."""
import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication,QObject
from PyQt6.QtWidgets import QMessageBox
from app.core import Workspace
from app.desktop import Window,find_torch_python
from test_bindings import two_inputs


class Harness(QObject):
    state=Window.state
    describe=Window.describe
    def __init__(self):
        super().__init__()
        self.workspace=Workspace();self.job=None;self.events=[]
        self.torch_python=sys.executable;self.rendering={}
        self.config=SimpleNamespace(values={'memory_limit_percent':10,'theme':'dark'})
    def recent_files(self):return []
    def broadcast(self,*args,**kwargs):self.events.append(args)
    def dispatch(self,action,**request):return Window.dispatch(self,{'action':action,**request},None)


app=QCoreApplication([])
h=Harness();d=two_inputs();h.workspace.put(d);mid=d['id']
try:
    report=h.describe(mid)
    assert report['valid'] and report['dataset_admission']['allowed']
    assert report['checkpoint_status']['status']=='none'
    d=h.workspace.models()[mid];d['result']={'mse':.5};h.workspace.put(d)
    assert h.describe(mid)['checkpoint_status']['status']=='unverified'
    h.workspace.record_run(mid,d['result'],h.workspace.models())
    d=h.workspace.models()[mid];dataset=copy.deepcopy(d['dataset'])
    d['graph']['input_order']=['a','b'];h.workspace.put(d)
    report=h.describe(mid)
    assert report['valid'] and not report['dataset_admission']['allowed']
    assert report['dataset_admission']['status']=='incompatible'
    assert h.workspace.models()[mid]['result'] is None
    with patch('app.desktop.TrainingJob') as launch:
        try:h.dispatch('train',id=mid)
        except ValueError as error:assert 'binding' in str(error)
        else:raise AssertionError('GUI admitted a silently reinterpreted dataset')
        launch.assert_not_called()
    before=h.workspace.models()
    with patch('app.desktop.QMessageBox.question',return_value=QMessageBox.StandardButton.Cancel):
        assert not h.dispatch('rebindDataset',id=mid)
    assert h.workspace.models()==before and not h.workspace.history.summary()['undo']
    with patch('app.desktop.QMessageBox.question',return_value=QMessageBox.StandardButton.Yes) as confirm:
        assert h.dispatch('rebindDataset',id=mid)
        assert confirm.call_args.args[1]=='Visual PyTorch — Rebind dataset'
    assert h.workspace.models()[mid]['dataset']==dataset
    assert h.workspace.models()[mid]['dataset_binding']['inputs'][0]['id']=='a'
    assert h.describe(mid)['dataset_admission']['allowed']
    assert h.dispatch('undo') and not h.describe(mid)['dataset_admission']['allowed']
    assert h.dispatch('redo') and h.describe(mid)['dataset_admission']['allowed']
    assert h.workspace.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==1
    h.job=SimpleNamespace(isRunning=lambda:True)
    try:h.dispatch('rebindDataset',id=mid)
    except ValueError as error:assert 'Stop' in str(error)
    else:raise AssertionError('Rebinding allowed during training')
    h.job=None
    d=h.workspace.models()[mid];d['graph']['input_order']=['b','a'];h.workspace.put(d)
    def concurrent_edit(*args):
        doc=h.workspace.models()[mid];doc['name']='Queued external edit';h.workspace.put(doc)
        return QMessageBox.StandardButton.Yes
    with patch('app.desktop.QMessageBox.question',side_effect=concurrent_edit):
        try:h.dispatch('rebindDataset',id=mid)
        except ValueError as error:assert 'changed during confirmation' in str(error)
        else:raise AssertionError('Stale modal confirmation admitted')
    assert h.workspace.models()[mid]['name']=='Queued external edit'
    assert not h.describe(mid)['dataset_admission']['allowed']
    # A worker envelope alone is insufficient to admit checkpoint contents.
    h.job=SimpleNamespace(mid=mid,models=copy.deepcopy(h.workspace.models()))
    models=h.workspace.models();before=copy.deepcopy(models)
    Window.training_complete(h,json.dumps({'mse':.1}))
    assert h.workspace.models()==before
    assert h.events[-1][0]=='trainingError'
    assert h.workspace.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==1
    # Retain a valid completed run against its original snapshot even when a
    # current-draft change means it must not become the current displayed result.
    h.job=None
    d=h.workspace.models()[mid]
    # Derive the current contract directly rather than reusing an older report.
    from app.evidence import binding_for
    from app.core import validate_model
    d['dataset_binding']=binding_for(validate_model(h.workspace.models(),mid));h.workspace.put(d)
    snapshot=h.workspace.models();python=find_torch_python();assert python
    import subprocess
    source=('import json,sys,torch;torch.set_num_threads(1);from app.core import train_model;'
            'r=json.load(sys.stdin);print(json.dumps(train_model(r["models"],r["id"])))')
    process=subprocess.run([python,'-c',source],input=json.dumps({'models':snapshot,'id':mid}),
                           text=True,capture_output=True,timeout=15)
    assert process.returncode==0,process.stderr
    h.job=SimpleNamespace(mid=mid,models=snapshot)
    d=h.workspace.models()[mid];d['dataset']['y'][0][0]+=1;h.workspace.put(d)
    Window.training_complete(h,process.stdout)
    assert h.workspace.models()[mid]['result'] is None
    assert h.workspace.db.execute('SELECT COUNT(*) FROM runs').fetchone()[0]==2
    assert any(event and event[0]=='externalConflict' for event in h.events)
    assert 'torch' not in sys.modules
finally:
    h.workspace.history.directory.cleanup();h.workspace.db.close()
print(json.dumps({'qt_binding_admission':'PASS','cancel_and_stale_confirmation':'PASS',
                  'explicit_rebind_undo_redo':'PASS','historical_runs_preserved':'PASS',
                  'invalid_checkpoint_rejected':'PASS','stale_completion_archived_not_current':'PASS',
                  'legacy_status':'unverified','torch_imported':False}))
