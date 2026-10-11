"""Real Qt dispatcher fences and one SQL/history edit for shared LSTM builder."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import QCoreApplication, QObject
from app.config import Config
from app.core import Workspace, default_model, validate_model
from app.desktop import Window
from app.lstm_authoring import CREATE_TOKEN, EDIT_TOKEN


class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,root):
        super().__init__();self.workspace=Workspace();self.job=None;self.torch_python=None;self.rendering={}
        self.config=Config(root/'config.json');self.config.save=Mock(side_effect=AssertionError('No config writes'))
        parent=default_model();parent['id']='parent';self.workspace.put(parent);self.workspace.dirty=False
    def recent_files(self):return []
    def broadcast(self,*args,**kwargs):pass
    def prepare(self,nid=None):
        s=self.state();context={k:s[k] for k in ('workspace_id','panel_scope')}
        info=Window.dispatch(self,{'action':'lstmEditing' if nid else 'lstmAuthoring','id':'parent',
            **context,**({'node':nid} if nid else {})},None)
        return {'action':'editLSTM' if nid else 'addLSTM',**context,
            **{k:info[k] for k in (EDIT_TOKEN if nid else CREATE_TOKEN)},'options_json':json.dumps(info['options'])}


app=QCoreApplication([])
with tempfile.TemporaryDirectory() as folder:
    h=Harness(Path(folder));before=h.workspace.models();raw=h.workspace.db.serialize();request=h.prepare()
    assert h.workspace.db.serialize()==raw and not h.workspace.dirty
    for patch in ({'workspace_id':'wrong'},{'panel_scope':'wrong'},{'root_graph':'wrong'},{'root_model':'wrong'},
                  {'extra':1},{'options_json':'{"version":1,"version":1}'},{'options_json':'NaN'},
                  {'options_json':'x'*65537},{'options_json':{}}):
        try:Window.dispatch(h,{**request,**patch},None)
        except ValueError:pass
        else:raise AssertionError('Malformed/stale LSTM request admitted')
        assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==0
    h.job=SimpleNamespace(isRunning=lambda:True,mid=None)
    try:Window.dispatch(h,request,None)
    except ValueError as error:assert 'Stop' in str(error)
    else:raise AssertionError('Busy mutation admitted')
    h.job=None;result=Window.dispatch(h,request,None);nid=result['node'];after=h.workspace.models()
    assert validate_model(after,'parent')['parameters']==419 and len(h.workspace.history.undo_stack)==1
    assert Window.dispatch(h,{'action':'undo'},None);assert h.workspace.models()==before
    assert Window.dispatch(h,{'action':'redo'},None);assert h.workspace.models()==after
    edit=h.prepare(nid);raw=h.workspace.db.serialize();count=len(h.workspace.history.undo_stack)
    Window.dispatch(h,edit,None);assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
    choices=json.loads(edit['options_json']);choices['label']='Native edit';edit['options_json']=json.dumps(choices)
    Window.dispatch(h,edit,None);assert len(h.workspace.history.undo_stack)==count+1
    try:Window.dispatch(h,edit,None)
    except ValueError as error:assert 'changed' in str(error)
    else:raise AssertionError('Stale replay admitted')
    path=Path(folder)/'private.vtw';h.workspace.save(path);stale=h.prepare(nid);old=h.workspace
    h.workspace=Workspace.load(path);old.db.close();raw=h.workspace.db.serialize()
    try:Window.dispatch(h,stale,None)
    except ValueError as error:assert 'Workspace changed' in str(error)
    else:raise AssertionError('Same UUID reload admitted stale instance')
    assert h.workspace.db.serialize()==raw;h.config.save.assert_not_called();h.workspace.db.close()
assert 'torch' not in sys.modules
print(json.dumps({'native_create_edit_undo_redo':'PASS','closed_json_stale_busy_reload':'PASS',
                  'noop_sql_history':'PASS','config_untouched':True,'host_torch':False}))
