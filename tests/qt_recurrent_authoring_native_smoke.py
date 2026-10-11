"""Native recurrent authoring scopes, stale closure refusal and atomic history."""
import copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6.QtCore import QCoreApplication,QObject
from app.config import Config
from app.core import Workspace,validate_model
from app.desktop import Window
from test_recurrent_authoring import authored

class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,path):
        super().__init__();self.workspace=Workspace();self.job=None;self.events=[]
        self.config=Config(path/'config.json');self.config.save=Mock(side_effect=AssertionError('No config writes'))
        self.torch_python=sys.executable;self.rendering={}
        self.models,self.token,self.options=authored()
        for model in self.models.values():self.workspace.put(model)
        self.workspace.dirty=False
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.events.append((kind,payload))
    def prepare(self):
        s=self.state();request={'action':'recurrentAuthoring','workspace_id':s['workspace_id'],
          'panel_scope':s['panel_scope'],'id':'parent','ref':self.token['ref']}
        info=Window.dispatch(self,request,None)
        return {**{k:info[k] for k in ('workspace_id','panel_scope','id','ref','root_graph','transition')},
                'action':'addRecurrent','options':copy.deepcopy(self.options)}

app=QCoreApplication([])
with tempfile.TemporaryDirectory() as tmp:
    h=Harness(Path(tmp));request=h.prepare();raw=h.workspace.db.serialize();history=h.workspace.history.summary()
    for changes in ({'workspace_id':'wrong'},{'panel_scope':'wrong'},{'root_graph':'wrong'},
                    {'transition':'wrong'},{'extra':1},{'options':{**h.options,'steps':True}}):
        try:Window.dispatch(h,{**request,**changes},None)
        except ValueError:pass
        else:raise AssertionError('Stale/malformed authoring admitted')
        assert h.workspace.db.serialize()==raw and h.workspace.history.summary()==history
    h.job=SimpleNamespace(isRunning=lambda:True,mid=None)
    try:Window.dispatch(h,request,None)
    except ValueError as error:assert 'Stop' in str(error)
    else:raise AssertionError('Busy graph edit admitted')
    assert h.workspace.db.serialize()==raw;h.job=None
    before=h.workspace.models();answer=Window.dispatch(h,request,None)
    after=h.workspace.models();nid=answer['node'];assert answer['id']=='parent'
    c=validate_model(after,'parent');assert c['parameters']==13
    description=Window.describe(h,'parent');assert description['valid'] and nid in description['interfaces']
    assert h.workspace.history.summary()['undo'] and len(h.workspace.history.undo_stack)==1
    assert Window.dispatch(h,{'action':'undo'},None);assert h.workspace.models()==before
    assert Window.dispatch(h,{'action':'redo'},None);assert h.workspace.models()==after
    try:Window.dispatch(h,request,None)
    except ValueError as error:assert 'changed' in str(error)
    else:raise AssertionError('Double-submit replay admitted')
    path=Path(tmp)/'saved.vtw';h.workspace.save(path);stale=h.prepare();old=h.workspace
    h.workspace=Workspace.load(path);old.db.close()
    try:Window.dispatch(h,stale,None)
    except ValueError as error:assert 'Workspace changed' in str(error)
    else:raise AssertionError('Same-UUID reload accepted a stale form')
    h.config.save.assert_not_called();h.workspace.db.close()
assert 'torch' not in sys.modules
print(json.dumps({'native_scopes_stale_graph_closure':'PASS','one_atomic_undo_redo':'PASS',
 'double_submit_and_same_uuid_reload':'PASS','busy_guard':'PASS','host_torch':False,'config_untouched':True}))
