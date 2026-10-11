"""Native panel-preset scopes/CAS, small broadcasts, no history or global writes."""
import copy,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PyQt6.QtCore import QCoreApplication,QObject
from app.config import Config
from app.core import Workspace
from app.desktop import Window
from test_panel_workspace import preset

class Harness(QObject):
    state=Window.state
    def __init__(self,path):
        super().__init__();self.workspace=Workspace.demo();self.job=None;self.events=[]
        self.config=Config(path/'config.json');self.config.save=Mock(side_effect=AssertionError('No global writes'))
        self.torch_python=sys.executable;self.rendering={}
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None):
        assert kind=='panelLayout','No whole-workspace broadcast for a panel preset'
        self.events.append((kind,payload))
    def request(self,value,**changes):
        s=self.state();req={'action':'setPanelLayout','workspace_id':s['workspace_id'],'panel_scope':s['panel_scope'],
                          'expected':s['panel_layout'],'layout':value};req.update(changes);return req
app=QCoreApplication([])
with tempfile.TemporaryDirectory() as tmp:
    h=Harness(Path(tmp));before=copy.deepcopy(h.workspace.models());request=h.request(preset())
    with patch.object(h.workspace,'models',side_effect=AssertionError('No full model/data read')):
        result=Window.dispatch(h,request,None)
    assert result['layout']==preset() and h.events[-1][1]==result and h.workspace.models()==before
    assert h.workspace.history.summary()=={'undo':False,'redo':False,'notice':None}
    assert h.workspace.dirty;h.workspace.dirty=False
    result=Window.dispatch(h,h.request(preset()),None);assert not result['dirty']
    raw=h.workspace.db.serialize()
    for bad in (dict(request,workspace_id='wrong'),dict(request,panel_scope='wrong'),
                dict(request,expected=None),dict(request,extra='css')):
        try:Window.dispatch(h,bad,None)
        except ValueError:pass
        else:raise AssertionError('Stale/malformed preset admitted')
        assert h.workspace.db.serialize()==raw
    path=Path(tmp)/'saved.vtw';h.workspace.save(path);old=h.workspace
    stale=h.request(None);h.workspace=Workspace.load(path);assert old.panel_scope!=h.workspace.panel_scope
    try:Window.dispatch(h,stale,None)
    except ValueError as error:assert 'Workspace changed' in str(error)
    else:raise AssertionError('Same-UUID reload admitted an old workspace-instance write')
    old.db.close()
    h.job=SimpleNamespace(isRunning=lambda:True,mid=None)
    current=h.state();req={'action':'setPanelLayout','workspace_id':current['workspace_id'],
       'panel_scope':current['panel_scope'],'expected':current['panel_layout'],'layout':None}
    assert Window.dispatch(h,req,None)['layout'] is None
    assert not h.workspace.history.undo_stack;h.config.save.assert_not_called()
    h.workspace.db.close();h.workspace=None
    try:Window.dispatch(h,req,None)
    except ValueError as error:assert 'workspace first' in str(error)
    else:raise AssertionError('No-workspace save admitted')
assert 'torch' not in sys.modules
print(json.dumps({'native_workspace_panel_preset':'PASS','same_uuid_instance_and_cas':'PASS',
 'no_history_or_full_model_broadcast':'PASS','busy_presentation_only':'PASS','global_config_untouched':'PASS','host_torch':False}))
