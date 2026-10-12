"""Native refresh avoids redundant whole-model reads, without changing text."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QCoreApplication
from app.core import Workspace,default_model
from app.config import Config
from app.desktop import Window

app=QCoreApplication([])
class Bridge:
    def __init__(self):self.events=[]
    def send(self,kind,value):self.events.append((kind,value))
class Tabs:
    def __init__(self):self.views=[type('View',(),{'mid':'model','bridge':Bridge()})() for _ in range(3)];self.names=[]
    def count(self):return 3
    def widget(self,i):return self.views[i]
    def setTabText(self,i,name):self.names.append(name)
class Harness:
    state=Window.state;describe=Window.describe;broadcast=Window.broadcast;tab_name=Window.tab_name
    def __init__(self,w,path):
        self.workspace=w;self.config=Config(path);self.job=None;self.rendering={};self.torch_python='';self.tabs=Tabs()
    def recent_files(self):return []
    def update_title(self):pass
    def update_history_actions(self):pass

with tempfile.TemporaryDirectory() as td:
    w=Workspace();d=default_model('Original label');d['id']='model';w.put(d);h=Harness(w,Path(td)/'config.json')
    try:
        with patch.object(w,'models',wraps=w.models) as reads:h.broadcast();assert reads.call_count==1
        assert h.tabs.names==['Original label']*3
        assert all(view.bridge.events[0][1]['models']['model']['name']=='Original label' for view in h.tabs.views)
        with patch.object(w,'models',wraps=w.models) as reads,patch.object(h,'state',side_effect=AssertionError('description must not gather all state')):
            assert h.describe('model')['valid'];assert reads.call_count==1
        snap=w.history.capture(w.db);d['name']='New label';w.put(d);w.history.record(snap,w.db)
        assert Window.dispatch(h,{'action':'undo'},None)
        assert h.tabs.names[-3:]==['Original label']*3
        assert all(view.bridge.events[-1][1]['models']['model']['name']=='Original label' for view in h.tabs.views)
        assert Window.dispatch(h,{'action':'redo'},None)
        assert h.tabs.names[-3:]==['New label']*3
        print(json.dumps({'ok':True,'one_models_read_per_broadcast':True,'description_no_state_reloads':True,'three_tabs_names_undo_redo_preserved':True}))
    finally:
        w.db.close()
        if w.history.directory:w.history.directory.cleanup()
