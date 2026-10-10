"""Native display settings and disk-update behavior, without Chromium startup."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication
from app.config import Config
from app.core import Workspace, default_model
from app.desktop import Window

app=QApplication([])
class Harness:
    def __init__(self,path,config):
        self.workspace=Workspace.load(path);self.job=None;self.external_notice=None
        self.events=[];self.config=Config(config)
        self.views=[SimpleNamespace(mid=next(iter(self.workspace.models())),zoom=None,
            setZoomFactor=lambda value:self.zooms.append(value),bridge=SimpleNamespace(send=lambda *v:self.events.append(v)))]
        self.zooms=[];self.tabs=SimpleNamespace(count=lambda:len(self.views),widget=lambda i:self.views[i])
    def broadcast(self,*args):self.events.append(args)

with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'live.vtw';w=Workspace();d=default_model();w.put(d);w.save(path);w.db.close()
    h=Harness(path,Path(td)/'config.json')
    external=Workspace.load(path);doc=external.models()[d['id']];doc['name']='Agent edit';external.put(doc);external.save(path)
    Window.poll_workspace(h)
    assert h.workspace.models()[d['id']]['name']=='Agent edit'
    assert ('externalReload','Workspace updated from disk (CLI or another editor).') in h.events
    local=h.workspace.models()[d['id']];local['name']='Unsaved user edit';h.workspace.put(local)
    doc['name']='Another agent edit';external.put(doc);external.save(path)
    Window.poll_workspace(h)
    assert h.workspace.models()[d['id']]['name']=='Unsaved user edit'
    assert any(e and e[0]=='externalConflict' for e in h.events)
    assert Window.set_display(h,ui_zoom=1.55,font_size=18)
    assert h.zooms[-1]==1.55 and Config(h.config.path).values['ui_zoom']==1.55
    h2=Harness(path,h.config.path)
    assert h2.config.values['ui_zoom']==1.55
    h.workspace.db.close();external.db.close();h2.workspace.db.close()
print(json.dumps({'live_disk_reload':'PASS','dirty_conflict_preserved':'PASS','display_persistence':'PASS'}))
