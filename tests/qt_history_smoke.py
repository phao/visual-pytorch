"""Native dispatcher undo/redo and failed-edit behavior without WebEngine."""
import json
from PyQt6.QtCore import QCoreApplication,QObject
from app.core import Workspace,default_model
from app.desktop import Window
app=QCoreApplication([])
class Harness(QObject):
    def __init__(self):
        super().__init__();self.workspace=Workspace();self.job=None;self.events=[]
        self.torch_python=None;self.rendering={}
    def broadcast(self,*args,**kw):self.events.append(args)
    def state(self):return Window.state(self)
    def recent_files(self):return []
    def describe(self,mid):return Window.describe(self,mid)
    def dispatch(self,**request):return Window.dispatch(self,request,None)
h=Harness();d=default_model();h.workspace.put(d)
original=h.workspace.models();d['graph']['nodes'][1]['x']=48
h.dispatch(action='updateModel',document=d)
assert h.workspace.history.summary()['undo']
assert h.dispatch(action='undo');assert h.workspace.models()==original
assert h.dispatch(action='redo');assert h.workspace.models()[d['id']]['graph']['nodes'][1]['x']==48
d['graph']['nodes'][1]['features']=0
try:h.dispatch(action='updateModel',document=d)
except ValueError:pass
else:raise AssertionError('Invalid edit accepted')
assert len(h.workspace.history.undo_stack)==1
assert h.dispatch(action='undo');assert not h.dispatch(action='undo')
h.workspace.history.directory.cleanup();h.workspace.db.close()
print(json.dumps({'dispatcher_undo_redo':'PASS','failed_edit_keeps_history':True}))
