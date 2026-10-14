"""Check real native menu actions and policy without initializing WebEngine."""
from types import SimpleNamespace
from PyQt6.QtCore import Qt,QPoint
from PyQt6.QtWidgets import QApplication,QWidget
from app.context_menu import build_context_menu,configure_context_menu
import json

app=QApplication([])
class View(QWidget):
    mid="model"
    def page(self):
        return SimpleNamespace(runJavaScript=self.run_script)
    def zoomFactor(self):return 1.35
    def run_script(self,script,callback=None):
        called.append(script)
        if callback:pending.append(callback)
called=[]
pending=[]
view=View()
window=SimpleNamespace(workspace=SimpleNamespace(models=lambda:{"model":{}}),
    save_workspace=lambda:called.append("save-workspace"),
    open_tab=lambda mid:called.append("tab:"+mid))
configure_context_menu(view,window)
assert view.contextMenuPolicy()==Qt.ContextMenuPolicy.CustomContextMenu
menu=build_context_menu(view,window)
actions={a.text():a for a in menu.actions() if not a.isSeparator()}
groups=[[]]
for action in menu.actions():
    if action.isSeparator():groups.append([])
    else:groups[-1].append(action.text())
assert all(group==sorted(group,key=str.casefold) for group in groups),groups
assert not any(s in " ".join(actions).lower() for s in ("save page","view page source","reload"))
assert 'Delete block' not in actions,"Empty space must not delete an earlier selection"
assert 'Inspect block' not in actions
palette=build_context_menu(view,window,{'palette':{'id':'add-linear','name':'Linear'}})
assert palette.actions()[0].text()=='About Linear' and palette.actions()[0].isEnabled()
assert palette.actions()[1].isSeparator()
assert not any(a.text() in ('Inspect block','Delete block','Automatic size') for a in palette.actions())
palette.actions()[0].trigger()
assert called[-1]=='window.loom && window.loom.perform("about-building-block","add-linear");'
block_menu=build_context_menu(view,window,{'id':'linear1','modelId':'model','canDelete':True,'canResize':True})
assert [a.text() for a in block_menu.actions()[:2]]==['Inspect block','Delete block']
assert block_menu.actions()[2].text()=='Automatic size'
assert block_menu.actions()[3].isSeparator()
automatic=block_menu.actions()[2];assert automatic.isEnabled()
automatic.trigger();assert called[-1]=='window.loom && window.loom.perform("automatic-block-size","linear1","model");'
inspect=block_menu.actions()[0]
assert inspect.isEnabled()
inspect.trigger();assert called[-1]=='window.loom && window.loom.perform("inspect-block","linear1","model");'
delete=next(a for a in block_menu.actions() if a.text()=='Delete block')
assert delete.isEnabled()
delete.trigger();assert called[-1]=='window.loom && window.loom.perform("delete-block","linear1","model");'
locked=build_context_menu(view,window,{'id':'linear1','modelId':'model','canDelete':False})
assert not next(a for a in locked.actions() if a.text()=='Delete block').isEnabled()
assert next(a for a in locked.actions() if a.text()=='Inspect block').isEnabled()
assert not next(a for a in locked.actions() if a.text()=='Automatic size').isEnabled()
other=build_context_menu(view,window,{'id':'linear1','modelId':'other','canDelete':True})
assert not next(a for a in other.actions() if a.text()=='Delete block').isEnabled()
assert not next(a for a in other.actions() if a.text()=='Inspect block').isEnabled()
assert not next(a for a in other.actions() if a.text()=='Automatic size').isEnabled()
assert 'Expand into simpler blocks' not in [a.text() for a in block_menu.actions()]
conv=build_context_menu(view,window,{'id':'conv','modelId':'model','canDelete':True,'canResize':True,'canExpand':True})
assert [a.text() for a in conv.actions()[2:4]]==['Automatic size','Expand into simpler blocks'] and conv.actions()[4].isSeparator()
conv.actions()[3].trigger();assert called[-1]=='window.loom && window.loom.perform("expand-block","conv","model");'
assert not next(a for a in build_context_menu(view,window,{'id':'conv','modelId':'other','canExpand':True}).actions() if a.text()=='Expand into simpler blocks').isEnabled()
actions["Building blocks panel"].trigger()
assert called[-1]=='window.loom && window.loom.perform("library");'
actions["Layout panel"].trigger()
assert called[-1]=='window.loom && window.loom.perform("layout");'
actions["Save workspace"].trigger();assert called[-1]=="save-workspace"
actions["Open architecture in new tab"].trigger();assert called[-1]=="tab:model"
assert 'Find…' not in actions
view.customContextMenuRequested.emit(QPoint(135,270))
assert called[-1]=='window.loom && window.loom.contextAt(100.0,200.0);'
assert not hasattr(view,'app_context_menu'),"Menu must wait for selection callback"
old=pending.pop()
view.customContextMenuRequested.emit(QPoint(27,54))
assert called[-1]=='window.loom && window.loom.contextAt(20.0,40.0);'
old({'id':'stale','modelId':'model','canDelete':True})
assert not hasattr(view,'app_context_menu'),"Stale callback must not open a menu"
pending.pop()({'id':'clicked','modelId':'model','canDelete':True})
assert view.app_context_menu.isVisible()
view.app_context_menu.actions()[0].trigger()
assert called[-1]=='window.loom && window.loom.perform("inspect-block","clicked","model");'
next(a for a in view.app_context_menu.actions() if a.text()=='Delete block').trigger()
assert called[-1]=='window.loom && window.loom.perform("delete-block","clicked","model");'
view.app_context_menu.close()
view.mid=None;home=build_context_menu(view,window)
assert not next(a for a in home.actions() if a.text()=="Fit graph").isEnabled()
assert 'Delete block' not in [a.text() for a in home.actions()]
assert 'Inspect block' not in [a.text() for a in home.actions()]
assert 'Automatic size' not in [a.text() for a in home.actions()]
print(json.dumps({"context_menu":"PASS","browser_actions_absent":True,"select_before_popup":True,
    "zoom_coordinates":True,"native_actions":list(actions)}))
