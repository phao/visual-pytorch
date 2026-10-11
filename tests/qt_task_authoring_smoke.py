"""Real sandboxed task forms create revisions used by actual numerical fits.

Private profiles/data only. The GUI interpreter must never import Torch.
"""
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace
from app.desktop import Bridge,Window,find_torch_python
from app.task_execution import training_protocol
from test_task_authoring import multi_fixture,sequence_fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['task-authoring-smoke']);view=QWebEngineView();view.resize(1500,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace()
        self.fixtures=[]
        for fixture in (multi_fixture(),sequence_fixture()):
            model,dataset,task=fixture;self.workspace.put(model)
            self.fixtures.append((self.workspace.freeze_model(model['id']),self.workspace.put_spec('dataset',dataset),task))
        assert self.workspace.list_specs('task')==[]
        self.job=None;self.torch_python=find_torch_python();assert self.torch_python
        self.rendering={};self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(3000);loop.exec();timer.stop()
    assert values,'JavaScript timeout';return values[0]
def until(script,timeout=30):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if js(script):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(f'UI timeout: {script}; console: {errors[:5]}')
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    js('document.getElementById('+json.dumps(sid)+').click()')
def set_id(sid,value):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    assert js('setTaskValue(document.getElementById('+json.dumps(sid)+'),'+json.dumps(value)+')'),errors
def set_binding(kind,index,label,value):
    until('Boolean(taskCard('+json.dumps(kind)+','+str(index)+')) && !document.getElementById("science-author-check").disabled')
    assert js('setTaskValue(taskControl('+json.dumps(kind)+','+str(index)+','+json.dumps(label)+'),'+json.dumps(value)+')'), (kind,index,label,errors)
    until('!document.getElementById("science-author-check").disabled')
def button(kind,index,name,transform=None):
    selector='taskCard('+json.dumps(kind)+','+str(index)+')'
    if transform is not None:selector+='.querySelector('+json.dumps("[data-transform-index='"+str(transform)+"']")+')'
    elif name in ('↑','↓','Remove binding'):selector+='.querySelector("[data-binding-order]")'
    until('Boolean('+selector+') && !document.getElementById("science-author-check").disabled')
    assert js('taskButton('+selector+','+json.dumps(name)+')'),(kind,index,name)
    until('!document.getElementById("science-author-check").disabled')
def add_transform(kind,index,step):
    until('Boolean(taskCard('+json.dumps(kind)+','+str(index)+')) && !document.getElementById("science-author-check").disabled')
    count=js('taskCard('+json.dumps(kind)+','+str(index)+').querySelectorAll("[data-transform-index]").length')
    assert js('addTaskTransform('+json.dumps(kind)+','+str(index)+','+json.dumps(step)+')')
    until('taskCard('+json.dumps(kind)+','+str(index)+').querySelectorAll("[data-transform-index]").length==='+str(count+1)+' && !document.getElementById("science-author-check").disabled')
def fill_task(task):
    set_id('science-author-name',task['name'])
    for kind,items in [('input',task['inputs']),('head',task['heads'])]:
        for index,binding in enumerate(items):
            assert js('taskButton(document.getElementById("science-task-bindings"),'+json.dumps('Add input binding' if kind=='input' else 'Add loss head')+')')
            until('Boolean(taskCard('+json.dumps(kind)+','+str(index)+')) && !document.getElementById("science-author-check").disabled')
            for key,label in ([('port','Input port ID'),('field','Source field ID')] if kind=='input' else
                    [('id','Head ID'),('port','Output port ID'),('field','Target field ID'),('loss','Loss'),('weight','Objective weight'),('mask','Mask field ID (optional)'),('sample_weight','Sample-weight field ID (optional)')]):
                if key in binding:set_binding(kind,index,label,str(binding[key]))
            if 'derived' in binding:
                set_binding(kind,index,'Derived input',binding['derived']);set_binding(kind,index,'Padded length',str(binding['length']))
            for ti,step in enumerate(binding['transforms']):
                add_transform(kind,index,step['kind'])
                for key,value in step.items():
                    if key!='kind':assert js('setTaskValue(taskTransform('+json.dumps(kind)+','+str(index)+','+str(ti)+','+json.dumps(key)+'),'+json.dumps(str(value))+')')
    if 'description' in task:assert js('setTaskValue([...document.querySelectorAll("#science-task-bindings label")].find(l=>l.firstChild.textContent==="Task description").querySelector("textarea"),'+json.dumps(task['description'])+')')
def choose_context(mref,dref):
    set_id('science-task-model',json.dumps(mref,separators=(',',':')))
    until('!document.getElementById("science-task-model").disabled')
    set_id('science-task-dataset',json.dumps(dref,separators=(',',':')))
    until('document.getElementById("science-task-context-status").textContent.startsWith("Exact context")')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);workspaces=[h.workspace]
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.fixtures[0][0]['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)')
        js('''window.setTaskValue=(control,value)=>{if(!control)throw Error("Missing task control");if(control.disabled)throw Error("Task control disabled");control.value=value;control.focus();control.dispatchEvent(new Event("change"));return true;};
          window.taskCard=(kind,index)=>document.querySelector("[data-task-"+kind+"='"+index+"']");
          window.taskControl=(kind,index,label)=>[...taskCard(kind,index).querySelectorAll("label")].find(l=>l.firstChild.textContent===label).querySelector("input,select,textarea");
          window.taskButton=(card,name)=>{const b=[...card.querySelectorAll("button")].find(b=>b.textContent===name);if(!b||b.disabled)throw Error("Missing/enabled task button "+name);b.click();return true;};
          window.taskTransform=(kind,index,ti,key)=>[...taskCard(kind,index).querySelector("[data-transform-index='"+ti+"']").querySelectorAll("label")].find(l=>l.firstChild.textContent===key).querySelector("input");
          window.addTaskTransform=(kind,index,step)=>{const card=taskCard(kind,index),select=card.querySelector("select[aria-label='Transform to add']");select.value=step;return taskButton(card,"Add transform");};''')
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"870px",width:"920px",top:"110px",left:"330px",bottom:"auto"});')
        retained=[]
        for fi,(mref,dref,expected) in enumerate(h.fixtures):
            click('science-author-new-task');until('!document.getElementById("science-author-task").hidden && document.getElementById("science-task-model").options.length>=2 && !document.getElementById("science-task-model").disabled')
            assert js('document.querySelectorAll("[data-task-input],[data-task-head]").length')==0
            choose_context(mref,dref);fill_task(expected)
            before=h.workspace.db.serialize()
            assert js('[...taskCard("input",0).querySelectorAll("button")].find(b=>b.title==="Move binding earlier").disabled'),errors
            # Reorder explicit bindings, then restore; IDs and chains survive.
            button('input',0,'↓');button('input',1,'↑')
            if fi==0:
                # Switching to CE must not discard a target transform silently.
                add_transform('head',0,'onehot');click('science-author-check')
                until('document.getElementById("science-author-status").textContent.includes("do not have transforms")')
                assert js('taskCard("head",0).querySelectorAll("[data-transform-index]").length')==1
                button('head',0,'Remove transform',0)
                button('input',0,'↓',0);button('input',0,'↑',1)
            else:
                # A derived control preserves conflicts until explicitly removed.
                add_transform('input',1,'standardize');click('science-author-check')
                until('document.getElementById("science-author-status").textContent.includes("no transforms")')
                button('input',1,'Remove transform',0)
                set_binding('input',1,'Padded length','2');click('science-task-compatibility')
                until('document.getElementById("science-task-compatibility-status").textContent.startsWith("incompatible")')
                set_binding('input',1,'Padded length','3')
            click('science-task-compatibility');until('document.getElementById("science-task-compatibility-status").textContent.startsWith("compatible")')
            assert h.workspace.db.serialize()==before
            # Typing (before blur/change) immediately invalidates old compatibility.
            click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid TaskSpec")')
            js('(()=>{const w=taskControl("head",0,"Objective weight");w.value="1e400";w.dispatchEvent(new Event("input"));})()')
            assert js('document.getElementById("science-task-compatibility-status").textContent')=='',errors
            assert js('document.getElementById("science-author-status").textContent.startsWith("Unsaved task draft")')
            js('taskControl("head",0,"Objective weight").dispatchEvent(new Event("change"))')
            click('science-task-context-refresh')
            until('!document.getElementById("science-task-model").disabled')
            assert js('taskControl("head",0,"Objective weight").value')=='1e400'
            click('science-author-save');assert not js('document.getElementById("science-author").hidden')
            assert h.workspace.db.serialize()==before
            set_binding('head',0,'Objective weight',str(expected['heads'][0]['weight']))
            if fi==0:
                # New browsing heads cannot silently advance exact context.
                context_model=js('document.getElementById("science-task-model").value')
                context_data=js('document.getElementById("science-task-dataset").value')
                model=h.workspace.models()[mref['id']];model['graph']['nodes'][2]['bias']=False
                h.workspace.put(model);h.workspace.freeze_model(model['id'])
                data=h.workspace.get_spec('dataset',dref['id']);data.pop('revision');data['name']='Later measured revision'
                h.workspace.put_spec('dataset',data)
                click('science-task-context-refresh');until('document.getElementById("science-task-model").options.length===4 && !document.getElementById("science-task-model").disabled')
                assert js('document.getElementById("science-task-model").value')==context_model
                assert js('document.getElementById("science-task-dataset").value')==context_data
                from app.science_browser import index_page
                # An explicitly unavailable pin from the page provider remains
                # selected, not substituted. This is a scoped metadata-provider
                # negative control; the retained bytes themselves still exist.
                def unavailable_pin(workspace,kind,**kwargs):
                    result=index_page(workspace,kind,**kwargs)
                    if kind=='model' and kwargs.get('preferred')==mref:
                        result['rows']=[r for r in result['rows'] if r['revision']!=mref['revision']]
                        result['pinned']=None;result['missing_pin']=mref
                    return result
                with patch('app.science_browser.index_page',side_effect=unavailable_pin):
                    click('science-task-context-refresh')
                    until('document.getElementById("science-task-model").selectedOptions[0].textContent.startsWith("Unavailable") && !document.getElementById("science-task-model").disabled')
                    assert js('document.getElementById("science-task-model").value')==context_model
                click('science-task-context-refresh');until('!document.getElementById("science-task-model").selectedOptions[0].textContent.startsWith("Unavailable") && !document.getElementById("science-task-model").disabled')
                before=h.workspace.db.serialize()
            for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
                h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
                until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
                assert js('parseFloat(getComputedStyle(taskControl("input",0,"Input port ID")).fontSize)')>=font-2
                assert js('taskCard("input",0).getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
            h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
            if fi==0 and os.environ.get('VTW_SMOKE_SCREENSHOT'):
                js('document.getElementById("toast-close").click();document.getElementById("science-author-title").scrollIntoView({block:"start"});')
                loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
            click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid TaskSpec")')
            assert h.workspace.db.serialize()==before
            click('science-author-save');until('document.getElementById("science-author").hidden && document.getElementById("science-kind").value==="task"')
            ref=json.loads(js('document.getElementById("science-object").value'));saved=h.workspace.get_spec('task',ref['id'],ref['revision'])
            assert saved['inputs']==expected['inputs'],(saved['inputs'],expected['inputs'])
            assert saved['heads']==expected['heads'];assert saved['name']==expected['name']
            assert 'model' not in saved and 'dataset' not in saved
            retained.append((mref,dref,ref))
            # Revision-only edits retain immutable bytes through undo/redo.
            click('science-author-edit');until('!document.getElementById("science-author").hidden')
            set_id('science-author-name','Renamed '+expected['name']);click('science-author-save');until('document.getElementById("science-author").hidden')
            updated=h.workspace.get_spec('task',ref['id']);assert updated['revision']!=ref['revision']
            assert h.dispatch({'action':'undo'},None);assert h.workspace.get_spec('task',ref['id'])['revision']==ref['revision']
            assert h.workspace.get_spec('task',ref['id'],updated['revision'])['name'].startswith('Renamed')
            assert h.dispatch({'action':'redo'},None);until('document.getElementById("science-object").options.length=== '+str((fi+1)*2))
        # Both GUI-authored exact tasks drive a real three-update numerical fit.
        for fi,(mref,dref,tref) in enumerate(retained):
            eref=h.workspace.put_spec('experiment',{'version':1,'id':'authored-fit-'+str(fi),'name':'Authored fit '+str(fi),
                'question':'Do exact visual task bindings execute?', 'models':[mref], 'data_tasks':[{'id':'observations','dataset':dref,'task':tref}],
                'splits':[{'id':'held-out','protocol':{'method':'row','seed':3,'fractions':[.5,.25,.25]}}],
                'initialization_seeds':[5], 'protocols':[{'id':'bounded','protocol':training_protocol(epochs=None,updates=3,batch_size=2,evaluate_every=1)}],
                'selection':{'kind':'crossed','pairs':[],'exclusions':[]}})
            coordinates={'model':mref,'data_task':'observations','split':'held-out','initialization':5,'protocol':'bounded'}
            h.dispatch({'action':'scienceTrain','reference':eref,'coordinates':coordinates,'run_id':'visual-task-'+str(fi),'name':'Visual authored task '+str(fi)},None)
            until('Boolean(window.loom.state.experiment_job)')
            until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")')
            record=h.workspace.get_spec('run','visual-task-'+str(fi));result=record['attempts'][-1]['result']
            assert record['attempts'][-1]['status']=='completed',record['attempts'][-1]
            assert result['counts']['completed_updates']==3
            assert set(result['final_metrics']['train']['heads'])=={head['id'] for head in h.fixtures[fi][2]['heads']}
            assert result['identity']['task']['revision']==tref['revision']
        # No-op and stale-head saves apply to task forms, not only datasets.
        set_id('science-kind','task');until('document.getElementById("science-author-edit").disabled===false')
        click('science-author-edit');until('!document.getElementById("science-author").hidden')
        before=h.workspace.db.serialize();history=copy.deepcopy(h.workspace.history.summary())
        click('science-author-save');until('document.getElementById("science-author").hidden')
        assert h.workspace.db.serialize()==before;assert h.workspace.history.summary()==history
        click('science-author-edit');until('!document.getElementById("science-author").hidden')
        set_id('science-author-name','Keep this unsaved task');ref=json.loads(js('document.getElementById("science-object").value'))
        newer=h.workspace.get_spec('task',ref['id']);newer.pop('revision');newer['name']='Concurrent task';h.workspace.put_spec('task',newer)
        before=h.workspace.db.serialize();click('science-author-save');until('document.getElementById("science-author-status").textContent.includes("head changed")')
        assert h.workspace.db.serialize()==before;assert js('document.getElementById("science-author-name").value')=='Keep this unsaved task'
        h.job=SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,mid=None,snapshot={'run_id':'busy'});h.broadcast()
        until('document.getElementById("science-author-save").disabled')
        try:h.dispatch({'action':'scienceAuthorSave'},None)
        except ValueError as error:assert 'Stop' in str(error)
        else:raise AssertionError('Pending terminal allowed task edit')
        h.job=None;h.broadcast();until('!document.getElementById("science-author-save").disabled')
        original=h.workspace;replacement=Workspace();workspaces.append(replacement);replacement.put(multi_fixture()[0])
        dispatch=h.dispatch
        def switch_during_context(request,target):
            value=dispatch(request,target)
            if request['action']=='scienceTaskContextPage':h.workspace=replacement;h.broadcast()
            return value
        # A genuine in-flight bridge response carries the old context after the
        # state signal changed workspace. It must not repopulate new suggestions.
        with patch.object(h,'dispatch',side_effect=switch_during_context):
            click('science-task-context-refresh')
            until('document.getElementById("science-author-status").textContent.includes("Workspace changed")')
        until('document.getElementById("science-author-status").textContent.includes("Workspace changed")')
        assert js('document.getElementById("science-task-compatibility").disabled')
        until('document.getElementById("science-task-input-ports").children.length===0')
        assert js('document.getElementById("science-task-compatibility-status").textContent')==''
        assert original.db.serialize()==before
        # A late error is also stale evidence, not a reason to overwrite the
        # retained Workspace changed status or display an old-context toast.
        h.workspace=original;h.broadcast();until('!document.getElementById("science-author-save").disabled')
        def fail_old_context(request,target):
            value=dispatch(request,target)
            if request['action']=='scienceTaskContextPage':
                h.workspace=replacement;h.broadcast();raise ValueError('Obsolete original-context failure')
            return value
        with patch.object(h,'dispatch',side_effect=fail_old_context):
            click('science-task-context-refresh')
            until('document.getElementById("science-author-status").textContent.includes("Workspace changed") && !document.getElementById("science-author-cancel").disabled')
        assert not js('document.getElementById("toast-message").textContent.includes("Obsolete original-context failure")')
        assert js('document.getElementById("science-task-input-ports").children.length')==0
        assert original.db.serialize()==before
        click('science-author-cancel')
        # Check/Save callbacks have the same identity fence, including a valid
        # Save already admitted to the original DB before callback delivery.
        for action,sid in [('scienceDraftCheck','science-author-check'),('scienceDraftCommit','science-author-save')]:
            h.workspace=original;h.broadcast()
            until('!document.getElementById("science-author-edit").disabled')
            click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
            if action=='scienceDraftCommit':set_id('science-author-name','Admitted to origin only')
            dispatch=h.dispatch
            def switch_during_author(request,target):
                value=dispatch(request,target)
                if request['action']==action:h.workspace=replacement;h.broadcast()
                return value
            with patch.object(h,'dispatch',side_effect=switch_during_author):
                click(sid);until('document.getElementById("science-author-status").textContent.includes("Workspace changed") && !document.getElementById("science-author-cancel").disabled')
            assert not js('document.getElementById("science-author").hidden')
            assert js('document.getElementById("science-author-save").disabled')
            assert h.workspace.list_specs('task',True)==[]
            if action=='scienceDraftCommit':assert original.get_spec('task',ref['id'])['name']=='Admitted to origin only'
            click('science-author-cancel')
        original.save(folder/'tasks.vtw');loaded=Workspace.load(folder/'tasks.vtw')
        try:
            for _,_,ref in retained:assert loaded.get_spec('task',ref['id'],ref['revision'])['inputs']==original.get_spec('task',ref['id'],ref['revision'])['inputs']
            assert len(loaded.list_specs('run',True))==2
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'typed_task_creation_empty_registry':'PASS','ordered_transform_derived_conflicts':'PASS','multihead_masks_weights':'PASS',
            'readonly_compatibility':'PASS','invalid_text_retention_stale_labels':'PASS','actual_authored_multihead_and_ragged_fits':'PASS',
            'history_noop_head_workspace_pending_guards':'PASS','exact_context_refresh_missing_choice_late_response':'PASS',
            'late_check_save_origin_fence':'PASS','save_reload_exact':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        if h.job and hasattr(h.job,'wait'):
            if h.job.isRunning():h.job.stop()
            h.job.wait(15000)
        for workspace in workspaces:
            if workspace.history.directory:workspace.history.directory.cleanup()
            workspace.db.close()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
