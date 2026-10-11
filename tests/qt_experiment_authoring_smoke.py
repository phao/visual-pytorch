"""Typed experiment authoring through real sandboxed Qt, followed by actual fits.

Models/data/tasks are exact fixtures; the experiment registry starts empty.
No user configuration/files, hidden JSON imports, or GUI-host Torch are used.
"""
import copy
import json
import os
import random
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
from app.core import Workspace,default_model
from app.desktop import Bridge,Window,find_torch_python
from app.experiment_objects import preview_experiment
from test_experiment_authoring import fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['experiment-authoring-smoke']);view=QWebEngineView();view.resize(1500,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace();self.expected=fixture(self.workspace)
        assert self.workspace.list_specs('experiment',True)==[]
        bad=default_model('Wrong-width negative control',inputs=2);self.workspace.put(bad);self.bad=self.workspace.freeze_model(bad['id'])
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
    raise AssertionError(f'UI timeout: {script}; status: {js("document.getElementById('science-author-status').textContent")}; console: {errors[:5]}')
def click(sid):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    js('document.getElementById('+json.dumps(sid)+').click()')
def set_id(sid,value):
    until('!document.getElementById('+json.dumps(sid)+').disabled')
    assert js('setExperimentValue(document.getElementById('+json.dumps(sid)+'),'+json.dumps(value)+')'),errors
def control(kind,index,label,value):
    until('Boolean(experimentCard('+json.dumps(kind)+','+str(index)+')) && !experimentControl('+json.dumps(kind)+','+str(index)+','+json.dumps(label)+').disabled')
    assert js('setExperimentValue(experimentControl('+json.dumps(kind)+','+str(index)+','+json.dumps(label)+'),'+json.dumps(value)+')'),(kind,index,label,errors)
def top(label,value):
    until('!experimentTop('+json.dumps(label)+').disabled')
    assert js('setExperimentValue(experimentTop('+json.dumps(label)+'),'+json.dumps(value)+')'),(label,errors)
def add(name):
    until('![...document.querySelectorAll("#science-experiment-bindings button")].find(b=>b.textContent==='+json.dumps(name)+').disabled')
    count=js('document.querySelectorAll("[data-experiment-axis]").length')
    assert js('experimentButton(document.getElementById("science-experiment-bindings"),'+json.dumps(name)+')'),(name,errors)
    until('document.querySelectorAll("[data-experiment-axis]").length==='+str(count+1)+' && !document.getElementById("science-author-check").disabled')
def entry_button(kind,index,name):
    until('![...experimentCard('+json.dumps(kind)+','+str(index)+').querySelector("[data-axis-order]").querySelectorAll("button")].find(b=>b.textContent==='+json.dumps(name)+').disabled')
    assert js('experimentButton(experimentCard('+json.dumps(kind)+','+str(index)+').querySelector("[data-axis-order]"),'+json.dumps(name)+')'),(kind,index,name,errors)
    until('!document.getElementById("science-author-check").disabled')
def initialization_seeds(values,create=False):
    for index,value in enumerate(values):
        if create:add('Add initialization seed')
        control('seed',index,'Initialization seed',str(value))
def ref_value(ref):return json.dumps({k:ref[k] for k in ('kind','id','revision')},separators=(',',':'))
def add_split(entry):
    js('[...document.querySelectorAll("select")].find(s=>s.getAttribute("aria-label")==="Split method to add").value='+json.dumps(entry['protocol']['method']))
    add('Add split');index=js('document.querySelectorAll("[data-experiment-axis=split]").length')-1
    control('split',index,'Split ID',entry['id']);p=entry['protocol']
    if p['method'] in ('row','group'):control('split',index,'Split seed',str(p['seed']))
    if p['method'] in ('group','temporal'):control('split',index,'Grouping field ID' if p['method']=='group' else 'Time field ID',p['field'])
    if p['method']=='external':
        for role in ('train','validation','test'):control('split',index,role+' observation IDs','\n'.join(p['roles'][role]))
        control('split',index,'Excluded observation IDs','\n'.join(p['excluded']))
    else:
        for label,value in zip(('Train fraction','Validation fraction','Test fraction'),p['fractions']):control('split',index,label,str(value))
def add_protocol(entry):
    add('Add training configuration');index=js('document.querySelectorAll("[data-experiment-axis=protocol]").length')-1;p=entry['protocol']
    control('protocol',index,'Configuration ID',entry['id'])
    for key,label in [('batch_size','Batch size'),('evaluate_every','Evaluate every updates')]:control('protocol',index,label,str(p[key]))
    control('protocol',index,'Retained curve points',str(p['history']['max_points']))
    for key,value in p['budget'].items():control('protocol',index,'Maximum '+key,'' if value is None else str(value))
    for key,label in [('lr','Learning rate'),('eps','Adam epsilon'),('weight_decay','Weight decay')]:control('protocol',index,label,str(p['optimizer'][key]))
    for i,value in enumerate(p['optimizer']['betas']):control('protocol',index,'Adam beta '+str(i+1),str(value))
    control('protocol',index,'Protocol initialization seed (axis overrides)',str(p['seeds']['initialization']))
    control('protocol',index,'Independent shuffle seed',str(p['seeds']['shuffle']))
    assert js('(()=>{const c=experimentControl("protocol",'+str(index)+',"Shuffle training observations ");c.checked='+json.dumps(p['shuffle'])+';c.dispatchEvent(new Event("change"));return true;})()')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);workspaces=[h.workspace];e=copy.deepcopy(h.expected)
    # Use non-default optimizer/seed settings to verify complete, not cosmetic,
    # configuration controls. Initialization axis intentionally overrides 180.
    p=e['protocols'][0]['protocol'];p['seeds']={'initialization':180,'shuffle':29};p['shuffle']=False
    p['optimizer'].update(lr=.01,betas=[.8,.95],eps=1e-7,weight_decay=.01);p['history']['max_points']=40
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+e['models'][0]['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)')
        js('''window.setExperimentValue=(c,v)=>{if(!c||c.disabled)throw Error("Missing/enabled experiment control");c.closest("details")?.setAttribute("open","");c.value=v;c.focus();c.dispatchEvent(new Event("change"));return true;};
          window.experimentCard=(kind,index)=>document.querySelector("[data-experiment-axis='"+kind+"'][data-axis-index='"+index+"']");
          window.experimentControl=(kind,index,label)=>[...experimentCard(kind,index).querySelectorAll("label")].find(l=>l.firstChild.textContent===label).querySelector("input,select,textarea");
          window.experimentTop=label=>[...document.querySelectorAll("#science-experiment-bindings label")].find(l=>l.firstChild.textContent===label).querySelector("input,select,textarea");
          window.experimentButton=(card,name)=>{const b=[...card.querySelectorAll("button")].find(b=>b.textContent===name);if(!b||b.disabled)throw Error("Missing/enabled button "+name);b.closest("details")?.setAttribute("open","");b.click();return true;};''')
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"870px",width:"920px",top:"110px",left:"330px",bottom:"auto"});')
        assert js('document.getElementById("panel-context").textContent')=='Exact revisions · explicit runs'
        js('document.querySelector("[data-panel=experiment]").click()')
        assert js('document.getElementById("panel-context").textContent')=='CPU · full-batch Adam · MSE'
        js('document.querySelector("[data-panel=science]").click()')
        click('science-author-new-experiment');until('!document.getElementById("science-author-experiment").hidden && !document.getElementById("science-author-name").disabled')
        assert js('document.querySelectorAll("[data-experiment-axis]").length')==0
        set_id('science-author-name',e['name']);top('Research question',e['question'])
        for index,mref in enumerate(e['models']):add('Add architecture selection');control('model',index,'Exact architecture',ref_value(mref))
        for index,pair in enumerate(e['data_tasks']):
            add('Add dataset/task pairing');control('data-task',index,'Pairing ID',pair['id'])
            control('data-task',index,'Exact dataset',ref_value(pair['dataset']));control('data-task',index,'Exact task',ref_value(pair['task']))
        add_split(e['splits'][0]);initialization_seeds([7,13],create=True);add_protocol(e['protocols'][0])
        # Semantic axes reorder explicitly, then restore without rebinding.
        entry_button('model',0,'↓');entry_button('model',1,'↑')
        before=h.workspace.db.serialize();rng=random.getstate()
        with patch('app.datasets._prepare_column',side_effect=AssertionError('Draft preview must not fit')):
            click('science-experiment-preview');until('document.querySelectorAll("[data-draft-candidate]").length===8')
        assert js('document.querySelectorAll("[data-draft-candidate=compatible]").length')==8
        assert h.workspace.db.serialize()==before;assert random.getstate()==rng
        assert h.workspace.list_specs('experiment',True)==[] and h.workspace.list_specs('run',True)==[]
        assert js('document.querySelectorAll("#science-experiment-preview-rows button").length')==0
        # Live invalid text clears preview but survives metadata refresh.
        js('(()=>{const c=experimentControl("protocol",0,"Maximum updates");c.value="1e400";c.dispatchEvent(new Event("input"));c.dispatchEvent(new Event("change"));})()')
        assert js('document.querySelectorAll("[data-draft-candidate]").length')==0
        click('science-experiment-refresh');until('!document.getElementById("science-experiment-refresh").disabled')
        assert js('experimentControl("protocol",0,"Maximum updates").value')=='1e400'
        click('science-author-save');assert not js('document.getElementById("science-author").hidden');assert h.workspace.db.serialize()==before
        control('protocol',0,'Maximum updates','3')
        # Exact selected revisions survive new heads and a provider-declared
        # unavailable pin; the negative control does not delete retained bytes.
        old=e['data_tasks'][0]['dataset'];data=h.workspace.get_spec('dataset',old['id']);data.pop('revision');data['name']='Newer dataset head'
        h.workspace.put_spec('dataset',data);click('science-experiment-refresh');until('!document.getElementById("science-experiment-refresh").disabled')
        assert js('experimentControl("data-task",0,"Exact dataset").value')==ref_value(old)
        from app.science_browser import index_page
        def unavailable_pin(workspace,kind,**kwargs):
            result=index_page(workspace,kind,**kwargs)
            if kind=='dataset' and kwargs.get('preferred')==old:
                result['rows']=[r for r in result['rows'] if r['revision']!=old['revision']]
                result['pinned']=None;result['missing_pin']=old
            return result
        with patch('app.science_browser.index_page',side_effect=unavailable_pin):
            click('science-experiment-refresh');until('experimentControl("data-task",0,"Exact dataset").selectedOptions[0].textContent.startsWith("Unavailable") && !document.getElementById("science-experiment-refresh").disabled')
            assert js('experimentControl("data-task",0,"Exact dataset").value')==ref_value(old)
        click('science-experiment-refresh');until('!experimentControl("data-task",0,"Exact dataset").selectedOptions[0].textContent.startsWith("Unavailable") && !document.getElementById("science-experiment-refresh").disabled')
        before=h.workspace.db.serialize()
        for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
            h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
            until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            assert js('parseFloat(getComputedStyle(experimentControl("model",0,"Exact architecture")).fontSize)')>=font-2
            assert js('experimentCard("protocol",0).getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
        h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            js('document.getElementById("toast-close").click();document.getElementById("science-author-title").scrollIntoView({block:"start"});')
            loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Valid ExperimentSpec")')
        assert h.workspace.db.serialize()==before
        click('science-author-save');until('document.getElementById("science-author").hidden && document.querySelectorAll(".science-candidate").length===8')
        ref=json.loads(js('document.getElementById("science-object").value'));e['id']=ref['id']
        retained=h.workspace.get_spec('experiment',ref['id'],ref['revision'])
        for key in e:assert retained[key]==e[key],(key,retained[key],e[key])
        # No-op and undo/redo retain bytes independently of later fit history.
        click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
        before=h.workspace.db.serialize();history=copy.deepcopy(h.workspace.history.summary())
        click('science-author-save');until('document.getElementById("science-author").hidden')
        assert h.workspace.db.serialize()==before;assert h.workspace.history.summary()==history
        click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
        set_id('science-author-name','Renamed experiment');click('science-author-save');until('document.getElementById("science-author").hidden')
        assert h.dispatch({'action':'undo'},None);assert h.workspace.get_spec('experiment',ref['id'])['revision']==ref['revision']
        assert h.dispatch({'action':'redo'},None);assert h.workspace.get_spec('experiment',ref['id'])['name']=='Renamed experiment'
        set_id('science-object',ref_value(ref))
        until('document.querySelectorAll(".science-candidate").length===8 && !document.getElementById("science-author-edit").disabled')
        # Eight explicit GUI clicks, not an implicit sweep: all declared cells
        # retain exact original experiment hashes and actual split IDs.
        manifests={}
        for i in range(8):
            until('document.querySelectorAll("[data-science-run]:not(:disabled)").length===8')
            old_runs={r['id'] for r in h.workspace.list_specs('run')}
            js('document.querySelectorAll("[data-science-run]")['+str(i)+'].click()')
            until('Boolean(window.loom.state.experiment_job)')
            until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")')
            new={r['id'] for r in h.workspace.list_specs('run')}-old_runs;assert len(new)==1
            run=h.workspace.get_spec('run',new.pop());result=run['attempts'][-1]['result']
            assert run['attempts'][-1]['status']=='completed',run['attempts'][-1]
            assert run['selection']['experiment']==ref;assert result['counts']['completed_updates']==3
            assert run['resolved']['protocol']['seeds']['shuffle']==29
            assert run['resolved']['protocol']['seeds']['initialization']==run['selection']['coordinates']['initialization']
            assert run['resolved']['protocol']['optimizer']==p['optimizer']
            pair=run['selection']['coordinates']['data_task'];manifest=run['resolved']['manifest']
            if pair in manifests:assert manifest==manifests[pair]
            manifests[pair]=manifest
        # Extend the exact same experiment through typed controls: all four
        # split kinds, another training configuration, a visible negative model,
        # and a reasoned exclusion. 96 rows require four bounded draft pages.
        click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
        add('Add architecture selection');control('model',2,'Exact architecture',ref_value(h.bad))
        extras=[{'id':'group','protocol':{'method':'group','seed':5,'fractions':[.5,.25,.25],'field':'subject'}},
            {'id':'temporal','protocol':{'method':'temporal','fractions':[.5,.25,.25],'field':'time'}},
            {'id':'external','protocol':{'method':'external','roles':{'train':['row3','row1','row2','row0'],'validation':['row4','row5'],'test':['row6']},'excluded':['row7']}}]
        for split in extras:add_split(split)
        second=copy.deepcopy(e['protocols'][0]);second['id']='another-adam';second['protocol']['budget']['updates']=4;second['protocol']['shuffle']=True;second['protocol']['seeds']['shuffle']=35
        add_protocol(second);initialization_seeds([13,7])
        add('Add exclusion');control('exclusions',0,'Selected exact architecture',ref_value(e['models'][0]));control('exclusions',0,'Dataset/task pairing ID','pair-1');control('exclusions',0,'Exclusion reason','Planned comparison exclusion')
        before=h.workspace.db.serialize();click('science-experiment-preview')
        until('document.getElementById("science-experiment-preview-status").textContent.includes("of 96")')
        statuses=[]
        for page_index in range(4):
            until('document.querySelectorAll("[data-draft-candidate]").length==='+str(25 if page_index<3 else 21))
            statuses += js('[...document.querySelectorAll("[data-draft-candidate]")].map(r=>r.dataset.draftCandidate)')
            if page_index<3:click('science-experiment-next');until('document.getElementById("science-experiment-preview-status").textContent.startsWith('+json.dumps(str((page_index+1)*25)+'–')+')')
        assert len(statuses)==96;assert statuses.count('compatible')==48;assert statuses.count('incompatible')==32;assert statuses.count('excluded')==16
        assert h.workspace.db.serialize()==before;assert js('document.getElementById("science-experiment-next").disabled')
        click('science-experiment-prev');until('document.getElementById("science-experiment-preview-status").textContent.startsWith("50–")')
        # Comparison-mode changes preserve conflicts for explicit correction.
        top('Comparison mode','paired');click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("paired needs explicit pairs")')
        for i,pair in enumerate(['pair-0','pair-1']):
            add('Add pair whitelist entry');control('pairs',i,'Selected exact architecture',ref_value(e['models'][i]));control('pairs',i,'Dataset/task pairing ID',pair)
        click('science-experiment-preview');until('document.querySelectorAll("[data-draft-candidate]").length===25')
        assert js('document.querySelectorAll("[data-draft-candidate=incompatible]").length')==0
        top('Comparison mode','crossed');click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Crossed uses no pair whitelist")')
        assert js('document.querySelectorAll("[data-experiment-axis=pairs]").length')==2
        entry_button('pairs',0,'Remove entry');entry_button('pairs',0,'Remove entry')
        # External duplicate IDs and all-blank budgets reject; text is retained.
        control('split',3,'train observation IDs','row3\nrow1\nrow2\nrow0\nrow0');click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("Duplicate observation")')
        control('split',3,'train observation IDs','row3\nrow1\nrow2\nrow0')
        control('protocol',0,'Maximum updates','');click('science-author-check');until('document.getElementById("science-author-status").textContent.includes("at least one finite training budget")')
        control('protocol',0,'Maximum updates','3')
        click('science-author-save');until('document.getElementById("science-author").hidden')
        complex_ref=json.loads(js('document.getElementById("science-object").value'));full=h.workspace.get_spec('experiment',ref['id'],complex_ref['revision'])
        assert len(full['splits'])==4;assert len(full['protocols'])==2;assert full['initialization_seeds']==[13,7]
        assert full['protocols'][1]==second;assert full['splits'][1:]==extras
        # Each non-row split is also executed, with its actually authored settings.
        for method in ('group','temporal','external'):
            coords={'model':e['models'][0],'data_task':'pair-0','split':method,'initialization':13,'protocol':e['protocols'][0]['id']}
            h.dispatch({'action':'scienceTrain','reference':complex_ref,'coordinates':coords,'run_id':'authored-'+method,'name':'Authored '+method},None)
            until('Boolean(window.loom.state.experiment_job)');until('!window.loom.state.experiment_job && document.getElementById("toast-message").textContent.includes("completed")')
            run=h.workspace.get_spec('run','authored-'+method);assert run['attempts'][-1]['status']=='completed',run['attempts'][-1]
            assert run['resolved']['manifest']['method']==method
            if method=='external':assert run['resolved']['manifest']['roles']['train']==extras[-1]['protocol']['roles']['train'];assert run['resolved']['manifest']['excluded']==['row7']
        # An in-flight prospective reply cannot repopulate another workspace.
        click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
        original=h.workspace;before=original.db.serialize();replacement=Workspace();workspaces.append(replacement)
        replacement.put(original.models()[e['models'][0]['id']]);dispatch=h.dispatch
        def switch_during_preview(request,target):
            value=dispatch(request,target)
            if request['action']=='scienceDraftExperimentPreview':h.workspace=replacement;h.broadcast()
            return value
        with patch.object(h,'dispatch',side_effect=switch_during_preview):
            click('science-experiment-preview');until('document.getElementById("science-author-status").textContent.includes("Workspace changed") && !document.getElementById("science-author-cancel").disabled')
        assert js('document.querySelectorAll("[data-draft-candidate]").length')==0
        assert original.db.serialize()==before;assert replacement.list_specs('run',True)==[]
        click('science-author-cancel');h.workspace=original;h.broadcast()
        until('!document.getElementById("science-author-edit").disabled')
        click('science-author-edit');until('!document.getElementById("science-author-name").disabled')
        set_id('science-author-name','Keep this unsaved experiment')
        other=original.get_spec('experiment',ref['id']);other.pop('revision');other['name']='Concurrent revision';original.put_spec('experiment',other)
        before=original.db.serialize();click('science-author-save');until('document.getElementById("science-author-status").textContent.includes("head changed")')
        assert original.db.serialize()==before;assert js('document.getElementById("science-author-name").value')=='Keep this unsaved experiment'
        completed=h.job;assert completed.wait(15000);completed.deleteLater()
        h.job=SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,mid=None,snapshot={'run_id':'busy'});h.broadcast()
        until('document.getElementById("science-author-save").disabled')
        try:h.dispatch({'action':'scienceDraftCommit'},None)
        except ValueError as error:assert 'Stop' in str(error)
        else:raise AssertionError('Pending terminal allowed experiment admission')
        h.job=None;h.broadcast();click('science-author-cancel')
        original.save(folder/'authored.vtw');loaded=Workspace.load(folder/'authored.vtw')
        try:
            assert loaded.get_spec('experiment',ref['id'],ref['revision'])==retained
            assert loaded.get_spec('experiment',ref['id'],complex_ref['revision'])==full
            assert len(loaded.list_specs('run',True))==11
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'typed_experiment_empty_registry':'PASS','exact_defaults_optimizer_seeds':'PASS','train_free_preview':'PASS',
            'two_models_two_datasets_two_seeds_actual_eight_gui_fits':'PASS','all_split_methods_actual_fits':'PASS','crossed_paired_exclusions_incompatibles':'PASS',
            'bounded_96_row_paging':'PASS','invalid_text_exact_refresh_noop_history':'PASS','late_preview_head_workspace_pending_guards':'PASS',
            'save_reload_exact':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        if h.job and hasattr(h.job,'wait'):
            if h.job.isRunning():h.job.stop()
            h.job.wait(15000)
        for workspace in workspaces:
            if workspace.history.directory:workspace.history.directory.cleanup()
            workspace.db.close()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
