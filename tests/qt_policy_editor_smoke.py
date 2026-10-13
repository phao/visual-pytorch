"""Actual sandboxed WebEngine policy controls, badges, history and private VTW."""
import json
import copy
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock
from PyQt6.QtCore import QObject,QEvent,QCoreApplication,QEventLoop,QTimer,QUrl,QPoint,Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.core import Workspace
from app.config import Config
from app.desktop import Window,Bridge
from app.policy_graph import plan
from policy_runtime_fixture import models as fixture

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['policy-ui-smoke']);view=QWebEngineView();view.resize(1500,1000)
profile=QWebEngineProfile(view);errors=[]
output=Path(sys.argv[1]).resolve();output.mkdir(parents=True,exist_ok=True)
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}:{message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe;dispatch=Window.dispatch
    def __init__(self,folder):
        super().__init__();self.config=Config(folder/'config.json');self.config.save=Mock(side_effect=AssertionError('No config writes'))
        self.workspace=Workspace();models,mid=fixture()
        positions={'x':(30,130),'base':(240,60),'A':(240,240),'B':(440,240),'scale':(630,240),'add':(830,130),'y':(1010,130)}
        for n in models[mid]['graph']['nodes']:n.update(x=positions[n['id']][0],y=positions[n['id']][1])
        self.workspace.put(models[mid]);self.workspace.save(folder/'initial.vtw')
        self.job=None;self.rendering={};self.torch_python='';self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kwargs):self.bridge.send(kind,self.state() if kind=='state' else payload)
def js(source):
    values=[];loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit)
    view.page().runJavaScript(source,lambda v:(values.append(v),loop.quit()));timer.start(5000);loop.exec();timer.stop()
    assert values,'JS callback timeout';return values[0]
def until(source):
    deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        assert not errors,errors
        if js(source):return
        QTest.qWait(25)
    raise AssertionError('UI timeout: '+source)
def click(selector):
    until('Boolean(document.querySelector('+json.dumps(selector)+')) && !document.querySelector('+json.dumps(selector)+').disabled')
    js('document.querySelector('+json.dumps(selector)+').scrollIntoView({block:"nearest"})')
    QTest.qWait(100)
    point=js('(()=>{const r=document.querySelector('+json.dumps(selector)+').getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,QPoint(round(point[0]),round(point[1])))
def inspect(nid):js('document.querySelector(".node[data-id='+nid+']").dispatchEvent(new MouseEvent("dblclick",{bubbles:true}))')

with tempfile.TemporaryDirectory() as folder:
    h=Harness(Path(folder));retained_source=None
    if len(sys.argv)>2:
        from policy_runtime_fixture import bundle
        from evidence_fixture import experiment
        from app.experiment_objects import execute_candidate, preview_experiment
        from app.parameter_sources import inventory,read
        b=bundle(updates=3); source_models,_=fixture()
        source_models['lora']['graph']['parameter_policy']['parameters']['["base","weight"]']['initializer']['values']=[
            [.6 if i==j else -0. for j in range(16)] for i in range(16)]
        source_data=copy.deepcopy(b['dataset']);source_data.pop('revision',None)
        source_task=copy.deepcopy(b['task']);source_task.pop('revision',None)
        ref=experiment(h.workspace,copy.deepcopy(source_models['lora']),source_data,source_task,'retained-policy-source',updates=3)
        exp=h.workspace.get_spec('experiment',ref['id'],ref['revision']);coordinates=preview_experiment(exp,h.workspace.get_spec)['candidates'][0]['coordinates']
        retained=execute_candidate(h.workspace,ref,coordinates,'policy-source-run','Retained policy source',python=sys.argv[2],timeout=20)
        assert retained['record']['attempts'][0]['status']=='completed',retained
        rows=inventory(h.workspace,retained['reference'],attempt=1);tensor=next(p for p in rows['parameters'] if p['path']=='layer_1.weight')
        retained_source={'run':rows['run'],'attempt':1,'checkpoint_id':rows['checkpoint_id'],'parameter':{k:tensor[k] for k in ('path','id')}}
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).parents[1]/'app/web/index.html'));url.setFragment('model=lora');view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document?.view.fitted)')
        inspect('base');until('document.querySelectorAll(".policy-owner").length===1')
        until('document.querySelector(".node[data-id=base]").dataset.policyStatus==="Frozen"')
        assert js('document.getElementById("parameter-policy-editor").textContent.includes("Frozen")')
        until('!document.getElementById("inspector-panel").hidden && document.getElementById("validation-message").textContent.includes("Valid")')
        js('document.querySelector(".policy-owner").scrollIntoView({block:"center"})')
        QTest.qWait(250)
        assert view.grab().save(str(output/'01-frozen-base-inspector.png'))
        click('.policy-owner input[type=checkbox]')
        until('window.loom.document.graph.parameter_policy.parameters[\'["base","weight"]\'].trainable===true')
        until('document.querySelector(".node[data-id=base]").dataset.policyStatus==="Trainable"')
        assert plan(h.workspace.models(),'lora')['resolved']['counts']['trainable']==320
        assert len(h.workspace.history.undo_stack)==1
        h.dispatch({'action':'undo'},h.bridge)
        until('document.querySelector(".node[data-id=base]").dataset.policyStatus==="Frozen"')
        h.dispatch({'action':'redo'},h.bridge)
        until('document.querySelector(".node[data-id=base]").dataset.policyStatus==="Trainable"')
        inspect('A');until('document.querySelector(".policy-owner-title")?.textContent.startsWith("A")')
        click('.policy-owner summary');until('Boolean(document.querySelector("[data-policy-field=std]"))')
        std=js('document.querySelector("[data-policy-field=std]").value');assert float(std)==.05
        js('document.querySelector("[data-policy-field=std]").value="0.02"')
        click('.policy-owner details button')
        until('window.loom.document.graph.parameter_policy.parameters[\'["A","weight"]\'].initializer.std===0.02')
        inspect('A');until('document.querySelector(".policy-owner-title")?.textContent.startsWith("A")')
        click('.policy-owner summary');js('document.querySelector(".policy-owner").scrollIntoView({block:"start"})');QTest.qWait(250);assert view.grab().save(str(output/'02-adapter-initializer.png'))
        raw=h.workspace.db.serialize();count=len(h.workspace.history.undo_stack)
        js('document.querySelector("[data-policy-field=std]").value="0"');click('.policy-owner details button')
        until('document.querySelector("[data-policy-owner-error]")?.textContent.includes("positive")')
        assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
        click('[data-policy-reset-block]')
        until('!Object.hasOwn(window.loom.document.graph.parameter_policy.parameters,\'["A","weight"]\')')
        assert len(h.workspace.history.undo_stack)==count+1
        h.dispatch({'action':'undo'},h.bridge)
        until('window.loom.document.graph.parameter_policy.parameters[\'["A","weight"]\']?.initializer.std===0.02')
        if retained_source:
            inspect('A');until('document.querySelector(".policy-owner-title")?.textContent.startsWith("A")')
            click('[data-parameter-source] summary')
            until('document.querySelectorAll("[data-binding-run] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-run]");e.value='+json.dumps(json.dumps(retained_source['run'],separators=(',',':')))+';e.dispatchEvent(new Event("change"));document.querySelector("[data-binding-attempt]").value="1"})()')
            click('[data-binding-read]');until('document.querySelectorAll("[data-binding-tensor] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-tensor]");const o=[...e.options].find(o=>o.textContent.startsWith("layer_1.weight "));e.value=o.value;e.dispatchEvent(new Event("change"))})()')
            before=len(h.workspace.history.undo_stack);click('[data-binding-apply]')
            until('window.loom.document.graph.parameter_policy.parameters[\'["A","weight"]\']?.initializer.kind==="values"')
            assert len(h.workspace.history.undo_stack)==before+1
            copied=h.workspace.models()['lora']['graph']['parameter_policy']['parameters']['["A","weight"]']['initializer']['values']
            assert copied==read(h.workspace,retained_source)['values']
            refs=h.workspace.list_specs('parameter-binding');assert len(refs)==1
            receipt=h.workspace.get_spec('parameter-binding',refs[0]['id'],refs[0]['revision'])
            assert receipt['source']['run']==retained_source['run'] and receipt['optimizer_resume'] is False
            assert plan(h.workspace.models(),'lora')['resolved']['owners']['["lora","A","weight"]']['trainable']
            h.dispatch({'action':'undo'},h.bridge)
            until('window.loom.document.graph.parameter_policy.parameters[\'["A","weight"]\']?.initializer.std===0.02')
            assert not h.workspace.list_specs('parameter-binding')
            assert h.workspace.get_spec('parameter-binding',refs[0]['id'],refs[0]['revision'])==receipt
            h.dispatch({'action':'redo'},h.bridge)
            until('window.loom.document.graph.parameter_policy.parameters[\'["A","weight"]\']?.initializer.kind==="values"')
            inspect('A');until('document.querySelector(".policy-owner-title")?.textContent.startsWith("A")')
            js('document.querySelector(".policy-owner").scrollIntoView({block:"start"})');QTest.qWait(250)
            assert view.grab().save(str(output/'03-bound-retained-tensor.png'))
            # A backend race/refusal must survive Inspector rebuilding.
            click('[data-parameter-source] summary');until('document.querySelectorAll("[data-binding-run] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-run]");e.value='+json.dumps(json.dumps(retained_source['run'],separators=(',',':')))+';e.dispatchEvent(new Event("change"));document.querySelector("[data-binding-attempt]").value="1"})()')
            click('[data-binding-read]');until('document.querySelectorAll("[data-binding-tensor] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-tensor]");e.value=[...e.options].find(o=>o.textContent.startsWith("layer_1.weight ")).value;e.dispatchEvent(new Event("change"))})()')
            original=copy.deepcopy(h.workspace.models()['lora']);changed=copy.deepcopy(original)
            changed['graph']['parameter_policy']['parameters']['["A","weight"]']['initializer']={'kind':'zeros'};h.workspace.put(changed)
            raw=h.workspace.db.serialize();count=len(h.workspace.history.undo_stack);click('[data-binding-apply]')
            until('!document.getElementById("toast").hidden && document.getElementById("toast-message").textContent.includes("Architecture changed")')
            assert h.workspace.db.serialize()==raw and len(h.workspace.history.undo_stack)==count
            h.workspace.put(original);h.broadcast()
            inspect('base');until('document.querySelector(".policy-owner-title")?.textContent.startsWith("base")')
            click('[data-parameter-source] summary');until('document.querySelectorAll("[data-binding-run] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-run]");e.value='+json.dumps(json.dumps(retained_source['run'],separators=(',',':')))+';e.dispatchEvent(new Event("change"));document.querySelector("[data-binding-attempt]").value="1"})()')
            click('[data-binding-read]');until('document.querySelectorAll("[data-binding-tensor] option").length>1')
            js('(()=>{const e=document.querySelector("[data-binding-tensor]");e.value=[...e.options].find(o=>o.textContent.startsWith("layer_0.weight ")).value;e.dispatchEvent(new Event("change"))})()')
            click('[data-binding-apply]')
            until('Object.is(window.loom.document.graph.parameter_policy.parameters[\'["base","weight"]\'].initializer.values[0][1],-0)')
            # Actual native geometry save must not recode authoritative policy floats/zeros.
            before_x=next(n for n in h.workspace.models()['lora']['graph']['nodes'] if n['id']=='base')['x']
            point=js('(()=>{const r=document.querySelector(".node[data-id=base] .node-title").getBoundingClientRect();return [r.x+r.width/2,r.y+r.height/2]})()')
            target=view.focusProxy() or view;start=QPoint(round(point[0]),round(point[1]));end=start+QPoint(30,15)
            QTest.mousePress(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,start);QTest.mouseMove(target,end,150);QTest.mouseRelease(target,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,end)
            deadline=time.monotonic()+10
            while next(n for n in h.workspace.models()['lora']['graph']['nodes'] if n['id']=='base')['x']==before_x and time.monotonic()<deadline:QTest.qWait(25)
            assert next(n for n in h.workspace.models()['lora']['graph']['nodes'] if n['id']=='base')['x']!=before_x
            import struct
            value=h.workspace.models()['lora']['graph']['parameter_policy']['parameters']['["base","weight"]']['initializer']['values'][0][1]
            assert struct.pack('<f',value)==b'\0\0\0\x80'
            from app.parameter_binding import current_bindings
            assert set(current_bindings(h.workspace,'lora'))=={'["base","weight"]','["A","weight"]'}
            inspect('base')
            until('document.querySelector(".policy-owner-title")?.textContent.startsWith("base") && document.getElementById("parameter-policy-editor").textContent.includes("Pinned tensor source")')
            js('document.querySelector(".policy-owner")?.scrollIntoView({block:"start"})');QTest.qWait(250)
            assert view.grab().save(str(output/'04-signed-zero-source.png'))
        h.workspace.save(Path(folder)/'saved.vtw');reopened=Workspace.load(Path(folder)/'saved.vtw')
        try:assert reopened.models()['lora']['graph']['parameter_policy']==h.workspace.models()['lora']['graph']['parameter_policy']
        finally:reopened.db.close()
        h.config.save.assert_not_called();assert not errors,errors
        assert 'torch' not in sys.modules and 'numpy' not in sys.modules
        print(json.dumps({'native_mouse_policy_controls':True,'badges':True,'undo_redo':True,
            'invalid_initializer_no_mutation':True,'atomic_block_reset_undo':True,'save_reload':True,'sandbox':True,
            'host_torch':False,'user_config_untouched':True,'captures':4 if retained_source else 2,'retained_source_binding':bool(retained_source),'signed_zero_geometry_save':bool(retained_source),'backend_refusal_notice':bool(retained_source),'console_errors':errors}))
    finally:
        h.workspace.db.close()
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        view.close();page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
