"""Real sandboxed saved-attempt presentation and exact multi-run GUI reports."""
import copy
import json
import os
import subprocess
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
from app.authoring import workspace_id
from app.revisions import model_document
from app.run_export import predictor_source
from evidence_fixture import populate

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['saved-evidence-smoke']);view=QWebEngineView();view.resize(1500,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def dispatch(self,req,target):
        before=self.workspace.db.total_changes;result=Window.dispatch(self,req,target)
        if self.workspace.db.total_changes!=before:self.writes.append(req['action'])
        return result
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace()
        self.torch_python=find_torch_python();assert self.torch_python
        self.f=populate(self.workspace,self.torch_python);assert 'torch' not in sys.modules
        self.job=None;self.rendering={};self.bridge=Bridge(self,view);self.writes=[]
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
def value(ref):return json.dumps({k:ref[k] for k in ('kind','id','revision')},separators=(',',':'))
def choose(ref):
    js('(()=>{const kind=document.getElementById("science-kind");kind.value='+json.dumps(ref['kind'])+';kind.dispatchEvent(new Event("change"));})()')
    until('[...document.getElementById("science-object").options].some(o=>o.value==='+json.dumps(value(ref))+')')
    js('(()=>{const object=document.getElementById("science-object");object.value='+json.dumps(value(ref))+';object.dispatchEvent(new Event("change"));})()')
    if ref['kind']=='run':until('document.getElementById("science-run-evidence").dataset.runEvidence==='+json.dumps(ref['revision'])+' && !document.getElementById("science-evidence-read").disabled')
    else:until('!document.getElementById("science-report").hidden && document.getElementById("science-report-page").textContent.includes("no training")')
def text(sid):return js('document.getElementById('+json.dumps(sid)+').textContent')
def report_choices():return js('[...document.querySelectorAll("[data-report-run]")].map(c=>({id:c.dataset.reportRun,revision:c.dataset.reportRevision}))')
def remove_first():js('document.querySelector("#science-report-selection button").click()')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder);workspaces=[h.workspace];f=h.f
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'))
    mid=f['first']['record']['selection']['coordinates']['model']['id']
    frozen_before=model_document(h.workspace.models(),mid)
    retained_before=h.workspace.list_specs(all_revisions=True)
    url.setFragment('model='+mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)')
        # route() schedules Fit in an animation frame; opening the scientific
        # tab is not a barrier for that queued presentation-only updateModel.
        # Observe its actual admission before establishing a read-only baseline.
        until('Boolean(window.loom.document.view.fitted)')
        deadline=time.monotonic()+10
        while not h.workspace.models()[mid]['view'].get('fitted'):
            assert time.monotonic()<deadline,'Startup Fit was never admitted'
            loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
        assert h.writes==['updateModel'],h.writes
        assert model_document(h.workspace.models(),mid)==frozen_before
        assert h.workspace.list_specs(all_revisions=True)==retained_before
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"880px",width:"940px",top:"110px",left:"330px",bottom:"auto"});')
        before=h.workspace.db.serialize();write_mark=len(h.writes)
        with patch('app.transport.run_json_worker',side_effect=AssertionError('Browsing/comparison must never execute')):
            choose(f['first']['reference'])
            assert 'Successful updates3' in text('science-run-evidence')
            assert js('document.querySelectorAll("#science-run-evidence [data-metric-ledger]").length')==6
            assert js('document.querySelectorAll("#science-run-evidence img,#science-run-evidence svg,#science-run-evidence script").length')==0
            assert not js('Boolean(window.evidenceInjected)')
            assert '0.75' in text('science-run-evidence') and '"mask":"valid"' in text('science-run-evidence')
            assert 'transformed target space' in text('science-run-evidence')
            assert h.workspace.db.serialize()==before,h.writes[write_mark:]
            # Preview/Qt chooser use this displayed exact attempt. Source is
            # literal text, and generation/export never executes or changes SQL.
            source=predictor_source(h.workspace,f['first']['reference'])
            click('science-predictor-view')
            until('!document.getElementById("science-predictor").hidden && document.getElementById("science-predictor-code").textContent.includes("saved_run_provenance")')
            assert text('science-predictor-code')==source['code']
            assert '3 successful updates' in text('science-predictor-status')
            assert js('document.querySelectorAll("#science-predictor-code img,#science-predictor-code svg,#science-predictor-code script").length')==0
            assert h.workspace.db.serialize()==before
            for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
                h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
                until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
                assert js('parseFloat(getComputedStyle(document.getElementById("science-predictor-code")).fontSize)')>=font-2
                assert js('document.getElementById("science-predictor-code").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
            h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
            if os.environ.get('VTW_PREDICTOR_SCREENSHOT'):
                js('document.getElementById("toast-close").click();document.getElementById("science-predictor").scrollIntoView({block:"start"});')
                loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_PREDICTOR_SCREENSHOT'])
            exported=folder/'retained_predictor.py'
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(exported),'Python source (*.py)')) as chooser:
                click('science-predictor-export')
                until('!document.getElementById("science-predictor-export").disabled && document.getElementById("toast-message").textContent.includes("not executed")')
                assert chooser.call_args.args[1]=='Export retained checkpoint predictor'
            assert exported.read_text()==source['code'];assert h.workspace.db.serialize()==before
            script="import sys,json,retained_predictor as p;assert not any(m=='app' or m.startswith('app.') for m in sys.modules);assert not p.model.training;o=p.predict_records([{'a':[3.],'b':None}]);print(json.dumps({'shapes':{k:list(v.shape) for k,v in o.items()},'provenance':p.saved_run_provenance}))"
            proc=subprocess.run([h.torch_python,'-c',script],cwd=folder,capture_output=True,text=True,timeout=20)
            assert proc.returncode==0,proc.stderr
            assert json.loads(proc.stdout)=={'shapes':{'category':[1,3],'estimate':[1,1]},'provenance':source['provenance']}
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(exported),'')):
                click('science-predictor-export');until('!document.getElementById("science-predictor-export").disabled && document.getElementById("toast-message").textContent.includes("File exists")')
            assert exported.read_text()==source['code'];assert h.workspace.db.serialize()==before
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=('','')):
                click('science-predictor-export');until('!document.getElementById("science-predictor-export").disabled')
            assert h.workspace.db.serialize()==before
            click('science-report-add');assert len(report_choices())==1
            choose(f['retry']['reference'])
            assert 'No numerical result retained' in text('science-run-evidence')
            assert 'Successful updates' not in text('science-run-evidence')
            assert js('document.getElementById("science-predictor-export").disabled && document.getElementById("science-predictor-view").disabled')
            before=h.workspace.db.serialize();click('science-report-add')
            until('document.getElementById("toast-message").textContent.includes("Another revision")')
            assert report_choices()==[{'id':f['first']['reference']['id'],'revision':f['first']['reference']['revision']}]
            assert h.workspace.db.serialize()==before
            js('document.getElementById("science-evidence-attempt").value="1"');click('science-evidence-read')
            until('document.getElementById("science-run-evidence").textContent.includes("attempt 1 of 2") && !document.getElementById("science-evidence-read").disabled')
            assert 'Successful updates3' in text('science-run-evidence')
            # Unread edit must not silently change the export target to attempt 2.
            js('document.getElementById("science-evidence-attempt").value="2"')
            earlier=folder/'explicit_earlier.py'
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',return_value=(str(earlier),'')):
                click('science-predictor-export');until('!document.getElementById("science-predictor-export").disabled')
            assert earlier.read_text()==predictor_source(h.workspace,f['retry']['reference'],attempt=1)['code']
            js('document.getElementById("science-evidence-attempt").value="3"');click('science-evidence-read')
            until('document.getElementById("toast-message").textContent.includes("finite/exact range")')
            assert js('document.getElementById("science-evidence-attempt").value')=='3'
            assert h.workspace.db.serialize()==before
            js('document.getElementById("science-evidence-attempt").value="2"');click('science-evidence-read')
            until('document.getElementById("science-run-evidence").textContent.includes("attempt 2 of 2") && !document.getElementById("science-evidence-read").disabled')
            assert 'No numerical result retained' in text('science-run-evidence')
            remove_first();click('science-report-add')
            assert report_choices()[0]['revision']==f['retry']['reference']['revision']
            remove_first();choose(f['first']['reference']);click('science-report-add')
            for ref in f['selected'][1:]:choose(ref);click('science-report-add')
            assert report_choices()==[{'id':r['id'],'revision':r['revision']} for r in f['selected']]
            js('document.getElementById("science-comparison-name").value="GUI exact comparison";document.getElementById("science-comparison-question").value="Which saved facts support this comparison?"')
            before=h.workspace.db.serialize();click('science-comparison-create')
            until('document.getElementById("science-report-page").textContent.startsWith("Exact saved runs 0–5 of 7")')
            gui_ref=json.loads(js('document.getElementById("science-object").value'))
            recipe=h.workspace.get_spec('report',gui_ref['id'],gui_ref['revision'])
            assert recipe['runs']==f['selected'];assert recipe['name']=='GUI exact comparison'
            assert recipe['analysis']=={'kind':'descriptive','inferential_unit':'run'}
            assert recipe['question']=='Which saved facts support this comparison?'
            assert js('document.querySelectorAll("#science-report-evidence [data-run-evidence]").length')==5
            assert 'No metrics are pooled across runs' in text('science-report-evidence')
            assert 'Cancelled saved trial' in text('science-report-evidence') and 'No effective supervision' in text('science-report-evidence')
            assert js('document.querySelectorAll("#science-report-evidence img,#science-report-evidence svg,#science-report-evidence script").length')==0
            before=h.workspace.db.serialize();click('science-report-next')
            until('document.getElementById("science-report-page").textContent.startsWith("Exact saved runs 5–7 of 7")')
            assert js('document.querySelectorAll("#science-report-evidence [data-run-evidence]").length')==2
            assert js('document.getElementById("science-report-next").disabled');assert h.workspace.db.serialize()==before
            click('science-report-prev');until('document.getElementById("science-report-page").textContent.startsWith("Exact saved runs 0–5 of 7")')
            # Drill down pins the report row, not the mutable retry head.
            js('document.querySelector("#science-report-evidence button").click()')
            until('document.getElementById("science-run-evidence").dataset.runEvidence==='+json.dumps(f['first']['reference']['revision'])+' && !document.getElementById("science-evidence-read").disabled')
            assert json.loads(js('document.getElementById("science-object").value'))==f['first']['reference']
            assert h.dispatch({'action':'undo'},None);assert h.dispatch({'action':'redo'},None)
            assert h.workspace.get_spec('report',gui_ref['id'],gui_ref['revision'])==recipe
            choose(f['many']['reference']);assert 'of 27' in text('science-run-evidence')
            assert js('document.querySelectorAll("#science-run-evidence [data-metric-ledger]").length')==25
            click('science-metrics-next');until('document.getElementById("science-run-evidence").textContent.includes("Metric rows 25–27 of 27") && !document.getElementById("science-evidence-read").disabled')
            assert js('document.querySelectorAll("#science-run-evidence [data-metric-ledger]").length')==2
            assert js('document.getElementById("science-metrics-next").disabled')
            click('science-metrics-prev');until('document.getElementById("science-run-evidence").textContent.includes("Metric rows 0–25 of 27") && !document.getElementById("science-evidence-read").disabled')
            for key,expected in [('cancelled','cancelled'),('zero','unavailable'),('excluded','Explicit negative control')]:
                choose(f[key]['reference']);assert expected in text('science-run-evidence')
                if key=='zero':assert 'Unavailable' in text('science-run-evidence')
            choose(f['first']['reference'])
            for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
                h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
                until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
                assert js('parseFloat(getComputedStyle(document.querySelector("#science-run-evidence td")).fontSize)')>=font-2
                assert js('document.getElementById("science-run-evidence").getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
                assert js('[...document.querySelectorAll("#science-run-evidence td,#science-run-evidence th")].every(c=>getComputedStyle(c).whiteSpace==="nowrap")')
                assert js('[...document.querySelectorAll("#science-run-evidence .science-evidence-table")].every(t=>t.clientWidth<=document.getElementById("science-run-evidence").clientWidth)')
            h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
            if os.environ.get('VTW_SMOKE_SCREENSHOT'):
                js('document.getElementById("toast-close").click();document.getElementById("science-run-evidence").scrollIntoView({block:"start"});')
                loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
            # Central guard remains stronger than visual controls.
            h.job=SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,mid=None,snapshot={'run_id':'busy'});h.broadcast()
            until('document.getElementById("science-comparison-create").disabled')
            before=h.workspace.db.serialize()
            assert js('document.getElementById("science-predictor-export").disabled && !document.getElementById("science-predictor-view").disabled')
            assert h.dispatch({'action':'scienceRunPredictor','reference':f['first']['reference'],
                'attempt':1,'workspace_id':workspace_id(h.workspace)},None)['code']==source['code']
            try:h.dispatch({'action':'scienceExportRunPredictor'},None)
            except ValueError as error:assert 'Stop' in str(error)
            else:raise AssertionError('Pending delivery admitted predictor export')
            try:h.dispatch({'action':'scienceResultReport'},None)
            except ValueError as error:assert 'Stop' in str(error)
            else:raise AssertionError('Pending delivery admitted a report')
            assert h.workspace.db.serialize()==before
            h.job=None;h.broadcast()
            # Workspace UUID gate rejects before dependency resolution/insertion.
            replacement=Workspace();replacement.put(default_model('Replacement'));workspaces.append(replacement)
            original=h.workspace;before=original.db.serialize()
            try:h.dispatch({'action':'scienceResultReport','workspace_id':workspace_id(replacement)},None)
            except ValueError as error:assert 'another workspace' in str(error)
            else:raise AssertionError('Foreign report selection admitted')
            assert original.db.serialize()==before
            dispatch=h.dispatch
            # Native modal guard checks the actual working copy again after a
            # chooser, before creating any destination (not just its old UUID).
            denied=folder/'not_exported.py'
            def replace_in_chooser(*args,**kwargs):h.workspace=replacement;h.broadcast();return str(denied),''
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',side_effect=replace_in_chooser):
                try:dispatch({'action':'scienceExportRunPredictor','reference':f['first']['reference'],
                    'attempt':1,'workspace_id':workspace_id(original)},None)
                except ValueError as error:assert 'Nothing exported' in str(error)
                else:raise AssertionError('Changed workspace exported origin predictor')
            assert not denied.exists();assert original.db.serialize()==before
            h.workspace=original;h.broadcast();choose(f['first']['reference'])
            def become_busy_in_chooser(*args,**kwargs):
                h.job=SimpleNamespace(isRunning=lambda:False,awaiting_terminal=True,mid=None,snapshot={'run_id':'modal-busy'})
                h.broadcast();return str(denied),''
            with patch('app.desktop_experiments.QFileDialog.getSaveFileName',side_effect=become_busy_in_chooser):
                try:dispatch({'action':'scienceExportRunPredictor','reference':f['first']['reference'],
                    'attempt':1,'workspace_id':workspace_id(original)},None)
                except ValueError as error:assert 'Nothing exported' in str(error)
                else:raise AssertionError('Busy modal loop exported source')
            assert not denied.exists();assert original.db.serialize()==before
            h.job=None;h.broadcast()
            def switch_on_source(req,target):
                result=dispatch(req,target)
                if req['action']=='scienceRunPredictor':h.workspace=replacement;h.broadcast()
                return result
            with patch.object(h,'dispatch',side_effect=switch_on_source):
                click('science-predictor-view');until('document.getElementById("science-run").hidden')
            assert js('document.getElementById("science-predictor-code").textContent')==''
            assert js('document.getElementById("science-predictor").hidden');assert original.db.serialize()==before
            h.workspace=original;h.broadcast();choose(f['first']['reference'])
            def switch_on_read(req,target):
                result=dispatch(req,target)
                if req['action']=='scienceRunEvidence':h.workspace=replacement;h.broadcast()
                return result
            with patch.object(h,'dispatch',side_effect=switch_on_read):
                click('science-evidence-read');until('document.querySelectorAll("[data-report-run]").length===0 && document.getElementById("science-run").hidden')
            assert original.db.serialize()==before
            assert js('document.getElementById("science-run-evidence").childElementCount')==0
            h.workspace=original;h.broadcast();choose(f['first']['reference']);click('science-report-add')
            def switch_on_create(req,target):
                result=dispatch(req,target)
                if req['action']=='scienceResultReport':h.workspace=replacement;h.broadcast()
                return result
            previous=len(original.list_specs('report',True))
            with patch.object(h,'dispatch',side_effect=switch_on_create):
                click('science-comparison-create');until('document.querySelectorAll("[data-report-run]").length===0 && document.getElementById("science-report").hidden')
            assert len(original.list_specs('report',True))==previous+1
            assert replacement.list_specs('report',True)==[]
            assert not js('document.getElementById("science-object").value')
            h.workspace=original;h.broadcast()
        original.save(folder/'evidence.vtw');loaded=Workspace.load(folder/'evidence.vtw')
        try:assert loaded.get_spec('report',gui_ref['id'],gui_ref['revision'])==recipe
        finally:loaded.db.close()
        assert not errors,errors;assert 'torch' not in sys.modules
        print(json.dumps({'actual_saved_weighted_metrics':'PASS','historical_attempt_no_substitution':'PASS',
            'completed_failed_cancelled_masked_excluded':'PASS','metric_and_report_paging':'PASS',
            'explicit_seven_run_recipe_no_training':'PASS','exact_duplicate_retry_and_drilldown':'PASS',
            'safe_literal_labels_fonts_themes':'PASS','history_reload_pending_workspace_late_reply_guards':'PASS',
            'predictor_source_preview_chooser_exact_displayed_attempt':'PASS',
            'standalone_export_actual_inference_no_app_import':'PASS','predictor_no_overwrite_cancel_busy_modal_late_guards':'PASS',
            'console_errors':errors,'host_torch':False}))
    finally:
        h.job=None
        for workspace in workspaces:
            if workspace.history.directory:workspace.history.directory.cleanup()
            workspace.db.close()
        view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
