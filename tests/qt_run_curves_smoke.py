"""Actual saved-attempt SVGs, native/CLI identities, fonts and cursor failures."""
import copy
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QObject,QEvent,QEventLoop,QTimer,QUrl,QCoreApplication
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage,QWebEngineProfile
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.authoring import workspace_id
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Bridge,Window,find_torch_python
from app.experiment_objects import execute_candidate,preview_experiment
from app.run_curves import curve_view
from evidence_fixture import populate,experiment
from test_authoring import measurements

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['saved-curve-smoke']);view=QWebEngineView();view.resize(1500,1100)
profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,path):
        super().__init__();self.config=Config(path/'config.json');self.workspace=Workspace()
        self.torch_python=find_torch_python();assert self.torch_python
        self.f=populate(self.workspace,self.torch_python);self.job=None;self.rendering={};self.bridge=Bridge(self,view)
        self.curves=[];self.fail_offset=None
        # One actual bounded fit supplies >100 history points and >25 heads.
        m=default_model('Curve paging fixture',inputs=1)
        m['graph']['nodes']=[n for n in m['graph']['nodes'] if n['type']!='Output']+[{'id':'out'+str(i),'type':'Output'} for i in range(30)]
        m['graph']['edges']=[e for e in m['graph']['edges'] if e['to']!='output']+[{'from':'linear1','to':'out'+str(i)} for i in range(30)]
        task={'version':1,'id':'curves','name':'Paged objectives','inputs':[{'port':'input','field':'x','transforms':[]}],
              'heads':[{'id':'head'+str(i),'port':'out'+str(i),'field':'y','loss':'mse','weight':1.,'transforms':[]} for i in range(30)]}
        ref=experiment(self.workspace,m,measurements(),task,'curve-pages',updates=106,history_limit=200)
        doc=self.workspace.get_spec('experiment',ref['id'],ref['revision'])
        coordinates=preview_experiment(doc,self.workspace.get_spec)['candidates'][0]['coordinates']
        self.long=execute_candidate(self.workspace,ref,coordinates,'paged-curves','Paged curve fit',python=self.torch_python,timeout=25)
        assert self.long['record']['attempts'][-1]['status']=='completed'
        assert len(self.long['record']['attempts'][-1]['result']['history'])==106
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,target):
        if req['action']=='scienceRunCurve' and self.fail_offset is not None and req.get('offset')==self.fail_offset:
            self.fail_offset=None;raise ValueError('Injected point navigation failure')
        result=Window.dispatch(self,req,target)
        if req['action']=='scienceRunCurve':self.curves.append((copy.deepcopy(req),copy.deepcopy(result)))
        return result
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
def choose(ref):
    # Actual visible selection controls retain exact history; no renderer injection.
    value=json.dumps({k:ref[k] for k in ('kind','id','revision')},separators=(',',':'))
    js('(()=>{const kind=document.getElementById("science-kind");kind.value='+json.dumps(ref['kind'])+';kind.dispatchEvent(new Event("change"));})()')
    until('[...document.getElementById("science-object").options].some(o=>o.value==='+json.dumps(value)+')')
    js('(()=>{const choice=document.getElementById("science-object");choice.value='+json.dumps(value)+';choice.dispatchEvent(new Event("change"));})()')
    if ref['kind']=='run':
        until('document.getElementById("science-run-evidence").dataset.runEvidence==='+json.dumps(ref['revision'])+' && !document.getElementById("science-evidence-read").disabled')
    else:
        until('!document.getElementById("science-report").hidden && document.getElementById("science-report-page").textContent.includes("no training")')
def open_curve():
    js('document.querySelector("#science-run-evidence .science-curve").open=true')
    until('document.querySelector("#science-run-evidence .science-curve [role=status]").textContent.includes("Saved evidence only")')
def control(index):
    js('document.querySelectorAll("#science-run-evidence .science-curve button")['+str(index)+'].click()')
def settled():until('!document.querySelectorAll("#science-run-evidence .science-curve button")[2].disabled')

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);h=Harness(folder)
    page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    mid=h.f['first']['record']['selection']['coordinates']['model']['id']
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+mid);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document.view.fitted)')
        # Startup camera completes locally; no SQL Fit write is expected.
        until('Boolean(window.loom?.document?.view.fitted)')
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"850px",width:"940px",top:"110px",left:"330px",bottom:"auto"});')
        before=h.workspace.db.serialize();history=copy.deepcopy(h.workspace.history.summary());rng=random.getstate()
        with patch('app.transport.run_json_worker',side_effect=AssertionError('Saved curves never execute')):
            choose(h.f['first']['reference']);assert not h.curves
            open_curve();first=h.curves[-1][1]
            assert first==curve_view(h.workspace,h.f['first']['reference'],attempt=1)
            assert js('document.querySelectorAll("#science-run-evidence .science-curve svg").length')==1
            assert not js('Boolean(window.evidenceInjected)')
            # Numeric coordinates in the real SVG equal the exact saved values.
            xy=js('document.querySelector("#science-run-evidence .science-curve polyline.science-curve-train").getAttribute("points")')
            expected=first['points']['rows']
            actual=[tuple(map(float,p.split(','))) for p in xy.split()]
            assert len(actual)==len(expected)
            for (x,y),point in zip(actual,expected):
                assert math.isclose(x,80+650*(point['update']/first['x_range'][1]))
                assert math.isclose(y,244-210*(point['metrics']['train']['value']/first['y_range'][1]))
            js('(()=>{const select=document.querySelector("#science-run-evidence .science-curve select");select.value="regression";select.dispatchEvent(new Event("change"));})()')
            settled();assert h.curves[-1][1]['head']['id']=='regression'
            assert '<svg' in h.curves[-1][1]['source_unit']
            assert js('document.querySelectorAll("#science-run-evidence .science-curve img, #science-run-evidence .science-curve script").length')==0
            # Exact failed latest attempt stays empty; explicitly inspect attempt1.
            choose(h.f['retry']['reference']);open_curve()
            assert h.curves[-1][1]['attempt']['number']==2;assert h.curves[-1][1]['points']['rows']==[]
            assert js('document.querySelectorAll("#science-run-evidence .science-curve svg").length')==0
            js('document.getElementById("science-evidence-attempt").value="1";document.getElementById("science-evidence-read").click()')
            until('document.getElementById("science-run-evidence").textContent.includes("attempt 1 of 2") && !document.getElementById("science-evidence-read").disabled')
            open_curve();assert h.curves[-1][1]['attempt']['number']==1
            choose(h.f['zero']['reference']);open_curve()
            assert js('document.querySelectorAll("#science-run-evidence .science-curve svg").length')==0
            assert js('document.querySelector("#science-run-evidence .science-curve").textContent.includes("not a zero-valued curve")')
            choose(h.long['reference']);open_curve();assert h.curves[-1][1]['points']['total']==106
            assert len(h.curves[-1][1]['points']['rows'])==100
            axes=h.curves[-1][1]['y_range'];control(4);settled()
            assert h.curves[-1][1]['points']['offset']==100;assert len(h.curves[-1][1]['points']['rows'])==6
            assert h.curves[-1][1]['y_range']==axes
            # A failed Back does not change page or history. Refresh uses100.
            h.fail_offset=0;control(3);settled()
            assert js('document.querySelector("#science-run-evidence .science-curve [role=status]").textContent.includes("Injected point navigation failure")')
            control(2);settled();assert h.curves[-1][0]['offset']==100
            control(3);settled();assert h.curves[-1][0]['offset']==0
            control(1);settled();assert h.curves[-1][1]['heads']['offset']==25
            assert js('document.querySelector("#science-run-evidence .science-curve select").value')=='head0'
            assert js('document.querySelector("#science-run-evidence .science-curve select").textContent.includes("selected outside this page")')
            js('(()=>{const select=document.querySelector("#science-run-evidence .science-curve select");select.value="head29";select.dispatchEvent(new Event("change"));})()');settled()
            assert h.curves[-1][1]['head']['id']=='head29';assert h.curves[-1][1]['points']['offset']==0
            for theme,font,spacing in [('dark',14,1),('github',18,4),('math',28,10)]:
                h.config.values.update(theme=theme,font_size=font,spacing=spacing);h.broadcast()
                until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
                assert js('document.querySelector(".science-curve-canvas").clientWidth<=document.getElementById("science-run-evidence").clientWidth')
                effective=js('document.querySelector(".science-curve svg").getBoundingClientRect().width/760*14')
                assert effective>=font-.1,(font,effective)
                assert js('[...document.querySelectorAll(".science-curve-canvas svg text")].every(t=>{const b=t.getBoundingClientRect(),s=t.closest("svg").getBoundingClientRect();return b.left>=s.left-1 && b.right<=s.right+1 && b.top>=s.top-1 && b.bottom<=s.bottom+1;})')
                assert js('getComputedStyle(document.querySelector(".science-curve-train")).stroke!==getComputedStyle(document.querySelector(".science-curve-validation")).stroke')
            h.config.values.update(theme='github',font_size=18,spacing=4);h.broadcast()
            if os.environ.get('VTW_CURVE_SCREENSHOT'):
                js('Object.assign(document.getElementById("experiment-dock").style,{width:"1210px",left:"240px"});')
                js('document.querySelector(".science-curve-canvas").scrollIntoView({block:"start"})')
                loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_CURVE_SCREENSHOT'])
            # Actual report cards attach independent, initially collapsed curves.
            mark=len(h.curves);choose(h.f['report'])
            assert len(h.curves)==mark
            assert js('document.querySelectorAll("#science-report-evidence .science-curve").length')==5
            assert js('document.querySelectorAll("#science-report-evidence .science-curve svg").length')==0
            js('[...document.querySelectorAll("#science-report-evidence .science-curve")].forEach(d=>d.open=true)')
            until('[...document.querySelectorAll("#science-report-evidence .science-curve [role=status]")].every(s=>s.textContent.includes("Saved evidence only"))')
            assert len(h.curves)==mark+5
            assert [v['reference'] for _,v in h.curves[mark:]]==h.f['selected'][:5]
            assert h.curves[mark][1]['reference']==h.f['first']['reference']
            assert h.curves[mark][1]['attempt']['number']==1
            assert js('document.querySelectorAll("#science-report-evidence .science-curve svg").length')==2
            mark=len(h.curves);js('document.getElementById("science-report-next").click()')
            until('document.querySelectorAll("#science-report-evidence .science-curve").length===2')
            assert len(h.curves)==mark
            js('[...document.querySelectorAll("#science-report-evidence .science-curve")].forEach(d=>d.open=true)')
            until('[...document.querySelectorAll("#science-report-evidence .science-curve [role=status]")].every(s=>s.textContent.includes("Saved evidence only"))')
            assert [v['reference'] for _,v in h.curves[mark:]]==h.f['selected'][5:]
            assert js('document.querySelectorAll("#science-report-evidence .science-curve svg").length')==0
            for req,result in h.curves:
                assert result['reference']==req['reference'] and result['attempt']['number']==req['attempt']
                assert len(result['points']['rows'])<=100 and len(result['heads']['rows'])<=25
            assert h.workspace.db.serialize()==before;assert h.workspace.history.summary()==history;assert random.getstate()==rng
            for bad in ({'workspace_id':'foreign'},{'offset':True},{'head':'foreign'},{'limit':101},{'surprise':1}):
                request={'action':'scienceRunCurve','workspace_id':workspace_id(h.workspace),'reference':h.long['reference'],'attempt':1}|bad
                try:h.dispatch(request,None)
                except ValueError:pass
                else:raise AssertionError('Invalid native curve request was admitted')
            assert not errors,errors;assert 'torch' not in sys.modules
            print(json.dumps({'actual_svg_saved_value_mapping':'PASS','historical_failed_attempts':'PASS','null_not_zero':'PASS',
                '106_retained_points_30_heads_pages':'PASS','failed_back_refresh_history':'PASS','exact_native_scope_guards':'PASS',
                'seven_report_curves_exact_recipe_order_and_attempts':'PASS',
                'read_only_sql_history_rng':'PASS','dark_github_math_fonts14_18_28':'PASS','curve_reads':len(h.curves),'console_errors':errors,'host_torch':False}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
