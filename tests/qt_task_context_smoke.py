"""Real paged task suggestions, large hidden vocabularies, no user state/fits."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from PyQt6.QtCore import QObject,QEventLoop,QTimer,QUrl,QCoreApplication,QEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEngineProfile,QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Bridge,Window
from app.science_browser import encoded

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['task-context-smoke']);view=QWebEngineView();view.resize(1500,1100);profile=QWebEngineProfile(view);errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(message)
class Harness(QObject):
    state=Window.state;describe=Window.describe
    def __init__(self,folder):
        super().__init__();self.workspace=Workspace();self.config=Config(folder/'config.json');self.job=None;self.torch_python=None;self.rendering={};self.actions=[];self.projections=[];self.fail_page=False
        self.model=default_model('Many exact task ports');nodes=[];edges=[]
        for i in range(32):
            nodes.extend([{'id':'i'+str(i),'type':'Input','label':'Input '+str(i),'features':1,'x':i*20,'y':0},
                {'id':'o'+str(i),'type':'Output','label':'Output '+str(i),'x':i*20,'y':200}]);edges.append({'from':'i'+str(i),'to':'o'+str(i)})
        self.model['graph']={'nodes':nodes,'edges':edges};self.workspace.put(self.model);self.mref=self.workspace.freeze_model(self.model['id'])
        self.dataset={'version':1,'id':'many-fields','name':'Many exact fields','storage':{'kind':'inline-records','version':1},
            'fields':[{'id':'f'+str(i),'name':'Feature '+str(i),'dtype':'float32','shape':[]} for i in range(600)],
            'records':[{'id':'row'+str(i),'values':{'f'+str(j):float(i+j) for j in range(600)}} for i in range(2)]}
        self.dataset['fields'][0].update(dtype='category',vocabulary={'version':1,'values':['class'+str(i) for i in range(20000)]},unit={'long':'λ😀'*20000})
        for record in self.dataset['records']:record['values']['f0']='class0'
        self.dataset['fields'][1].update(shape=[1]*8,axes=['axis'+str(i) for i in range(8)])
        self.smp_id='😀'*198;self.dataset['fields'][-1]['id']=self.smp_id
        for record in self.dataset['records']:
            value=record['values']['f1']
            for _ in range(8):value=[value]
            record['values']['f1']=value;record['values'][self.smp_id]=record['values'].pop('f599')
        self.old=self.workspace.put_spec('dataset',self.dataset);new=copy.deepcopy(self.dataset);new['name']='New browsing head';self.new=self.workspace.put_spec('dataset',new)
        self.bridge=Bridge(self,view)
    def recent_files(self):return []
    def broadcast(self,kind='state',payload=None,**kw):self.bridge.send(kind,self.state() if kind=='state' else payload)
    def dispatch(self,req,source=None):
        self.actions.append(req['action'])
        if req['action']=='scienceTaskContextPage' and self.fail_page:self.fail_page=False;raise ValueError('Fixture: transient context page failure')
        value=Window.dispatch(self,req,source)
        if req['action']=='scienceTaskContextPage':assert len(encoded(value))<=65536;self.projections.append(value)
        return value
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()));timer.start(5000);loop.exec();timer.stop();assert values,'JS timeout';return values[0]
def until(condition):
    end=time.monotonic()+30
    while time.monotonic()<end:
        if js(condition):return
        loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    raise AssertionError(condition+'; '+str(errors)+'; '+str(js('document.getElementById("science-task-context-status").textContent')))
def click(sid):until('!document.getElementById('+json.dumps(sid)+').disabled');js('document.getElementById('+json.dumps(sid)+').click()')
def group(kind):return 'document.querySelector("[data-context-collection='+kind+']")'
def control(kind,tag,index):return group(kind)+'.querySelectorAll("'+tag+'")[ '+str(index)+' ]'
def search(kind,query,pin=''):
    js('(()=>{const box='+group(kind)+';const inputs=box.querySelectorAll("input");inputs[0].value='+json.dumps(query)+';inputs[1].value='+json.dumps(pin)+';box.querySelector("button").click()})()')
    until('!'+control(kind,'button',0)+'.disabled')

with tempfile.TemporaryDirectory() as folder:
    h=Harness(Path(folder));page=Page(profile,view);view.setPage(page);channel=QWebChannel(page);channel.registerObject('host',h.bridge);page.setWebChannel(channel)
    url=QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html'));url.setFragment('model='+h.model['id']);view.setUrl(url);view.show()
    try:
        until('Boolean(window.loom?.document)');until('Boolean(window.loom.document.view?.fitted)')
        js('document.querySelector("[data-panel=science]").click();Object.assign(document.getElementById("experiment-dock").style,{height:"880px",width:"1000px",top:"120px",left:"230px",bottom:"auto"})')
        before=h.workspace.db.serialize();click('science-author-new-task');until('!document.getElementById("science-task-dataset").disabled && document.getElementById("science-task-dataset").options.length===3')
        js('document.getElementById("science-task-dataset").value=JSON.stringify('+json.dumps(h.old)+');document.getElementById("science-task-dataset").dispatchEvent(new Event("change"))')
        until('!'+control('fields','button',0)+'.disabled && document.getElementById("science-task-context-status").textContent.startsWith("Exact context")')
        for sid in ('science-task-fields','science-task-input-ports','science-task-output-ports'):assert js('document.getElementById('+json.dumps(sid)+').children.length')<=25
        assert h.projections[-1]['dataset']==h.old
        js('[...document.querySelectorAll("#science-task-bindings button")].find(n=>n.textContent==="Add input binding").click()')
        until('Boolean(document.querySelector("[data-task-input]")) && !document.getElementById("science-author-check").disabled')
        js('(()=>{const n=document.querySelector("[data-task-input] input[list=science-task-fields]");n.value="unfinished manual field";n.dispatchEvent(new Event("change"))})()')
        search('fields','Feature 599','f0');assert js('document.getElementById("science-task-fields").children.length')==2
        assert '20000 fixed categories' in js('document.querySelector("#science-task-fields option[value=f0]").title')
        search('fields','Feature 599','f1');assert 'preview, rank 8' in js('document.querySelector("#science-task-fields option[value=f1]").textContent')
        search('fields','Feature 599',h.smp_id)
        assert js('document.querySelector("#science-task-fields option").value')==h.smp_id
        assert js(control('fields','input',1)+'.maxLength')==400
        search('fields','Feature 599','missing-field');assert 'ID unavailable: missing-field' in js(group('fields')+'.querySelector("p").textContent')
        assert js('document.querySelector("[data-task-input] input[list=science-task-fields]").value')=='unfinished manual field'
        js('[...document.querySelectorAll("#science-task-bindings button")].find(n=>n.textContent==="Add loss head").click()')
        until('Boolean(document.querySelector("[data-task-head]")) && !document.getElementById("science-author-check").disabled')
        js('(()=>{const label=[...document.querySelectorAll("[data-task-head] label")].find(n=>n.firstChild.textContent==="Objective weight"),n=label.querySelector("input");n.value="not-a-number";n.dispatchEvent(new Event("change"))})()')
        search('fields','Feature 599','f0')
        assert js('document.querySelector("[data-task-head] [aria-invalid=true]").value')=='not-a-number'
        assert h.workspace.db.serialize()==before
        # Port navigation does not touch field search or manual bindings.
        js(control('input_ports','button',2)+'.click()');until('!'+control('input_ports','button',0)+'.disabled');assert h.projections[-1]['offset']==25
        h.fail_page=True;js(control('input_ports','button',1)+'.click()');until('!'+control('input_ports','button',0)+'.disabled');assert not js(control('input_ports','button',1)+'.disabled')
        js(control('input_ports','button',1)+'.click()');until('!'+control('input_ports','button',0)+'.disabled');assert h.projections[-1]['offset']==0
        assert js(control('fields','input',0)+'.value')=='Feature 599'
        request={'action':'scienceTaskContextPage','workspace_id':h.state()['workspace_id'],'model':h.mref,'dataset':h.old,'collection':'fields'}
        for change in ({'extra':True},{'workspace_id':'wrong'},{'limit':False}):
            try:h.dispatch({**request,**change},h.bridge)
            except ValueError:pass
            else:raise AssertionError('Invalid context page request accepted')
        for theme,font in [('dark',14),('github',18),('math',28)]:
            h.config.values.update(theme=theme,font_size=font);h.broadcast();until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==='+json.dumps(str(font)+'px'))
            js(group('fields')+'.open=true;'+group('fields')+'.scrollIntoView({block:"start"})')
            assert js('parseFloat(getComputedStyle('+control('fields','input',0)+').fontSize)')>=font-2
            assert js(group('fields')+'.getBoundingClientRect().width<=document.getElementById("science-panel").getBoundingClientRect().width')
            search('fields','Feature 599','λ'*200)
            assert js(group('fields')+'.querySelector("p").scrollWidth<='+group('fields')+'.querySelector("p").clientWidth+1')
        if os.environ.get('VTW_SMOKE_SCREENSHOT'):
            h.config.values.update(theme='github',font_size=18);h.broadcast();until('getComputedStyle(document.body).getPropertyValue("--ui-font-size").trim()==="18px"')
            search('fields','Feature 599','missing-field')
            js('document.getElementById("toast-close").click()');loop=QEventLoop();QTimer.singleShot(300,loop.quit);loop.exec();assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
        assert 'scienceTaskContext' not in h.actions and 'scienceIndex' not in h.actions
        assert all('records' not in value and 'vocabulary' not in row for value in h.projections for row in value['rows'])
        assert h.workspace.db.serialize()==before and 'torch' not in sys.modules and not errors,errors
        print(json.dumps({'bounded_actual_context':'PASS','ports':32,'fields':600,'hidden_vocabulary':20000,'historical_exact':'PASS','manual_text':'PASS',
            'independent_search_pin_paging_retry':'PASS','closed_requests':'PASS','fonts':'PASS','read_only':'PASS','projections':len(h.projections),
            'max_bytes':max(len(encoded(p)) for p in h.projections),'host_torch':False,'console_errors':errors}))
    finally:
        if h.workspace.history.directory:h.workspace.history.directory.cleanup()
        h.workspace.db.close();view.close();view.setPage(QWebEnginePage(view));page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        profile.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
