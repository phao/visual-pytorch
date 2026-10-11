"""Actual native window/WebEngine header access under zoom/font/viewport changes.

Only private configuration, recent-file store and workspace. Chromium sandbox on.
No screenshot alone substitutes for hit tests, focus access or immutable SQL.
"""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
from PyQt6.QtCore import QEvent,QEventLoop,QTimer,QCoreApplication,Qt,QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from app.config import Config
from app.core import Workspace,default_model
from app.desktop import Window

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['window-geometry-smoke'])
errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:errors.append(f'{source}:{line}: {message}')
class Settings:
    def __init__(self,*args):self.data={}
    def value(self,key,default=None,**kwargs):return self.data.get(key,default)
    def setValue(self,key,value):self.data[key]=value

def js(view,script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda result:(values.append(result),loop.quit()))
    timer.start(4000);loop.exec();timer.stop();assert values,'JavaScript callback timed out';return values[0]
def until(view,script):
    deadline=time.monotonic()+15
    while time.monotonic()<deadline:
        if js(view,script):return
        QTest.qWait(20)
    raise AssertionError('UI timeout: '+script)

with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);config=Config(folder/'preferences.json');config.save()
    with patch('app.desktop.Config',return_value=config),patch('app.desktop.QSettings',Settings),patch('app.desktop.QWebEnginePage',Page):
        window=Window()
    window.file_timer.stop();window.resize(1480,960);window.show()
    try:
        view=window.tabs.widget(0)
        until(view,'Boolean(window.loom)')
        window.workspace=Workspace()
        model=default_model(('A long architecture name / '+('projection-'*18))[:100])
        window.workspace.put(model);window.workspace.save(folder/('a-long-workspace-name-'*5+'.vtw'))
        window.broadcast();view.bridge.send('navigate',model['id'])
        until(view,'Boolean(window.loom?.document?.view.fitted)')
        before=window.workspace.db.serialize();disk=Path(window.workspace.path).read_bytes()
        history=copy.deepcopy(window.workspace.history.summary());config_bytes=config.path.read_bytes()
        samples=[];failures=[]
        # Common displays and accessibility extremes, not every OS/font engine.
        settings=[(width,height,font,zoom,spacing,theme,tab_count)
                  for tab_count in (1,2)
                  for width,height in ((1480,960),(1000,700))
                  for font,zoom,spacing in ((14,1.15,4),(18,1.35,4),(28,1.35,10),(28,2,1),(28,3,10))
                  for theme in ('dark','github','math')]
        controls=['home','back','forward','new-tab','rename-model','fit','zoom-out','zoom-in',
                  'validate','reset-panels','constrained-mode','open-workspace','save-workspace']
        for width,height,font,zoom,spacing,theme,tab_count in settings:
            if window.tabs.count()!=tab_count:
                with patch('app.desktop.QWebEnginePage',Page):window.open_tab(model['id'])
                view=window.current_view();until(view,'Boolean(window.loom?.document?.view.fitted)')
            assert window.tabs.tabBar().isVisible()==(tab_count>1)
            js(view,"""if(!window.__geometryClicks){window.__geometryClicks=0;
                document.getElementById('validate').addEventListener('click',()=>window.__geometryClicks++);}""")
            window.resize(width,height);window.set_display(persist=False,font_size=font,ui_zoom=zoom,spacing=spacing,theme=theme)
            until(view,f'getComputedStyle(document.body).fontSize=="{font}px" && document.documentElement.dataset.theme=={json.dumps(theme)}')
            QTest.qWait(50)
            geometry=js(view,"""(() => {
                const r=e=>{const b=e.getBoundingClientRect();return {x:b.x,y:b.y,width:b.width,height:b.height,right:b.right,bottom:b.bottom}};
                return {width:innerWidth,height:innerHeight,header:r(document.querySelector('.topbar')),
                    canvas:r(document.getElementById('canvas')),footer:r(document.querySelector('footer')),
                    bodyOverflow:document.body.scrollWidth>innerWidth+1,
                    headerOverflow:document.querySelector('.topbar').scrollWidth>document.querySelector('.topbar').clientWidth+1};
            })()""")
            sample={'width':width,'height':height,'font':font,'zoom':zoom,'spacing':spacing,'theme':theme,'tabs':tab_count,'geometry':geometry}
            if geometry['bodyOverflow'] or geometry['headerOverflow'] or geometry['canvas']['height']<50 or geometry['canvas']['bottom']>geometry['height']+1:
                failures.append({'sample':sample,'reason':'whole-window bounds'})
            # Reveal/focus each control within the header, then hit-test its center.
            # No hidden duplicate controls, synthetic click or font reduction.
            for identifier in controls:
                result=js(view,"""(() => {
                    const e=document.getElementById(%s);e.focus({preventScroll:true});e.scrollIntoView({block:'nearest',inline:'nearest'});
                    const r=e.getBoundingClientRect(),h=document.querySelector('.topbar').getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2;
                    const hit=document.elementFromPoint(x,y);
                    return {id:e.id,x,y,left:r.left,right:r.right,top:r.top,bottom:r.bottom,
                        hit:hit?.id||hit?.tagName,accessible:x>=0&&x<innerWidth&&y>=h.top&&y<h.bottom&&Boolean(hit&&(hit===e||e.contains(hit))),
                        focused:document.activeElement===e};
                })()"""%json.dumps(identifier))
                if not result['accessible'] or not result['focused']:failures.append({'sample':sample,'reason':'control access','control':result})
            # Native mouse click, with coordinate conversion at WebEngine zoom.
            point=js(view,"""(() => {const e=document.getElementById('validate');e.scrollIntoView({block:'nearest'});
                const r=e.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};})()""")
            click_before=js(view,'window.__geometryClicks')
            QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
                             QPoint(round(point['x']*zoom),round(point['y']*zoom)))
            until(view,f'window.__geometryClicks==={click_before+1} && document.getElementById("toast").classList.contains("toast-visible") && document.getElementById("toast-message").textContent.includes("Valid graph")')
            js(view,'document.getElementById("toast-close").click()')
            js(view,'document.getElementById("zoom-in").focus()')
            QTest.keyClick(view.focusProxy() or view,Qt.Key.Key_Tab)
            until(view,'document.activeElement.id==="validate"')
            assert js(view,"""(() => {const r=document.activeElement.getBoundingClientRect(),h=document.querySelector('.topbar').getBoundingClientRect();
                return r.top>=h.top-1 && r.bottom<=h.bottom+1;})()"""),'Tab did not reveal the focused toolbar control'
            # Launcher buttons remain reachable via their own scroll region.
            js(view,'window.loom.perform("reset-panels")')
            for name in ('architectures-panel','library-panel','experiment-dock','inspector-panel','layout-panel'):
                result=js(view,"""(() => {const e=document.querySelector('[data-open-panel=%s]');e.scrollIntoView({block:'nearest'});
                    const r=e.getBoundingClientRect(),x=r.x+r.width/2,y=r.y+r.height/2,hit=document.elementFromPoint(x,y);
                    return {name:%s,x,y,accessible:x>=0&&x<innerWidth&&y>=document.getElementById('editor').getBoundingClientRect().top&&y<innerHeight&&Boolean(hit&&(hit===e||e.contains(hit)))};
                })()"""%(name,json.dumps(name)))
                if not result['accessible']:failures.append({'sample':sample,'reason':'launcher access','control':result})
                else:
                    QTest.mouseClick(view.focusProxy() or view,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
                                     QPoint(round(result['x']*zoom),round(result['y']*zoom)))
                    until(view,f'!document.getElementById({json.dumps(name)}).hidden')
                    js(view,'window.loom.perform("reset-panels")')
            capture=os.environ.get('VTW_CHROME_SCREENSHOTS')
            if capture and tab_count==2 and ((width,font,zoom,theme)==(1480,18,1.35,'dark') or (width,font,zoom,theme)==(1000,28,3,'github')):
                js(view,'window.loom.fit()');QTest.qWait(100)
                name='chrome-readable.png' if width==1480 else 'chrome-extreme.png'
                assert window.grab().save(str(Path(capture)/name))
            samples.append(sample)
        print(json.dumps({'samples':samples,'failures':failures},sort_keys=True),flush=True)
        assert not failures, f'{len(failures)} inaccessible/window-overflow cases'
        assert window.workspace.db.serialize()==before and Path(window.workspace.path).read_bytes()==disk
        assert window.workspace.history.summary()==history and not window.workspace.dirty
        assert config.path.read_bytes()==config_bytes and 'torch' not in sys.modules
        assert not errors,errors
        print(json.dumps({'native_window_geometry':'PASS','samples':len(samples),'focused_hit_tests':len(samples)*len(controls),
                          'launcher_hit_tests':5*len(samples),'real_validate_clicks':len(samples),'native_tab_reveal':len(samples),
                          'sql_disk_history_config_unchanged':'PASS','console_errors':errors,'host_torch':False}))
    finally:
        if window.workspace:window.workspace.dirty=False
        window.close()
        for index in range(window.tabs.count()):
            v=window.tabs.widget(index);page=v.page();v.setPage(QWebEnginePage(v));page.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
        window.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
