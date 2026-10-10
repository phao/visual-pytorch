"""Real Chromium CSS cascade across every block type and semantic theme.

Private offscreen webview; no workspace, credentials or user preferences.
"""
import json
import os
import time
from pathlib import Path
from PyQt6.QtCore import QEventLoop, QTimer, QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.themes import THEMES, resolve_theme

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
app=QApplication(['theme-block-cascade-smoke'])
view=QWebEngineView();view.resize(1150,820)
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);result=[]
    def receive(value):result.append(value);loop.quit()
    view.page().runJavaScript(script,receive);timer.start(3000);loop.exec();timer.stop()
    assert result,'JavaScript callback timeout'
    return result[0]
view.setUrl(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html')));view.show()
deadline=time.monotonic()+10
while not js('Boolean(document.getElementById("canvas"))'):
    assert time.monotonic()<deadline
    loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
# Reuse real node selectors and SVG structure, without numerical/model side effects.
mapping={'Input':'node-input','Concat':'node-input','Linear':'node-linear','Conv2d':'node-linear',
    'Module':'node-module','Splitter':'node-module','Replicator':'node-module',
    'Activation':'node-output','Output':'node-output','AvgPool2d':'node-output',
    'MaxPool2d':'node-output','AdaptiveAvgPool2d':'node-output',
    'Reshape':'node-reshape','Flatten':'node-reshape','Permute':'node-reshape'}
js('document.getElementById("welcome").hidden=true;document.getElementById("editor").hidden=false;')
js("""
const ns="http://www.w3.org/2000/svg", group=document.getElementById("nodes");
group.replaceChildren();
TYPES.forEach((type,i)=>{
 const node=document.createElementNS(ns,"g");node.setAttribute("class","node "+type);
 node.setAttribute("transform","translate("+(180+(i%4)*220)+","+(40+Math.floor(i/4)*155)+")");
 const rect=document.createElementNS(ns,"rect");rect.setAttribute("width",190);rect.setAttribute("height",120);node.append(rect);
 for(const [cls,label,y] of [["node-header",type.toUpperCase(),20],["node-title",type,45],["node-shape","[B, 2] → [B, 4]",70],["node-params","Shape / parameters",96]]){
  const text=document.createElementNS(ns,"text");text.setAttribute("class",cls);text.setAttribute("x",10);text.setAttribute("y",y);text.textContent=label;node.append(text);
 }
 group.append(node);
});
""".replace('TYPES',json.dumps(list(mapping))))
def rgb(hexvalue):return 'rgb('+', '.join(str(int(hexvalue[i:i+2],16)) for i in (1,3,5))+')'
checked=0
for name,tokens in THEMES.items():
    js('document.documentElement.dataset.theme='+json.dumps(name)+';Object.entries('+json.dumps(tokens)+').forEach(([k,v])=>document.documentElement.style.setProperty("--"+k,v));')
    values=js('Object.fromEntries([...document.querySelectorAll("#nodes .node")].map(n=>[n.classList[1],{fill:getComputedStyle(n.querySelector("rect")).fill,ink:getComputedStyle(n.querySelector(".node-title")).fill,muted:getComputedStyle(n.querySelector(".node-params")).fill,stroke:getComputedStyle(n.querySelector("rect")).stroke}]))')
    for kind,token in mapping.items():
        assert values[kind]['fill']==rgb(tokens[token]),(name,kind,values[kind],tokens[token])
        assert values[kind]['ink']==rgb(tokens['ink']),(name,kind,'title')
        assert values[kind]['muted']==rgb(tokens['muted']),(name,kind,'details')
        checked+=1
    assert values['Concat']['stroke']==rgb(tokens['blue']),(name,values['Concat'])
    js('document.querySelector(".node.Concat").classList.add("selected")')
    assert js('getComputedStyle(document.querySelector(".node.Concat rect")).stroke')==rgb(tokens['ink'])
    assert js('getComputedStyle(document.querySelector(".node.Concat rect")).fill')==rgb(tokens['node-input'])
    js('document.querySelector(".node.Concat").classList.remove("selected")')
for system in ('light','dark'):
    assert resolve_theme('system',system) in THEMES
if os.environ.get('VTW_SMOKE_SCREENSHOT'):
    tokens=THEMES['github']
    js('document.documentElement.dataset.theme="github";Object.entries('+json.dumps(tokens)+').forEach(([k,v])=>document.documentElement.style.setProperty("--"+k,v));')
    loop=QEventLoop();QTimer.singleShot(250,loop.quit);loop.exec()
    assert view.grab().save(os.environ['VTW_SMOKE_SCREENSHOT'])
view.close()
print(json.dumps({'real_webengine_theme_cascade':'PASS','themes':len(THEMES),'block_types':len(mapping),'type_theme_pairs':checked,'concat_selected_fill_and_border':'PASS','system_resolution':'PASS'}))

