"""Actual menu group containment/order and themed framing; no user state."""
import json
import os
import time
from pathlib import Path
from PyQt6.QtCore import QEventLoop,QTimer,QUrl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView
from app.themes import THEMES

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS','')
app=QApplication(['palette-groups-smoke']);view=QWebEngineView();view.resize(1100,900)
errors=[]
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self,level,message,line,source):
        if level==self.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            errors.append(f'{source}:{line}: {message}')
view.setPage(Page(view))
def js(script):
    loop=QEventLoop();timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(loop.quit);values=[]
    view.page().runJavaScript(script,lambda value:(values.append(value),loop.quit()))
    timer.start(3000);loop.exec();timer.stop();assert values,'JavaScript timeout';return values[0]
view.setUrl(QUrl.fromLocalFile(str(Path(__file__).resolve().parents[1]/'app/web/index.html')));view.show()
deadline=time.monotonic()+10
while not js('Boolean(window.LoomActivations && document.getElementById("add-tensor-Conv2d"))'):
    assert time.monotonic()<deadline
    loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
js('document.getElementById("welcome").hidden=true;document.getElementById("editor").hidden=false;document.getElementById("library-panel").hidden=false;')
assert js('document.querySelectorAll("#library-panel .palette-group").length')==3  # activation, MLP, architecture reference
# Alphabetical within each heading's section (headings themselves keep their fixed order).
assert js('(()=>{const s=[[]];for(const n of document.querySelector("#library-scroll .sidebar-section").children){if(n.classList.contains("eyebrow"))s.push([]);else if(n.matches(".palette-group,.palette"))s.at(-1).push(n.querySelector("b").textContent)}return s.every(a=>a.every((v,i)=>!i||a[i-1].localeCompare(v)<=0))})()')
assert js('!["add-linear","add-reshape","add-concat","add-input","add-output"].some(id=>document.getElementById(id).closest(".palette-group"))')
checked=0
for theme,tokens in THEMES.items():
    js('document.documentElement.dataset.theme='+json.dumps(theme)+';Object.entries('+json.dumps(tokens)+').forEach(([k,v])=>document.documentElement.style.setProperty("--"+k,v));')
    for font in (14,18,28):
        for spacing in (1,4,10):
            js(f'document.documentElement.style.setProperty("--ui-font-size","{font}px");document.documentElement.style.setProperty("--menu-unit","{spacing}px");')
            # Select-then-button groups; the MLP group is a button that opens a form (checked in qt_mlp_block_smoke).
            result=js('''(()=>Array.from(document.querySelectorAll(".palette-group[aria-labelledby]")).map(g=>{
                const c=getComputedStyle(g),r=g.getBoundingClientRect(),s=g.querySelector("select"),b=g.querySelector("button"),sr=s.getBoundingClientRect(),br=b.getBoundingClientRect();
                return {border:c.borderTopWidth,color:c.borderTopColor,line:getComputedStyle(document.documentElement).getPropertyValue("--line").trim(),padding:parseFloat(c.paddingTop),gap:parseFloat(c.rowGap),
                  contains:sr.left>=r.left&&sr.right<=r.right+.1&&br.left>=r.left&&br.right<=r.right+.1&&sr.top>=r.top&&br.bottom<=r.bottom+.1,
                  ordered:sr.bottom<=br.top&&Boolean(s.compareDocumentPosition(b)&Node.DOCUMENT_POSITION_FOLLOWING),label:g.getAttribute("aria-labelledby"),labelled:!!document.getElementById(g.getAttribute("aria-labelledby"))};}))()''')
            for row in result:
                expected='rgb('+', '.join(str(int(tokens['line'][i:i+2],16)) for i in (1,3,5))+')'
                assert row['border']=='1px' and row['color']==expected,(theme,row)
                assert row['padding']==spacing and row['gap']==spacing,(font,spacing,row)
                assert row['contains'] and row['ordered'] and row['labelled'],(theme,font,spacing,row)
                checked+=1
assert not errors,errors
view.close()
print(json.dumps({'actual_palette_group_framing':'PASS','group_samples':checked,'themes':len(THEMES),'fonts':[14,18,28],'spacing':[1,4,10],'alphabetical_group_units':'PASS','standalone_unframed':'PASS','console_errors':errors}))
