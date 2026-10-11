"""Render actual real-kernel notebook HTML and numerical SVG in sandboxed Qt."""
import json
import os
import sys
import time
from pathlib import Path
from PyQt6.QtCore import QEventLoop, QTimer, QUrl, QCoreApplication, QEvent
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, QWebEngineUrlRequestInterceptor
from PyQt6.QtWebEngineWidgets import QWebEngineView

assert not os.environ.get('QTWEBENGINE_DISABLE_SANDBOX')
assert '--no-sandbox' not in os.environ.get('QTWEBENGINE_CHROMIUM_FLAGS', '')
folder = Path(sys.argv[1]).resolve(); evidence = json.loads((folder / 'render.json').read_text())
app = QApplication(['rendered-report-smoke']); view = QWebEngineView(); view.resize(1000, 980)
profile = QWebEngineProfile(view); errors = []; remote = []
class LocalOnly(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, request):
        if request.requestUrl().scheme() not in ('file', 'data', 'about'):
            remote.append(request.requestUrl().toString()); request.block(True)
class Page(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, line, source):
        if level == self.JavaScriptConsoleMessageLevel.ErrorMessageLevel: errors.append(message)
interceptor = LocalOnly(profile); profile.setUrlRequestInterceptor(interceptor)
page = Page(profile, view); view.setPage(page); view.show()
def js(source):
    values = []; loop = QEventLoop(); timer = QTimer(); timer.setSingleShot(True); timer.timeout.connect(loop.quit)
    page.runJavaScript(source, lambda value: (values.append(value), loop.quit()))
    timer.start(3000); loop.exec(); timer.stop(); assert values, 'JS callback timed out'; return values[0]
def until(source):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if js(source): return
        loop = QEventLoop(); QTimer.singleShot(20, loop.quit); loop.exec()
    raise AssertionError('Render timed out: ' + source)
def load(name, ready):
    page.setUrl(QUrl.fromLocalFile(str(folder / name))); until(ready)
try:
    load('curves.html', 'document.querySelectorAll("svg").length===11')
    probe = js('''(() => {
      const plots=[...document.querySelectorAll('svg')],overflow=[],overlap=[];
      plots.forEach((svg,index)=>{const box=svg.getBoundingClientRect();
        for(const text of svg.querySelectorAll('text')){const b=text.getBoundingClientRect();
          if(b.left<box.left-.5||b.right>box.right+.5||b.top<box.top-.5||b.bottom>box.bottom+.5)
            overflow.push({case:svg.closest('section').dataset.case,text:text.textContent,left:b.left-box.left,right:b.right-box.left});}
        const labels=[...svg.querySelectorAll('text')];
        for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){
          const a=labels[i].getBoundingClientRect(),b=labels[j].getBoundingClientRect();
          if(Math.min(a.right,b.right)-Math.max(a.left,b.left)>1&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>1)
            overlap.push({case:svg.closest('section').dataset.case,first:labels[i].textContent,second:labels[j].textContent});}
      });
      const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);
      return {plots:plots.length,overflow,overlap,duplicate_ids:ids.filter((id,i)=>ids.indexOf(id)!==i)};
    })()''')
    (folder / 'geometry.json').write_text(json.dumps(probe, indent=2), encoding='utf-8')
    if '--probe' not in sys.argv:
        assert not probe['overflow'], probe
        assert not probe['overlap'], probe
        assert not probe['duplicate_ids'], probe
    if os.environ.get('VTW_REPORT_CAPTURE'):
        js('document.querySelector("[data-case=wide-latin]").scrollIntoView()')
        loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec()
        assert view.grab().save(str(folder / 'long-label-curve.png'))
    for mode in ('result-report', 'rerun'):
        load(mode + '.html', 'document.querySelectorAll(".cell").length===' + str(evidence[mode]['notebook_cells']))
        text = js('document.body.textContent')
        assert 'Literal <script> evidence.' in text and not js('Boolean(window.renderInjected)')
        assert all(section in text for section in ('Question', 'Models', 'Datasets', 'Protocol', 'Training', 'Evaluation', 'Failures', 'Environment', 'Checkpoints'))
        assert not js('document.querySelectorAll("script,iframe").length')
        assert evidence[mode]['requested_training'] == (mode == 'rerun')
        assert evidence[mode]['statuses'] == (['completed', 'failed'] if mode == 'result-report' else ['completed'] * 8)
        if '--probe' not in sys.argv:
            assert js('document.querySelectorAll("figure.vpw-curve").length') == evidence[mode]['curves']
            assert js('document.querySelectorAll(".vpw-curve figcaption").length') == evidence[mode]['curves']
            assert 'Exact plotted rows' in text and 'denominator' in text
            assert not js('document.querySelectorAll(' + json.dumps('img[alt="No description has been provided for this image"]') + ').length')
            assert js('''[...document.querySelectorAll('.vpw-curve')].every(f=>{
              const caption=document.getElementById(f.getAttribute('aria-labelledby'));
              const svg=f.querySelector('svg');return caption===f.querySelector('figcaption')&&
                svg.getAttribute('aria-labelledby').split(' ').every(id=>svg.querySelector('[id="'+id+'"]'));
            })''')
            # Expand through a real keyboard gesture, preserving the plotted data.
            from PyQt6.QtCore import Qt
            from PyQt6.QtTest import QTest
            view.activateWindow();view.setFocus();QCoreApplication.processEvents()
            js('document.querySelector(".vpw-curve summary").focus()')
            assert js('document.activeElement===document.querySelector(".vpw-curve summary")')
            QTest.keyClick(view.focusProxy() or view, Qt.Key.Key_Return)
            until('document.querySelector(".vpw-curve details").open')
            assert js('document.querySelectorAll(".vpw-curve details:first-of-type table th[scope=col]").length') >= 8
            assert js('''[...document.querySelectorAll('.vpw-curve table th,.vpw-curve table td')].every(e=>
                getComputedStyle(e).whiteSpace==='nowrap'&&getComputedStyle(e).overflowWrap==='normal')''')
            view.resize(520, 980)
            until('window.innerWidth<=520')
            assert js('''[...document.querySelectorAll('.vpw-curve')].every(f=>{
              const box=f.getBoundingClientRect(),plot=f.querySelector('svg').parentElement;
              return box.width<=document.documentElement.clientWidth&&getComputedStyle(plot).overflowX==='auto'&&
                f.querySelector('svg').getAttribute('width')==='760';
            })''')
            view.resize(1000, 980);until('window.innerWidth>=990')
            js('document.body.style.fontSize="28px"')
            assert js('parseFloat(getComputedStyle(document.querySelector(".vpw-curve figcaption")).fontSize)') == 28
            js('document.body.style.fontSize="18px"')
        if os.environ.get('VTW_REPORT_CAPTURE'):
            js('document.querySelector(".vpw-curve, .output_svg").scrollIntoView()')
            loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec();assert view.grab().save(str(folder / (mode + '-report.png')))
            if mode == 'result-report':
                js('document.body.style.filter="grayscale(1)"')
                loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec();assert view.grab().save(str(folder / 'saved-report-grayscale.png'))
                js('document.body.style.filter="none"')
                js('document.querySelector(".vpw-curve details").scrollIntoView()')
                loop=QEventLoop();QTimer.singleShot(100,loop.quit);loop.exec();assert view.grab().save(str(folder / 'saved-report-table.png'))
    assert not remote, remote
    assert not errors, errors
    assert not any(name in sys.modules for name in ('torch', 'nbclient', 'nbformat', 'nbconvert'))
    print(json.dumps({'rendered_notebooks': 'PASS', 'offline': 'PASS', 'geometry': probe,
                      'host_numerical_stack': False, 'console_errors': errors}))
finally:
    view.close();page.deleteLater();QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete);QCoreApplication.processEvents()
