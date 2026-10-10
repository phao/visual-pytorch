"""Native palette, tab colour and preference choices, without WebEngine startup."""
import json
from PyQt6.QtGui import QPalette,QFont,QFontInfo
from PyQt6.QtWidgets import QApplication,QTabWidget,QWidget,QMainWindow
from types import SimpleNamespace
from app.themes import THEMES,native_palette,SERIF_FAMILIES
from app.qt_style import configure_tabs
from app.desktop import Window
app=QApplication([]);window=QMainWindow();tabs=QTabWidget();tabs.addTab(QWidget(),'Model')
window.tabs=tabs;window.setCentralWidget(tabs);window.resize(640,300)
menu=window.menuBar().addMenu('File');menu.addAction('Save workspace');menu.addAction('Unavailable').setEnabled(False)
window.show()
baseline=QFont(app.font())
for name,tokens in THEMES.items():
    window.config=SimpleNamespace(values={'theme':name})
    Window.apply_theme(window);app.processEvents()
    assert tabs.tabBar().palette().color(QPalette.ColorRole.Window).name()==tokens['chrome']
    assert not tabs.tabBar().isVisible()
    assert app.palette().color(QPalette.ColorRole.Text).name()==tokens['ink']
    if name=='math':
        assert app.font().styleHint()==QFont.StyleHint.Serif
        assert QFontInfo(app.font()).family() in SERIF_FAMILIES, QFontInfo(app.font()).family()
    else:
        assert app.font().families()==baseline.families(), (name,app.font().families())
    bar=window.menuBar();pixel=bar.grab().toImage().pixelColor(bar.width()-10,bar.height()//2).name()
    assert pixel==tokens['chrome'],(name,pixel,tokens['chrome'])
    assert f"QMenu {{background:{tokens['panel']}" in bar.styleSheet()
print(json.dumps({'native_themes':'PASS','themes':list(THEMES),'single_tab_hidden':True,'menu_bar_pixels':'PASS','serif_restoration':'PASS'}))
