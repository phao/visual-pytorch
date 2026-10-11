"""Real native Preview/Cancel/Save flow without initializing Chromium."""
import json
import tempfile
from pathlib import Path
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication,QMainWindow,QWidget,QTabWidget,QDialogButtonBox,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox
from app.config import Config
from app.desktop import Window
from app.themes import THEMES

app=QApplication([])
class View(QWidget):
    def setZoomFactor(self,value):self.zoom=value
class Harness(Window):
    def __init__(self,path):
        QMainWindow.__init__(self)
        self.config=Config(path);self.tabs=QTabWidget();self.tabs.addTab(View(),'Model')
        self.setCentralWidget(self.tabs);self.events=[]
        self.apply_theme()
    def broadcast(self,*args):self.events.append(dict(self.config.values))

with tempfile.TemporaryDirectory() as td:
    h=Harness(Path(td)/'preferences.json');h.config.save()
    original=dict(h.config.values);before=h.config.path.read_bytes()
    original_font=QFont(app.font())
    def change_and_preview(theme,font,zoom,spacing=4,box_spacing=35,grid_snap=True):
        dialog=app.activeModalWidget()
        combo=dialog.findChild(QComboBox,'display-theme');combo.setCurrentIndex(combo.findData(theme))
        dialog.findChild(QSpinBox,'display-spacing').setValue(spacing)
        dialog.findChild(QSpinBox,'display-font').setValue(font);dialog.findChild(QDoubleSpinBox).setValue(zoom)
        dialog.findChild(QSpinBox,'display-box-spacing').setValue(box_spacing)
        dialog.findChild(QCheckBox,'display-grid-snap').setChecked(grid_snap)
        dialog.findChild(QPushButton,'preview-display').click()
        assert dialog.isVisible(),"Preview must keep dialog open"
        assert h.config.path.read_bytes()==before,"Preview must not persist"
        assert h.tabs.widget(0).zoom==zoom
        assert app.font().pixelSize()==font
        assert h.config.values['theme']==theme
        assert h.config.values['spacing']==spacing
        assert h.config.values['box_spacing']==box_spacing
        assert h.config.values['grid_snap'] is grid_snap
        assert f"background:{THEMES[theme]['chrome']}" in h.menuBar().styleSheet()
        return dialog
    def cancel():
        dialog=change_and_preview('light',18,1.35)
        change_and_preview('matrix',20,1.5,10,100,False)
        change_and_preview('math',18,1.35)
        assert app.font().styleHint()==QFont.StyleHint.Serif
        change_and_preview('metallic',18,1.35)
        assert app.font().families()==original_font.families()
        dialog.reject()
    QTimer.singleShot(0,cancel);h.display_options()
    assert h.config.values==original and h.config.path.read_bytes()==before
    assert h.tabs.widget(0).zoom==original['ui_zoom']
    assert app.font().pixelSize()==original['font_size']
    assert app.font().families()==original_font.families()
    def save():
        dialog=change_and_preview('math',16,1.45,1,100)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save).click()
    QTimer.singleShot(0,save);h.display_options()
    saved=Config(h.config.path).values
    assert saved['theme']=='math' and saved['font_size']==16 and saved['ui_zoom']==1.45
    assert saved['spacing']==1
    assert saved['box_spacing']==100
    assert saved['grid_snap'] is True
    assert h.events and h.events[-1]==saved
    assert app.font().styleHint()==QFont.StyleHint.Serif
print(json.dumps({'display_preview':'PASS','cancel_restores':'PASS','save_persists':'PASS','preview_does_not_write':True}))
