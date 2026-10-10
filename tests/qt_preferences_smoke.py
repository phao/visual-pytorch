"""Native preference dialogs remain wide/readable at the reported 18px font."""
import json
from types import SimpleNamespace
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QDialog, QDoubleSpinBox,QComboBox,QDialogButtonBox,QPushButton,QSpinBox
from app.desktop import Window
from app.themes import THEME_NAMES

app=QApplication([])
font=app.font();font.setPixelSize(18);app.setFont(font)
h=QMainWindow();h.config=SimpleNamespace(values={'font_size':18,'ui_zoom':1.35,'memory_limit_percent':10.0});h.job=None
observed=[]

def inspect():
    dialog=app.activeModalWidget()
    assert isinstance(dialog,QDialog)
    assert dialog.windowTitle().startswith('Visual PyTorch — ')
    assert dialog.width() >= 540
    assert dialog.width() >= dialog.fontMetrics().horizontalAdvance(dialog.windowTitle()) + 80
    observed.append({'title':dialog.windowTitle(),'width':dialog.width()})
    if 'Run safety' in dialog.windowTitle():
        assert dialog.findChild(QDoubleSpinBox).value()==10
    else:
        combo=dialog.findChild(QComboBox,'display-theme')
        assert [combo.itemText(i) for i in range(combo.count())]==[name.title() for name in THEME_NAMES]
        assert dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Save)
        assert dialog.findChild(QPushButton,'preview-display')
        spacing=dialog.findChild(QSpinBox,'display-spacing')
        assert (spacing.minimum(),spacing.maximum(),spacing.value())==(1,10,4)
        boxes=dialog.findChild(QSpinBox,'display-box-spacing')
        assert (boxes.minimum(),boxes.maximum(),boxes.value())==(1,100,35)
    dialog.reject()

QTimer.singleShot(0,inspect);Window.display_options(h)
QTimer.singleShot(0,inspect);Window.safety_options(h)
assert len(observed)==2
print(json.dumps({'preferences_layout':'PASS','font_px':18,'dialogs':observed}))
