"""Render the actual vector app icon with native Qt, without Chromium."""
import json
from pathlib import Path
import sys
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from app.identity import APP_NAME,DESKTOP_ID,ICON_FILE
app=QApplication([])
app.setApplicationName(APP_NAME);app.setDesktopFileName(DESKTOP_ID)
icon=QIcon(str(Path(__file__).resolve().parents[1]/ICON_FILE))
assert not icon.isNull()
app.setWindowIcon(icon)
for size in (16,24,32,48,64,192):
    rendered=icon.pixmap(size,size)
    assert not rendered.isNull() and rendered.width()==rendered.height()==size
pixmap=icon.pixmap(192,192)
assert not pixmap.isNull()
if len(sys.argv)>1:assert pixmap.save(sys.argv[1])
assert app.applicationName()=='Visual PyTorch' and app.desktopFileName()=='tensor-loom'
assert not app.windowIcon().isNull()
print(json.dumps({'icon':'PASS','width':pixmap.width(),'height':pixmap.height(),'sizes':[16,24,32,48,64,192],
    'application_name':app.applicationName(),'desktop_id':app.desktopFileName()}))
