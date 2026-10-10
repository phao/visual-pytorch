"""Test native tab visibility and unused-strip color without starting Chromium."""
import json
import sys

from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget
from app.qt_style import configure_tabs

app = QApplication([])
tabs = QTabWidget()
configure_tabs(tabs)
tabs.resize(900, 300)
tabs.addTab(QWidget(), "Linear regression")
tabs.show()
app.processEvents()
assert tabs.tabBar().isHidden(), "Single tab should be hidden"
tabs.addTab(QWidget(), "MLP regression")
app.processEvents()
assert tabs.tabBar().isVisible(), "Two tabs should be visible"
bar = tabs.tabBar()
strip = tabs.grab().toImage()
# QTabWidget owns the unused space to the right of the shorter QTabBar.
color = strip.pixelColor(tabs.width() - 12, max(1, bar.height() // 2)).name()
assert color == "#0d1723", f"Unused strip should be dark, got {color}"
if len(sys.argv) > 1:
    tabs.grab().save(sys.argv[1])
tabs.removeTab(1)
app.processEvents()
assert tabs.tabBar().isHidden(), "Closing to a single tab should hide the bar again"
print(json.dumps({"tabs": "PASS", "single_tab_hidden": True,
                  "two_tabs_visible": True, "strip_color": color}))
tabs.close()
