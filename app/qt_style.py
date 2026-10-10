"""Native tab styling, independent of WebEngine for widget-level checks."""
from PyQt6.QtGui import QColor, QPalette
from .themes import THEMES


def configure_menubar(bar, tokens):
    """Theme native menus explicitly; a fixed stylesheet overrides QPalette."""
    bar.setStyleSheet(f"""
        QMenuBar {{background:{tokens['chrome']};color:{tokens['ink']};}}
        QMenuBar::item {{padding:5px 9px;background:transparent;}}
        QMenuBar::item:selected {{background:{tokens['hover']};color:{tokens['ink']};}}
        QMenuBar::item:pressed {{background:{tokens['selected']};}}
        QMenu {{background:{tokens['panel']};color:{tokens['ink']};border:1px solid {tokens['line']};padding:5px;}}
        QMenu::item {{padding:7px 22px;border-radius:4px;}}
        QMenu::item:selected {{background:{tokens['selected']};color:{tokens['ink']};}}
        QMenu::item:disabled {{color:{tokens['muted']};}}
        QMenu::separator {{height:1px;background:{tokens['line']};margin:5px;}}
    """)


def configure_tabs(tabs, tokens=None):
    tokens = tokens or THEMES['dark']
    tabs.setDocumentMode(True)
    tabs.setTabsClosable(True)
    tabs.setMovable(True)
    tabs.setTabBarAutoHide(True)
    # The unused strip behind the tabs uses Window, not the tab subcontrol.
    palette = tabs.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens['chrome']))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens['chrome']))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens['panel']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens['ink']))
    tabs.setPalette(palette)
    tabs.tabBar().setPalette(palette)
    tabs.tabBar().setAutoFillBackground(True)
    sheet = """
        QTabWidget {background:#0d1723;}
        QTabWidget::pane {border:0;background:#09111b;}
        QTabBar {background:#0d1723;color:#e5edf6;}
        QTabBar::tab {
            background:#101c29;color:#8396ab;padding:8px 16px;
            min-width:100px;border:0;border-top:2px solid transparent;
        }
        QTabBar::tab:selected {background:#18312f;color:#e5edf6;border-top-color:#70cfb5;}
        QTabBar::tab:hover:!selected {background:#1c2d3d;color:#c9d9e9;}
        QTabBar QToolButton {background:#0d1723;color:#e5edf6;border:0;}
    """
    for original,key in {'#0d1723':'chrome','#09111b':'bg','#e5edf6':'ink','#101c29':'panel',
                         '#8396ab':'muted','#18312f':'selected','#70cfb5':'mint',
                         '#1c2d3d':'hover','#c9d9e9':'ink'}.items():
        sheet=sheet.replace(original,tokens[key])
    tabs.setStyleSheet(sheet)
