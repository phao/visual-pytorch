"""App-only WebEngine context menu; also testable on an ordinary Qt widget."""
import json
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMenu,QApplication
from .themes import THEMES,resolve_theme


def build_context_menu(view, window, block=None):
    menu=QMenu(view)
    system='light' if QApplication.instance().styleHints().colorScheme()==Qt.ColorScheme.Light else 'dark'
    choice=window.config.values.get('theme','dark') if hasattr(window,'config') else 'dark'
    tokens=THEMES[resolve_theme(choice,system)]
    sheet="""
        QMenu {background:#10212d;color:#e5edf6;border:1px solid #3c5966;padding:5px;}
        QMenu::item {padding:7px 22px;border-radius:4px;}
        QMenu::item:selected {background:#24483f;color:#c8eee0;}
        QMenu::item:disabled {color:#627887;}
        QMenu::separator {height:1px;background:#304451;margin:5px;}
    """
    for original,key in {'#10212d':'panel','#e5edf6':'ink','#3c5966':'line','#24483f':'selected',
                         '#c8eee0':'ink','#627887':'muted','#304451':'line'}.items():sheet=sheet.replace(original,tokens[key])
    menu.setStyleSheet(sheet)
    has_model=bool(window.workspace and view.mid in window.workspace.models())
    # Only the block under this click is targeted; empty space never deletes a
    # previously selected block. The editor rechecks existence and busy state.
    if isinstance(block, dict) and isinstance(block.get('id'), str) and isinstance(block.get('modelId'), str):
        inspect=menu.addAction("Inspect block")
        inspect.setEnabled(has_model and view.mid==block['modelId'])
        inspect.triggered.connect(lambda checked=False, nid=block['id'], mid=block['modelId']:
            view.page().runJavaScript('window.loom && window.loom.perform("inspect-block",'+json.dumps(nid)+','+json.dumps(mid)+');'))
        delete=menu.addAction("Delete block")
        delete.setEnabled(has_model and view.mid==block['modelId'] and block.get('canDelete') is True)
        delete.triggered.connect(lambda checked=False, nid=block['id'], mid=block['modelId']:
            view.page().runJavaScript('window.loom && window.loom.perform("delete-block",'+json.dumps(nid)+','+json.dumps(mid)+');'))
        menu.addSeparator()
    for label,command in sorted([
        ("Fit graph","fit"),("Validate architecture","validate"),
        ("Architectures panel","architectures"),("Building blocks panel","library"),("Inspector panel","inspector"),
        ("Experiment panel","experiment"),("PyTorch forward","code"),
        ("Layout panel","layout"),
        ("Learned parameters","results"),("Reset floating menus","reset-panels"),
    ], key=lambda item:item[0].casefold()):
        action=menu.addAction(label)
        action.setEnabled(has_model)
        action.triggered.connect(lambda checked=False,c=command:
            view.page().runJavaScript("window.loom && window.loom.perform("+json.dumps(c)+");"))
    menu.addSeparator()
    menu.addAction("Find…").triggered.connect(lambda:window.find())
    tab=menu.addAction("Open architecture in new tab")
    tab.setEnabled(has_model)
    tab.triggered.connect(lambda:window.open_tab(view.mid))
    save=menu.addAction("Save workspace")
    save.setEnabled(bool(window.workspace))
    save.triggered.connect(lambda:window.save_workspace())
    # Editing shortcuts remain available via the keyboard. No browser-source,
    # page-saving or reload actions are inherited from the default web menu.
    return menu


def configure_context_menu(view,window):
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    def show(position):
        request=getattr(view,'context_menu_request',0)+1
        view.context_menu_request=request
        def selected(block):
            if request!=view.context_menu_request:return
            menu=build_context_menu(view,window,block)
            view.app_context_menu=menu
            menu.aboutToHide.connect(menu.deleteLater)
            menu.popup(view.mapToGlobal(position))
        zoom=view.zoomFactor() if hasattr(view,'zoomFactor') else 1
        script=f'window.loom && window.loom.contextAt({position.x()/zoom},{position.y()/zoom});'
        view.page().runJavaScript(script,selected)
    view.customContextMenuRequested.connect(show)
