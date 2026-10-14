"""Run with: python -m app.desktop [workspace.vtw]."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from .core import (GraphError, Workspace, default_model, default_mlp, generate_code, generate_dataset,
                   validate_model, preflight, safety_report)
from .memory import DEFAULT_MEMORY_LIMIT_PERCENT
from .themes import THEME_NAMES, THEMES, resolve_theme, native_palette, native_font
from .qt_style import configure_tabs,configure_menubar
from .rendering import configure_renderer
from .context_menu import configure_context_menu
from .config import Config
from .identity import APP_NAME, DESKTOP_ID, ICON_FILE
from .file_sync import revision, stamp
from .job_state import job_busy

from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, QUrl, Qt, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt6.QtGui import QAction, QKeySequence, QIcon, QFont, QDesktopServices
from PyQt6.QtWidgets import (QApplication, QFileDialog, QInputDialog, QMainWindow,
                            QMessageBox, QTabWidget, QDialog, QFormLayout, QDoubleSpinBox,
                            QSpinBox, QDialogButtonBox, QComboBox, QCheckBox)
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineProfile, qWebEngineVersion
from PyQt6.QtWebEngineWidgets import QWebEngineView
BINDING = "PyQt6"


# Closed allow-list: the webview names a resource; only these URLs open, in the user's browser.
HELP_RESOURCES = {
    'pytorch-basics': 'https://docs.pytorch.org/tutorials/beginner/basics/intro.html',
    'pytorch-nn': 'https://docs.pytorch.org/docs/stable/nn.html',
    'torch.nn.Linear': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html',
    'torch.nn.ReLU': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html',
    'torch.nn.Tanh': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html',
    'torch.nn.Sigmoid': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html',
    'torch.nn.GELU': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html',
    'torch.nn.Softplus': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html',
    'torch.nn.Softmax': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html',
    'torch.nn.LogSoftmax': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html',
    'torch.reshape': 'https://docs.pytorch.org/docs/stable/generated/torch.reshape.html',
    'torch.add': 'https://docs.pytorch.org/docs/stable/generated/torch.add.html',
    'torch.cat': 'https://docs.pytorch.org/docs/stable/generated/torch.cat.html',
    'torch.split': 'https://docs.pytorch.org/docs/stable/generated/torch.split.html',
    'torch.Tensor': 'https://docs.pytorch.org/docs/stable/tensors.html',
    'torch.nn.Module': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html',
    'torch.nn.LSTM': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html',
    'torch.nn.RNN': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html',
    'torch.full': 'https://docs.pytorch.org/docs/stable/generated/torch.full.html',
    'torch.nn.Parameter': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html',
    'torch.logsumexp': 'https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html',
    'torch.nn.Conv2d': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html',
    'torch.nn.MaxPool2d': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html',
    'torch.nn.AvgPool2d': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html',
    'torch.nn.AdaptiveAvgPool2d': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html',
    'torch.nn.Flatten': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.flatten.Flatten.html',
    'torch.permute': 'https://docs.pytorch.org/docs/stable/generated/torch.permute.html',
    'torch.mul': 'https://docs.pytorch.org/docs/stable/generated/torch.mul.html',
    'torch.nn.functional.softplus': 'https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html',
    'torch.exp': 'https://docs.pytorch.org/docs/stable/generated/torch.exp.html',
    'torch.Tensor.unfold': 'https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html',
}

DEFAULT_PORT_NAMES = {"Input": "Input x", "Output": "Prediction ŷ"}

class TrainingJob(QThread):
    point = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, models, mid, parent, python, constrained=True,
                 memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT):
        super().__init__(parent)
        self.models = copy.deepcopy(models)
        self.mid = mid
        self.constrained = constrained
        self.memory_limit_percent = memory_limit_percent
        self.cancel = threading.Event()
        self.python = python
        self.process = None

    def stop(self):
        self.cancel.set()

    def run(self):
        try:
            from .transport import run_json_worker
            result=run_json_worker(
                [self.python,"-u","-m","app.worker"],
                {"models":self.models,"id":self.mid,"constrained":self.constrained,
                 "memory_limit_percent":self.memory_limit_percent},
                on_progress=lambda point:self.point.emit(json.dumps({"id":self.mid,**point})),
                cancel=self.cancel,cwd=str(Path(__file__).resolve().parent.parent),
                on_process=lambda process:setattr(self,'process',process))
            self.completed.emit(json.dumps(result, allow_nan=False))
        except Exception as exc:
            self.failed.emit(str(exc))


class TaskTrainingJob(QThread):
    """Frozen task worker adapter using the bundle's exact admitted version."""
    point = Signal(str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, bundle, parent, python, constrained=True,
                 memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT):
        super().__init__(parent)
        self.bundle = copy.deepcopy(bundle)
        self.python = python
        self.constrained = constrained
        self.memory_limit_percent = memory_limit_percent
        self.cancel = threading.Event()
        self.process = None

    def stop(self):
        self.cancel.set()

    def run(self):
        try:
            from .transport import run_json_worker
            from .task_execution import validate_bundle
            validate_bundle(self.bundle)
            result = run_json_worker([self.python, '-u', '-m', 'app.worker'],
                {'kind': 'task-v'+str(self.bundle['version']), 'bundle': self.bundle, 'constrained': self.constrained,
                 'memory_limit_percent': self.memory_limit_percent},
                on_progress=lambda point: self.point.emit(json.dumps(point, allow_nan=False)),
                cancel=self.cancel, cwd=str(Path(__file__).resolve().parent.parent),
                on_process=lambda process: setattr(self, 'process', process))
            self.completed.emit(json.dumps(result, allow_nan=False))
        except Exception as error:
            self.failed.emit(str(error))


class Bridge(QObject):
    # Preserve the QWebChannel name without shadowing QObject.event(), which is
    # called for DeferredDelete and other native Qt events on the bridge itself.
    notification = Signal(str, name='event')

    def __init__(self, window, view):
        super().__init__(view)
        self.window, self.view = window, view

    @Slot(str, result=str)
    def request(self, payload):
        try:
            if len(payload) > 16_000_000:
                raise ValueError("This request is too large for the prototype.")
            req = json.loads(payload)
            if isinstance(req, dict) and (req.get('action', '').startswith(('scienceDraft', 'scienceFamily')) or
                                         req.get('action') in {'lstmAuthoring','lstmEditing','addLSTM','editLSTM',
                                             'parameterPolicy','editParameterPolicy','resetBlockParameterPolicy','bindParameter','parameterSourceInventory'}):
                from .revisions import _parse
                req = _parse(payload)  # New editor transport rejects duplicate/NaN JSON before mutation.
            value = self.window.dispatch(req, self)
            return json.dumps({"ok": True, "value": value}, allow_nan=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)})

    def send(self, kind, payload=None):
        self.notification.emit(json.dumps({"kind": kind, "payload": payload}, allow_nan=False))


class Window(QMainWindow):
    def __init__(self, path=None, smoke_path=None, torch_python=None, rendering=None):
        super().__init__()
        self.workspace = None
        self.job = None
        self.rendering = rendering or {"mode": "native", "reason": "Qt default rendering"}
        self.torch_python = torch_python or find_torch_python()
        self.settings_store = QSettings("VisualTorch", "TensorLoom")  # Preserve recent-file history across the rename.
        self.config = Config()
        font = QApplication.instance().font()
        font.setPixelSize(self.config.values['font_size'])
        QApplication.instance().setFont(font)
        self.external_notice = None
        # Only the .vtw file persists work. The WebEngine profile is off the record.
        self.profile = QWebEngineProfile(self)
        self.tabs = QTabWidget()
        configure_tabs(self.tabs)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)
        self.apply_theme()
        QApplication.instance().styleHints().colorSchemeChanged.connect(lambda _: self.system_theme_changed())
        self.resize(1480, 960)
        self.setMinimumSize(1000, 700)
        self.build_menus()
        self.open_tab()
        self.file_timer = QTimer(self)
        self.file_timer.setInterval(1000)
        self.file_timer.timeout.connect(self.poll_workspace)
        self.file_timer.start()
        self.update_title()
        if self.config.error:
            QTimer.singleShot(0, lambda: QMessageBox.warning(self, 'Invalid application configuration',
                'Using defaults. Your configuration was preserved.\n'+self.config.error))
        if path:
            self.load_workspace(path)
        if smoke_path:
            self.smoke_path = Path(smoke_path)
            QTimer.singleShot(16000, self.smoke_timeout)
            self.tabs.widget(0).loadFinished.connect(self.smoke_ready)

    def build_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        entries = [
            ("New workspace", "Ctrl+N", lambda: self.new_workspace()),
            ("Open workspace…", "Ctrl+O", lambda: self.open_dialog()),
            ("Save workspace", "Ctrl+S", lambda: self.save_workspace()),
            ("Save workspace as…", "Ctrl+Shift+S", lambda: self.save_workspace(True)),
        ]
        for title, key, callback in entries:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            file_menu.addAction(action)
        file_menu.addAction('Reload workspace from disk…').triggered.connect(self.reload_workspace)
        edit = self.menuBar().addMenu('&Edit')
        self.undo_action=edit.addAction('Undo workspace edit (Ctrl+Z)')
        self.redo_action=edit.addAction('Redo workspace edit (Ctrl+Shift+Z / Ctrl+Y)')
        self.undo_action.triggered.connect(lambda: self.current_view().page().runJavaScript('window.loom && window.loom.perform("undo");'))
        self.redo_action.triggered.connect(lambda: self.current_view().page().runJavaScript('window.loom && window.loom.perform("redo");'))
        self.update_history_actions()
        nav = self.menuBar().addMenu("&Navigate")
        for title, key, callback in [
            ("Back", "Alt+Left", lambda: self.current_view().back()),
            ("Forward", "Alt+Right", lambda: self.current_view().forward()),
            ("Open current architecture in new tab", "Ctrl+T", self.duplicate_tab),
        ]:
            action = QAction(title, self)
            action.setShortcut(QKeySequence(key))
            action.triggered.connect(callback)
            nav.addAction(action)
        options = self.menuBar().addMenu('&Options')
        options.addAction('Display preferences…').triggered.connect(self.display_options)
        options.addAction('Run safety…').triggered.connect(self.safety_options)
        for title, keys, callback in [
            ('Larger interface', ['Ctrl++','Ctrl+='], lambda: self.set_display(ui_zoom=min(3, round(self.config.values['ui_zoom']+.1, 2)))),
            ('Smaller interface', ['Ctrl+-'], lambda: self.set_display(ui_zoom=max(.5, round(self.config.values['ui_zoom']-.1, 2)))),
            ('Reset interface zoom', ['Ctrl+0'], lambda: self.set_display(ui_zoom=1.15)),
        ]:
            action = options.addAction(title)
            action.setShortcuts([QKeySequence(key) for key in keys])
            action.triggered.connect(callback)

    def current_view(self):
        return self.tabs.currentWidget()

    def open_tab(self, mid=None, *, clear_menus=False):
        view = QWebEngineView()
        view.mid = mid
        configure_context_menu(view, self)
        page = QWebEnginePage(self.profile, view)
        view.setPage(page)
        view.setZoomFactor(self.config.values['ui_zoom'])
        bridge = Bridge(self, view)
        channel = QWebChannel(page)
        channel.registerObject("host", bridge)
        page.setWebChannel(channel)
        view.bridge, view.channel = bridge, channel
        url = QUrl.fromLocalFile(str(Path(__file__).parent / "web" / "index.html"))
        if clear_menus:url.setQuery('menus=hidden')
        if mid:
            url.setFragment("model=" + mid)
        view.setUrl(url)
        self.tabs.addTab(view, self.tab_name(mid))
        self.tabs.setCurrentWidget(view)
        view.urlChanged.connect(lambda url, v=view: self.url_changed(v, url))
        return view

    def set_display(self, *, persist=True, **changes):
        try:
            if persist:self.config.save(**changes)
            else:
                values={**self.config.values,**changes}
                Config.validate(values)
                self.config.values=values
        except Exception as error:
            QMessageBox.warning(self, 'Cannot save preferences', str(error))
            return False
        for index in range(self.tabs.count()):
            self.tabs.widget(index).setZoomFactor(self.config.values['ui_zoom'])
        font = QApplication.instance().font()
        font.setPixelSize(self.config.values['font_size'])
        QApplication.instance().setFont(font)
        if hasattr(self,'apply_theme'):self.apply_theme()
        self.broadcast()
        return True

    def apply_theme(self):
        app=QApplication.instance()
        if not hasattr(self, '_default_ui_font'):
            self._default_ui_font = QFont(app.font())
        system='light' if app.styleHints().colorScheme()==Qt.ColorScheme.Light else 'dark'
        name=resolve_theme(self.config.values.get('theme','dark'),system)
        app.setFont(native_font(name, self.config.values.get('font_size', 14), self._default_ui_font))
        app.setPalette(native_palette(THEMES[name]))
        configure_tabs(self.tabs,THEMES[name])
        configure_menubar(self.menuBar(),THEMES[name])

    def system_theme_changed(self):
        if self.config.values.get('theme')=='system':
            self.apply_theme();self.broadcast()

    def display_options(self):
        original={key:self.config.values.get(key,{'theme':'dark','spacing':4,'box_spacing':35,'grid_snap':False,'menu_grid_snap':False}.get(key)) for key in ('ui_zoom','font_size','theme','spacing','box_spacing','grid_snap','menu_grid_snap')}
        dialog = QDialog(self)
        dialog.setWindowTitle(APP_NAME+' — Display preferences')
        dialog.setMinimumWidth(max(480, self.config.values['font_size'] * 30))
        dialog.setSizeGripEnabled(True)
        layout = QFormLayout(dialog)
        zoom = QDoubleSpinBox(dialog)
        zoom.setRange(.5, 3); zoom.setSingleStep(.1); zoom.setValue(self.config.values['ui_zoom'])
        font = QSpinBox(dialog)
        font.setObjectName('display-font')
        font.setRange(12, 28); font.setSuffix(' px'); font.setValue(self.config.values['font_size'])
        layout.addRow('Interface zoom', zoom)
        layout.addRow('Base font size', font)
        theme=QComboBox(dialog)
        theme.setObjectName('display-theme')
        for name in THEME_NAMES:theme.addItem(name.title(),name)
        theme.setCurrentIndex(THEME_NAMES.index(self.config.values.get('theme','dark')))
        layout.addRow('Stylistic theme',theme)
        spacing=QSpinBox(dialog);spacing.setObjectName('display-spacing')
        spacing.setRange(1,10);spacing.setValue(self.config.values.get('spacing',4))
        spacing.setToolTip('Independent of NN box spacing. 1 is densest, 10 roomiest; text size is unchanged.')
        layout.addRow('Menu spacing (1–10)',spacing)
        boxes=QSpinBox(dialog);boxes.setObjectName('display-box-spacing')
        boxes.setRange(1,100);boxes.setValue(self.config.values.get('box_spacing',35))
        boxes.setToolTip('Independent of menu spacing. 1 is densest, 100 roomiest; text size is unchanged.')
        layout.addRow('NN box spacing (1–100)',boxes)
        snap=QCheckBox('Snap blocks to grid by default',dialog)
        snap.setObjectName('display-grid-snap')
        snap.setChecked(self.config.values.get('grid_snap',False))
        snap.setToolTip('Ctrl temporarily inverts snapping for block movement and resizing. Independent of menus.')
        layout.addRow(snap)
        menu_snap=QCheckBox('Snap menus to grid by default',dialog)
        menu_snap.setObjectName('display-menu-grid-snap')
        menu_snap.setChecked(self.config.values.get('menu_grid_snap',False))
        menu_snap.setToolTip('Ctrl temporarily inverts menu/pack movement and resizing. Shift-drag still moves the menu layer. Independent of blocks.')
        layout.addRow(menu_snap)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, dialog)
        preview=buttons.addButton('Preview',QDialogButtonBox.ButtonRole.ActionRole)
        preview.setObjectName('preview-display')
        preview.setToolTip('Try these settings without saving; Cancel restores the previous appearance.')
        def values():return {'ui_zoom':zoom.value(),'font_size':font.value(),'theme':theme.currentData(),'spacing':spacing.value(),'box_spacing':boxes.value(),'grid_snap':snap.isChecked(),'menu_grid_snap':menu_snap.isChecked()}
        def preview_now():
            self.set_display(persist=False,**values())
            dialog.setMinimumWidth(max(480,self.config.values['font_size']*30))
            dialog.adjustSize()
        preview.clicked.connect(preview_now)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        accepted=dialog.exec()==QDialog.DialogCode.Accepted
        if accepted and self.set_display(**values()):return
        if {key:self.config.values.get(key,{'theme':'dark','spacing':4,'box_spacing':35,'grid_snap':False,'menu_grid_snap':False}.get(key)) for key in original}!=original:
            self.set_display(persist=False,**original)

    def safety_options(self):
        if job_busy(self.job):
            QMessageBox.warning(self, 'Stop training first', 'Stop the active run before changing run safety.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(APP_NAME+' — Run safety')
        dialog.setMinimumWidth(max(480, self.config.values['font_size'] * 30))
        dialog.setSizeGripEnabled(True)
        layout = QFormLayout(dialog)
        percentage = QDoubleSpinBox(dialog)
        percentage.setDecimals(2); percentage.setRange(.01, 100)
        percentage.setSuffix(' %'); percentage.setValue(self.config.values['memory_limit_percent'])
        layout.addRow('Maximum share of available RAM', percentage)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, dialog)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_display(memory_limit_percent=percentage.value())

    def poll_workspace(self):
        workspace = self.workspace
        if not workspace or not workspace.path or job_busy(self.job):
            return
        try:
            current_stamp = stamp(workspace.path)
            if current_stamp == workspace.disk_stamp:
                return
            current_revision = revision(workspace.path)
        except OSError as error:
            if self.external_notice != str(error):
                self.external_notice = str(error)
                self.broadcast('externalConflict', 'Cannot read changed workspace: '+str(error))
            return
        if current_revision == workspace.disk_revision:
            workspace.disk_stamp = current_stamp
            return
        if workspace.dirty:
            if self.external_notice != ('conflict', current_revision, current_stamp):
                self.external_notice = ('conflict', current_revision, current_stamp)
                self.broadcast('externalConflict', 'The workspace changed on disk. Your unsaved edits remain intact. Save As a separate file or use File → Reload workspace from disk.')
            return
        try:
            candidate = Workspace.load(workspace.path)
        except Exception as error:
            if self.external_notice != (current_stamp, str(error)):
                self.external_notice = (current_stamp, str(error))
                self.broadcast('externalConflict', 'Cannot reload changed workspace: '+str(error))
            return
        self.workspace = candidate
        workspace.db.close()
        self.external_notice = None
        self.broadcast()
        for index in range(self.tabs.count()):
            view = self.tabs.widget(index)
            if view.mid not in candidate.models():
                view.bridge.send('navigate', next(iter(candidate.models()), None))
        self.broadcast('externalReload', 'Workspace updated from disk (CLI or another editor).')

    def reload_workspace(self):
        if self.workspace and self.workspace.path:
            if job_busy(self.job):
                QMessageBox.warning(self, 'Stop training first', 'Stop the active run before reloading the workspace.')
                return False
            return self.load_workspace(self.workspace.path)
        return False

    def url_changed(self, view, url):
        fragment = url.fragment()
        view.mid = fragment[6:] if fragment.startswith("model=") else None
        index = self.tabs.indexOf(view)
        if index >= 0:
            self.tabs.setTabText(index, self.tab_name(view.mid))

    def tab_name(self, mid, models=None):
        models=(self.workspace.models() if self.workspace else {}) if models is None else models
        if mid in models:
            return models[mid]["name"]
        return "Home"

    def close_tab(self, index):
        if self.tabs.count() == 1:
            self.current_view().bridge.send("navigate", None)
            return
        view = self.tabs.widget(index)
        self.tabs.removeTab(index)
        view.deleteLater()

    def duplicate_tab(self):
        self.open_tab(self.current_view().mid)

    def recent_files(self):
        raw = self.settings_store.value("recentFiles", "[]")
        try:
            return json.loads(raw)[:12]
        except (ValueError, TypeError):
            return []

    def touch_recent(self, path):
        recent = [r for r in self.recent_files() if r["path"] != path]
        recent.insert(0, {"path": path, "name": Path(path).stem})
        self.settings_store.setValue("recentFiles", json.dumps(recent[:12]))

    def state(self):
        from .recovery_views import header as recovery_header
        models = self.workspace.models() if self.workspace else {}
        app=QApplication.instance()
        system='light' if isinstance(app,QApplication) and app.styleHints().colorScheme()==Qt.ColorScheme.Light else 'dark'
        choice=self.config.values.get('theme','dark') if hasattr(self,'config') else 'dark'
        theme=resolve_theme(choice,system)
        return {
            'history':self.workspace.history.summary() if self.workspace else {'undo':False,'redo':False},
            'theme':theme,'theme_tokens':THEMES[theme],
            'theme_scheme':'light' if theme in ('light','github','math','retrofuturism','bauhaus') else 'dark',
            "binding": BINDING, "path": self.workspace.path if self.workspace else None,
            "workspace_id": self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0] if self.workspace else None,
            "dirty": bool(self.workspace and self.workspace.dirty),
            "models": models, "recent": self.recent_files(),
            "training_id": self.job.mid if job_busy(self.job) else None,
            "experiment_job": self.job.snapshot.get('run_id') if job_busy(self.job) and hasattr(self.job, 'snapshot') else None,
            "notebook_job": {**{k: self.job.snapshot[k] for k in ('id', 'workspace_id', 'selection')},
                             'operation': self.job.snapshot.get('operation', 'execute')} if job_busy(self.job)
                and hasattr(self.job, 'snapshot') and 'selection' in self.job.snapshot and 'timeout' in self.job.snapshot else None,
            "experiment_recoveries": recovery_header(self),
            "torch_python": self.torch_python,
            "rendering": self.rendering,
            "constrained_mode": self.workspace.constrained_mode if self.workspace else True,
            "layout_preferences": self.workspace.layout_preferences if self.workspace else None,
            "panel_layout": self.workspace.panel_layout if self.workspace else None,
            "panel_scope": self.workspace.panel_scope if self.workspace else None,
            "memory_limit_percent": self.config.values['memory_limit_percent'] if hasattr(self, 'config') else DEFAULT_MEMORY_LIMIT_PERCENT,
            "display": {k: v for k, v in self.config.values.items() if k != 'layout'} if hasattr(self, 'config') else {'font_size':14,'ui_zoom':1.15},
        }

    def broadcast(self, kind="state", payload=None, exclude=None):
        self.update_history_actions()
        value = self.state() if kind == "state" else payload
        models=value['models'] if kind=='state' else self.workspace.models() if self.workspace else {}
        for index in range(self.tabs.count()):
            bridge = self.tabs.widget(index).bridge
            if bridge != exclude:
                bridge.send(kind, value)
        for index in range(self.tabs.count()):
            view = self.tabs.widget(index)
            self.tabs.setTabText(index, self.tab_name(view.mid,models))
        self.update_title()

    def update_title(self):
        path = self.workspace.path if self.workspace else None
        name = Path(path).name if path else ("Untitled workspace" if self.workspace else "Welcome")
        dirty = " • unsaved" if self.workspace and self.workspace.dirty else ""
        self.setWindowTitle(f"{name}{dirty} — {APP_NAME}")

    def can_replace(self):
        if job_busy(self.job):
            QMessageBox.information(self, "Training in progress", "Stop training before changing workspaces.")
            return False
        if self.workspace and self.workspace.dirty:
            answer = QMessageBox.question(
                self, "Unsaved workspace", "Save changes to this workspace?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                return False
            if answer == QMessageBox.StandardButton.Save:
                return self.save_workspace()
        return True

    def reset_tabs(self, mid=None):
        while self.tabs.count() > 1:
            view = self.tabs.widget(1)
            self.tabs.removeTab(1)
            view.deleteLater()
        self.current_view().bridge.send("state", self.state())
        self.current_view().bridge.send("navigate", mid)
        self.update_title()

    def new_workspace(self):
        if not self.can_replace():
            return False
        old = self.workspace
        self.workspace = Workspace.demo()
        if old:
            old.db.close()
        self.reset_tabs(next(iter(self.workspace.models())))
        return True

    def open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Visual Torch Workspace", "", "Visual Torch Workspace (*.vtw)")
        if path:
            return self.load_workspace(path)
        return False

    def load_workspace(self, path):
        if not self.can_replace():
            return False
        try:
            candidate = Workspace.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open workspace", str(exc))
            return False
        old = self.workspace
        self.workspace = candidate
        if old:
            old.db.close()
        self.touch_recent(candidate.path)
        self.reset_tabs(next(iter(candidate.models()), None))
        return True

    def save_workspace(self, save_as=False):
        if not self.workspace:
            return False
        path = self.workspace.path
        if save_as or not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Visual Torch Workspace", path or "Untitled.vtw",
                "Visual Torch Workspace (*.vtw)",
            )
            if not path:
                return False
        try:
            self.workspace.save(path)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot save workspace", str(exc))
            return False
        self.touch_recent(self.workspace.path)
        self.broadcast()
        return True

    def update_history_actions(self):
        if not hasattr(self,'undo_action'):return
        history=self.workspace.history.summary() if self.workspace else {'undo':False,'redo':False}
        active=job_busy(self.job)
        self.undo_action.setEnabled(history['undo'] and not active)
        self.redo_action.setEnabled(history['redo'] and not active)

    def dispatch(self, req, bridge):
        action=req.get('action')
        if action in {'undo','redo'}:
            if not self.workspace:return False
            if job_busy(self.job):raise ValueError('Stop training before undo/redo.')
            changed=self.workspace.history.restore(self.workspace,redo=action=='redo')
            if changed:
                self.broadcast()
                if hasattr(self,'tabs'):
                    models=self.workspace.models()
                    for index in range(self.tabs.count()):
                        view=self.tabs.widget(index)
                        if view.mid not in models:view.bridge.send('navigate',next(iter(models),None))
            return changed
        from .desktop_experiments import EDITS
        from .desktop_construction import EDITS as CONSTRUCTION_EDITS
        edits={'newModel','deleteModel','updateModel','editParameterPolicy','resetBlockParameterPolicy','bindParameter','addRecurrent','editRecurrent','addLSTM','editLSTM','addGaussian','editGaussian','generateData','setConstrainedMode','rebindDataset','setLayoutPreferences'} | EDITS | CONSTRUCTION_EDITS
        if job_busy(self.job) and action in edits | {'train', 'scienceTrain'}:
            raise ValueError('Stop the current run before editing or starting another run.')
        workspace=self.workspace
        if action == 'scienceDraftCommit' and workspace:
            # Admission before the potentially large undo SQL snapshot, not just
            # after dispatch reaches the draft service. No bytes are admitted here.
            from .authoring_sessions import _json
            from .memory import memory_percent
            req = _json(req)
            sessions = getattr(self,'authoring_sessions',None)
            if sessions is None: raise ValueError('Unknown draft session for this owner.')
            configuration = getattr(self,'config',None)
            sessions.memory_limit_percent = memory_percent(configuration.values.get('memory_limit_percent',10.) if configuration else 10.)
            sessions.admit_commit(workspace,req.get('session'),bridge if bridge is not None else self,req.get('sequence'))
        before=workspace.history.capture(workspace.db) if workspace and action in edits else None
        changes=workspace.db.total_changes if workspace else 0
        try:result=Window._dispatch(self,req,bridge)
        except Exception:
            if workspace:workspace.history.discard(before)
            raise
        if workspace and action in edits:
            if workspace.db.total_changes!=changes or action in CONSTRUCTION_EDITS and result.get('changed',False):workspace.history.record(before,workspace.db)
            else:workspace.history.discard(before)
            if hasattr(self,'update_history_actions'):self.update_history_actions()
            self.broadcast('history',workspace.history.summary())
        return result

    def _dispatch(self, req, bridge):
        action = req.get("action")
        from .desktop_construction import ACTIONS,dispatch as construction_dispatch
        if action in ACTIONS:return construction_dispatch(self,req)
        if action == 'openHelpResource':
            resources=HELP_RESOURCES
            if set(req) != {'action','id'} or not isinstance(req['id'],str) or req['id'] not in resources:
                raise ValueError('Unknown help resource.')
            if not QDesktopServices.openUrl(QUrl(resources[req['id']])):
                raise ValueError('Could not open the help resource in your browser.')
            return True
        if action == "state":
            return self.state()
        if action == "newWorkspace":
            return self.new_workspace()
        if action == "openWorkspace":
            return self.load_workspace(req["path"]) if req.get("path") else self.open_dialog()
        if action == "save":
            return self.save_workspace(req.get("saveAs", False))
        if action == "home":
            bridge.send("navigate", None)
            return True
        if action == 'setLayoutPreferences':
            if not self.workspace:
                raise ValueError('Open a workspace before saving its layout preferences.')
            if job_busy(self.job):
                raise ValueError('Stop training before changing workspace layout preferences.')
            self.workspace.set_layout_preferences(req['layout'])
            self.broadcast()
            return self.workspace.layout_preferences
        if not self.workspace:
            raise ValueError("Create or open a workspace first.")
        if action == 'setPanelLayout':
            if set(req) != {'action', 'workspace_id', 'panel_scope', 'expected', 'layout'}:
                raise ValueError('Panel arrangement request has missing or unsupported fields.')
            from .presentation import validate_panel_layout
            wid=self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            if req['workspace_id'] != wid or req['panel_scope'] != self.workspace.panel_scope:
                raise ValueError('Workspace changed; review its panel arrangement before saving.')
            if validate_panel_layout(req['expected']) != self.workspace.panel_layout:
                raise ValueError('Saved panel arrangement changed; restore or review it before overwriting.')
            self.workspace.set_panel_layout(req['layout'])
            result={'workspace_id':wid,'panel_scope':self.workspace.panel_scope,
                    'layout':self.workspace.panel_layout,'dirty':self.workspace.dirty}
            self.broadcast('panelLayout',result)
            return result
        from .desktop_experiments import ACTIONS, dispatch as science_dispatch
        if action in ACTIONS: return science_dispatch(self, req, bridge)
        models = self.workspace.models()
        if action=='layoutParameterCounts':
            if set(req)!={'action','workspace_id','panel_scope','id','graph'}:
                raise ValueError('Parameter layout request has missing or unsupported fields.')
            wid=self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            mid=req['id']
            if req['workspace_id']!=wid or req['panel_scope']!=self.workspace.panel_scope or mid not in models:
                raise ValueError('Workspace changed; retry parameter sizing.')
            expected=copy.deepcopy(models[mid]['graph'])
            for n in expected['nodes']:
                n.setdefault('x',0);n.setdefault('y',0)
                if n['type']=='Linear':n.setdefault('bias',True)
            if req['graph']!=expected:raise ValueError('Architecture changed; retry parameter sizing after its edits are saved.')
            from .policy_authoring import trainable_block_counts
            return {'workspace_id':wid,'panel_scope':self.workspace.panel_scope,'id':mid,
                    'counts':trainable_block_counts(models,mid)}
        if action == "setConstrainedMode":
            if job_busy(self.job):
                raise ValueError("Stop the current run before changing its safety mode.")
            self.workspace.set_constrained_mode(req["enabled"])
            self.broadcast()
            return self.workspace.constrained_mode
        if action == "openModel":
            mid = req["id"]
            if mid not in models:
                raise ValueError("That architecture no longer exists.")
            if req.get("newTab"):
                if req.get('clearMenus') is True:self.open_tab(mid,clear_menus=True)
                else:self.open_tab(mid)
            else:
                bridge.send('navigateClean' if req.get('clearMenus') is True else 'navigate',mid)
            return True
        if action == "newModel":
            if job_busy(self.job):
                raise ValueError("Stop training before creating an architecture.")
            kind = req.get("kind", "linear")
            if kind == "mlp":
                doc = default_mlp(req.get("name", "MLP regression"), req.get("inputs", 2),
                                  req.get("outputs", 1), req.get("hidden", [16, 16]),
                                  req.get("activation", "Tanh"))
            elif kind == "linear":
                doc = default_model(req.get("name", "Untitled architecture"))
            else:
                raise ValueError("Choose a linear or MLP architecture.")
            # GUI-created architectures start with readable port names; the box
            # title is the port name. CLI/test defaults keep their port IDs.
            for node in doc["graph"]["nodes"]:
                if node["type"] in DEFAULT_PORT_NAMES:
                    node.setdefault("label", DEFAULT_PORT_NAMES[node["type"]])
            self.workspace.put(doc)
            self.broadcast()
            bridge.send("navigate", doc["id"])
            return doc["id"]
        if action == "deleteModel":
            if job_busy(self.job):
                raise ValueError("Stop training before deleting an architecture.")
            self.workspace.delete(req["id"])
            self.broadcast()
            bridge.send("navigate", next(iter(self.workspace.models())))
            return True
        if action == "updateModel":
            doc = copy.deepcopy(req["document"])
            if doc["id"] not in models:
                raise ValueError("That architecture no longer exists.")
            if job_busy(self.job):
                raise ValueError("Stop training before editing an architecture.")
            if 'preserve_parameter_policy' in req:
                if type(req['preserve_parameter_policy']) is not bool: raise ValueError('Preserve-parameter-policy flag must be boolean.')
                if req['preserve_parameter_policy']:
                    policy = models[doc['id']]['graph'].get('parameter_policy')
                    if policy is None: doc['graph'].pop('parameter_policy', None)
                    else: doc['graph']['parameter_policy'] = copy.deepcopy(policy)
            self.workspace.put(doc)
            self.broadcast(exclude=bridge)
            return self.describe(doc["id"])
        if action in {'parameterPolicy', 'editParameterPolicy', 'resetBlockParameterPolicy', 'bindParameter', 'parameterSourceInventory'}:
            from .policy_authoring import prepare, build, reset_block, TOKEN
            from .evidence import digest
            writing = action in {'editParameterPolicy', 'resetBlockParameterPolicy', 'bindParameter'}
            expected = {'action', 'workspace_id', 'panel_scope', 'id', 'node'}
            if writing: expected.add('architecture')
            if action == 'editParameterPolicy': expected |= {'selector', 'changes_json'}
            if action == 'bindParameter': expected |= {'selector', 'source'}
            if action == 'parameterSourceInventory': expected |= {'architecture', 'run', 'attempt', 'offset'}
            if set(req) != expected:
                raise ValueError('Parameter policy request has missing or unsupported fields.')
            wid = self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            if req['workspace_id'] != wid or req['panel_scope'] != self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen parameter policy.')
            if action == 'parameterSourceInventory':
                from .parameter_sources import inventory_page
                info = prepare(models, req['id'], req['node'])
                if req['architecture'] != info['architecture']: raise ValueError('Architecture changed; reopen parameter policy.')
                return inventory_page(self.workspace, req['run'], attempt=req['attempt'], offset=req['offset'])
            if writing:
                token = {key: req[key] for key in TOKEN}
                if action == 'bindParameter':
                    from .parameter_binding import admit
                    receipt = admit(self.workspace, token, req['selector'], req['source'])
                else:
                    doc = reset_block(models, token) if action == 'resetBlockParameterPolicy' else build(models, token, req['selector'], req['changes_json'])
                    if digest(doc) != digest(models[req['id']]): self.workspace.put(doc)
                self.broadcast(exclude=bridge)
            current = self.workspace.models()
            result = prepare(current, req['id'], req['node'])
            if result['policy_error'] is None:
                from .parameter_binding import current_bindings
                bindings = current_bindings(self.workspace, req['id'])
                for owner in result['owners']: owner['initial_source'] = bindings.get(owner['selector'])
            if writing: result['document'] = current[req['id']]
            if action == 'bindParameter': result['binding'] = receipt
            return result
        if action=='addGaussian':
            if set(req)!={'action','id','architecture','workspace_id','panel_scope'}:
                raise ValueError('Gaussian component authoring needs exact workspace and architecture context.')
            if req['workspace_id']!=self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0] or req['panel_scope']!=self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen Gaussian authoring.')
            from .gaussian_authoring import add
            document,nid=add(models,req['id'],req['architecture'])
            self.workspace.put(document);self.broadcast()
            return {'id':req['id'],'node':nid,'scope':'exact HMM p=0 component; no task/training or neural MS-VAR'}
        if action=='editGaussian':
            if set(req)!={'action','id','node','architecture','fields','workspace_id','panel_scope'}:
                raise ValueError('Filter editing needs exact workspace/node context and closed fields.')
            if req['workspace_id']!=self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0] or req['panel_scope']!=self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen filter inspection.')
            from .gaussian_authoring import edit
            document=edit(models,req['id'],req['node'],req['architecture'],req['fields'])
            self.workspace.put(document);self.broadcast()
            return {'id':req['id'],'node':req['node']}
        if action in {'lstmAuthoring','lstmEditing','addLSTM','editLSTM'}:
            from .lstm_authoring import prepare, prepare_edit, build, build_edit, CREATE_TOKEN, EDIT_TOKEN
            from .revisions import _parse
            editing = action in {'lstmEditing','editLSTM'}
            writing = action in {'addLSTM','editLSTM'}
            fields = EDIT_TOKEN if editing else CREATE_TOKEN
            expected = {'action','workspace_id','panel_scope','id'} | ({'node'} if editing else set())
            if writing:
                expected |= {'root_graph','root_model','options_json'}
            if set(req) != expected:
                raise ValueError('LSTM authoring has missing or unsupported fields.')
            wid = self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            if req['workspace_id'] != wid or req['panel_scope'] != self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen LSTM authoring.')
            if not writing:
                info = prepare_edit(models, req['id'], req['node']) if editing else prepare(models, req['id'])
                return {**info,'workspace_id':wid,'panel_scope':self.workspace.panel_scope}
            if not isinstance(req['options_json'],str) or len(req['options_json']) > 65536:
                raise ValueError('LSTM options must be JSON text of at most 65536 characters.')
            document,nid = (build_edit if editing else build)(models,{k:req[k] for k in fields},_parse(req['options_json']))
            if document != models[document['id']]:
                self.workspace.put(document);self.broadcast()
            return {'id':document['id'],'node':nid,'workspace_id':wid,'panel_scope':self.workspace.panel_scope}
        if action in {'recurrentEditing', 'editRecurrent'}:
            from .recurrent_reconfiguration import prepare_edit, build_edit, TOKEN
            from .revisions import _parse
            expected = {'action', 'workspace_id', 'panel_scope', 'id', 'ref', 'node'}
            if action == 'editRecurrent':
                expected |= {'root_graph', 'root_model', 'transition', 'options_json'}
            if set(req) != expected:
                raise ValueError('Recurrent editing has missing or unsupported fields.')
            wid = self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            if req['workspace_id'] != wid or req['panel_scope'] != self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen recurrent editing.')
            if action == 'recurrentEditing':
                return {**prepare_edit(models, req['id'], req['node'], req['ref']),
                        'workspace_id': wid, 'panel_scope': self.workspace.panel_scope}
            if not isinstance(req['options_json'], str) or len(req['options_json']) > 65536:
                raise ValueError('Recurrent interface JSON must be text of at most 65536 characters.')
            document, nid = build_edit(models, {key: req[key] for key in TOKEN}, _parse(req['options_json']))
            if document != models[document['id']]:
                self.workspace.put(document)
                self.broadcast()
            return {'id': document['id'], 'node': nid, 'workspace_id': wid, 'panel_scope': self.workspace.panel_scope}
        if action in {'recurrentAuthoring', 'addRecurrent'}:
            from .recurrent_authoring import prepare, build
            expected = {'action', 'workspace_id', 'panel_scope', 'id', 'ref'}
            if action == 'addRecurrent':expected |= {'root_graph', 'transition', 'options'}
            if set(req) != expected:
                raise ValueError('Recurrent authoring has missing or unsupported fields.')
            wid = self.workspace.db.execute("SELECT value FROM metadata WHERE key='workspace_id'").fetchone()[0]
            if req['workspace_id'] != wid or req['panel_scope'] != self.workspace.panel_scope:
                raise ValueError('Workspace changed; reopen recurrent authoring.')
            if action == 'recurrentAuthoring':
                return {**prepare(models, req['id'], req['ref']), 'workspace_id': wid, 'panel_scope': self.workspace.panel_scope}
            token = {key: req[key] for key in ('id', 'ref', 'root_graph', 'transition')}
            document, nid = build(models, token, req['options'])
            self.workspace.put(document)
            self.broadcast()
            return {'id': document['id'], 'node': nid, 'workspace_id': wid, 'panel_scope': self.workspace.panel_scope}
        if action == "describe":
            return self.describe(req["id"])
        if action == "generateData":
            if job_busy(self.job):
                raise ValueError("Stop training before changing the dataset.")
            doc = models[req["id"]]
            data = generate_dataset(models, doc["id"], req.get("samples", 256),
                                    req.get("noise", 0.), req.get("seed", 7), req.get("kind", "auto"),
                                    constrained=self.workspace.constrained_mode,
                                    memory_limit_percent=self.state()['memory_limit_percent'])
            doc.update(dataset=data, result=None)
            doc["training"]["seed"] = data["seed"]
            self.workspace.put(doc)
            self.broadcast()
            return {"rows": len(data["x"]), "kind": data["truth"]["kind"]}
        if action == "compile":
            return generate_code(models, req["id"])
        if action == 'rebindDataset':
            if job_busy(self.job):
                raise ValueError('Stop training before rebinding dataset columns.')
            from .evidence import binding_for, dataset_report
            doc=models[req['id']]
            contract=validate_model(models,doc['id'])
            report=dataset_report(doc,contract)
            if not report['can_rebind']:
                raise ValueError('Rebinding is only needed for unbound or incompatible columns; '+report['reason'])
            answer=QMessageBox.question(self, APP_NAME+' — Rebind dataset',
                'Use the current declared input/output port order for these existing numeric columns? '
                'This reinterprets column meaning without moving values. The current result will be cleared; '
                'historical runs remain unchanged. Review Column bindings before continuing.',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer!=QMessageBox.StandardButton.Yes:return False
            # A modal dialog can process queued external workspace notifications.
            if self.workspace.models().get(doc['id'])!=doc:
                raise ValueError('Architecture changed during confirmation. Review its bindings again.')
            doc.update(dataset_binding=binding_for(contract),result=None)
            self.workspace.put(doc)
            self.broadcast()
            return True
        if action == "copyText":
            # Read-only: no workspace edit, undo step or dirty flag.
            text = req.get("text")
            if not isinstance(text, str) or len(text) > 8 * 1024 * 1024:
                raise ValueError("Copy needs text of at most 8 MiB.")
            QApplication.clipboard().setText(text)
            return True
        if action == "exportCode":
            code = generate_code(models, req["id"])
            path, _ = QFileDialog.getSaveFileName(self, "Export PyTorch model", "model.py", "Python module (*.py)")
            if path:
                Path(path).write_text(code, encoding="utf-8")
            return bool(path)
        if action == "train":
            if job_busy(self.job):
                raise ValueError("A training run is already active.")
            doc = models[req["id"]]
            data = doc.get("dataset")
            preflight(models, req["id"], self.workspace.constrained_mode,
                      samples=len(data["x"]) if isinstance(data, dict) and isinstance(data.get("x"), list) else None,
                      epochs=doc.get("training", {}).get("epochs", 400),
                      memory_limit_percent=self.state()['memory_limit_percent'])
            from .evidence import dataset_report
            admission=dataset_report(doc,validate_model(models,doc['id']))
            if not admission['allowed']:raise ValueError(admission['reason'])
            if not self.torch_python:
                raise ValueError("Select a Python with PyTorch using --torch-python /path/to/python.")
            if self.job:
                self.job.deleteLater()
            self.job = TrainingJob(models, req["id"], self, self.torch_python,
                                   constrained=self.workspace.constrained_mode,
                                   memory_limit_percent=self.state()['memory_limit_percent'])
            self.job.point.connect(lambda payload: self.broadcast("progress", json.loads(payload)))
            self.job.completed.connect(self.training_complete)
            self.job.failed.connect(lambda error: self.broadcast("trainingError", error))
            self.job.finished.connect(lambda: self.broadcast())
            self.job.start()
            self.broadcast()
            return True
        if action == "stop":
            if self.job:
                self.job.stop()
            return True
        raise ValueError("Unknown command.")

    def describe(self, mid):
        models = self.workspace.models()
        try:
            percentage=self.config.values['memory_limit_percent'] if hasattr(self,'config') else DEFAULT_MEMORY_LIMIT_PERCENT
            from .evidence import dataset_report, checkpoint_report, identity
            contract=validate_model(models,mid)
            from .policy_authoring import block_summary
            interfaces = {n["id"]: validate_model(models, n["ref"]) for n in models[mid]["graph"]["nodes"] if n["type"] in {"Module", "Recurrent"}}
            return {"valid": True, "contract": contract, "interfaces": interfaces,
                    'architecture':identity(models,mid)['architecture'],
                    'parameter_blocks': block_summary(models, mid),
                    'dataset_admission':dataset_report(models[mid],contract),
                    'checkpoint_status':checkpoint_report(models,mid),
                    "code": generate_code(models, mid),
                    "model_safety": safety_report(models, mid, self.workspace.constrained_mode,
                        memory_limit_percent=percentage),
                    "run_safety": safety_report(models, mid, self.workspace.constrained_mode,
                        samples=len(models[mid]["dataset"]["x"]) if models[mid].get("dataset") else None,
                        epochs=models[mid].get("training", {}).get("epochs", 400),
                        memory_limit_percent=percentage)}
        except Exception as exc:
            invalid = {"valid": False, "error": str(exc)}
            if isinstance(exc, GraphError) and mid in models:
                # Display-only: locate problems and keep determinable shapes visible.
                from .graph import partial_shapes
                invalid.update(problems=exc.located(), partial_shapes=partial_shapes(models, mid))
            return invalid

    def training_complete(self, payload):
        result = json.loads(payload)
        mid = self.job.mid
        from .evidence import normalize_models, check_checkpoint
        try:
            check_checkpoint(normalize_models(self.job.models,mid),mid,result)
        except ValueError as error:
            self.broadcast('trainingError','Worker returned an invalid checkpoint: '+str(error))
            return
        doc = self.workspace.models()[mid]
        doc["result"] = result
        self.workspace.put(doc)
        self.workspace.record_run(mid, result, self.job.models)
        current=self.workspace.models()[mid].get('result')
        if current is not None:self.broadcast("complete", {"id": mid, "result": current})
        else:self.broadcast('externalConflict','Run retained in history; its snapshot no longer matches the current draft.')
        self.broadcast()

    def closeEvent(self, event):
        if getattr(self.job, 'awaiting_terminal', False):
            # Do not block the GUI event loop that must admit queued evidence.
            self.job.stop()
            self.broadcast('externalConflict', 'Stopping active work; close again after its terminal evidence has been retained.')
            event.ignore()
            return
        from .desktop_experiments import recovery_rows
        if any(not row['exported_to'] for row in recovery_rows(self)):
            answer=QMessageBox.warning(self, 'Unexported experiment recovery evidence',
                'Some terminal evidence could not be admitted. It is only retained in this session. '
                'Export it from Experiment → Experiments & history before closing, or explicitly discard it.',
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Discard:
                event.ignore(); return
        if self.job and self.job.isRunning():
            self.job.stop()
            self.job.wait()
        if self.can_replace():
            event.accept()
        else:
            event.ignore()

    def smoke_ready(self, ok):
        if not ok:
            return
        # Exercise the actual JS→QWebChannel→Python path, not just HTML loading.
        script = """
        (async () => {
            for (let i=0; i<40 && !window.loom; i++)
                await new Promise(r => setTimeout(r, 100));
            await window.loom.call('newWorkspace');
            await new Promise(r => setTimeout(r, 600));
            const s = await window.loom.call('state');
            const id = Object.keys(s.models)[0];
            const d = await window.loom.call('describe', {id});
            return JSON.stringify({bridge:true, modelCount:Object.keys(s.models).length,
                valid:d.valid, title:document.title, nodes:document.querySelectorAll('.node').length});
        })()
        """
        # Qt does not unwrap JS Promises; resolve via a temporary DOM attribute.
        script = script.rstrip() + ".then(r => document.body.dataset.smoke = r)"
        self.current_view().page().runJavaScript(script)
        QTimer.singleShot(2500, self.smoke_read)

    def smoke_read(self):
        self.current_view().page().runJavaScript(
            "document.body.dataset.smoke || ''", self.smoke_finish)

    def smoke_finish(self, value):
        if not value:
            QTimer.singleShot(500, self.smoke_read)
            return
        self.smoke_path.write_text(value, encoding="utf-8")
        self.workspace.dirty = False
        QApplication.instance().quit()

    def smoke_timeout(self):
        if not self.smoke_path.exists():
            self.smoke_path.write_text('{"error":"Qt smoke test timed out"}', encoding="utf-8")
        if self.workspace:
            self.workspace.dirty = False
        QApplication.instance().quit()


def find_torch_python():
    explicit = os.environ.get("VTW_TORCH_PYTHON")
    candidates = [explicit, sys.executable, shutil.which("python"), shutil.which("python3")]
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            check = subprocess.run(
                [candidate, "-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"],
                capture_output=True, timeout=8,
            )
            if check.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description=APP_NAME+" visual linear/MLP model editor")
    parser.add_argument("workspace", nargs="?", help="Visual Torch Workspace (.vtw)")
    parser.add_argument("--torch-python", help="Python executable for the PyTorch training worker")
    parser.add_argument("--renderer", choices=("auto", "software", "native"), default="auto",
                        help="Webview rendering: auto applies a scoped Qt 6.9.2/Wayland workaround; "
                             "software disables WebEngine GPU acceleration; native uses Qt defaults")
    parser.add_argument("--smoke-test", metavar="RESULT_JSON", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rendering = configure_renderer(args.renderer, qWebEngineVersion())
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setDesktopFileName(DESKTOP_ID)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parents[1] / ICON_FILE)))
    window = Window(args.workspace, args.smoke_test, args.torch_python, rendering)
    print(f"{APP_NAME}: {BINDING}; SQLite .vtw workspace; PyTorch training", flush=True)
    print(f"WebEngine {rendering['webengine_version']}: {rendering['mode']} — {rendering['reason']}", flush=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
