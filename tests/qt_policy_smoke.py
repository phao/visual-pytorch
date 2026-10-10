"""Real Qt dispatch and worker snapshot checks without starting WebEngine."""
import json
import math
from unittest.mock import patch
from types import SimpleNamespace
from PyQt6.QtCore import QObject, QCoreApplication, QTimer
from app.core import Workspace, default_model, generate_dataset
from app.desktop import Window, find_torch_python


class Harness(QObject):
    def __init__(self):
        super().__init__()
        self.workspace=Workspace()
        self.job=None
        self.torch_python=find_torch_python()
        self.rendering={}
        self.output={}
        self.config=SimpleNamespace(values={'ui_zoom':1.15,'font_size':14,'memory_limit_percent':12.5})

    def recent_files(self):
        return []

    def state(self):
        return Window.state(self)

    def broadcast(self,*args,**kwargs):
        pass

    def describe(self,mid):
        return Window.describe(self,mid)

    def training_complete(self,payload):
        self.output=json.loads(payload)
        Window.training_complete(self,payload)

    def dispatch(self,action,**args):
        return Window.dispatch(self,{"action":action,**args},None)


app=QCoreApplication([])
ram=patch('app.core.available_memory',return_value=512*1024**2);ram.start()
h=Harness()
d=default_model(inputs=128,outputs=3)
d["training"]["epochs"]=2
h.workspace.put(d)
report=h.dispatch("describe",id=d["id"])
assert report["valid"] and not report["model_safety"]["allowed"]
assert h.dispatch("state")["constrained_mode"] is True
try:
    h.dispatch("train",id=d["id"])
except ValueError as error:
    assert "Constrained mode" in str(error)
else:
    raise AssertionError("Unsafe job dispatched")
assert h.job is None
assert h.dispatch("setConstrainedMode",enabled=False) is False
assert h.dispatch("describe",id=d["id"])["model_safety"]["allowed"]
d["dataset"]=generate_dataset(h.workspace.models(),d["id"],12,constrained=False)
h.workspace.put(d)
assert h.dispatch("train",id=d["id"])
assert h.job.constrained is False
assert h.job.memory_limit_percent == 12.5
try:
    h.dispatch("setConstrainedMode",enabled=True)
except ValueError as error:
    assert "Stop" in str(error)
else:
    raise AssertionError("Policy changed during active job")
h.job.failed.connect(lambda error:h.output.update(error=error))
h.job.finished.connect(app.quit)
QTimer.singleShot(15000,h.job.stop)
app.exec()
h.job.wait()
assert "error" not in h.output,h.output
assert math.isfinite(h.output["mse"])
assert len(h.output["A"][0])==128
before=h.workspace.models()
assert h.dispatch("setConstrainedMode",enabled=True) is True
assert h.workspace.models()==before
assert not h.dispatch("describe",id=d["id"])["run_safety"]["allowed"]
assert h.workspace.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]==1
print(json.dumps({"qt_policy":"PASS","input_features":128,"worker_mode":"unconstrained",
    "reenabled_preserves_result":True,"epochs":h.output["epochs"]}))
h.workspace.db.close()
ram.stop()
