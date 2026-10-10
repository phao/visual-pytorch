"""Exercise the real PyQt signal path and separate PyTorch interpreter."""
import json
import random
import sys

from PyQt6.QtCore import QCoreApplication, QTimer
from app.core import Workspace, generate_dataset
from app.desktop import TrainingJob, find_torch_python

app = QCoreApplication([])
workspace = Workspace.demo()
models = workspace.models()
nonlinear = "--mlp" in sys.argv
mid = next(key for key, doc in models.items() if any(n["type"] == "Activation" for n in doc["graph"]["nodes"])) if nonlinear else next(iter(models))
random_source = random.Random(7)
x = [[random_source.uniform(-1, 1), random_source.uniform(-1, 1)] for _ in range(100)]
models[mid]["dataset"] = {"x": x, "y": [[2 * r[0] - r[1] + .2] for r in x]}
if nonlinear:
    models[mid]["dataset"] = generate_dataset(models, mid)
python = find_torch_python()
if not python:
    raise SystemExit("No PyTorch interpreter found.")
job = TrainingJob(models, mid, app, python)
output = {"python": python, "points": 0}
job.point.connect(lambda _: output.update(points=output["points"] + 1))
job.completed.connect(lambda text: output.update(result=json.loads(text)))
job.failed.connect(lambda text: output.update(error=text))
job.finished.connect(app.quit)
QTimer.singleShot(30000, job.stop)
job.start()
app.exec()
job.wait()
workspace.db.close()
print(json.dumps({"engine": "PyQt6 worker", "model": "MLP" if nonlinear else "linear", "points": output["points"],
                  "mse": output.get("result", {}).get("mse"), "error": output.get("error")}))
if "result" not in output or output["result"]["mse"] > (.002 if nonlinear else 1e-6):
    raise SystemExit(1)
