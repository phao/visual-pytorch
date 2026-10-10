import copy
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from app.core import (GraphError, Workspace, compile_model, default_model,
                      generate_code, train_model, validate_dataset, validate_model)


def observations(n=256):
    rng = random.Random(31)
    x = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    return {"x": x, "y": [[2 * v[0] - 1.5 * v[1] + .4] for v in x],
            "truth": {"A": [[2., -1.5]], "b": [.4]}}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Workspace.demo()
        self.models = self.workspace.models()
        self.mid = next(iter(self.models))

    def tearDown(self):
        self.workspace.db.close()

    def test_affine_contract_and_parameter_count(self):
        contract = validate_model(self.models, self.mid)
        self.assertEqual((contract["inputs"], contract["outputs"], contract["parameters"]), (2, 1, 3))
        model = compile_model(self.models, self.mid)
        self.assertEqual(tuple(model(torch.randn(5, 2)).shape), (5, 1))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3)

    def test_nested_reference_export_matches_compiler(self):
        mid = list(self.models)[2]
        model = compile_model(self.models, mid)
        namespace = {}
        exec(compile(generate_code(self.models, mid), "<generated>", "exec"), namespace)
        exported = namespace["model"]
        exported.load_state_dict(model.state_dict())
        x = torch.randn(5, 2)
        torch.testing.assert_close(exported(x), model(x))
        self.assertEqual(sum(p.numel() for p in model.parameters()), validate_model(self.models, mid)["parameters"])

    def test_each_reference_has_independent_weights(self):
        block = default_model("Square projection", 2, 2)
        main = default_model()
        main["graph"]["nodes"].insert(1, {"id": "ref1", "type": "Module", "ref": block["id"], "x": 250, "y": 170})
        main["graph"]["nodes"].insert(2, {"id": "ref2", "type": "Module", "ref": block["id"], "x": 500, "y": 170})
        main["graph"]["edges"] = [
            {"from": "input", "to": "ref1"}, {"from": "ref1", "to": "ref2"},
            {"from": "ref2", "to": "linear1"}, {"from": "linear1", "to": "output"}]
        models = {block["id"]: block, main["id"]: main}
        model = compile_model(models, main["id"])
        self.assertNotEqual(model.layer_0.layer_0.weight.data_ptr(), model.layer_1.layer_0.weight.data_ptr())

    def test_recover_known_affine_parameters(self):
        self.models[self.mid]["dataset"] = observations()
        result = train_model(self.models, self.mid)
        self.assertLess(result["mse"], 1e-8)
        self.assertLess(result["A_error"], 1e-4)
        self.assertLess(result["b_error"], 1e-4)
        self.assertEqual(result["train_rows"] + result["test_rows"], 256)

    def test_composed_model_recovery(self):
        mid = list(self.models)[2]
        self.models[mid]["dataset"] = observations()
        self.models[mid]["training"]["epochs"] = 500
        result = train_model(self.models, mid)
        self.assertLess(result["mse"], 1e-7)
        self.assertLess(result["A_error"], .001)

    def test_disconnected_graph_rejected(self):
        self.models[self.mid]["graph"]["edges"].pop()
        with self.assertRaisesRegex(GraphError, "Connect every block"):
            validate_model(self.models, self.mid)

    def test_ordinary_multiple_inputs_need_explicit_join(self):
        self.models[self.mid]["graph"]["edges"].append({"from": "input", "to": "output"})
        with self.assertRaisesRegex(GraphError, "use Concat"):
            validate_model(self.models, self.mid)

    def test_reference_shape_mismatch_rejected(self):
        composed = self.models[list(self.models)[2]]
        composed["graph"]["nodes"][0]["features"] = 4
        with self.assertRaisesRegex(GraphError, "needs 2 features"):
            validate_model(self.models, composed["id"])

    def test_reference_cycle_rejected(self):
        a, b = default_model("A"), default_model("B")
        for owner, target in [(a, b), (b, a)]:
            owner["graph"]["nodes"][1] = {
                "id": "linear1", "type": "Module", "ref": target["id"], "x": 340, "y": 170}
        with self.assertRaisesRegex(GraphError, "reference cycle"):
            validate_model({a["id"]: a, b["id"]: b}, a["id"])

    def test_sqlite_roundtrip_collection_data_and_runs(self):
        doc = self.models[self.mid]
        doc["dataset"] = observations(20)
        doc["result"] = {"mse": 0.0}
        self.workspace.put(doc)
        self.workspace.record_run(self.mid, {"mse": 0.0}, self.models)
        with tempfile.TemporaryDirectory() as folder:
            path = self.workspace.save(Path(folder) / "trial.vtw")
            self.assertEqual(Path(path).read_bytes()[:16], b"SQLite format 3\x00")
            loaded = Workspace.load(path)
            try:
                self.assertEqual(loaded.models(), self.workspace.models())
                self.assertEqual(loaded.db.execute("SELECT COUNT(*) FROM architecture_refs").fetchone()[0], 1)
                self.assertEqual(loaded.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)
                self.assertFalse(loaded.dirty)
            finally:
                loaded.db.close()

    def test_referenced_architecture_cannot_be_deleted(self):
        with self.assertRaisesRegex(GraphError, "referenced"):
            self.workspace.delete(list(self.models)[1])

    def test_reference_edit_invalidates_owner_result(self):
        owner = self.models[list(self.models)[2]]
        owner["result"] = {"mse": 0.0}
        self.workspace.put(owner)
        block = self.models[list(self.models)[1]]
        block["graph"]["nodes"][1]["features"] = 4
        self.workspace.put(block)
        self.assertIsNone(self.workspace.models()[owner["id"]]["result"])

    def test_layout_edit_preserves_result(self):
        doc = self.models[self.mid]
        doc["result"] = {"mse": 0.0}
        self.workspace.put(doc)
        doc["graph"]["nodes"][1]["x"] += 50
        self.workspace.put(doc)
        self.assertEqual(self.workspace.models()[self.mid]["result"], {"mse": 0.0})

    def test_unknown_workspace_version_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.workspace.save(Path(folder) / "future.vtw")
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA user_version=999")
            with self.assertRaisesRegex(GraphError, "supported"):
                Workspace.load(path)

    def test_dataset_nan_and_shape_rejected(self):
        data = observations(8)
        data["x"][0][0] = float("nan")
        with self.assertRaisesRegex(GraphError, "finite"):
            validate_dataset(data, 2, 1)
        with self.assertRaisesRegex(GraphError, "3 values"):
            validate_dataset(observations(8), 3, 1)

    def test_worker_protocol_runs_real_training(self):
        self.models[self.mid]["dataset"] = observations(32)
        self.models[self.mid]["training"]["epochs"] = 10
        process = subprocess.run(
            [sys.executable, "-m", "app.worker"],
            input=json.dumps({"models": self.models, "id": self.mid}),
            capture_output=True, text=True, timeout=40)
        self.assertEqual(process.returncode, 0, process.stderr)
        events = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(events[-1]["kind"], "result")
        self.assertTrue(any(e["kind"] == "progress" for e in events))

    def test_cancellation_returns_partial_result(self):
        self.models[self.mid]["dataset"] = observations(32)
        cancel = threading.Event()
        result = train_model(self.models, self.mid, progress=lambda _: cancel.set(), cancel=cancel)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["epochs"], 1)


if __name__ == "__main__":
    unittest.main()
