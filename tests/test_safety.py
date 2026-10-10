"""Pre-allocation policy, persistence and deliberately small unconstrained runs."""
import ast
import copy
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from app.core import (GraphError, Workspace, check_document, compile_model, default_model,
                      default_mlp, generate_code, generate_dataset, preflight,
                      safety_report, train_model, validate_model)


def chain_references(block, repeats, name="Repeated"):
    d = default_model(name, block["graph"]["nodes"][0]["features"], 1)
    refs = [{"id": f"ref{i}", "type": "Module", "ref": block["id"], "x": 200+i*200, "y": 100}
            for i in range(repeats)]
    d["graph"]["nodes"][1:1] = refs
    nodes = d["graph"]["nodes"]
    d["graph"]["edges"] = [{"from": a["id"], "to": b["id"]} for a,b in zip(nodes,nodes[1:])]
    return d


class SafetyTests(unittest.TestCase):
    def setUp(self):
        ram = patch('app.core.available_memory', return_value=8 * 1024**3)
        ram.start(); self.addCleanup(ram.stop)

    def test_large_specifications_are_valid_saveable_and_exportable_by_default(self):
        w = Workspace()
        self.addCleanup(w.db.close)
        d = default_mlp(inputs=512, outputs=128, hidden=[1024,512])
        w.put(d)
        self.assertTrue(w.constrained_mode)
        c = validate_model(w.models(),d["id"])
        self.assertEqual((c["inputs"],c["outputs"]),(512,128))
        ast.parse(generate_code(w.models(),d["id"]))  # Do not instantiate this model.
        with patch('app.core.available_memory', return_value=512 * 1024**2):
            self.assertFalse(safety_report(w.models(),d["id"])["allowed"])
        self.assertTrue(safety_report(w.models(),d["id"])["allowed"])
        with tempfile.TemporaryDirectory() as folder:
            path=w.save(Path(folder)/"large.vtw")
            loaded=Workspace.load(path)
            self.addCleanup(loaded.db.close)
            self.assertEqual(loaded.models(),w.models())
            self.assertTrue(loaded.constrained_mode)

    def test_toggle_is_persisted_and_reenable_preserves_models_data_and_runs(self):
        w=Workspace()
        self.addCleanup(w.db.close)
        d=default_model(inputs=128)
        d["dataset"]={"x":[[0]*128,[1]*128],"y":[[0],[1]]}
        d["result"]={"weights":{"a":[1,2,3]},"mse":0}
        w.put(d);w.record_run(d["id"],d["result"],{d["id"]:d})
        before=w.models()
        w.set_constrained_mode(False)
        with tempfile.TemporaryDirectory() as folder:
            path=w.save(Path(folder)/"mode.vtw")
            self.assertFalse(w.dirty)
            w.set_constrained_mode(False)
            self.assertFalse(w.dirty)
            loaded=Workspace.load(path)
            self.addCleanup(loaded.db.close)
            self.assertFalse(loaded.constrained_mode)
            loaded.set_constrained_mode(True)
            self.assertTrue(loaded.dirty)
            self.assertEqual(before,loaded.models())
            self.assertEqual(loaded.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0],1)
            loaded.save(path)
            reopened=Workspace.load(path)
            self.addCleanup(reopened.db.close)
            self.assertTrue(reopened.constrained_mode)
            self.assertEqual(before,reopened.models())

    def test_legacy_missing_mode_defaults_on_and_bad_metadata_is_rejected(self):
        w=Workspace.demo();self.addCleanup(w.db.close)
        with tempfile.TemporaryDirectory() as folder:
            path=w.save(Path(folder)/"legacy.vtw")
            with sqlite3.connect(path) as db:
                db.execute("DELETE FROM metadata WHERE key='constrained_mode'")
            legacy=Workspace.load(path);self.addCleanup(legacy.db.close)
            self.assertTrue(legacy.constrained_mode)
            with sqlite3.connect(path) as db:
                db.execute("INSERT INTO metadata VALUES ('constrained_mode','yes')")
            with self.assertRaisesRegex(GraphError,"metadata"):
                Workspace.load(path)

    def test_policy_boolean_and_structural_width_validation_are_not_bypassable(self):
        w=Workspace();self.addCleanup(w.db.close)
        d=default_model()
        for value in ("false",0,None):
            with self.subTest(value=value),self.assertRaises(GraphError):
                w.set_constrained_mode(value)
            with self.assertRaises(GraphError):
                preflight({d["id"]:d},d["id"],value)
        for width in (0,True,1.2,2**53):
            d["graph"]["nodes"][0]["features"]=width
            with self.subTest(width=width),self.assertRaises(GraphError):
                check_document(d)

    def test_large_model_rejected_before_torch_allocation_and_data_generation(self):
        d=default_model(inputs=10**9)
        models={d["id"]:d}
        with patch("torch.nn.Linear",side_effect=AssertionError("allocated")) as allocation:
            with self.assertRaisesRegex(GraphError,"Constrained mode"):
                compile_model(models,d["id"])
            allocation.assert_not_called()
        with patch("app.core.random.Random",side_effect=AssertionError("generated")) as rng:
            with self.assertRaisesRegex(GraphError,"Constrained mode"):
                generate_dataset(models,d["id"],samples=10000)
            rng.assert_not_called()
        with patch("app.core.compile_model",side_effect=AssertionError("allocated")) as compilation:
            with self.assertRaisesRegex(GraphError,"Constrained mode"):
                train_model(models,d["id"])
            compilation.assert_not_called()

    def test_worker_defaults_safe_and_rejects_giant_model_without_result(self):
        d=default_model(inputs=10**9)
        p=subprocess.run([sys.executable,"-m","app.worker"],
            input=json.dumps({"models":{d["id"]:d},"id":d["id"]}),
            capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,1,p.stderr)
        events=[json.loads(line) for line in p.stdout.splitlines()]
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]["kind"],"error")
        self.assertIn("Constrained mode",events[0]["payload"])

    def test_inputs_are_checked_after_concat_not_individually(self):
        d=default_model(inputs=40)
        d["graph"]["nodes"].insert(1,{"id":"extra","type":"Input","features":40})
        d["graph"]["nodes"].insert(2,{"id":"join","type":"Concat","sources":["input","extra"]})
        d["graph"]["edges"]=[{"from":"input","to":"join"},{"from":"extra","to":"join"},
                             {"from":"join","to":"linear1"},{"from":"linear1","to":"output"}]
        models={d["id"]:d}
        self.assertEqual(validate_model(models,d["id"])["inputs"],80)
        report=safety_report(models,d["id"])
        self.assertTrue(report['allowed'])  # No fixed 64-input cap.
        self.assertEqual(report['memory']['parameter_bytes'], 4 * 81)

    def test_limits_count_independent_reference_occurrences_and_nested_widths(self):
        block=default_model("Square",50,50)
        owner=chain_references(block,40)
        models={block["id"]:block,owner["id"]:owner}
        c = validate_model(models,owner['id'])
        self.assertEqual(c['parameters'], 40 * 2550 + 51)
        self.assertEqual(safety_report(models,owner['id'])['memory']['parameter_bytes'], 4 * c['parameters'])
        block=default_mlp("Wide internal",2,2,[128])
        owner=chain_references(block,1)
        models={block["id"]:block,owner["id"]:owner}
        self.assertEqual(validate_model(models,owner["id"])["max_width"],128)
        self.assertTrue(safety_report(models,owner["id"])["allowed"])
        block=default_mlp("Many small ops",1,1,[1]*8)
        owner=chain_references(block,20)
        models={block["id"]:block,owner["id"]:owner}
        self.assertGreater(validate_model(models,owner['id'])['operations'], 256)
        self.assertTrue(safety_report(models,owner['id'])['allowed'])

    def test_dataset_memory_counts_but_epochs_are_not_a_fixed_run_cap(self):
        d=default_model(inputs=64,outputs=64)
        models={d["id"]:d}
        report=safety_report(models,d["id"],samples=10000,epochs=5001)
        self.assertTrue(report['allowed'])
        with patch('app.core.available_memory', return_value=1024**3):
            self.assertEqual({v['rule'] for v in safety_report(models,d['id'],samples=10000,epochs=5001)['violations']},{'memory'})
        self.assertTrue(safety_report(models,d["id"],False,samples=10000,epochs=5001)["allowed"])
        d=default_model(inputs=1)
        models={d["id"]:d}
        data=generate_dataset(models,d["id"],10001)
        self.assertEqual(len(data["x"]),10001)
        d["dataset"]=generate_dataset(models,d["id"],4)
        d["training"]["epochs"]=5001
        cancel=threading.Event();cancel.set()
        result=train_model(models,d["id"],cancel=cancel)
        self.assertEqual(result["epochs"],0)
        self.assertTrue(result["stopped"])

    def test_small_unconstrained_training_and_export_work_above_64_without_identity(self):
        d=default_model(inputs=128,outputs=3)
        models={d["id"]:d}
        d["dataset"]=generate_dataset(models,d["id"],12,constrained=False)
        d["training"]["epochs"]=2
        with patch("torch.eye",side_effect=AssertionError("quadratic identity")):
            r=train_model(models,d["id"],constrained=False)
        self.assertTrue(math.isfinite(r["mse"]))
        self.assertEqual(len(r["A"][0]),128)
        m=compile_model(models,d["id"],constrained=False)
        namespace={}
        exec(generate_code(models,d["id"]),namespace)
        exported=namespace["model"]
        state={k:torch.tensor(v) for k,v in r["weights"].items()}
        m.load_state_dict(state);exported.load_state_dict(state)
        x=torch.randn(2,128)
        torch.testing.assert_close(m(x),exported(x))
        torch.testing.assert_close(m(x),x@torch.tensor(r["A"]).T+torch.tensor(r["b"]))

    def test_ram_budget_blocks_and_unconstrained_can_override(self):
        d=default_model(inputs=10,outputs=10)
        models={d["id"]:d}
        with patch('app.core.available_memory', return_value=1024**3):
            report=safety_report(models,d['id'],samples=10000,epochs=5000)
            self.assertEqual([v['rule'] for v in report['violations']],['memory'])
            with self.assertRaisesRegex(GraphError,'available RAM'):
                preflight(models,d['id'],samples=10000,epochs=5000)
            self.assertTrue(preflight(models,d['id'],False,samples=10000,epochs=5000)['allowed'])
            self.assertTrue(preflight(models,d['id'],samples=10000,epochs=5000,memory_limit_percent=50)['allowed'])
