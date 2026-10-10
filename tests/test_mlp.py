import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from app.core import (ACTIVATIONS, GraphError, Workspace, check_document, compile_model,
                      default_model, default_mlp, generate_code, generate_dataset,
                      layer_descriptions, train_model, validate_model)

torch.set_num_threads(1)


def grouped_model(mode="Splitter", activation="Tanh"):
    doc = default_mlp(inputs=3, outputs=3, hidden=[8, 8], activation=activation)
    # Exercise the exact boundary-construction code used by the webview.
    source = """
    let text=''; process.stdin.on('data', x => text+=x); process.stdin.on('end', () => {
      const d=JSON.parse(text), g=require('./app/web/graph-tools.js');
      const extra=g.addInput(d);
      d.graph.nodes.find(n=>n.id==='input').features=2;
      d.graph.nodes.find(n=>n.id==='input').label='measurements';
      d.graph.nodes.find(n=>n.id===extra).label='context';
      g.addRouter(d, process.argv[1], 3);
      process.stdout.write(JSON.stringify(d));
    });
    """
    process = subprocess.run(["node", "-e", source, mode], input=json.dumps(doc),
                             text=True, capture_output=True, check=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=10)
    return json.loads(process.stdout)


def exported(models, mid, compiled):
    scope = {}
    exec(compile(generate_code(models, mid), "<generated>", "exec"), scope)
    result = scope["model"]
    result.load_state_dict(compiled.state_dict())
    return result


class MLPTests(unittest.TestCase):
    def test_mlp_shapes_count_and_all_activation_exports(self):
        for activation in ACTIVATIONS:
            with self.subTest(activation=activation):
                d = default_mlp(inputs=3, outputs=2, hidden=[8, 5], activation=activation)
                models = {d["id"]: d}
                c = validate_model(models, d["id"])
                self.assertFalse(c["affine"])
                self.assertEqual(c["parameters"], 3*8+8 + 8*5+5 + 5*2+2)
                m = compile_model(models, d["id"])
                x = torch.randn(7, 3)
                self.assertEqual(tuple(m(x).shape), (7, 2))
                torch.testing.assert_close(m(x), exported(models, d["id"], m)(x))
                self.assertEqual(sum(p.numel() for p in m.parameters()), c["parameters"])

    def test_unknown_activation_and_bad_hidden_width_rejected(self):
        for hidden in ([], [True], [0], [2**53], [8]*9, "16,16"):
            with self.subTest(hidden=hidden), self.assertRaises(GraphError):
                default_mlp(hidden=hidden)
        with self.assertRaises(GraphError):
            default_mlp(activation="eval")
        d = default_mlp()
        next(n for n in d["graph"]["nodes"] if n["type"] == "Activation")["activation"] = "eval"
        with self.assertRaises(GraphError):
            check_document(d)

    def test_nonlinear_teacher_is_deterministic(self):
        d = default_mlp()
        models = {d["id"]: d}
        a = generate_dataset(models, d["id"], samples=32)
        self.assertEqual(a, generate_dataset(models, d["id"], samples=32))
        self.assertEqual(a["truth"]["kind"], "nonlinear")
        self.assertNotEqual(a, generate_dataset(models, d["id"], samples=32, seed=8))
        self.assertEqual(len(a["x"][0]), 2)
        self.assertEqual(len(a["y"][0]), 1)

    def test_mlp_learns_nonlinear_teacher_without_fake_effective_matrix(self):
        d = default_mlp()
        models = {d["id"]: d}
        d["dataset"] = generate_dataset(models, d["id"], samples=256)
        result = train_model(models, d["id"])
        self.assertLess(result["mse"], .002)
        self.assertLess(result["mse"], result["history"][0]["test"] / 20)
        self.assertNotIn("A", result)
        self.assertNotIn("b", result)
        self.assertNotIn("A_error", result)
        self.assertEqual(result["kind"], "nonlinear")
        self.assertEqual(len(result["weights"]), 6)
        self.assertEqual(len(result["layers"]), 5)

    def test_grouped_inputs_preserve_names_order_and_compiler_export(self):
        d = grouped_model()
        models = {d["id"]: d}
        c = validate_model(models, d["id"])
        self.assertEqual([(g["name"], g["features"], g["start"]) for g in c["input_groups"]],
                         [("measurements", 2, 0), ("context", 1, 2)])
        m = compile_model(models, d["id"])
        e = exported(models, d["id"], m)
        x = torch.randn(7, 3)
        parts = {"context": x[:, 2:], "measurements": x[:, :2]}
        torch.testing.assert_close(m.forward_tensor(parts), m.forward_tensor(x))
        torch.testing.assert_close(m(parts), e(parts))
        with self.assertRaises(GraphError):
            m({"measurements": x})
        with self.assertRaises(GraphError):
            m({"measurements": x, "context": x[:, :1]})

    def test_splitter_has_disjoint_views_and_correct_gradient(self):
        d = grouped_model()
        models = {d["id"]: d}
        m = compile_model(models, d["id"])
        x = torch.randn(4, 3, requires_grad=True)
        parts = m(x)
        self.assertEqual([tuple(p.shape) for p in parts.values()], [(4, 1), (4, 2)])
        torch.testing.assert_close(torch.cat(list(parts.values()), dim=1), m.forward_tensor(x))
        sum(p.sum() for p in parts.values()).backward()
        grad = x.grad.clone()
        x.grad = None
        m.forward_tensor(x).sum().backward()
        torch.testing.assert_close(grad, x.grad)

    def test_replicator_aliases_full_tensor_and_accumulates_gradient(self):
        d = grouped_model("Replicator")
        models = {d["id"]: d}
        m = compile_model(models, d["id"])
        e = exported(models, d["id"], m)
        x = torch.randn(4, 3, requires_grad=True)
        parts = m(x)
        values = list(parts.values())
        self.assertIs(values[0], values[1])
        self.assertEqual(tuple(values[0].shape), (4, 3))
        torch.testing.assert_close(parts, e(x))
        sum(p.sum() for p in parts.values()).backward()
        grad = x.grad.clone()
        x.grad = None
        m.forward_tensor(x).sum().backward()
        torch.testing.assert_close(grad, 2*x.grad)

    def test_bad_split_sizes_and_duplicate_port_names_rejected(self):
        d = grouped_model()
        router = next(n for n in d["graph"]["nodes"] if n["type"] == "Splitter")
        router["parts"][0]["features"] = 3
        with self.assertRaisesRegex(GraphError, "sum to 3"):
            validate_model({d["id"]: d}, d["id"])
        d = grouped_model()
        for n in d["graph"]["nodes"]:
            if n["type"] == "Input":
                n["label"] = "duplicate"
        with self.assertRaisesRegex(GraphError, "unique"):
            validate_model({d["id"]: d}, d["id"])

    def test_concat_order_is_explicit_not_node_storage_order(self):
        d = grouped_model()
        join = next(n for n in d["graph"]["nodes"] if n["type"] == "Concat")
        join["sources"].reverse()
        models = {d["id"]: d}
        contract = validate_model(models, d["id"])
        self.assertEqual([g["name"] for g in contract["input_groups"]], ["context", "measurements"])
        m = compile_model(models, d["id"])
        context, measurements = torch.randn(4, 1), torch.randn(4, 2)
        torch.testing.assert_close(m.forward_tensor({"measurements": measurements, "context": context}),
                                   m.forward_tensor(torch.cat([context, measurements], dim=1)))

    def test_invalid_boundary_wiring_and_internal_branch_rejected(self):
        original = grouped_model()
        router = next(n for n in original["graph"]["nodes"] if n["type"] == "Splitter")
        join = next(n for n in original["graph"]["nodes"] if n["type"] == "Concat")
        mutations = [
            lambda d: next(n for n in d["graph"]["nodes"] if n["id"] == join["id"])["sources"].pop(),
            lambda d: d["graph"]["edges"].append({"from": "linear1", "to": "readout"}),
            lambda d: next(n for n in d["graph"]["nodes"] if n["id"] == router["id"])["parts"][0].update(output="missing"),
            lambda d: d["graph"]["edges"].append(copy.deepcopy(d["graph"]["edges"][0])),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                d = copy.deepcopy(original)
                mutation(d)
                with self.assertRaises(GraphError):
                    validate_model({d["id"]: d}, d["id"])

    def test_parameter_free_activation_exports_but_cannot_train(self):
        d = default_model()
        d["graph"]["nodes"][1].update(type="Activation", activation="Tanh")
        models = {d["id"]: d}
        m = compile_model(models, d["id"])
        x = torch.randn(4, 2)
        torch.testing.assert_close(m(x), exported(models, d["id"], m)(x))
        with self.assertRaisesRegex(GraphError, "trainable parameters"):
            train_model(models, d["id"])

    def test_nested_mlp_reference_keeps_interface_and_packed_forward(self):
        block = grouped_model()
        owner = default_model(inputs=3, outputs=1)
        first = owner["graph"]["nodes"][1]
        first.update(type="Module", ref=block["id"])
        models = {block["id"]: block, owner["id"]: owner}
        contract = validate_model(models, owner["id"])
        self.assertFalse(contract["affine"])
        self.assertEqual(contract["outputs"], 3)
        m = compile_model(models, owner["id"])
        x = torch.randn(4, 3)
        self.assertEqual(tuple(m(x).shape), (4, 3))
        torch.testing.assert_close(m(x), exported(models, owner["id"], m)(x))
        paths = [layer["path"] for layer in layer_descriptions(models, owner["id"])]
        self.assertTrue(all(path.startswith("layer_0.") for path in paths))

    def test_grouped_training_roundtrip_and_saved_weight_predictions(self):
        for mode in ("Splitter", "Replicator"):
            with self.subTest(mode=mode):
                d = grouped_model(mode)
                w = Workspace()
                try:
                    w.put(d)
                    d["dataset"] = generate_dataset(w.models(), d["id"], samples=32)
                    d["training"]["epochs"] = 12
                    result = train_model({d["id"]: d}, d["id"])
                    d["result"] = result
                    w.put(d)
                    w.record_run(d["id"], result, w.models())
                    with tempfile.TemporaryDirectory() as folder:
                        path = w.save(Path(folder)/"mlp.vtw")
                        loaded = Workspace.load(path)
                        try:
                            self.assertEqual(loaded.models(), w.models())
                            c = compile_model(loaded.models(), d["id"])
                            c.load_state_dict({k: torch.tensor(v) for k, v in result["weights"].items()})
                            for row in result["preview"]:
                                torch.testing.assert_close(c.forward_tensor(torch.tensor([row["x"]]))[0],
                                                           torch.tensor(row["prediction"]), atol=1e-6, rtol=1e-5)
                        finally:
                            loaded.db.close()
                finally:
                    w.db.close()

    def test_old_linear_workspace_remains_compatible(self):
        # A legacy document has no activation, labels, grouped boundaries or new result fields.
        d = default_model()
        d["result"] = {"A": [[1., 2.]], "b": [0.], "mse": 0.}
        w = Workspace()
        try:
            w.put(d)
            with tempfile.TemporaryDirectory() as folder:
                loaded = Workspace.load(w.save(Path(folder)/"old.vtw"))
                try:
                    self.assertTrue(validate_model(loaded.models(), d["id"])["affine"])
                    self.assertEqual(loaded.models()[d["id"]]["result"], d["result"])
                finally:
                    loaded.db.close()
        finally:
            w.db.close()


if __name__ == "__main__":
    unittest.main()
