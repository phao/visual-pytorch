"""Graph contracts, SQLite workspaces, PyTorch code generation and training."""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import random
import sqlite3
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from .memory import (DEFAULT_MEMORY_LIMIT_PERCENT, memory_percent, available_memory,
                     estimate_memory, format_bytes)
from .operators import ARITHMETIC_OPERATORS, BLOCK_TYPES, TENSOR_OPERATORS, LAYER_TYPES, REFERENCE_TYPES, normalized, axes_for, constructor_code
from .activations import ACTIVATIONS, activation_name, normalize_activation, softmax_axis
from .activations import constructor_code as activation_constructor_code

FORMAT = "visual-torch-workspace"
VERSION = 23
SUPPORTED_VERSIONS = frozenset(range(1, 21)) | {23}
MAX_INTEGER = 2**53 - 1  # Exact round-trip through the JavaScript editor.


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def default_model(name="Linear regression", inputs=2, outputs=1):
    return {
        "id": uuid.uuid4().hex, "name": name,
        "graph": {
            "nodes": [
                {"id": "input", "type": "Input", "features": inputs, "x": 90, "y": 170},
                {"id": "linear1", "type": "Linear", "features": outputs, "bias": True, "x": 340, "y": 170},
                {"id": "output", "type": "Output", "x": 600, "y": 170},
            ],
            "edges": [
                {"from": "input", "to": "linear1"},
                {"from": "linear1", "to": "output"},
            ],
        },
        "dataset": None, "result": None,
        "training": {"epochs": 400, "lr": 0.03, "seed": 7},
        "view": {"x": 0, "y": 0, "zoom": 1},
    }


class GraphError(ValueError):
    pass


def default_mlp(name="MLP regression", inputs=2, outputs=1, hidden=(16, 16), activation="Tanh"):
    feature_count(inputs)
    feature_count(outputs)
    if not isinstance(hidden, (list, tuple)) or not 1 <= len(hidden) <= 8:
        raise GraphError("Use 1–8 hidden layers with positive integer feature counts.")
    for width in hidden:
        feature_count(width)
    activation_name(activation)
    doc = default_model(name, inputs, outputs)
    nodes = [{"id": "input", "type": "Input", "features": inputs}]
    for i, width in enumerate(hidden, 1):
        nodes.extend([
            {"id": f"linear{i}", "type": "Linear", "features": width, "bias": True},
            {"id": f"activation{i}", "type": "Activation", "activation": activation},
        ])
    nodes.extend([
        {"id": "readout", "type": "Linear", "features": outputs, "bias": True},
        {"id": "output", "type": "Output"},
    ])
    for i, node in enumerate(nodes):
        node.update(x=90 + 224 * i, y=170)
    doc["graph"] = {"nodes": nodes, "edges": [
        {"from": a["id"], "to": b["id"]} for a, b in zip(nodes, nodes[1:])
    ]}
    doc["training"] = {"epochs": 800, "lr": 0.01, "seed": 7}
    return doc


def default_cnn(name="Spatial regression"):
    """Small explicit CHW image-regression example, not a hidden composite block."""
    doc = default_model(name)
    nodes = [
        {'id': 'input', 'type': 'Input', 'shape': [1, 6, 5], 'axes': ['C', 'H', 'W']},
        normalized({'id': 'conv', 'type': 'Conv2d', 'out_channels': 2, 'kernel_size': [3, 2]}),
        {'id': 'activation', 'type': 'Activation', 'activation': 'Tanh'},
        normalized({'id': 'pool', 'type': 'AdaptiveAvgPool2d', 'output_size': [1, 1]}),
        normalized({'id': 'flatten', 'type': 'Flatten'}),
        {'id': 'head', 'type': 'Linear', 'features': 1, 'bias': True},
        {'id': 'output', 'type': 'Output'},
    ]
    for i, node in enumerate(nodes):
        node.update(x=90 + 210 * i, y=170)
    doc['graph'] = {'nodes': nodes, 'edges': [
        {'from': a['id'], 'to': b['id']} for a, b in zip(nodes, nodes[1:])]}
    doc['training'] = {'epochs': 450, 'lr': .03, 'seed': 7}
    return doc


def feature_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_INTEGER:
        raise GraphError("Feature counts must be positive integers representable exactly by the editor.")
    return value


def tensor_shape(value, *, infer=False):
    """Sample dimensions only; the batch axis is never user-specified."""
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise GraphError("Specify 1–8 sample dimensions; do not include B.")
    for dim in value:
        if infer and dim == -1 and isinstance(dim, int) and not isinstance(dim, bool):
            continue
        feature_count(dim)
    if value.count(-1) > 1:
        raise GraphError("A Reshape may infer at most one dimension (-1).")
    if math.prod(d for d in value if d != -1) > MAX_INTEGER:
        raise GraphError("The sample element count must be exactly representable by the editor.")
    return list(value)


def input_shape(node):
    return tensor_shape(node["shape"]) if "shape" in node else [feature_count(node.get("features"))]


def reshape_shape(before, target):
    target = tensor_shape(target, infer=True)
    size, known = math.prod(before), math.prod(d for d in target if d != -1)
    if -1 in target:
        if size % known:
            raise GraphError("Reshape inferred dimension must divide the incoming sample element count.")
        target[target.index(-1)] = size // known
    if math.prod(target) != size:
        raise GraphError(f"Reshape must preserve {size} elements per sample; batch B is unchanged.")
    return target


def ordered_nodes(model):
    from .graph import topology
    return topology(model)


def operation_nodes(model):
    return [n for n in ordered_nodes(model) if n["type"] in LAYER_TYPES]


def boundary_groups(nodes, widths):
    groups, start, names = [], 0, set()
    for node, width in zip(nodes, widths):
        name = node.get("label", node["id"]).strip()
        if not name or name in names:
            raise GraphError("Input and Output port names must be nonempty and unique on each boundary.")
        names.add(name)
        groups.append({"id": node["id"], "name": name, "features": width,
                       "start": start, "stop": start + width})
        start += width
    return groups


def validate_model(models, model_id, ancestry=(), _cache=None):
    from .graph import contract
    return contract(models, model_id, ancestry, _cache)


def safety_report(models, mid, constrained=True, *, samples=None, epochs=None,
                  memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, kind='auto'):
    """No tensor allocations: compare the whole-run estimate to current RAM."""
    if not isinstance(constrained, bool):
        raise GraphError("Constrained mode must be true or false.")
    c = validate_model(models, mid)
    percentage = memory_percent(memory_limit_percent)
    if samples is not None:
        if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= MAX_INTEGER:
            raise GraphError("Samples must be an integer of at least 2.")
    if epochs is not None:
        feature_count(epochs)
    memory = estimate_memory(c, samples=samples, training=epochs is not None, kind=kind)
    available = available_memory()
    budget = math.floor(available * percentage / 100) if available is not None else None
    memory.update(available_bytes=available, budget_bytes=budget, limit_percent=percentage)
    violations = []
    if available is None:
        violations.append({'rule': 'memory_unavailable', 'actual': None, 'limit': None,
                           'message': 'Available RAM could not be measured; constrained runs are blocked.'})
    elif memory['estimated_bytes'] > budget:
        violations.append({'rule': 'memory', 'actual': memory['estimated_bytes'], 'limit': budget,
                           'message': f"Estimated {memory['operation']} memory {format_bytes(memory['estimated_bytes'])} exceeds {percentage:g}% of available RAM ({format_bytes(budget)} of {format_bytes(available)})."})
    return {"constrained": constrained, "allowed": not constrained or not violations,
            "violations": violations, "memory": memory}


def preflight(models, mid, constrained=True, *, samples=None, epochs=None,
              memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT, kind='auto'):
    report = safety_report(models, mid, constrained, samples=samples, epochs=epochs,
                           memory_limit_percent=memory_limit_percent, kind=kind)
    if not report["allowed"]:
        raise GraphError("Constrained mode blocked this run. " +
                         " ".join(v["message"] for v in report["violations"]) +
                         " Reduce the run, change the user RAM percentage in Options, or uncheck Constrained mode. The architecture is unchanged.")
    return report


def check_document(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("id"), str):
        raise GraphError("The model document needs an ID.")
    if not isinstance(doc.get("name"), str) or not 1 <= len(doc["name"].strip()) <= 100:
        raise GraphError("Architecture names must contain 1–100 characters.")
    graph = doc.get("graph", {})
    if not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise GraphError("A graph needs node and edge lists.")
    if not 2 <= len(graph["nodes"]) <= 64 or len(graph["edges"]) > 128:
        raise GraphError("This prototype accepts 2–64 blocks.")
    ids = set()
    if 'input_order' in graph and (not isinstance(graph['input_order'],list) or
            any(not isinstance(n,str) for n in graph['input_order']) or
            len(graph['input_order'])!=len(set(graph['input_order']))):
        raise GraphError('input_order must contain unique Input IDs.')
    for node in graph["nodes"]:
        if not isinstance(node.get("id"), str) or node["id"] in ids:
            raise GraphError("Block IDs must be unique strings.")
        ids.add(node["id"])
        if node.get("type") not in BLOCK_TYPES:
            raise GraphError("Unsupported block type.")
        if node['type'] in TENSOR_OPERATORS | ARITHMETIC_OPERATORS | {'Recurrent', 'LSTM'}:
            normalized(node)
        if "label" in node and (not isinstance(node["label"], str) or not 1 <= len(node["label"].strip()) <= 60):
            raise GraphError("Port labels must contain 1–60 characters.")
        if node["type"] in {"Concat", "Replicator"}:
            key = "sources" if node["type"] == "Concat" else "outputs"
            if not isinstance(node.get(key), list) or any(not isinstance(v, str) for v in node[key]):
                raise GraphError(f"{node['type']} needs an ordered list of port IDs.")
        if node['type'] == 'Concat' and not isinstance(node.get('flatten', True), bool):
            raise GraphError('Concat flatten must be true or false.')
        if node["type"] == "Splitter":
            if not isinstance(node.get("parts"), list) or any(not isinstance(p, dict) or
                    not isinstance(p.get("output"), str) for p in node["parts"]):
                raise GraphError("Splitter needs named Output IDs and feature counts.")
            for part in node["parts"]:
                feature_count(part.get("features"))
        if node["type"] == "Input":
            axes_for(node, len(input_shape(node)))
        elif 'axes' in node:
            raise GraphError('Only Input declares axes; adapters propagate or discard their meaning.')
        if node["type"] in {"Reshape", "Output"} and "shape" in node:
            tensor_shape(node["shape"], infer=node["type"] == "Reshape")
        if node["type"] == "Reshape" and "shape" not in node:
            raise GraphError("Reshape needs a sample shape.")
        if node["type"] == "Linear":
            feature_count(node.get("features"))
        if node["type"] == "Linear" and not isinstance(node.get("bias", True), bool):
            raise GraphError("Bias must be true or false.")
        if node["type"] == "Activation":
            normalize_activation(node)
        if node["type"] in REFERENCE_TYPES and not isinstance(node.get("ref"), str):
            raise GraphError("An architecture block needs a reference ID.")
        for key in ("x", "y"):
            v = node.get(key, 0)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 100000:
                raise GraphError("Invalid block position.")
        for key in ('width', 'height'):
            if key in node:
                v = node[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 < v <= 100000:
                    raise GraphError('Block width/height must be finite positive numbers at most 100000.')
        if 'collapsed' in node and not isinstance(node['collapsed'], bool):
            raise GraphError('Block collapsed must be true or false.')
    for edge in graph["edges"]:
        if edge.get("from") not in ids or edge.get("to") not in ids:
            raise GraphError("A wire refers to a missing block.")
        if 'from_port' in edge and not isinstance(edge['from_port'], str):
            raise GraphError('Wire from_port must be a declared string port ID.')
    from .graph import parameter_owners
    parameter_owners(doc)
    json.dumps(doc, allow_nan=False)


class Workspace:
    """An in-memory working database; explicit Save writes an atomic snapshot."""
    def __init__(self):
        from .history import History
        self.history = History()
        self.db = sqlite3.connect(":memory:")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(f"""
            PRAGMA user_version={VERSION};
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE architectures(
                id TEXT PRIMARY KEY, name TEXT NOT NULL,
                document TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE architecture_refs(
                owner_id TEXT NOT NULL REFERENCES architectures(id) ON DELETE CASCADE,
                node_id TEXT NOT NULL,
                target_id TEXT NOT NULL REFERENCES architectures(id),
                PRIMARY KEY(owner_id,node_id));
            CREATE TABLE datasets(
                architecture_id TEXT PRIMARY KEY REFERENCES architectures(id) ON DELETE CASCADE,
                payload TEXT NOT NULL);
            CREATE TABLE runs(
                id INTEGER PRIMARY KEY, architecture_id TEXT NOT NULL
                    REFERENCES architectures(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL, graph_snapshot TEXT NOT NULL,
                result TEXT NOT NULL);
        """)
        from .revisions import create_schema
        create_schema(self.db)
        self.db.executemany("INSERT INTO metadata VALUES (?,?)", [
            ("format", FORMAT), ("workspace_id", uuid.uuid4().hex),
            ("created_at", timestamp()), ("name", "Untitled workspace"),
            ("constrained_mode", "true"),
        ])
        self.db.commit()
        self.path = None
        self.dirty = False
        self.disk_revision = None
        self.disk_stamp = None
        self.panel_scope = uuid.uuid4().hex  # Ephemeral UI owner; never stored in .vtw.

    @property
    def panel_layout(self):
        from .presentation import validate_panel_layout
        row = self.db.execute("SELECT value FROM metadata WHERE key='panel_layout'").fetchone()
        try:
            return validate_panel_layout(json.loads(row[0]) if row else None)
        except (ValueError, TypeError) as error:
            raise GraphError('Invalid workspace panel_layout metadata: ' + str(error)) from error

    def set_panel_layout(self, value):
        from .presentation import validate_panel_layout
        value = validate_panel_layout(value)
        if value != self.panel_layout:
            with self.db:
                if value is None:
                    self.db.execute("DELETE FROM metadata WHERE key='panel_layout'")
                else:
                    self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('panel_layout', ?)",
                                    (json.dumps(value, sort_keys=True, allow_nan=False),))
            self.dirty = True

    @property
    def layout_preferences(self):
        from .presentation import LAYOUT_DEFAULTS, validate_layout
        row = self.db.execute("SELECT value FROM metadata WHERE key='layout_preferences'").fetchone()
        try:
            return validate_layout(json.loads(row[0]) if row else LAYOUT_DEFAULTS)
        except (ValueError, TypeError) as error:
            raise GraphError('Invalid workspace layout_preferences metadata: ' + str(error)) from error

    def set_layout_preferences(self, value):
        from .presentation import validate_layout
        value = validate_layout(value)
        if value != self.layout_preferences:
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('layout_preferences', ?)",
                                (json.dumps(value, sort_keys=True, allow_nan=False),))
            self.dirty = True

    @property
    def constrained_mode(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='constrained_mode'").fetchone()
        if row is None:
            return True  # Legacy workspaces open conservatively.
        if row[0] not in {"true", "false"}:
            raise GraphError("Invalid workspace constrained_mode metadata.")
        return row[0] == "true"

    def set_constrained_mode(self, enabled):
        if not isinstance(enabled, bool):
            raise GraphError("Constrained mode must be true or false.")
        if enabled != self.constrained_mode:
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('constrained_mode', ?)",
                                ("true" if enabled else "false",))
            self.dirty = True

    @classmethod
    def demo(cls):
        w = cls()
        main = default_model()
        block = default_model("Feature projection", 2, 3)
        composed = default_model("Composed linear model", 2, 1)
        composed["graph"]["nodes"].insert(1, {
            "id": "projection", "type": "Module", "ref": block["id"], "x": 310, "y": 170})
        composed["graph"]["nodes"][2]["x"] = 570
        composed["graph"]["nodes"][-1]["x"] = 830
        composed["graph"]["edges"] = [
            {"from": "input", "to": "projection"},
            {"from": "projection", "to": "linear1"},
            {"from": "linear1", "to": "output"},
        ]
        mlp = default_mlp()
        for doc in (main, block, composed, mlp):
            w.put(doc)
        w.dirty = True
        return w

    def models(self):
        result = {}
        for mid, document in self.db.execute("SELECT id,document FROM architectures ORDER BY rowid"):
            doc = json.loads(document)
            data = self.db.execute("SELECT payload FROM datasets WHERE architecture_id=?", (mid,)).fetchone()
            doc["dataset"] = json.loads(data[0]) if data else None
            result[mid] = doc
        return result

    def put(self, doc):
        check_document(doc)
        for node in doc["graph"]["nodes"]:
            if node["type"] in REFERENCE_TYPES:
                if not self.db.execute("SELECT 1 FROM architectures WHERE id=?", (node["ref"],)).fetchone():
                    raise GraphError("The referenced architecture does not exist in this workspace.")
        document = copy.deepcopy(doc)
        document['graph']['nodes'] = [normalized(n) for n in document['graph']['nodes']]
        data = document.pop("dataset", None)
        old = self.db.execute("SELECT document FROM architectures WHERE id=?", (doc["id"],)).fetchone()
        from .evidence import pin_input_order, graph_signature, binding_for, identity, digest
        previous=json.loads(old[0]) if old else None
        pin_input_order(document['graph'],previous['graph'] if previous else None)
        old_data=self.db.execute('SELECT payload FROM datasets WHERE architecture_id=?',(doc['id'],)).fetchone()
        data_changed=bool(old and digest(json.loads(old_data[0]) if old_data else None)!=digest(data))
        graph_changed=bool(old and graph_signature(previous['graph'])!=graph_signature(document['graph']))
        if data is None:
            document['dataset_binding']=None
        elif data_changed or not document.get('dataset_binding'):
            if not data_changed and previous and previous.get('dataset_binding'):
                document['dataset_binding']=previous['dataset_binding']
            else:
                candidates=self.models()
                candidates[doc['id']]={**document,'dataset':data}
                try:document['dataset_binding']=binding_for(validate_model(candidates,doc['id']))
                except GraphError:
                    # Invalid drafts retain the old binding rather than guessing new columns.
                    document['dataset_binding']=previous.get('dataset_binding') if previous else None
        models=self.models();models[doc['id']]={**document,'dataset':data}
        candidate=document.get('result')
        if candidate and candidate.get('identity') and candidate['identity']!=identity(models,doc['id']):
            document["result"] = None
        elif old and (graph_changed or data_changed or previous.get('dataset_binding')!=document.get('dataset_binding') or
                      previous.get('training',{})!=document.get('training',{})):
            if not candidate or not candidate.get('identity'):
                document['result']=None
        with self.db:
            self.db.execute("""
                INSERT INTO architectures VALUES (?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                    document=excluded.document, updated_at=excluded.updated_at
            """, (doc["id"], doc["name"].strip(), json.dumps(document, allow_nan=False), timestamp()))
            self.db.execute("DELETE FROM architecture_refs WHERE owner_id=?", (doc["id"],))
            for node in doc["graph"]["nodes"]:
                if node["type"] in REFERENCE_TYPES:
                    self.db.execute("INSERT INTO architecture_refs VALUES (?,?,?)",
                                    (doc["id"], node["id"], node["ref"]))
            if data is not None:
                self.db.execute("""
                    INSERT INTO datasets VALUES (?,?)
                    ON CONFLICT(architecture_id) DO UPDATE SET payload=excluded.payload
                """, (doc["id"], json.dumps(data, allow_nan=False)))
            else:
                self.db.execute("DELETE FROM datasets WHERE architecture_id=?", (doc["id"],))
            if graph_changed:
                # A saved result belongs to its exact graph, including referenced graphs.
                # Historical runs keep their original snapshots.
                dependents = self.db.execute("""
                    WITH RECURSIVE owners(id) AS (
                        SELECT owner_id FROM architecture_refs WHERE target_id=?
                        UNION
                        SELECT r.owner_id FROM architecture_refs r JOIN owners o ON r.target_id=o.id
                    ) SELECT id FROM owners
                """, (doc["id"],)).fetchall()
                for (mid,) in dependents:
                    raw = self.db.execute("SELECT document FROM architectures WHERE id=?", (mid,)).fetchone()
                    dependent = json.loads(raw[0])
                    dependent["result"] = None
                    self.db.execute("UPDATE architectures SET document=?,updated_at=? WHERE id=?",
                                    (json.dumps(dependent), timestamp(), mid))
        self.dirty = True

    def delete(self, mid):
        if self.db.execute("SELECT 1 FROM architecture_refs WHERE target_id=?", (mid,)).fetchone():
            raise GraphError("This architecture is referenced by another model. Remove its references first.")
        if len(self.models()) <= 1:
            raise GraphError("Keep at least one architecture in the workspace.")
        with self.db:
            self.db.execute("DELETE FROM architectures WHERE id=?", (mid,))
        self.dirty = True

    def record_run(self, mid, result, graph_snapshot):
        with self.db:
            self.db.execute("INSERT INTO runs(architecture_id,created_at,graph_snapshot,result) VALUES (?,?,?,?)",
                            (mid, timestamp(), json.dumps(graph_snapshot), json.dumps(result, allow_nan=False)))
        self.dirty = True

    def put_spec(self, kind, document, select=True):
        from .revisions import RevisionStore
        return RevisionStore(self).put(kind, document, select)

    def get_spec(self, kind, sid, revision=None):
        from .revisions import RevisionStore
        return RevisionStore(self).get(kind, sid, revision)

    def list_specs(self, kind=None, all_revisions=False):
        from .revisions import RevisionStore
        return RevisionStore(self).list(kind, all_revisions)

    def select_spec(self, kind, sid, revision):
        from .revisions import RevisionStore
        RevisionStore(self).select(kind, sid, revision)

    def archive_spec(self, kind, sid):
        from .revisions import RevisionStore
        RevisionStore(self).archive(kind, sid)

    def freeze_model(self, mid):
        from .revisions import model_document
        return self.put_spec('model', model_document(self.models(), mid))

    def save(self, path, *, require_new=False):
        from .file_sync import revision, save_lock, stamp
        from .revisions import validate_all
        validate_all(self.db)
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() != ".vtw":
            path = path.with_suffix(".vtw")
        with save_lock(path):
            if require_new and path.exists():
                raise GraphError('Refusing to overwrite an existing workspace.')
            if self.path == str(path) and revision(path) != self.disk_revision:
                raise GraphError("Workspace changed on disk. Reload external changes or Save As a new file; unsaved work was not overwritten.")
            handle, temporary = tempfile.mkstemp(prefix=".vtw-save-", suffix=".sqlite", dir=path.parent)
            os.close(handle)
            try:
                destination = sqlite3.connect(temporary)
                try:
                    self.db.backup(destination)
                    destination.commit()
                finally:
                    destination.close()
                with open(temporary, "rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.disk_revision = revision(path)
            self.disk_stamp = stamp(path)
        self.path, self.dirty = str(path), False
        return str(path)

    @classmethod
    def load(cls, path):
        from .file_sync import revision, stamp
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise GraphError("That workspace file no longer exists.")
        before = revision(path)
        source = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        w = cls()
        try:
            version = source.execute("PRAGMA user_version").fetchone()[0]
            magic = source.execute("SELECT value FROM metadata WHERE key='format'").fetchone()
            if version not in SUPPORTED_VERSIONS or magic != (FORMAT,):
                raise GraphError("This is not a supported Visual Torch Workspace file.")
            source.backup(w.db)
            if w.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise GraphError("The workspace database is damaged.")
            if w.db.execute("PRAGMA foreign_key_check").fetchall():
                raise GraphError("The workspace contains broken architecture references.")
            loaded = w.models()
            expected_refs = set()
            for doc in loaded.values():
                check_document(doc)
                expected_refs.update((doc['id'], n['id'], n['ref']) for n in doc['graph']['nodes'] if n['type'] in REFERENCE_TYPES)
            if expected_refs != set(w.db.execute('SELECT owner_id,node_id,target_id FROM architecture_refs')):
                raise GraphError('Architecture reference index does not match the graph documents.')
            for mid, name in w.db.execute('SELECT id,name FROM architectures'):
                if loaded[mid]['id'] != mid or loaded[mid]['name'].strip() != name:
                    raise GraphError('Architecture row ID/name does not match its document.')
            w.constrained_mode
            w.layout_preferences
            w.panel_layout
            if version<3:
                # Working-copy migration only; the disk snapshot stays untouched until Save.
                for doc in loaded.values():w.put(doc)
            from .revisions import migrate_schema, validate_all
            if version < 5:
                migrate_schema(w.db)
            if version < 6 and w.db.execute("SELECT 1 FROM spec_revisions WHERE kind IN ('experiment','run','report') LIMIT 1").fetchone():
                raise GraphError('Experiment/run/report revisions require a version-6 container.')
            if version < 7 and w.db.execute("SELECT 1 FROM spec_revisions WHERE kind IN ('family','specialization') LIMIT 1").fetchone():
                raise GraphError('Family/specialization revisions require a version-7 container.')
            validate_all(w.db)
            if version < 23:
                from .revisions import _parse
                if any(n['type'] in ARITHMETIC_OPERATORS for m in loaded.values() for n in m['graph']['nodes']):
                    raise GraphError('Add/Scale blocks require a version-23 container.')
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('model','family','specialization')"):
                    if _parse(text)['version'] >= {'model': 5, 'family': 9, 'specialization': 8}[kind]:
                        raise GraphError('Arithmetic model/family contracts require a version-23 container.')
            if version < 20:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('experiment','run')"):
                    if _parse(text)['version'] >= 6:
                        raise GraphError('Temporal-v2 experiment/run contracts require a version-20 container.')
            if version < 19:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization')"):
                    if _parse(text)['version'] >= (8 if kind == 'family' else 7):
                        raise GraphError('Temporal-v2 family contracts require a version-19 container.')
            if version < 18:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization')"):
                    if _parse(text)['version'] >= (7 if kind == 'family' else 6):
                        raise GraphError('LSTM family contracts require a version-18 container.')
            if version < 8:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('experiment','run')"):
                    if _parse(text)['version'] == 2:
                        raise GraphError('Version-2 experiment/run revisions require a version-8 container.')
            if version < 9:
                from .revisions import _parse
                if any('share_with' in n for doc in loaded.values() for n in doc['graph']['nodes']):
                    raise GraphError('Explicit shared parameters require a version-9 container.')
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','model')"):
                    if _parse(text)['version'] == 2:
                        raise GraphError('Version-2 family/model revisions require a version-9 container.')
            if version < 10:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization','experiment','run')"):
                    doc_version = _parse(text)['version']
                    if doc_version >= (2 if kind == 'specialization' else 3):
                        raise GraphError('Typed constructor choices require a version-10 container.')
            if version < 11:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization')"):
                    if _parse(text)['version'] >= (3 if kind == 'specialization' else 4):
                        raise GraphError('Scoped constructor bindings require a version-11 container.')
            if version < 12:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization')"):
                    if _parse(text)['version'] >= (4 if kind == 'specialization' else 5):
                        raise GraphError('Runtime family contracts require a version-12 container.')
            if version < 14:
                from .revisions import _parse
                if any(n['type'] == 'Recurrent' for m in w.models().values() for n in m['graph']['nodes']):
                    raise GraphError('Recurrent graph references require a version-14 container.')
                for text, in w.db.execute("SELECT document FROM spec_revisions WHERE kind='model'"):
                    if _parse(text)['version'] >= 3:
                        raise GraphError('Recurrent ModelRevisions require a version-14 container.')
            if version < 17:
                from .revisions import _parse
                if any(n['type'] == 'LSTM' for m in w.models().values() for n in m['graph']['nodes']):
                    raise GraphError('LSTM graph blocks require a version-17 container.')
                for text, in w.db.execute("SELECT document FROM spec_revisions WHERE kind='model'"):
                    if _parse(text)['version'] >= 4:
                        raise GraphError('LSTM ModelRevisions require a version-17 container.')
            if version < 16:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('experiment','run')"):
                    if _parse(text)['version'] >= 5:
                        raise GraphError('Temporal experiment execution requires a version-16 container.')
            if version < 15:
                from .revisions import _parse
                for kind, text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('family','specialization')"):
                    if _parse(text)['version'] >= (6 if kind == 'family' else 5):
                        raise GraphError('Temporal recurrent family contracts require a version-15 container.')
            if version < 13:
                from .revisions import _parse
                for kind,text in w.db.execute("SELECT kind,document FROM spec_revisions WHERE kind IN ('experiment','run')"):
                    if _parse(text)['version']>=4:
                        raise GraphError('Runtime task execution requires a version-13 container.')
            w.db.execute(f'PRAGMA user_version={VERSION}')
            w.disk_revision = revision(path)
            if before != w.disk_revision:
                raise GraphError('Workspace changed while opening it. Retry.')
            w.disk_stamp = stamp(path)
        except Exception:
            w.db.close()
            raise
        finally:
            source.close()
        w.path, w.dirty = str(path), False
        return w


def _pack_inputs(x, groups, sample_shape):
    """Native and exported forward share this exact batch/shape boundary contract."""
    import torch
    if isinstance(x, dict):
        if set(x) != {g['name'] for g in groups}:
            raise ValueError('Supply exactly the declared named input ports.')
        parts = [x[g['name']] for g in groups]
        if any(list(p.shape[1:]) != g['shape'] for p, g in zip(parts, groups)):
            raise ValueError('Named input tensor sample shapes must match their declared ports.')
        if len({p.shape[0] for p in parts}) != 1:
            raise ValueError('All named inputs must have the same batch size B.')
        x = parts[0] if len(parts) == 1 else torch.cat(
            [p.reshape(p.shape[0], g['features']) for p, g in zip(parts, groups)], dim=1)
    if list(x.shape[1:]) != sample_shape:
        raise ValueError('Input must have shape [B, *declared_sample_shape]. Use an explicit Reshape.')
    return x


def _graph_values(module, x, contract):
    import torch
    x = _pack_inputs(x, contract['input_groups'], contract['input_shape'])
    values = {}
    for item in contract['plan']:
        nid, kind = item['id'], item['type']
        parts = [values[s['node']][s['port']] if s['port'] is not None else values[s['node']]
                 for s in item['sources']]
        if kind == 'Input':
            g = item['group']
            value = x if len(contract['input_groups']) == 1 else x[:, g['start']:g['stop']].reshape(x.shape[0], *g['shape'])
        elif 'layer' in item:
            layer = getattr(module, item['layer'])
            if kind == 'Recurrent':
                # The app-free emitter captures this helper under the same name.
                value = _recurrent_graph_forward(layer, parts, item['recurrent'])
            elif kind == 'LSTM':
                value = _lstm_graph_forward(layer, parts, item['lstm'])
            elif kind == 'Module' and 'runtime_contract' in item:
                child = item['runtime_contract']
                value = _graph_tensor(_graph_values(layer, parts[0], child), child)
            elif kind == 'Reshape' and 'runtime_reshape' in item:
                value = layer(parts[0], shape=item['runtime_reshape'])
            else:
                value = layer.forward_tensor(parts[0]) if kind == 'Module' else layer(parts[0])
        elif kind == 'Add':
            if len(parts) < 2 or any(not isinstance(p, torch.Tensor) or p.dtype != torch.float32 or p.device.type != 'cpu' or p.shape != parts[0].shape for p in parts):
                raise ValueError('Add requires at least two identical-shape CPU float32 tensors; no broadcasting.')
            value = parts[0]
            for part in parts[1:]: value = value + part
        elif kind == 'Scale':
            if not isinstance(parts[0], torch.Tensor) or parts[0].dtype != torch.float32 or parts[0].device.type != 'cpu':
                raise ValueError('Scale requires CPU float32 input.')
            value = parts[0] * item['coefficient']
        elif kind == 'Concat':
            value = torch.cat([p.flatten(start_dim=1) for p in parts] if item['flatten'] else parts,
                              dim=1 if item['flatten'] else -1)
        elif kind == 'Splitter':
            value = dict(zip(item['ports'], torch.split(parts[0], item['sizes'], dim=-1)))
        elif kind == 'Replicator':
            value = {port: parts[0] for port in item['ports']}
        else:
            value = parts[0]  # Output is an identity boundary.
        values[nid] = value
    return values


def _recurrent_graph_forward(layer, parts, plan):
    from .recurrent_nodes import graph_forward
    return graph_forward(layer, parts, plan)


def _lstm_graph_forward(layer, parts, plan):
    from .lstm_nodes import graph_forward
    return graph_forward(layer, parts, plan)


def _graph_tensor(values, contract):
    import torch
    if contract['tensor_source'] is not None:
        return values[contract['tensor_source']]
    groups = {g['start']: g for g in contract['output_groups']}
    parts = [values[g['id']].reshape(values[g['id']].shape[0], g['features'])
             for _, g in sorted(groups.items())]
    return torch.cat(parts, dim=1)


def generate_code(models, model_id):
    validate_model(models, model_id)
    lines = ["import torch", "from torch import nn", "", inspect.getsource(_pack_inputs),
             inspect.getsource(_graph_values), inspect.getsource(_graph_tensor),
             "class _Reshape(nn.Module):", "    def __init__(self, shape):",
             "        super().__init__()", "        self.shape = tuple(shape)",
             "    def forward(self, x):", "        return x.reshape(x.shape[0], *self.shape)", "",
             "class _Permute(nn.Module):", "    def __init__(self, dims):",
             "        super().__init__()", "        self.dims = tuple(dims)",
             "    def forward(self, x):", "        return x.permute(*self.dims)", ""]
    emitted = set()
    def has_recurrence(mid, seen):
        if mid in seen: return False
        seen.add(mid)
        return any(n['type'] == 'Recurrent' or
                   (n['type'] in REFERENCE_TYPES and has_recurrence(n['ref'], seen))
                   for n in models[mid]['graph']['nodes'])
    if has_recurrence(model_id, set()):
        from .memory import MIB
        from .recurrent import (_recurrent_inputs, _recurrent_run, _runner_class, recurrent_memory_report)
        from .recurrent_nodes import _control, graph_forward
        from .task_execution import _helper_source
        lines.extend(['from pathlib import Path', 'import math', 'GraphError = ValueError',
                      'MAX_INTEGER = 2**53 - 1', f'MIB = {MIB!r}'])
        for helper in (feature_count, memory_percent, available_memory, format_bytes, recurrent_memory_report,
                       _recurrent_inputs, _recurrent_run, _runner_class, _control, graph_forward):
            lines.append(_helper_source(helper))
        lines.append('_recurrent_graph_forward = graph_forward')

    def has_lstm(mid, seen):
        if mid in seen: return False
        seen.add(mid)
        return any(n['type'] == 'LSTM' or
                   (n['type'] in REFERENCE_TYPES and has_lstm(n['ref'], seen))
                   for n in models[mid]['graph']['nodes'])
    if has_lstm(model_id, set()):
        from .lstm_nodes import graph_forward as lstm_forward, lstm_memory_report
        from .task_execution import _helper_source
        from .memory import MIB
        lines.extend(['from pathlib import Path', 'import math', f'MIB = {MIB!r}'])
        for helper in (memory_percent, available_memory, format_bytes, lstm_memory_report, lstm_forward):
            lines.append(_helper_source(helper))
        lines.append('_lstm_graph_forward = graph_forward')

    def cname(mid):
        return "Architecture_" + str(uuid.uuid5(uuid.NAMESPACE_URL, mid)).replace("-", "")[:12]

    def emit(mid):
        if mid in emitted:
            return
        model, contract = models[mid], validate_model(models, mid)
        order = operation_nodes(model)
        for n in order:
            if n["type"] in REFERENCE_TYPES:
                emit(n["ref"])
        lines.extend([
            f"class {cname(mid)}(nn.Module):",
            "    def __init__(self):", "        super().__init__()",
        ])
        for index, node in enumerate(order):
            if 'parameter_owner' in contract['shapes'][node['id']]:
                continue
            if node["type"] == "Linear":
                s = contract["shapes"][node["id"]]
                op = f'nn.Linear({s["input"]}, {s["output"]}, bias={node.get("bias", True)}, device="cpu", dtype=torch.float32)'
            elif node["type"] == "Activation":
                op = activation_constructor_code(node, len(contract['shapes'][node['id']]['input_shape']))
            elif node["type"] == "Reshape":
                op = f"_Reshape({contract['shapes'][node['id']]['output_shape']!r})"
            elif node['type'] in TENSOR_OPERATORS:
                op = constructor_code(contract['shapes'][node['id']]['operator'])
            elif node['type'] == 'Recurrent':
                op = f'_runner_class(nn)({cname(node["ref"])}(), {contract["shapes"][node["id"]]["recurrent"]["contract"]!r}, constrained=False)'
            elif node['type'] == 'LSTM':
                kwargs = dict(contract['shapes'][node['id']]['lstm']['kwargs'])
                kwargs.pop('dtype')
                op = f'nn.LSTM(**{kwargs!r}, dtype=torch.float32)'
            else:
                op = f'{cname(node["ref"])}()'
            lines.append(f"        self.layer_{index} = {op}")
        lines.extend([f"        self.contract = {contract!r}", "", "    def forward_tensor(self, x):",
                      "        return _graph_tensor(_graph_values(self, x, self.contract), self.contract)",
                      "", "    def forward(self, x):",
                      "        values = _graph_values(self, x, self.contract)",
                      "        groups = self.contract['output_groups']",
                      "        if len(groups) == 1:", "            return values[groups[0]['id']]",
                      "        return {g['name']: values[g['id']] for g in groups}"])
        lines.extend(["", ""])
        emitted.add(mid)

    emit(model_id)
    lines.extend([f"model = {cname(model_id)}()", ""])
    return "\n".join(lines)


def compile_model(models, mid, constrained=True, *, memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT,
                  runtime_inputs=None, initial_shapes=None, runtime_temporal=None, runtime_temporal_v2=None):
    if runtime_temporal_v2 is not None:
        if runtime_inputs is not None or runtime_temporal is not None:
            raise GraphError('Spatial and versioned temporal runtime policies are distinct; select exactly one.')
        from .temporal_shapes_v2 import compile_temporal_model
        return compile_temporal_model(models, mid, runtime_temporal_v2, constrained=constrained,
                                      memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes)
    if runtime_temporal is not None:
        if runtime_inputs is not None:
            raise GraphError('Spatial and temporal runtime policies are distinct; select exactly one.')
        from .temporal_shapes import compile_temporal_model
        return compile_temporal_model(models, mid, runtime_temporal, constrained=constrained,
                                      memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes)
    if runtime_inputs is not None:
        from .runtime_shapes import compile_runtime_model
        return compile_runtime_model(models, mid, runtime_inputs, constrained=constrained,
                                     memory_limit_percent=memory_limit_percent, initial_shapes=initial_shapes)
    if initial_shapes is not None:
        raise GraphError('Initial runtime shapes require an explicit runtime input declaration.')
    preflight(models, mid, constrained, memory_limit_percent=memory_limit_percent)
    import torch
    import torch.nn as nn
    contract = validate_model(models, mid)
    class Reshape(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = tuple(shape)

        def forward(self, x, *, shape=None):
            return x.reshape(x.shape[0], *(self.shape if shape is None else shape))

    class Permute(nn.Module):
        def __init__(self, dims):
            super().__init__()
            self.dims = tuple(dims)

        def forward(self, x):
            return x.permute(*self.dims)

    def recurrent_reference(current, seen=None):
        # Legacy plain Modules keep their construction-only nested policy.
        # A recurrent descendant must retain per-call admission through wrappers.
        seen = set() if seen is None else seen
        if current in seen:
            return False
        seen.add(current)
        return any(n['type'] in {'Recurrent', 'LSTM'} or
                   (n['type'] == 'Module' and recurrent_reference(n['ref'], seen))
                   for n in models[current]['graph']['nodes'])

    layers = OrderedDict()
    for index, node in enumerate(operation_nodes(models[mid])):
        if 'parameter_owner' in contract['shapes'][node['id']]:
            continue
        if node["type"] == "Linear":
            shape = contract["shapes"][node["id"]]
            layer = nn.Linear(shape["input"], shape["output"], bias=node.get("bias", True),
                              device='cpu', dtype=torch.float32)
        elif node["type"] == "Activation":
            name = activation_name(node.get("activation"))
            layer = (nn.Softmax(dim=softmax_axis(node, len(contract['shapes'][node['id']]['input_shape'])) + 1)
                     if name == 'Softmax' else getattr(nn, name)())
        elif node["type"] == "Reshape":
            layer = Reshape(contract["shapes"][node["id"]]["output_shape"])
        elif node['type'] in TENSOR_OPERATORS:
            spec = contract['shapes'][node['id']]['operator']
            kwargs = dict(spec['kwargs'])
            if 'dtype' in kwargs:
                kwargs['dtype'] = torch.float32
            layer = (Permute if node['type'] == 'Permute' else getattr(nn, spec['constructor']))(**kwargs)
        elif node['type'] == 'Recurrent':
            from .recurrent import _runner_class
            c = contract['shapes'][node['id']]['recurrent']['contract']
            layer = _runner_class(nn)(compile_model(models, node['ref'], constrained=constrained,
                                                     memory_limit_percent=memory_limit_percent), c,
                                      constrained=constrained, memory_limit_percent=memory_limit_percent)
        elif node['type'] == 'LSTM':
            kwargs = dict(contract['shapes'][node['id']]['lstm']['kwargs'])
            kwargs['dtype'] = torch.float32
            layer = nn.LSTM(**kwargs)
            layer.constrained = constrained
            layer.memory_limit_percent = memory_limit_percent
        else:
            layer = compile_model(models, node["ref"],
                                  constrained=constrained and recurrent_reference(node['ref']),
                                  memory_limit_percent=memory_limit_percent)
        layers[f"layer_{index}"] = layer

    class GraphModule(nn.Module):
        def __init__(self,layers):
            super().__init__()
            for name,layer in layers.items():self.add_module(name,layer)
            self.contract=contract

        def intermediate_values(self,x):
            return _graph_values(self,x,contract)

        def forward_tensor(self, x):
            try:
                return _graph_tensor(_graph_values(self, x, contract), contract)
            except ValueError as error:
                raise GraphError(str(error)) from error

        def forward(self, x):
            try:
                values = _graph_values(self, x, contract)
            except ValueError as error:
                raise GraphError(str(error)) from error
            groups = contract["output_groups"]
            if len(groups) == 1:
                return values[groups[0]['id']]
            return {g["name"]: values[g["id"]] for g in groups}

    return GraphModule(layers)


def layer_descriptions(models, mid, prefix="", stable_prefix=None):
    """Flatten referenced operations without losing their state-dict paths."""
    contract = validate_model(models, mid)
    layers = []
    stable_prefix=mid if stable_prefix is None else stable_prefix
    for i, node in enumerate(operation_nodes(models[mid])):
        if 'parameter_owner' in contract['shapes'][node['id']]:
            continue
        path = prefix + f"layer_{i}"
        if node["type"] in REFERENCE_TYPES:
            layers.extend(layer_descriptions(models, node["ref"], path + ('.transition.' if node['type'] == 'Recurrent' else '.'),stable_prefix+'/'+node['id']))
        else:
            layers.append({"path": path, "node_id": node["id"], "architecture": models[mid]["name"],
                           "stable_id":stable_prefix+'/'+node['id'],
                           "type": node.get("activation", node["type"]),
                           **({'bias': node.get('bias', True)} if node['type'] in {'Linear', 'Conv2d', 'LSTM'} else {}),
                           **contract["shapes"][node["id"]]})
    return layers


def generate_dataset(models, mid, samples=256, noise=0., seed=7, kind="auto", *, constrained=True,
                     memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT):
    """Small reproducible regression trials; no Torch import needed by the UI host."""
    contract = validate_model(models, mid)
    if isinstance(noise, bool) or not isinstance(noise, (int, float)) or not math.isfinite(noise) or not 0 <= noise <= 1:
        raise GraphError("Noise must be between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise GraphError("Seed must be an integer from 0 to 2³¹ − 1.")
    if kind == "auto":
        from .tensor_data import spatial_eligible
        has_conv = any(layer['type'] == 'Conv2d' for layer in layer_descriptions(models, mid))
        kind = "spatial" if has_conv and spatial_eligible(contract) else "affine" if contract["affine"] else "nonlinear"
    if kind not in {"affine", "nonlinear", "spatial"}:
        raise GraphError("Choose an affine, nonlinear or spatial synthetic target.")
    preflight(models, mid, constrained, samples=samples, memory_limit_percent=memory_limit_percent, kind=kind)
    rng = random.Random(seed)
    ni, no = contract["inputs"], contract["outputs"]
    if kind == 'spatial':
        from .tensor_data import spatial_teacher
        truth, target = spatial_teacher(contract, rng)
    elif kind == "affine":
        a = [[round(rng.uniform(-2, 2), 2) for _ in range(ni)] for _ in range(no)]
        b = [round(rng.uniform(-1, 1), 2) for _ in range(no)]
        truth = {"kind": kind, "A": a, "b": b}
        def target(row):
            return [sum(v * x for v, x in zip(weights, row)) + bias for weights, bias in zip(a, b)]
    else:
        hidden = 4
        w1 = [[rng.uniform(-1.5, 1.5) for _ in range(ni)] for _ in range(hidden)]
        b1 = [rng.uniform(-.5, .5) for _ in range(hidden)]
        w2 = [[rng.uniform(-1, 1) for _ in range(hidden)] for _ in range(no)]
        b2 = [rng.uniform(-.2, .2) for _ in range(no)]
        truth = {"kind": kind, "W1": w1, "b1": b1, "W2": w2, "b2": b2,
                 "equation": "y = W2 tanh(W1 x + b1) + b2"}
        def target(row):
            h = [math.tanh(sum(v * x for v, x in zip(weights, row)) + bias)
                 for weights, bias in zip(w1, b1)]
            return [sum(v * x for v, x in zip(weights, h)) + bias for weights, bias in zip(w2, b2)]
    x = [[rng.uniform(-1, 1) for _ in range(ni)] for _ in range(samples)]
    y = [[value + noise * rng.gauss(0, 1) for value in target(row)] for row in x]
    data = {"x": x, "y": y, "truth": truth, "source": "synthetic", "seed": seed, "noise": noise}
    if len(contract['input_groups']) == 1 and len(contract['input_shape']) > 1:
        from .tensor_data import adapter_for
        data['adapter'] = adapter_for(contract)
    return data


def validate_dataset(data, n_in, n_out):
    if not isinstance(data, dict) or not isinstance(data.get("x"), list) or not isinstance(data.get("y"), list):
        raise GraphError("Generate synthetic data or import x/y data first.")
    if len(data["x"]) < 2 or len(data["x"]) != len(data["y"]):
        raise GraphError("Supply matching x/y rows with at least 2 samples.")
    for key, width in (("x", n_in), ("y", n_out)):
        for row in data[key]:
            if not isinstance(row, list) or len(row) != width:
                raise GraphError(f"{key} rows must have {width} values.")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1e6 for v in row):
                raise GraphError("Data values must be finite numbers with absolute value at most 1,000,000.")


def affine_summary(model, contract):
    """Optional serial-vector analysis; never mistake arbitrary DAGs for chains."""
    import torch
    if not contract['matrix_affine']:
        return {'status':'unavailable','reason':'Nonlinear, branching or tensor geometry is outside the serial-vector analyzer.'}
    effective_a=effective_b=None
    def occurrences(current):
        for item in current.contract['plan']:
            if item['type'] == 'Linear':
                yield getattr(current, item['layer'])
            elif item['type'] == 'Module':
                yield from occurrences(getattr(current, item['layer']))
    with torch.no_grad():
        for layer in occurrences(model):
            bias=layer.bias if layer.bias is not None else layer.weight.new_zeros(layer.out_features)
            if effective_a is None:effective_a,effective_b=layer.weight.detach(),bias.detach()
            else:effective_a,effective_b=layer.weight @ effective_a,layer.weight @ effective_b+bias
    if effective_a is None or tuple(effective_a.shape)!=(contract['outputs'],contract['inputs']):
        return {'status':'unavailable','reason':'The serial composition does not match the canonical packed map.'}
    return {'status':'available','method':'serial-vector-composition','A':effective_a.tolist(),'b':effective_b.tolist()}


def restore_predictor(models, mid, result, *, constrained=True):
    from .evidence import normalize_models, check_checkpoint
    models=normalize_models(models,mid)
    check_checkpoint(models,mid,result)
    import torch
    model=compile_model(models,mid,constrained=constrained)
    model.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in result['weights'].items()},strict=True)
    model.eval()
    return model


def generate_predictor(models, mid, result):
    """Standalone trained inference export, rather than a freshly initialized template."""
    from .evidence import normalize_models, check_checkpoint
    models=normalize_models(models,mid)
    check_checkpoint(models,mid,result)
    return generate_code(models,mid)+(
        "\n# Frozen checkpoint; use model.eval() for inference. No retraining occurs.\n"
        f"_checkpoint = {result['weights']!r}\n"
        "model.load_state_dict({k: torch.tensor(v, dtype=torch.float32) for k,v in _checkpoint.items()}, strict=True)\n"
        "model.eval()\n"
        f"checkpoint_identity = {result['identity']!r}\n")


def train_model(models, mid, progress=None, cancel=None, *, constrained=True,
                memory_limit_percent=DEFAULT_MEMORY_LIMIT_PERCENT):
    from .evidence import normalize_models, check_binding, identity
    models=normalize_models(models,mid)
    contract = validate_model(models, mid)
    if not contract["parameters"]:
        raise GraphError("Add a layer with trainable parameters, such as Linear or Conv2d, before training.")
    doc = models[mid]
    data = doc.get("dataset")
    check_binding(doc,contract)
    config = doc.get("training", {})
    epochs = config.get("epochs", 400)
    lr = config.get("lr", 0.03)
    seed = config.get("seed", 7)
    preflight(models, mid, constrained,
              samples=len(data["x"]) if isinstance(data, dict) and isinstance(data.get("x"), list) else None,
              epochs=epochs, memory_limit_percent=memory_limit_percent)
    validate_dataset(data, contract["inputs"], contract["outputs"])
    from .tensor_data import check_adapter
    check_adapter(data, contract)
    if isinstance(lr, bool) or not isinstance(lr, (float, int)) or not math.isfinite(lr) or not 0 < lr <= 1:
        raise GraphError("Learning rate must be greater than 0 and at most 1.")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**31 - 1:
        raise GraphError("Seed must be an integer from 0 to 2³¹ − 1.")
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = compile_model(models, mid, constrained=constrained, memory_limit_percent=memory_limit_percent)
    def predict(x):
        return model.forward_tensor(x).reshape(x.shape[0], contract["outputs"])
    x = torch.tensor(data["x"], dtype=torch.float32).reshape(len(data["x"]), *contract["input_shape"])
    y = torch.tensor(data["y"], dtype=torch.float32)
    train_rows = max(1, int(len(x) * 0.8))
    indices = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    training_indices, test_indices = indices[:train_rows], indices[train_rows:]
    x_train, y_train = x[training_indices], y[training_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    stopped = False
    completed=0
    started=time.monotonic()

    def observation(epoch, optimization_loss=None):
        mode=model.training
        model.eval()
        with torch.no_grad():
            training_value=float(torch.nn.functional.mse_loss(predict(x_train),y_train))
            test_value=float(torch.nn.functional.mse_loss(predict(x_test),y_test))
        model.train(mode)
        if not math.isfinite(training_value) or not math.isfinite(test_value):
            raise GraphError('Training diverged. Reduce the learning rate or normalize the data.')
        point={'epoch':epoch,'update':epoch,'train':training_value,'test':test_value,
               'timing':'post-update','mode':'eval'}
        if optimization_loss is not None:point['optimization_loss']=optimization_loss
        history.append(point)
        if progress:progress(point)

    for epoch in range(1, epochs + 1):
        if cancel is not None and cancel.is_set():
            stopped = True
            break
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(predict(x_train), y_train)
        if not torch.isfinite(loss):
            raise GraphError("Training diverged. Reduce the learning rate or normalize the data.")
        loss.backward()
        optimizer.step()
        completed=epoch
        if epoch == 1 or epoch % max(1, epochs // 80) == 0 or epoch == epochs:
            observation(epoch,float(loss.detach()))
    if not history or history[-1]['epoch']!=completed:
        observation(completed)
    model.eval()
    with torch.no_grad():
        predicted = predict(x_test)
        mse = float(torch.nn.functional.mse_loss(predicted, y_test))
        if not math.isfinite(mse) or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise GraphError("Training diverged. Reduce the learning rate or normalize the data.")
    result = {
        "engine": "PyTorch CPU / Adam", "history": history, "mse": mse,
        "kind": "affine" if contract["affine"] else "nonlinear", "matrix_affine": contract["matrix_affine"],
        "input_shape": contract["input_shape"], "output_shape": contract["output_shape"],
        "input_groups": contract["input_groups"], "output_groups": contract["output_groups"],
        "output_mode": contract["output_mode"],
        "epochs": completed, "stopped": stopped,
        "requested_epochs":epochs,"completed_updates":completed,
        "examples_processed":completed*train_rows,"elapsed_seconds":time.monotonic()-started,
        "stop_reason":"cancelled" if stopped else "budget_completed",
        "metric_protocol":{"loss":"MSE","timing":"post-update","mode":"eval","dtype":"float32"},
        "split_indices":{"train":training_indices.tolist(),"held_out":test_indices.tolist()},
        "identity":identity(models,mid),"checkpoint_version":1,
        "preprocessing":{"kind":"identity","input_adapter":"declared-row-major","binding":doc.get('dataset_binding')},
        "parameter_ids":{item['path']+'.'+parameter:item['stable_id']+'/'+parameter
                         for item in layer_descriptions(models,mid) for parameter in ('weight','bias')
                         if item['path']+'.'+parameter in model.state_dict()},
        "layers": layer_descriptions(models, mid),
        "weights": {key: value.tolist() for key, value in model.state_dict().items()},
        "preview": [
            {"x": x_test[i].reshape(-1).tolist(), "target": y_test[i].tolist(), "prediction": predicted[i].tolist()}
            for i in range(min(5, len(x_test)))
        ],
        "train_rows": train_rows, "test_rows": len(x_test),
        "split": "seeded shuffled 80/20",
    }
    try:
        analysis=affine_summary(model,contract)
        if analysis['status']=='available':
            result.update(A=analysis.pop('A'),b=analysis.pop('b'))
        result['affine_analysis']=analysis
    except Exception as error:
        # Checkpoint and execution evidence must survive analysis failure.
        result['affine_analysis']={'status':'error','reason':str(error)}
    truth = data.get("truth")
    if "A" in result and isinstance(truth, dict) and truth.get("A") is not None and truth.get("b") is not None:
        try:
            a = torch.tensor(truth["A"], dtype=torch.float32)
            b = torch.tensor(truth["b"], dtype=torch.float32)
            effective_a=torch.tensor(result['A']);effective_b=torch.tensor(result['b'])
            if a.shape == effective_a.shape and b.shape == effective_b.shape and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()):
                result["A_error"] = float((a - effective_a).abs().max())
                result["b_error"] = float((b - effective_b).abs().max())
            else:result['truth_comparison']={'status':'unavailable','reason':'Teacher coefficient shape/values do not match the packed map.'}
        except Exception as error:
            result['truth_comparison']={'status':'error','reason':str(error)}
    return result
