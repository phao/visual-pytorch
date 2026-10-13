"""Shared actual operator/format/port discovery and stdlib-only CLI checks."""
import copy
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from app import discovery, operators
from app.core import Workspace, VERSION, SUPPORTED_VERSIONS, GraphError, check_document, validate_model
import vpw_cli


class DiscoveryTests(unittest.TestCase):
    def cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out): result = vpw_cli.main(list(args))
        return result, json.loads(out.getvalue())

    def test_authoritative_registry_sorted_complete_and_unavailable_not_admitted(self):
        value = discovery.capabilities()
        self.assertEqual(value['operators'], sorted(operators.BLOCK_TYPES))
        self.assertEqual(value['version'], 1); self.assertEqual(value['container_version'], VERSION)
        for kind in ('Constant', 'TrainableParameter'):
            self.assertNotIn(kind, value['operators']); self.assertIn(kind, value['unavailable'])
            with self.assertRaises(GraphError): discovery.operator_schema(kind)

    def test_every_example_roundtrips_actual_checker_frozen_revision_and_workspace(self):
        from app.revisions import model_document, frozen_document
        for kind in sorted(operators.BLOCK_TYPES):
            with self.subTest(kind=kind):
                schema = json.loads(json.dumps(discovery.operator_schema(kind), allow_nan=False))
                example = schema['example']; models = example['models']; root = example['root']
                for model in models.values(): check_document(model)
                c = validate_model(models, root)
                self.assertEqual(c['input_groups'], schema['example_contract']['inputs'])
                self.assertEqual(c['output_groups'], schema['example_contract']['outputs'])
                frozen = model_document(models, root); frozen_document('model', frozen)
                with tempfile.TemporaryDirectory() as td:
                    path = Path(td) / 'schema.vtw'; w = Workspace()
                    try:
                        # Dependencies first, never alphabetize ordered patches.
                        for mid, model in models.items(): w.put(model)
                        w.save(path); before = path.read_bytes(); loaded = Workspace.load(path)
                        try: self.assertEqual(validate_model(loaded.models(), root)['output_groups'], c['output_groups'])
                        finally: loaded.db.close()
                        self.assertEqual(path.read_bytes(), before)
                    finally: w.db.close()

    def test_field_sources_defaults_and_required_examples_are_not_conflated(self):
        for kind in operators.TENSOR_OPERATORS:
            schema = discovery.operator_schema(kind)
            self.assertEqual(schema['fields']['semantic'], sorted(operators.FIELDS[kind]))
        self.assertNotIn('kernel_size', discovery.operator_schema('Conv2d')['defaults'])
        self.assertEqual(discovery.operator_schema('Conv2d')['defaults']['stride'], [1, 1])
        self.assertEqual(discovery.operator_schema('AvgPool2d')['defaults']['stride'], [2, 2])
        self.assertEqual(discovery.operator_schema('Linear')['defaults'], {'bias': True})
        self.assertEqual(discovery.operator_schema('Activation')['defaults'], {'dim': -1})
        self.assertEqual(discovery.operator_schema('LSTM')['defaults']['num_layers'], 1)
        self.assertIn('hidden_size', discovery.operator_schema('LSTM')['fields']['required'])
        self.assertTrue(any('does not prove' in message for message in discovery.capabilities()['limitations']))

    def test_mutation_of_returned_data_cannot_change_next_discovery(self):
        value = discovery.operator_schema('Conv2d'); previous = copy.deepcopy(value)
        value['example']['models'][value['example']['root']]['graph']['nodes'][0]['shape'][0] = 999
        value['defaults']['stride'][0] = 999
        self.assertEqual(discovery.operator_schema('Conv2d'), previous)

    def test_unknown_boolean_empty_and_wrong_case_operator_refuse(self):
        for value in (True, None, {}, '', 'linear', '__import__'):
            with self.subTest(value=value), self.assertRaises(GraphError): discovery.operator_schema(value)
        for value in ('Future', 'linear'):
            self.assertEqual(self.cli('operator-schema', value)[0], 1)

    def test_capabilities_refuses_missing_metadata_for_future_registry_entry(self):
        with patch.object(operators, 'BLOCK_TYPES', operators.BLOCK_TYPES | {'Future'}), self.assertRaises(GraphError):
            discovery.capabilities()

    def test_format_sql_is_actual_creator_with_no_persisted_workspace(self):
        result = discovery.format_schema(); w = Workspace()
        try:
            expected = [{'name': n, 'type': t, 'sql': s} for n, t, s in w.db.execute(
                'SELECT name,type,sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY name')]
            self.assertEqual(result['sqlite_schema'], expected)
        finally: w.db.close()
        self.assertEqual(result['reader_versions'], sorted(SUPPORTED_VERSIONS))
        self.assertIn('parameter-binding', result['independent_kinds'])
        self.assertIn('spec_revisions_no_delete', [s['name'] for s in expected])

    def test_application_queries_do_not_load_workspace_read_config_or_modify_rng(self):
        rng = random.getstate()
        with patch('vpw_cli.Workspace.load', side_effect=AssertionError('No workspace load')), \
             patch('vpw_cli.ram_percentage', side_effect=AssertionError('No user config')):
            for command in (('capabilities',), ('format-schema',), ('operator-schema', 'LSTM')):
                self.assertEqual(self.cli(*command)[0], 0)
                self.assertEqual(self.cli('/missing/arbitrary.vtw', *command)[0], 0)
        self.assertEqual(rng, random.getstate())
        self.assertEqual(self.cli('ignored.vtw', '--expect-revision', '0' * 64, 'capabilities')[0], 1)

    def test_fresh_cli_without_site_libraries_or_workspace_leaves_directory_empty(self):
        path = Path(__file__).parents[1] / 'vpw_cli.py'
        with tempfile.TemporaryDirectory() as td:
            for command in (('capabilities',), ('format-schema',), ('operator-schema', 'Recurrent')):
                script = ('import sys,runpy;sys.path.insert(0,' + repr(str(path.parent)) + ');'
                    'sys.argv=' + repr([str(path), *command]) + ';'
                    'try: runpy.run_path(' + repr(str(path)) + ',run_name="__main__")\n'
                    'except SystemExit as e: assert e.code==0\n'
                    'assert not any(n.split(".")[0] in ("torch","numpy","PyQt6") for n in sys.modules)')
                # Compound try cannot follow a semicolon: put it on its own line.
                script = script.replace(';try:', '\ntry:')
                p = subprocess.run(['/usr/bin/python3', '-I', '-S', '-B', '-c', script], cwd=td,
                                   capture_output=True, text=True, timeout=30)
                self.assertEqual(p.returncode, 0, p.stderr); self.assertTrue(json.loads(p.stdout)['ok'])
                self.assertEqual(list(Path(td).iterdir()), [])

    def test_ports_cli_preserves_file_and_exact_packing_and_control_bindings(self):
        value = discovery.operator_schema('LSTM')['example']; w = Workspace()
        try:
            for model in value['models'].values(): w.put(model)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'ports.vtw'; w.save(path); before = path.read_bytes()
                code, result = self.cli(str(path), 'ports', value['root'])
                self.assertEqual(code, 0, result); self.assertEqual(path.read_bytes(), before)
                self.assertEqual(result['value'], discovery.port_contract(w.models(), value['root']))
                op = next(n for n in result['value']['nodes'] if n['id'] == 'op')
                self.assertEqual(op['bindings']['inputs']['hidden'], 'initial-h')
                self.assertEqual([g['start'] for g in result['value']['inputs']], [0, 6, 8])
                self.assertEqual(self.cli(str(path), '--expect-revision', '0' * 64, 'ports', value['root'])[0], 1)
                self.assertEqual(path.read_bytes(), before)
        finally: w.db.close()

    def test_splitter_multidimensional_contract_and_semantic_warning(self):
        value = discovery.operator_schema('Splitter')['example']; root = value['root']
        value['models'][root]['graph']['nodes'][0]['shape'] = [3, 2]
        c = discovery.port_contract(value['models'], root)
        self.assertEqual([g['shape'] for g in c['outputs']], [[3, 1], [3, 1]])
        self.assertIn('not explicitly declared', c['output_representation'])
        self.assertIn('last sample axis', discovery.operator_schema('Splitter')['shape_rule'])

    def test_referenced_cell_interfaces_and_role_plan_are_complete(self):
        schema = discovery.operator_schema('Recurrent')
        c = schema['example_contract']; child = c['referenced_interfaces'][0]
        self.assertEqual([g['id'] for g in child['inputs']], ['x', 'state'])
        self.assertEqual([g['id'] for g in child['outputs']], ['next'])
        plan = next(p for p in c['execution_plan'] if p['id'] == 'op')
        self.assertEqual(plan['recurrent']['states'], {'hidden': 1})
        self.assertEqual(plan['recurrent']['ports'], ['sequence', 'hidden'])

    def test_unequal_router_heads_do_not_get_alphabetized_or_changed_by_geometry(self):
        example = discovery.operator_schema('Splitter')['example']; root = example['root']
        graph = example['models'][root]['graph']; graph['nodes'][0]['shape'] = [3]
        router = next(n for n in graph['nodes'] if n['id'] == 'op')
        router['parts'] = [{'output': 'right', 'features': 2}, {'output': 'left', 'features': 1}]
        before = discovery.port_contract(example['models'], root)
        self.assertEqual([g['id'] for g in before['outputs']], ['other-y', 'y'])
        self.assertEqual([g['shape'] for g in before['outputs']], [[2], [1]])
        for i, node in enumerate(graph['nodes']): node.update(x=100-i*17, y=i*12, collapsed=True)
        after = discovery.port_contract(example['models'], root)
        self.assertEqual(before, after)


if __name__ == '__main__': unittest.main()
