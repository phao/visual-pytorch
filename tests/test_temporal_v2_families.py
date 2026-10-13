"""Pure retained v2 family contracts; no tensors or inferred state/data."""
import copy
import itertools
import json
import random
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, VERSION, Workspace, default_model
from app.evidence import digest
from app.families import specialize, validate_family
from app.family_export import family_source
from app.family_records import definition as record_definition, retain, validate_specialization
from app.revisions import validate_all
from examples.runtime_lstm_family import definition
from examples.lstm_family import definition as fixed_definition
from test_lstm_families import parameter_count
from app.runtime_shapes import shape_contract

ROOT = Path(__file__).resolve().parents[1]


class TemporalV2FamilyTests(unittest.TestCase):
    def setUp(self):
        self.w = Workspace(); self.addCleanup(self.w.db.close)

    def record(self,ref):
        return self.w.get_spec('specialization',ref['id'],ref['revision'])

    def test_sixty_four_constructors_at_three_times_have_exact_ports_counts_and_proof(self):
        doc = definition(); before = copy.deepcopy(doc); rng = random.getstate()
        for layers,directions,projection,bias,width,hidden in itertools.product((1,2),(1,2),(0,1),(False,True),(1,3),(2,4)):
            bindings = {'time':4,'layers':layers,'directions':directions==2,'projection':projection,
                        'bias':bias,'width':width,'cell-width':hidden}
            result = specialize(doc,bindings)
            self.assertEqual(result['version'],6)
            self.assertEqual(result['contract']['parameters'],parameter_count(width,hidden,projection,layers,directions,bias))
            spec = result['runtime_specification']; self.assertEqual(spec['version'],2)
            self.assertEqual(result['runtime'],{'version':3,'kind':'runtime-temporal-v2',
                             'inputs':{'sequence':[0]},'specification_sha256':digest(spec)})
            for t in (1,5,13):
                shapes = {g['id']:copy.deepcopy(g['shape']) for g in result['contract']['input_groups']}
                shapes['sequence'][0]=t
                actual = shape_contract(result['models'],result['root'],spec,shapes)
                self.assertEqual(actual['shapes']['lstm']['ports'],{
                    'sequence':[t,directions*(projection or hidden)],'hidden':[layers*directions,projection or hidden],
                    'cell':[layers*directions,hidden]})
        self.assertEqual(doc,before); self.assertEqual(random.getstate(),rng)

    def test_exact_retained_versions_model_and_dependency_hash(self):
        result = retain(self.w,family=definition(),materialize=False)
        s = self.record(result['specialization'])
        self.assertEqual(s['version'],7); self.assertEqual(s['runtime'],result['resolved']['runtime'])
        self.assertEqual(self.w.get_spec('model',s['model']['id'])['version'],4)
        validate_all(self.w.db)

    def test_nullable_family8_is_fixed_not_an_inferred_runtime_contract(self):
        doc = fixed_definition(); doc['version']=8
        result = retain(self.w,family=doc,materialize=False)
        self.assertIsNone(result['resolved']['runtime']); self.assertIsNone(result['resolved']['runtime_specification'])
        self.assertEqual(self.record(result['specialization'])['version'],7)

    def test_policy_grammar_exactness_and_old_lstm_policy_refusal(self):
        cases = [{'version':True,'kind':'runtime-temporal-v2','inputs':{'sequence':[0]}},
                 {'version':1,'kind':'runtime-temporal-v1','inputs':{'sequence':[0]}},
                 {'version':2,'kind':'runtime-temporal-v1','inputs':{'sequence':[0]}},
                 {'version':2,'kind':'runtime-temporal-v2','inputs':{'hidden':[0]}},
                 {'version':2,'kind':'runtime-temporal-v2','inputs':{'sequence':[True]}},
                 {'version':2,'kind':'runtime-temporal-v2','inputs':{'sequence':[0]},'extra':1}]
        before = self.w.db.serialize()
        for policy in cases:
            doc = definition(); doc['runtime_policy']=policy
            with self.subTest(policy=policy),self.assertRaises(GraphError): self.w.put_spec('family',doc)
            self.assertEqual(self.w.db.serialize(),before)
        for version in (6,7):
            doc = definition(); doc['version']=version
            with self.assertRaises(GraphError): validate_family(doc)

    def test_fixed_output_annotation_cannot_be_silently_widened(self):
        doc = fixed_definition(); doc.update(version=8,runtime_policy=definition()['runtime_policy'])
        result = specialize(doc)
        shapes = {g['id']:copy.deepcopy(g['shape']) for g in result['contract']['input_groups']}
        shapes['sequence'][0]=9
        with self.assertRaises(GraphError): shape_contract(result['models'],result['root'],result['runtime_specification'],shapes)

    def test_time_to_learned_dimension_refuses_before_atomic_retention(self):
        doc = definition()
        # Width symbol is used in a learned axis; declaring that Input's last
        # axis as runtime must never become parameter recreation.
        doc['runtime_policy']['inputs']['sequence']=[1]
        before = self.w.db.serialize()
        with self.assertRaises(GraphError): retain(self.w,family=doc,materialize=False)
        self.assertEqual(self.w.db.serialize(),before)

    def test_witness_version_kind_hash_or_context_forgery_is_atomic(self):
        result = retain(self.w,family=definition(),materialize=False)
        original = record_definition(self.record(result['specialization'])); before=self.w.db.serialize()
        for mutation in ('version','kind','hash','axis','context','bindings','missing','extra'):
            bad = copy.deepcopy(original)
            if mutation=='version':bad['runtime']['version']=2
            elif mutation=='kind':bad['runtime']['kind']='runtime-temporal-v1'
            elif mutation=='hash':bad['runtime']['specification_sha256']='0'*64
            elif mutation=='axis':bad['runtime']['inputs']={'cell':[0]}
            elif mutation=='context':bad['instances'][0]['bindings']['time']=9
            elif mutation=='bindings':bad['bindings']['time']=9
            elif mutation=='missing':bad.pop('runtime')
            else:bad['runtime']['extra']=True
            with self.subTest(mutation=mutation),self.assertRaises(GraphError):self.w.put_spec('specialization',bad)
            self.assertEqual(self.w.db.serialize(),before)

    def test_specialization_version_must_match_exact_family(self):
        result=retain(self.w,family=definition(),materialize=False)
        bad=record_definition(self.record(result['specialization']));bad['version']=6
        bad['runtime']=None
        with self.assertRaisesRegex(GraphError,'version must match'):self.w.put_spec('specialization',bad)
        bad['version']=8
        with self.assertRaisesRegex(GraphError,'version must match'):self.w.put_spec('specialization',bad)
        for version in (True,10):
            bad['version']=version
            with self.assertRaises(GraphError):validate_specialization(bad)

    def test_historical_revision_and_private_copies_remain_exact(self):
        doc=definition();first=retain(self.w,family=doc,materialize=False)
        record=self.record(first['specialization']); doc['name']='New head';self.w.put_spec('family',doc)
        self.assertEqual(self.record(first['specialization']),record)
        record['runtime']['inputs']['sequence'][0]=99
        self.assertEqual(self.record(first['specialization'])['runtime']['inputs'],{'sequence':[0]})

    def test_late_materialization_failure_restores_every_row_and_dirty_state(self):
        before=self.w.db.serialize();dirty=self.w.dirty;real=self.w.put
        def fail(doc):real(doc);raise RuntimeError('late insert')
        with patch.object(self.w,'put',side_effect=fail),self.assertRaisesRegex(RuntimeError,'late insert'):
            retain(self.w,family=definition(),materialize=True)
        self.assertEqual(self.w.db.serialize(),before);self.assertEqual(self.w.dirty,dirty)

    def test_vtw19_fences_family_only_and_specialization_without_disk_mutation(self):
        for retained in (False,True):
            w=Workspace()
            try:
                if retained:retain(w,family=definition(),materialize=False)
                else:w.put_spec('family',definition())
                with tempfile.TemporaryDirectory() as td:
                    path=Path(td)/'new.vtw';w.save(path)
                    with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=18');db.commit()
                    before=path.read_bytes()
                    with self.assertRaisesRegex(GraphError,'version-19'):Workspace.load(path)
                    self.assertEqual(path.read_bytes(),before)
            finally:w.db.close()

    def test_old18_open_only_changes_working_copy(self):
        self.w.put(default_model())
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'old.vtw';self.w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=18');db.commit()
            before=path.read_bytes();opened=Workspace.load(path)
            try:self.assertEqual(opened.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            finally:opened.db.close()
            self.assertEqual(path.read_bytes(),before)

    def test_pure_host_proof_export_and_import_dependencies(self):
        code=('import sys;from examples.runtime_lstm_family import definition;'
              'from app.families import specialize;from app.family_export import family_source;'
              'doc=definition();assert specialize(doc)["runtime_specification"]["version"]==2;'
              'source=family_source(doc);compile(source["code"],"<family>","exec");'
              'assert source["provenance"]["family_grammar"]==8;'
              'assert not any(m in sys.modules for m in ("torch","numpy","PyQt6"))')
        ran=subprocess.run(['/usr/bin/python3','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(ran.returncode,0,ran.stderr)

    def test_real_example_cli_exact_preview_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'example.vtw';command=['/usr/bin/python3','examples/runtime_lstm_family.py',str(path)]
            ran=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(ran.returncode,0,ran.stderr);result=json.loads(ran.stdout);before=path.read_bytes()
            family=result['family']
            for args in (['preview-stored-family',family['id'],family['revision']],
                         ['export-family',family['id'],family['revision'],'--output',str(Path(td)/'family.py')]):
                checked=subprocess.run(['/usr/bin/python3','vpw_cli.py',str(path),*args],cwd=ROOT,capture_output=True,text=True,timeout=30)
                self.assertEqual(checked.returncode,0,checked.stdout+checked.stderr)
                self.assertTrue(json.loads(checked.stdout)['ok']);self.assertEqual(path.read_bytes(),before)
            refused=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(refused.returncode,0);self.assertEqual(path.read_bytes(),before)

    def test_owned_family7_to8_upgrade_is_atomic_no_sql_until_retention(self):
        from app.authoring_sessions import Sessions
        doc=definition();doc.update(version=7,runtime_policy=None)
        ref=self.w.put_spec('family',doc);sessions=Sessions();owner=object()
        h=sessions.open(self.w,owner,'family',base=ref);before=self.w.db.serialize()
        ops=[{'op':'property','target':{'kind':'document'},'key':'version','text':'8'},
             {'op':'property','target':{'kind':'document'},'key':'runtime_policy','text':json.dumps(definition()['runtime_policy'])}]
        h=sessions.family_edit(self.w,h['session'],owner,0,ops)
        self.assertEqual(h['sequence'],1);self.assertEqual(self.w.db.serialize(),before)
        checked=sessions.family_preview(self.w,h['session'],owner,1)
        self.assertEqual(checked['status'],'verified');self.assertEqual(self.w.db.serialize(),before)
        saved=sessions.commit(self.w,h['session'],owner,1)
        self.assertEqual(self.w.get_spec('family',saved['id'],saved['revision'])['version'],8)

    def test_owned_family8_cannot_downgrade_or_accept_unknown_version(self):
        from app.authoring_sessions import Sessions
        ref=self.w.put_spec('family',definition());sessions=Sessions();owner=object()
        h=sessions.open(self.w,owner,'family',base=ref)
        for version in (7,11,True):
            before=sessions.read(h['session'],owner,0)
            with self.subTest(version=version),self.assertRaisesRegex(GraphError,'preserve stable'):
                sessions.family_edit(self.w,h['session'],owner,0,[{'op':'property','target':{'kind':'document'},'key':'version','text':json.dumps(version)}])
            self.assertEqual(sessions.read(h['session'],owner,0),before)
        upgraded=sessions.family_edit(self.w,h['session'],owner,0,[{'op':'property','target':{'kind':'document'},'key':'version','text':'9'}])
        self.assertEqual(upgraded['sequence'],1)
        renamed=sessions.family_edit(self.w,h['session'],owner,1,[{'op':'property','target':{'kind':'document'},'key':'name','text':'"Explicit family9"'}])
        self.assertEqual(renamed['sequence'],2)
        self.assertEqual(sessions.family_preview(self.w,h['session'],owner,2)['status'],'verified')
