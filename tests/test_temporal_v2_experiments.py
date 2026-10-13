"""Pure retained experiment6 selection, owned atomic upgrades and file20 gates."""
import copy
import json
import random
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from app.authoring_sessions import Sessions
from app.core import GraphError, Workspace, VERSION
from app.evidence import digest
from app.experiment_objects import candidate_bundle, preview_experiment, selection_at
from app.task_execution import validate_bundle, run_identity, resolve_run
from app.revisions import validate_all
from runtime_v2_experiment_fixture import experiment

ROOT=Path(__file__).resolve().parents[1]


class TemporalV2ExperimentTests(unittest.TestCase):
    def make(self,**kwargs):
        w,e,r=experiment(**kwargs);self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        return w,e,r

    def test_explicit_two_dataset_times_one_constructor_exact_provenance_no_SQL_or_RNG(self):
        w,e,_=self.make();before=w.db.serialize();rng=random.getstate()
        rows=preview_experiment(e,w.get_spec)['candidates']
        self.assertEqual([r['status'] for r in rows],['compatible']*2)
        bs=[candidate_bundle(e,r['coordinates'],w.get_spec) for r in rows]
        self.assertEqual([b['version'] for b in bs],[4,4]);self.assertEqual(bs[0]['model'],bs[1]['model'])
        self.assertEqual(bs[0]['runtime'],bs[1]['runtime'])
        self.assertEqual([validate_bundle(b)['shapes']['lstm']['ports']['sequence'] for b in bs],[[3,4],[7,4]])
        self.assertEqual([run_identity(b)['compiler'] for b in bs],['native-runtime-temporal-v2']*2)
        self.assertEqual(w.db.serialize(),before);self.assertEqual(random.getstate(),rng)

    def test_old_versions_reject_v2_and_omitted_execution_remains_concrete(self):
        w,e,_=self.make();before=w.db.serialize()
        for version in (1,2,3,4,5,999,True):
            bad=copy.deepcopy(e);bad.pop('revision');bad['version']=version
            with self.subTest(version=version),self.assertRaises(GraphError):w.put_spec('experiment',bad)
            self.assertEqual(w.db.serialize(),before)
        w,e,_=self.make(times=(5,),execution=None)
        b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        self.assertEqual(b['version'],1);self.assertNotIn('runtime',b)
        self.assertEqual(run_identity(b)['compiler'],'native-graph-v1')

    def test_exact_family8_null_and_older_policy_mismatch_stays_incompatible(self):
        for mode in ('null','v1','spatial'):
            w,e,_=self.make();doc=copy.deepcopy(e);doc.pop('revision')
            if mode=='null':
                f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision');f['runtime_policy']=None
                doc['models'][0]['family']=w.put_spec('family',f)
            else:doc['models'][0]['execution']='runtime-temporal-v1' if mode=='v1' else 'runtime-spatial-v1'
            r=w.put_spec('experiment',doc);doc=w.get_spec('experiment',r['id'],r['revision'])
            self.assertEqual(preview_experiment(doc,w.get_spec)['candidates'][0]['status'],'incompatible')

    def test_typed_constructor_axes_keep_order_and_invalid_state_width_is_visible(self):
        w,e,_=self.make(axes=[{'symbol':'bias','values':[False,True]}])
        rows=preview_experiment(e,w.get_spec)['candidates']
        self.assertEqual([r['coordinates']['model']['bindings']['bias'] for r in rows],[False,False,True,True])
        self.assertEqual([r['status'] for r in rows],['compatible']*4)
        w,e,_=self.make(times=(3,),axes=[{'symbol':'projection','values':[0,2]}])
        rows=preview_experiment(e,w.get_spec)['candidates']
        self.assertEqual([r['status'] for r in rows],['incompatible','compatible'])
        self.assertIn('shape',rows[0]['reason'])

    def test_historical_family_and_experiment_revisions_never_follow_new_heads(self):
        w,e,ref=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        f=w.get_spec('family',e['models'][0]['family']['id']);f.pop('revision');f['name']='New family head';w.put_spec('family',f)
        newer=copy.deepcopy(e);newer.pop('revision');newer['name']='New trial head';w.put_spec('experiment',newer)
        old=w.get_spec('experiment',ref['id'],ref['revision'])
        self.assertEqual(candidate_bundle(old,selection_at(old,0),w.get_spec),b)
        validate_all(w.db)

    def test_owned_3_to6_upgrade_is_explicit_atomic_no_rows_before_commit(self):
        w,e,ref=self.make(version=3,execution=None);store=Sessions();owner=object();self.addCleanup(store.close_owner,owner)
        h=store.open(w,owner,'experiment',base=ref);before=w.db.serialize()
        ops=[{'op':'property','target':{'kind':'document'},'key':'version','text':'6'},
             {'op':'property','target':{'kind':'entry','collection':'models','ordinal':0},'key':'execution','text':'"runtime-temporal-v2"'}]
        h=store.experiment_edit(w,h['session'],owner,0,ops)
        preview=store.experiment_preview(w,h['session'],owner,h['sequence'])
        self.assertEqual([c['status'] for c in preview['candidates']],['compatible']*2)
        self.assertEqual(w.db.serialize(),before)
        for value in ('5','999','true'):
            with self.subTest(value=value),self.assertRaises(GraphError):
                store.experiment_edit(w,h['session'],owner,h['sequence'],[dict(ops[0],text=value)])
            self.assertEqual(store.describe(h['session'],owner),h);self.assertEqual(w.db.serialize(),before)
        retained=store.commit(w,h['session'],owner,h['sequence'])
        self.assertEqual(w.get_spec('experiment',retained['id'],retained['revision'])['version'],6)

    def test_failed_owned_execution_patch_preserves_session_sequence_and_SQL(self):
        w,e,ref=self.make(version=5,execution='concrete-v1');store=Sessions();owner=object();self.addCleanup(store.close_owner,owner)
        h=store.open(w,owner,'experiment',base=ref);before=w.db.serialize()
        bad=[{'op':'property','target':{'kind':'entry','collection':'models','ordinal':0},'key':'execution','text':'"runtime-temporal-v2"'}]
        # Draft may temporarily contain an incompatible version/policy pair;
        # Check/Commit remain the semantic admission boundary, with text retained.
        h=store.experiment_edit(w,h['session'],owner,0,bad)
        with self.assertRaises(GraphError):store.check(w,h['session'],owner,h['sequence'])
        self.assertEqual(store.describe(h['session'],owner),h);self.assertEqual(w.db.serialize(),before)

    def test_vtw20_unmaterialized_experiment_fence_and_old19_working_copy_only(self):
        w,e,ref=self.make();self.assertEqual(w.list_specs('specialization'),[])
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'new.vtw';w.save(path)
            with closing(sqlite3.connect(path)) as db:db.execute('PRAGMA user_version=19');db.commit()
            before=path.read_bytes()
            with self.assertRaisesRegex(GraphError,'version-20'):Workspace.load(path)
            self.assertEqual(path.read_bytes(),before)
            old=Workspace();old_path=Path(td)/'old.vtw'
            try:old.save(old_path)
            finally:old.db.close()
            with closing(sqlite3.connect(old_path)) as db:db.execute('PRAGMA user_version=19');db.commit()
            before=old_path.read_bytes();opened=Workspace.load(old_path)
            try:self.assertEqual(opened.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
            finally:opened.db.close()
            self.assertEqual(old_path.read_bytes(),before)

    def test_real_CLI_preview_exact_readonly_and_stdlib_only(self):
        w,e,ref=self.make(mode='pad');before=w.db.serialize()
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'lab.vtw';w.save(path);before=path.read_bytes()
            probe="import sys,vpw_cli;code=vpw_cli.main(sys.argv[1:]);assert not any(n in sys.modules for n in ('torch','numpy','PyQt6'));raise SystemExit(code)"
            result=subprocess.run([sys.executable,'-c',probe,str(path),'preview-experiment',ref['id'],ref['revision']],
                                  cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);rows=json.loads(result.stdout)['value']['candidates']
            self.assertEqual([r['status'] for r in rows],['compatible']*2);self.assertEqual(path.read_bytes(),before)

    def test_real_CLI_stdin_admission_and_failed_batch_are_atomic_and_stdlib_only(self):
        w,e,_=self.make();good=copy.deepcopy(e);good.pop('revision');good['id']='agent-runtime-v2'
        bad=copy.deepcopy(good);bad['id']='invalid-runtime-v2';bad['version']=5
        probe="import sys,vpw_cli;code=vpw_cli.main(sys.argv[1:]);assert not any(n in sys.modules for n in ('torch','numpy','PyQt6'));raise SystemExit(code)"
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'agent.vtw';w.save(path);before=path.read_bytes()
            command=[sys.executable,'-c',probe,str(path),'apply','-']
            edits=[{'op':'put-spec','kind':'experiment','document':d} for d in (good,bad)]
            failed=subprocess.run(command,input=json.dumps(edits),cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(failed.returncode,1,failed.stdout+failed.stderr);self.assertEqual(path.read_bytes(),before)
            success=subprocess.run(command,input=json.dumps(edits[:1]),cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(success.returncode,0,success.stdout+success.stderr)
            loaded=Workspace.load(path)
            try:
                value=loaded.get_spec('experiment',good['id']);self.assertEqual(value['version'],6)
                self.assertEqual(value['models'][0]['execution'],'runtime-temporal-v2')
                self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION);validate_all(loaded.db)
            finally:loaded.db.close()


if __name__=='__main__':unittest.main()
