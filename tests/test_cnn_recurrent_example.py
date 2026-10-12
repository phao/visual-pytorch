"""Pure original-D integrated CNN/state/padding/grouped-time example contracts."""
import copy
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.core import GraphError, Workspace, VERSION
from app.datasets import check_manifest, make_split
from app.experiment_objects import candidate_bundle, preview_experiment, selection_at
from app.families import specialize
from app.revisions import validate_all
from app.task_execution import resolve_run
from examples.cnn_recurrent_experiment import family, observations, populate

ROOT=Path(__file__).resolve().parents[1]


class CNNRecurrentExampleTests(unittest.TestCase):
    def make(self,**kwargs):
        w=Workspace();self.addCleanup(w.db.close)
        self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        info=populate(w,materialize=False,**kwargs);r=info['experiment'];e=w.get_spec('experiment',r['id'],r['revision'])
        return w,e,r,info

    def test_scoped_channels_activation_baseline_and_runtime_padded_time_are_distinct(self):
        doc=family();before=copy.deepcopy(doc);rng=random.getstate()
        for width in (1,2,4):
            for activation in ('ReLU','Tanh'):
                result=specialize(doc,{'features':width,'activation':activation})
                self.assertEqual(result['contract']['parameters'],4*width+6)
                child=next(row for row in result['instances'] if row['template']=='cnn-cell')
                self.assertEqual(child['bindings'],{'cell-features':width,'cell-activation':activation})
                self.assertEqual(result['runtime']['kind'],'runtime-temporal-v2')
        self.assertEqual(doc,before);self.assertEqual(random.getstate(),rng)
        w,e,_,info=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        p,_=resolve_run(b,constrained=False)
        self.assertEqual(info['parameters'],14);self.assertEqual(b['version'],4)
        self.assertEqual(p['contract']['shapes']['scan']['recurrent']['steps'],7)
        self.assertEqual(b['runtime']['specialization']['bindings']['time'],5)

    def test_ragged_lengths_padding_execution_masks_reset_and_loss_masks_are_explicit(self):
        w,e,_,_=self.make();b=candidate_bundle(e,selection_at(e,0),w.get_spec);p,_=resolve_run(b,constrained=False)
        rows={r['id']:r['values'] for r in b['dataset']['records']}
        for role,pop in p['populations'].items():
            for i,rid in enumerate(pop['observation_ids']):
                row=rows[rid];length=len(row['frames'])
                self.assertEqual(pop['inputs']['lengths'][i],[float(length)])
                self.assertEqual(pop['inputs']['frames'][i][:length],row['frames'])
                self.assertEqual(pop['inputs']['frames'][i][length:],[[[[0.]*5 for _ in range(4)]]]*(7-length))
                self.assertEqual(pop['inputs']['step_mask'][i],row['step_mask'])
                self.assertEqual(pop['inputs']['step_reset'][i],row['step_reset'])
                self.assertEqual(pop['masks']['trajectory'][i],row['valid'])
                self.assertEqual(pop['masks']['final'][i],row['valid_final'])
        self.assertIn(0,[len(r['values']['frames']) for r in b['dataset']['records']])

    def test_global_temporal_ties_across_subjects_replicates_and_ragged_lengths_stay_together(self):
        w,e,_,_=self.make(split='temporal');b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        manifest=b['manifest'];rows={r['id']:r['values'] for r in b['dataset']['records']};owners={}
        for role,ids in manifest['roles'].items():
            for rid in ids:self.assertEqual(owners.setdefault(rows[rid]['time'],role),role)
        self.assertEqual(len(owners),8);self.assertEqual([len(manifest['roles'][r]) for r in ('train','validation','test')],[32,16,16])
        bad=copy.deepcopy(manifest);bad['roles']['train'][0],bad['roles']['test'][0]=bad['roles']['test'][0],bad['roles']['train'][0]
        with self.assertRaisesRegex(GraphError,'chronological'):check_manifest(b['dataset'],bad)

    def test_subject_holdout_is_separate_and_never_silently_means_within_subject_chronology(self):
        w,e,_,_=self.make(split='group');b=candidate_bundle(e,selection_at(e,0),w.get_spec)
        rows={r['id']:r['values'] for r in b['dataset']['records']};owners={}
        for role,ids in b['manifest']['roles'].items():
            for rid in ids:self.assertEqual(owners.setdefault(rows[rid]['subject'],role),role)
        self.assertEqual(len(owners),4)
        for role,ids in b['manifest']['roles'].items():self.assertEqual({rows[i]['time'] for i in ids},set(range(0,80,10)))
        bad=copy.deepcopy(b['manifest']);bad['roles']['train'][0],bad['roles']['test'][0]=bad['roles']['test'][0],bad['roles']['train'][0]
        with self.assertRaisesRegex(GraphError,'group'):check_manifest(b['dataset'],bad)

    def test_physical_record_reordering_does_not_reinterpret_either_split(self):
        d=observations();other=copy.deepcopy(d);other['records'].reverse()
        for protocol in ({'method':'temporal','field':'time','fractions':[.5,.25,.25]},
                         {'method':'group','field':'subject','seed':5,'fractions':[.5,.25,.25]}):
            a=make_split(d,protocol);b=make_split(other,protocol)
            self.assertEqual(a['roles'],b['roles']);self.assertEqual(a['protocol'],b['protocol'])
            self.assertNotEqual(a['dataset_revision'],b['dataset_revision'])

    def test_constructor_invalid_bindings_before_retention_and_preview_is_pure(self):
        w,e,_,_=self.make();before=w.db.serialize();rng=random.getstate()
        self.assertEqual(preview_experiment(e,w.get_spec)['candidates'][0]['status'],'compatible')
        for values in ({'features':0},{'features':True},{'activation':'unknown'},{'time':0},{'time':33}):
            with self.subTest(values=values),self.assertRaises(GraphError):specialize(family(),values)
        self.assertEqual(w.db.serialize(),before);self.assertEqual(random.getstate(),rng)

    def test_public_example_real_stdlib_CLI_materializes_complete_closure_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cnn-rnn.vtw'
            # runpy passes the script path as argv[0], not a second model argument.
            probe="import runpy,sys;path=sys.argv.pop(1);sys.argv[0]=path;runpy.run_path(path,run_name='__main__');assert not any(n in sys.modules for n in ('torch','numpy','PyQt6'))"
            command=[sys.executable,'-c',probe,str(ROOT/'examples/cnn_recurrent_experiment.py'),str(path),'--split','temporal']
            result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr);data=json.loads(result.stdout)
            self.assertEqual(data['parameters'],14);self.assertIn('not-trained',data['status'])
            loaded=Workspace.load(path)
            try:
                self.assertEqual(len(loaded.models()),2);self.assertEqual(loaded.db.execute('PRAGMA user_version').fetchone()[0],VERSION)
                validate_all(loaded.db)
            finally:loaded.db.close()
            before=path.read_bytes();again=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(again.returncode,0);self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
