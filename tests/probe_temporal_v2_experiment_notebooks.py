"""Persist receipts from genuine retained runtime-v2 saved/rerun kernels.

Run in numerical Python with PYTHONPATH=.:tests and a NEW private output folder.
Full portable bundles stay in that folder; no user workspace/config is opened.
"""
import json
import sys
from pathlib import Path

from app.core import Workspace, VERSION
from app.experiment_objects import candidate_bundle, execute_candidate, selection_at
from app.notebooks import create_bundle, execute_bundle, verify_bundle
from app.revisions import validate_all
from runtime_v2_experiment_fixture import experiment


def check(folder):
    w,e,ref=experiment(times=(7,),mode='pad',updates=3)
    try:
        done=execute_candidate(w,ref,selection_at(e,0),'kernel-proof','Actual runtime-v2 evidence',
                               python=sys.executable,timeout=30)
        result=done['record']['attempts'][0]['result']
        assert done['record']['version']==6 and done['record']['attempts'][0]['status']=='completed'
        assert result['identity']['compiler']=='native-runtime-temporal-v2'
        views=[]
        for mode in ('result-report','rerun'):
            recipe=w.put_spec('report',{'version':1,'id':mode,'name':'Actual retained runtime-v2 kernel',
                'question':'Do saved and rerun kernels preserve exact checkpoints/metrics?',
                'mode':mode,'experiment':ref if mode=='rerun' else None,'runs':[] if mode=='rerun' else [done['reference']],
                'sections':['models','training','evaluation','checkpoints','failures'],
                'analysis':{'kind':'descriptive','inferential_unit':'run'}})
            before=w.db.serialize();bundle=create_bundle(w,recipe,folder/mode);verify_bundle(bundle['path'])
            actual=execute_bundle(bundle['path'],timeout=60)
            assert actual['execution']['status']=='completed',actual['execution']['error']
            view=json.loads((Path(actual['path'])/'report-data.json').read_text());views.append(view)
            assert view['runs'][0]['checkpoint_id']==result['checkpoint_id']
            assert view['notebook_execution']['training_requested']==(mode=='rerun')
            assert len(list(Path(actual['path']).glob('runtime-architecture-*.py')))==1
            assert len(list(Path(actual['path']).glob('predictor-*.py')))==1
            assert w.db.serialize()==before
            (folder/(mode+'-execution.json')).write_text(json.dumps(actual,sort_keys=True,indent=2)+'\n')
            (folder/(mode+'-report-data.json')).write_text(json.dumps(view,sort_keys=True,indent=2)+'\n')
        assert views[0]['runs'][0]['metrics']==views[1]['runs'][0]['metrics']
        validate_all(w.db);w.save(folder/'runtime-v2.vtw');loaded=Workspace.load(folder/'runtime-v2.vtw')
        try:
            assert loaded.db.execute('PRAGMA user_version').fetchone()[0]==20
            assert loaded.get_spec('run',done['record']['id'],done['record']['revision'])==done['record']
            assert candidate_bundle(e,selection_at(e,0),loaded.get_spec)['version']==4
        finally:loaded.db.close()
        print(json.dumps({'ok':True,'experiment_run_version':6,'family_version':8,'specialization_version':7,
            'bundle_identity_version':4,'container_version':VERSION,'actual_kernels':['result-report','rerun'],
            'checkpoint_id':result['checkpoint_id'],'metrics_exact':True,'companions_present':True,'output':str(folder)}))
    finally:
        w.db.close()
        if w.history.directory:w.history.directory.cleanup()


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: probe_temporal_v2_experiment_notebooks.py NEW_OUTPUT_FOLDER')
    path=Path(sys.argv[1]).absolute();path.mkdir(exist_ok=False);check(path)
