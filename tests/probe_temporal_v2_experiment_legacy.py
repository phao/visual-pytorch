"""Exact old1–5 experiment selections against pre-v2 retained-route modules.

Pinned modules reuse current unchanged dependencies; this is contract/identity
parity, not a complete historical deployment or authenticity proof.
"""
import copy
import json
import subprocess
import types
from pathlib import Path

from app import experiment_objects, family_axes
from app.task_execution import run_identity
from test_family_axes import family_experiment
from test_runtime_tasks import runtime_experiment
from test_temporal_tasks import temporal_experiment

ROOT=Path(__file__).resolve().parents[1]
PIN='6d8d7a9db934a13642ac00790c25eb30cba6e91d'


def pinned(name):
    raw=subprocess.check_output(['git','show',PIN+':app/'+name+'.py'],cwd=ROOT,text=True)
    m=types.ModuleType('app._pre_experiment_v2_'+name);m.__package__='app'
    exec(compile(raw,PIN+':app/'+name+'.py','exec'),m.__dict__)
    return m


def main():
    old=pinned('experiment_objects');axes=pinned('family_axes');points=bundles=0
    cases=[]
    w,family=family_experiment();cases += [(w,w.get_spec('experiment','experiment')),(w,family)]
    for version in (3,4):
        source=copy.deepcopy(family);source.pop('revision');source['version']=version;source['id']='legacy-'+str(version)
        r=w.put_spec('experiment',source);cases.append((w,w.get_spec('experiment',r['id'],r['revision'])))
    spatial,e,_=runtime_experiment();cases.append((spatial,e))
    temporal,e,_=temporal_experiment();cases.append((temporal,e))
    try:
        for workspace,e in cases:
            assert old.selection_count(e)==experiment_objects.selection_count(e)
            previous=old.preview_experiment(e,workspace.get_spec);current=experiment_objects.preview_experiment(e,workspace.get_spec)
            assert previous==current
            for i,r in enumerate(current['candidates']):
                coordinate=experiment_objects.selection_at(e,i)
                assert coordinate==old.selection_at(e,i)
                if e['version']>=2:
                    assert axes.resolve_model(e['models'],coordinate['model'],workspace.get_spec,version=e['version'])==family_axes.resolve_model(e['models'],coordinate['model'],workspace.get_spec,version=e['version'])
                points+=1
                if r['status']=='compatible':
                    a=old.candidate_bundle(e,coordinate,workspace.get_spec);b=experiment_objects.candidate_bundle(e,coordinate,workspace.get_spec)
                    assert a==b and run_identity(a)==run_identity(b);bundles+=1
        import sys
        assert not any(n in sys.modules for n in ('torch','numpy','PyQt6'))
        print(json.dumps({'ok':True,'pin':PIN,'grammars':[1,2,3,4,5],
                          'preview_selection_points':points,'exact_bundle_identity_points':bundles,'heavy_imports':False}))
    finally:
        for workspace in (w,spatial,temporal):
            workspace.db.close()
            if workspace.history.directory:workspace.history.directory.cleanup()


if __name__=='__main__':main()
