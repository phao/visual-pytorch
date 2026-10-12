"""Pure TaskBundle1/2 identities and spatial policy parity against pre-temporal code."""
import copy
import json
import subprocess
import types
from pathlib import Path

from app import runtime_tasks, task_execution
from app.experiment_objects import candidate_bundle,selection_at
from test_runtime_tasks import runtime_experiment

ROOT=Path(__file__).resolve().parents[1]
PIN='3079bfddce9a9df334318c48b79c9de0d28de4a0'


def pinned(name):
    raw=subprocess.check_output(['git','show',PIN+':app/'+name+'.py'],cwd=ROOT,text=True)
    module=types.ModuleType('app._pinned_'+name);module.__package__='app'
    exec(compile(raw,PIN+':app/'+name+'.py','exec'),module.__dict__)
    return module


def main():
    old=pinned('task_execution');policy=pinned('runtime_tasks');checked=0
    for execution,shape in ((None,(1,2,2)),('runtime-spatial-v1',(1,3,5)),('runtime-spatial-v1',(1,7,2))):
        w,e,_=runtime_experiment(execution=execution,shape=shape)
        try:
            bundle=candidate_bundle(e,selection_at(e,0),w.get_spec)
            reconstructed=old.task_bundle(*(bundle[k] for k in ('model','dataset','task','manifest','protocol')),runtime=bundle.get('runtime'))
            assert reconstructed==bundle
            assert old.run_identity(bundle)==task_execution.run_identity(bundle)
            if bundle['version']==2:
                assert policy.input_policy(bundle)==runtime_tasks.input_policy(bundle)
                spec={'dataset':bundle['dataset'],'task':bundle['task'],'runtime_policy':runtime_tasks.input_policy(bundle)}
                for h,width in ((1,2),(3,5),(9,3)):
                    rows=[{'x':[[[.1]*width for _ in range(h)]]}]
                    assert policy.record_shapes(spec,rows)==runtime_tasks.record_shapes(spec,rows)
            checked+=1
        finally:w.db.close()
    print(json.dumps({'ok':True,'pinned':PIN,'exact_bundle_identity_cases':checked,'spatial_actual_shape_cases':6}))


if __name__=='__main__':main()
