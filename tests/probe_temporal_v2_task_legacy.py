"""Compare old TaskBundle3 meaning with the committed pre-v2 task modules.

Pinned modules reuse unchanged dependencies; this is exact pure contract parity,
not an assertion that a complete historical environment has been reproduced.
"""
import json
import subprocess
import types
from pathlib import Path

from app import runtime_tasks, task_execution
from app.experiment_objects import candidate_bundle, selection_at
from test_temporal_tasks import temporal_experiment

ROOT=Path(__file__).resolve().parents[1]
PIN='271a82fad10b834393a21b4821914ce4d035705e'


def pinned(name):
    raw=subprocess.check_output(['git','show',PIN+':app/'+name+'.py'],cwd=ROOT,text=True)
    module=types.ModuleType('app._before_v2_task_'+name);module.__package__='app'
    exec(compile(raw,PIN+':app/'+name+'.py','exec'),module.__dict__)
    return module


def main():
    old=pinned('task_execution');policy=pinned('runtime_tasks');points=shapes=0
    for cnn in (False,True):
        w,e,_=temporal_experiment(times=(3,7))
        try:
            if cnn:
                from test_temporal_families import family
                e.pop('revision');e['models'][0]['family']=w.put_spec('family',family(cnn=True,scoped=True))
                for pair in e['data_tasks']:
                    d=w.get_spec('dataset',pair['dataset']['id']);d.pop('revision')
                    t=d['fields'][0]['shape'][0];d['fields'][0].update(shape=[t,1,4,5],axes=['T','C','H','W'])
                    for r in d['records']:
                        r['values']['sequence']=[[[[v[0]+i/50+j/100 for j in range(5)] for i in range(4)]] for v in r['values']['sequence']]
                    pair['dataset']=w.put_spec('dataset',d)
                ref=w.put_spec('experiment',e);e=w.get_spec('experiment',ref['id'],ref['revision'])
            for i in (0,1):
                b=candidate_bundle(e,selection_at(e,i),w.get_spec)
                assert b['version']==3
                assert old.task_bundle(*(b[k] for k in ('model','dataset','task','manifest','protocol')),runtime=b['runtime'])==b
                assert old.run_identity(b)==task_execution.run_identity(b)
                assert old.resolve_run(b,constrained=False,available_bytes=10**9)==task_execution.resolve_run(b,constrained=False,available_bytes=10**9)
                assert policy.input_policy(b)==runtime_tasks.input_policy(b)
                for steps in (1,5,11):
                    row={k:v for k,v in b['dataset']['records'][1]['values'].items()}
                    row.update(sequence=([[[[.1]*5 for _ in range(4)]]] if cnn else [[.1,.2]])*steps,
                               mask=[1.]*steps,reset=[0.]*steps,lengths=[float(steps)])
                    spec={'dataset':b['dataset'],'task':b['task'],'runtime_policy':runtime_tasks.input_policy(b)}
                    assert policy.record_shapes(spec,[row])==runtime_tasks.record_shapes(spec,[row]);shapes+=1
                points+=1
        finally:w.db.close()
    print(json.dumps({'ok':True,'pin':PIN,'exact_bundle_identity_preparation_memory_cases':points,'actual_shape_cases':shapes}))


if __name__=='__main__':main()
