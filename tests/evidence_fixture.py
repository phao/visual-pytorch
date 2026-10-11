"""Exact real-run fixtures; imports remain pure for the Qt GUI host."""
import copy
import threading
from app.core import default_model
from app.experiment_objects import execute_candidate, preview_experiment
from app.experiment_ui import result_report
from app.task_execution import training_protocol
from test_authoring import measurements
from test_task_authoring import multi_fixture


def experiment(workspace, model, dataset, task, sid, *, updates=3, history_limit=2):
    model['id']=sid+'-model';dataset['id']=sid+'-data';task['id']=sid+'-task'
    workspace.put(model)
    doc={'version':1,'id':sid,'name':sid,'question':'Inspect exact retained evidence, not fresh execution',
         'models':[workspace.freeze_model(model['id'])],
         'data_tasks':[{'id':'measured','dataset':workspace.put_spec('dataset',dataset),'task':workspace.put_spec('task',task)}],
         'splits':[{'id':'split','protocol':{'method':'row','seed':3,'fractions':[.5,.25,.25]}}],
         'initialization_seeds':[7,13],
         'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=updates,batch_size=2,evaluate_every=1,history_limit=history_limit)}],
         'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
    return workspace.put_spec('experiment',doc)


def populate(workspace, python):
    workspace.set_constrained_mode(False)
    m,d,t=multi_fixture();m['name']="<img src=x onerror='window.evidenceInjected=true'>"
    next(f for f in d['fields'] if f['id']=='y')['unit']="<svg onload='window.evidenceInjected=true'>"
    ref=experiment(workspace,m,d,t,'weighted')
    def coords(eref):
        e=workspace.get_spec('experiment',eref['id'],eref['revision'])
        return preview_experiment(e,workspace.get_spec)['candidates'][0]['coordinates']
    first=execute_candidate(workspace,ref,coords(ref),'weighted-run','Weighted saved trial',python=python,timeout=20)
    assert first['record']['attempts'][-1]['status']=='completed',first
    retry=execute_candidate(workspace,ref,coords(ref),'weighted-run','Weighted saved trial',python='/missing/evidence-python',retry=True)
    # Actual cancellation with a real worker and a finite upper budget.
    cm,cd,ct=multi_fixture();cref=experiment(workspace,cm,cd,ct,'cancel',updates=5000);cancel=threading.Event()
    cancelled=execute_candidate(workspace,cref,coords(cref),'cancelled-run','Cancelled saved trial',python=python,
        progress=lambda p:cancel.set() if p['update']>=1 else None,cancel=cancel,timeout=20)
    assert cancelled['record']['attempts'][-1]['status']=='cancelled',cancelled
    zm,zd,zt=multi_fixture()
    for row in zd['records']:row['values']['valid']=False
    zref=experiment(workspace,zm,zd,zt,'zero-mask')
    zero=execute_candidate(workspace,zref,coords(zref),'zero-run','No effective supervision',python=python,timeout=20)
    assert zero['record']['attempts'][-1]['status']=='unavailable',zero
    excluded=workspace.get_spec('experiment',ref['id'],ref['revision']);excluded.pop('revision');excluded['id']='excluded'
    excluded['selection']['exclusions']=[{'model':excluded['models'][0],'data_task':'measured','reason':'Explicit negative control'}]
    xref=workspace.put_spec('experiment',excluded)
    unavailable=execute_candidate(workspace,xref,coords(xref),'excluded-run','Deliberately excluded')
    failed=[execute_candidate(workspace,ref,coords(ref),'failed-'+str(i),'Worker missing '+str(i),python='/missing/evidence-python') for i in range(3)]
    selected=[first['reference'],cancelled['reference'],zero['reference'],unavailable['reference']]+[r['reference'] for r in failed]
    report=result_report(workspace,selected,sid='saved-comparison',name='Saved comparison',question='What is retained, including unsuccessful trials?')
    # Nine explicit output heads produce 27 final metric rows for paging tests.
    pm=default_model('Nine explicit heads',inputs=1)
    pm['graph']['nodes']=[n for n in pm['graph']['nodes'] if n['type']!='Output']+[{'id':'out'+str(i),'type':'Output'} for i in range(9)]
    pm['graph']['edges']=[edge for edge in pm['graph']['edges'] if edge['to']!='output']+[{'from':'linear1','to':'out'+str(i)} for i in range(9)]
    pt={'version':1,'id':'many','name':'Nine heads','inputs':[{'port':'input','field':'x','transforms':[]}],
        'heads':[{'id':'head'+str(i),'port':'out'+str(i),'field':'y','loss':'mse','weight':1.,'transforms':[]} for i in range(9)]}
    pref=experiment(workspace,pm,measurements(),pt,'many-heads')
    many=execute_candidate(workspace,pref,coords(pref),'many-run','Many metric rows',python=python,timeout=20)
    assert many['record']['attempts'][-1]['status']=='completed',many
    return {'first':first,'retry':retry,'cancelled':cancelled,'zero':zero,'excluded':unavailable,
            'failed':failed,'selected':selected,'report':report,'many':many}
