"""Stdlib-only explicit retained v2 tasks; no worker/experiment inference."""
import copy
import math

from app.core import Workspace
from app.datasets import make_split
from app.family_records import retain
from app.runtime_tasks import proof
from app.task_execution import task_bundle, training_protocol
from examples.runtime_lstm_family import definition


def bundle(*, steps=3, baseline=5, updates=3, unavailable=False, mode='rectangular', bindings=None):
    w=Workspace()
    try:
        values={'time':baseline,'width':2,'cell-width':3,'projection':2,'layers':1,
                'directions':True,'bias':True,**(bindings or {})}
        saved=retain(w,family=definition(),bindings=values,materialize=False)
        s=w.get_spec('specialization',saved['specialization']['id'],saved['specialization']['revision'])
        f=w.get_spec('family',saved['family']['id'],saved['family']['revision'])
        model=w.get_spec('model',s['model']['id'],s['model']['revision'])
        runtime=proof(f,s,model,execution='runtime-temporal-v2')
        fixed=saved['resolved']['contract'];groups=copy.deepcopy(fixed['input_groups'])
        next(g for g in groups if g['id']=='sequence')['shape'][0]=steps
        fields=[{'id':g['id'],'dtype':'float32','shape':g['shape'],
                 **({'axes':g['axes']} if g.get('axes') else {})} for g in groups]
        output_shapes={g['id']:copy.deepcopy(g['shape']) for g in fixed['output_groups']}
        output_shapes['out_sequence'][0]=steps
        for key,shape in output_shapes.items():
            output=next(g for g in fixed['output_groups'] if g['id']==key)
            fields.append({'id':'target_'+key,'dtype':'float32','shape':shape,
                           **({'axes':output['axes']} if output.get('axes') else {})})
        fields += [{'id':'valid','dtype':'bool','shape':output_shapes['out_sequence']},
                   {'id':'valid_state','dtype':'bool','shape':[]},
                   {'id':'subject','dtype':'string','shape':[]}, {'id':'time','dtype':'int64','shape':[]}]
        def tensor(shape,fn):
            return [tensor(shape[1:],fn) for _ in range(shape[0])] if shape else fn()
        rows=[]
        for i in range(12):
            length=i%(steps+1)
            row={g['id']:tensor(g['shape'],lambda:.07*math.sin(i+.3)) for g in groups}
            row['sequence']=[[.3*math.sin(i+t+.1*j) for j in range(values['width'])] for t in range(steps)]
            row['lengths']=[float(length)]
            row.update({'target_'+k:tensor(shape,lambda:0.) for k,shape in output_shapes.items()})
            row.update(valid=[[t<length and not unavailable]*output_shapes['out_sequence'][-1] for t in range(steps)],
                       valid_state=not unavailable,subject='s'+str(i//3),time=i)
            rows.append({'id':'r'+str(i),'values':row})
        inputs=[{'port':g['id'],'field':g['id'],'transforms':[]} for g in groups]
        if mode=='pad':
            next(f for f in fields if f['id']=='sequence')['shape'][0]=-1
            for r in rows:r['values']['sequence']=r['values']['sequence'][:int(r['values']['lengths'][0])]
            next(b for b in inputs if b['port']=='sequence')['transforms']=[{'kind':'pad','length':steps,'value':0.}]
            next(b for b in inputs if b['port']=='lengths').update(field='sequence',derived='length',length=steps)
        elif mode=='standardize':
            next(b for b in inputs if b['port']=='sequence')['transforms']=[{'kind':'standardize'}]
        dataset={'version':1,'id':'runtime-lstm-data','name':'Explicit LSTM observations',
                 'storage':{'version':1,'kind':'inline-records'},'fields':fields,'records':rows,
                 'provenance':{'procedure':'deterministic stdlib sequence/state fixture; zero targets'}}
        task={'version':1,'id':'runtime-lstm-task','name':'All three explicit LSTM objectives','inputs':inputs,
              'heads':[{'id':k,'port':k,'field':'target_'+k,'loss':'mse','weight':1. if k=='out_sequence' else .2,
                        'transforms':[],'mask':'valid' if k=='out_sequence' else 'valid_state'} for k in output_shapes]}
        manifest=make_split(dataset,{'method':'external','roles':{'train':['r'+str(i) for i in range(8)],
                            'validation':['r8','r9'],'test':['r10','r11']},'excluded':[]})
        protocol=training_protocol(epochs=None,updates=updates,batch_size=8,lr=.03,shuffle=False,
                                   evaluate_every=1,history_limit=12)
        return task_bundle(model,dataset,task,manifest,protocol,runtime=runtime)
    finally:
        w.db.close()


def rebuild(value, *, dataset=None, task=None, protocol=None, manifest=None):
    """Re-freeze edited definitions; never repair hashes in a forged payload."""
    d=copy.deepcopy(dataset if dataset is not None else value['dataset']);d.pop('revision',None)
    t=copy.deepcopy(task if task is not None else value['task']);t.pop('revision',None)
    split={k:v for k,v in value['manifest']['protocol'].items() if k not in ('algorithm','allocation_unit')}
    m=manifest if manifest is not None else make_split(d,split)
    return task_bundle(value['model'],d,t,m,protocol or value['protocol'],runtime=value['runtime'])
