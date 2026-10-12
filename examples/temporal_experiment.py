"""Create a NEW portable temporal workspace; fitting is explicit via --train.

python -m examples.temporal_experiment /path/to/new-temporal.vtw
python -m examples.temporal_experiment /path/to/new-temporal.vtw --train
Creation needs only the standard library; --train needs a PyTorch environment.
"""
import argparse
import math
import sys
from pathlib import Path

from app.core import Workspace
from app.experiment_objects import execute_candidate,selection_at
from app.task_execution import training_protocol


def architectures():
    body={'id':'cell','name':'Tanh transition','graph':{'nodes':[
        {'id':'x','type':'Input','features':2,'label':'frame'},
        {'id':'h','type':'Input','features':2,'label':'previous state'},
        {'id':'join','type':'Concat','sources':['x','h'],'flatten':True},
        {'id':'mix','type':'Linear','features':2,'bias':True},
        {'id':'tanh','type':'Activation','activation':'Tanh'},
        {'id':'next','type':'Output','label':'next state'}],
        'edges':[{'from':a,'to':b} for a,b in [('x','join'),('h','join'),('join','mix'),('mix','tanh'),('tanh','next')]],
        'input_order':['x','h']}}
    root={'id':'sequence','name':'Sequence predictor','graph':{'nodes':[
        {'id':'frames','type':'Input','shape':[4,2],'axes':['T','D0'],'label':'frames'},
        {'id':'initial','type':'Input','shape':[2],'label':'initial'},
        {'id':'scan','type':'Recurrent','ref':'cell','sequence_inputs':[{'input':'x','source':'frames'}],
         'states':[{'input':'h','output':'next','port':'state','source':'initial'}],
         'sequence_outputs':[{'output':'next','port':'trajectory'}]},
        {'id':'trajectory','type':'Output','label':'trajectory'},
        {'id':'final','type':'Output','label':'final'}],
        'edges':[{'from':'frames','to':'scan'},{'from':'initial','to':'scan'},
                 {'from':'scan','from_port':'trajectory','to':'trajectory'},
                 {'from':'scan','from_port':'state','to':'final'}],'input_order':['frames','initial']}}
    return {'cell':body,'sequence':root}


def build(workspace,updates):
    models=architectures()
    for model in models.values():workspace.put(model)
    family=workspace.put_spec('family',{'version':6,'id':'temporal-family','name':'One cell, explicit runtime T',
        'root':'sequence','models':models,'symbols':[],'assignments':[],'repeats':[],
        'runtime_policy':{'version':1,'kind':'runtime-temporal-v1','inputs':{'frames':[0]}}})
    pairs=[]
    for steps in (3,7):
        rows=[]
        for i in range(12):
            frames=[[math.sin((i+1)*(t+1)*.13),math.cos((i+2)*(t+1)*.11)] for t in range(steps)]
            initial=[i*.01,-i*.008];h,k=initial;targets=[]
            for a,b in frames:
                h,k=math.tanh(.4*a-.2*b+.25*h+.1*k+.1),math.tanh(-.1*a+.5*b-.2*h+.3*k-.1)
                targets.append([h,k])
            rows.append({'id':'r'+str(i),'values':{'frames':frames,'initial':initial,'target':targets,'target_final':[h,k]}})
        dataset=workspace.put_spec('dataset',{'version':1,'id':'T'+str(steps),'name':'Independent teacher T='+str(steps),
            'storage':{'version':1,'kind':'inline-records'},'fields':[
                {'id':'frames','dtype':'float32','shape':[steps,2],'axes':['T','D0']},
                {'id':'initial','dtype':'float32','shape':[2]},
                {'id':'target','dtype':'float32','shape':[steps,2],'axes':['T','D0']},
                {'id':'target_final','dtype':'float32','shape':[2]}],
            'records':rows,'provenance':{'procedure':'independent explicit Python tanh transition'}})
        task=workspace.put_spec('task',{'version':1,'id':'task-T'+str(steps),'name':'Explicit trajectory and final-state heads',
            'inputs':[{'port':k,'field':k,'transforms':[]} for k in ('frames','initial')],
            'heads':[{'id':'trajectory','port':'trajectory','field':'target','loss':'mse','weight':1.,'transforms':[]},
                     {'id':'final','port':'final','field':'target_final','loss':'mse','weight':.2,'transforms':[]}]})
        pairs.append({'id':'T'+str(steps),'dataset':dataset,'task':task})
    return workspace.put_spec('experiment',{'version':5,'id':'temporal-experiment','name':'Two explicit time cases',
        'question':'Can the same transition architecture learn both T=3 and T=7?',
        'models':[{'id':'cell','family':family,'bindings':{},'axes':[],'execution':'runtime-temporal-v1'}],
        'data_tasks':pairs,'initialization_seeds':[7],
        'splits':[{'id':'split','protocol':{'method':'external','roles':{'train':['r'+str(i) for i in range(8)],
                    'validation':['r8','r9'],'test':['r10','r11']},'excluded':[]}}],
        'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=updates,batch_size=8,lr=.04,
                    evaluate_every=4,history_limit=50)}],'selection':{'kind':'crossed','pairs':[],'exclusions':[]}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path);parser.add_argument('--train',action='store_true')
    parser.add_argument('--updates',type=int,default=20);args=parser.parse_args()
    if args.path.exists() or args.path.is_symlink():parser.error('Destination must be a new file; nothing overwritten.')
    workspace=Workspace()
    try:
        reference=build(workspace,args.updates)
        if args.train:
            experiment=workspace.get_spec('experiment',reference['id'],reference['revision'])
            for i in range(2):
                result=execute_candidate(workspace,reference,selection_at(experiment,i),'run-'+str(i),'Explicit T case '+str(i),
                                         python=sys.executable,timeout=60)
                attempt=result['record']['attempts'][-1]
                print('Run',i,attempt['status'],attempt['error'] or '')
                if attempt['status']!='completed':raise RuntimeError('The bounded demonstration did not complete.')
        workspace.save(args.path)
        print('Created',args.path,'— open Experiment → Experiments & history and select Two explicit time cases.')
    finally:
        workspace.db.close()
        if workspace.history.directory:workspace.history.directory.cleanup()


if __name__=='__main__':main()
