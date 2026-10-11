"""Create a retained CNN→recurrent experiment; stdlib only, no training.

Ragged monochrome frames are explicitly padded, lengths are explicitly derived,
step masks/reset differ from objective masks, and split policies are explicit.
Run: python examples/cnn_recurrent_experiment.py NEW.vtw [--split temporal|group]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

if __package__ is None:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from app.core import Workspace, default_model
from app.family_records import retain
from app.task_execution import training_protocol

TEACHER={'conv_weight':[.6,-.4],'conv_bias':[.03,-.02],
         'mix_weight':[[.4,-.2,.25,.1],[-.1,.5,-.2,.3]],'mix_bias':[.1,-.1]}


def family():
    cell=default_model('CNN recurrent cell');cell['id']='cnn-cell'
    cell['graph']={'input_order':['image','previous'],'nodes':[
        {'id':'image','type':'Input','label':'Image frame','shape':[1,4,5],'axes':['C','H','W'],'x':40,'y':60},
        {'id':'previous','type':'Input','label':'Previous hidden','shape':[2],'x':40,'y':280},
        {'id':'conv','type':'Conv2d','label':'Learned frame features','out_channels':2,'kernel_size':1,'bias':True,'x':260,'y':60},
        {'id':'pool','type':'AdaptiveAvgPool2d','output_size':[1,1],'x':460,'y':60},
        {'id':'flat','type':'Flatten','start_dim':0,'end_dim':2,'x':650,'y':60},
        {'id':'join','type':'Concat','sources':['flat','previous'],'flatten':True,'x':850,'y':140},
        {'id':'mix','type':'Linear','label':'Frame + feedback','features':2,'bias':True,'x':1050,'y':140},
        {'id':'activate','type':'Activation','activation':'Tanh','x':1240,'y':140},
        {'id':'next','type':'Output','label':'Next hidden','x':1440,'y':140}],
        'edges':[{'from':a,'to':b} for a,b in [('image','conv'),('conv','pool'),('pool','flat'),
             ('flat','join'),('previous','join'),('join','mix'),('mix','activate'),('activate','next')]]}
    root=default_model('CNN to explicit recurrent state');root['id']='cnn-sequence'
    controls={'lengths':{'source':'lengths','adapter':'exact-int64-v1'},
              'mask':{'source':'step_mask','adapter':'exact-bool-v1'},
              'reset':{'source':'step_reset','adapter':'exact-bool-v1'}}
    root['graph']={'input_order':['frames','initial','lengths','step_mask','step_reset'],'nodes':[
        {'id':'frames','type':'Input','label':'Ragged image sequence','shape':[5,1,4,5],'axes':['T','C','H','W'],'x':40,'y':40},
        {'id':'initial','type':'Input','label':'Initial hidden','shape':[2],'x':40,'y':160},
        {'id':'lengths','type':'Input','label':'Valid lengths','shape':[1],'x':40,'y':280},
        *[{'id':role,'type':'Input','label':role,'shape':[5],'axes':['T'],'x':40,'y':y}
           for role,y in [('step_mask',400),('step_reset',520)]],
        {'id':'scan','type':'Recurrent','label':'CNN → recurrent cell','ref':cell['id'],
         'sequence_inputs':[{'input':'image','source':'frames'}],
         'states':[{'input':'previous','output':'next','port':'hidden','source':'initial'}],
         'sequence_outputs':[{'output':'next','port':'trajectory'}],'controls':controls,'x':400,'y':160,'height':210},
        {'id':'prediction','type':'Output','label':'Predicted trajectory','x':760,'y':120},
        {'id':'final','type':'Output','label':'Returned final hidden','x':760,'y':310}],
        'edges':[{'from':source,'to':'scan'} for source in ('frames','initial','lengths','step_mask','step_reset')]+
                [{'from':'scan','from_port':port,'to':target} for port,target in [('trajectory','prediction'),('hidden','final')]]}
    models={root['id']:root,cell['id']:cell};strip=lambda m:{k:copy.deepcopy(m[k]) for k in ('id','name','graph')}
    return {'version':8,'id':'cnn-recurrent-family','name':'Scoped CNN/recurrent family','root':root['id'],
        'models':{k:strip(v) for k,v in models.items()},
        'symbols':[{'id':'time','scope':root['id'],'name':'Constructor baseline T','default':5,'minimum':1,'maximum':32},
                   {'id':'features','scope':root['id'],'name':'CNN channels','default':2,'minimum':1,'maximum':8},
                   {'id':'activation','scope':root['id'],'name':'Cell activation','type':'enum','values':['ReLU','Tanh'],'default':'Tanh'},
                   {'id':'cell-features','scope':cell['id'],'name':'CNN channels','minimum':1,'maximum':8},
                   {'id':'cell-activation','scope':cell['id'],'name':'Cell activation','type':'enum','values':['ReLU','Tanh']}],
        'assignments':[{'model':root['id'],'node':nid,'field':'shape','value':[{'symbol':'time'},*tail]}
                       for nid,tail in [('frames',[1,4,5]),('step_mask',[]),('step_reset',[])]]+
                      [{'model':cell['id'],'node':'conv','field':'out_channels','value':{'symbol':'cell-features'}},
                       {'model':cell['id'],'node':'activate','field':'activation','value':{'symbol':'cell-activation'}}],
        'constructors':[{'model':root['id'],'node':'scan','symbol':child,'value':{'symbol':parent}}
                        for child,parent in [('cell-features','features'),('cell-activation','activation')]],
        'repeats':[],'runtime_policy':{'version':2,'kind':'runtime-temporal-v2',
                                     'inputs':{'frames':[0],'step_mask':[0],'step_reset':[0]}}}


def observations(*, steps=7):
    if type(steps) is not int or not 1<=steps<=32:raise ValueError('Example padded T must be1–32; this is a bounded generator, not an architecture limit.')
    fields=[{'id':'frames','dtype':'float32','shape':[-1,1,4,5],'axes':['T','C','H','W']},
            {'id':'initial','dtype':'float32','shape':[2]},
            *[{'id':k,'dtype':'float32','shape':[steps],'axes':['T']} for k in ('step_mask','step_reset')],
            {'id':'target','dtype':'float32','shape':[steps,2],'axes':['T','D0']},
            {'id':'valid','dtype':'bool','shape':[steps,2]},
            {'id':'target_final','dtype':'float32','shape':[2]}, {'id':'valid_final','dtype':'bool','shape':[]},
            {'id':'subject','dtype':'string','shape':[]}, {'id':'time','dtype':'int64','shape':[]}]
    rows=[]
    for subject in range(4):
        for clock in range(8):
            for replicate in range(2):
                length=(7*subject+3*clock+replicate)%(steps+1)
                initial=[.03*(subject+1),-.02*(replicate+1)];state=list(initial)
                images=[[[[.4*math.sin((subject+1)*(clock+1)*.17+t*.19+i*.11+j*.07+replicate*.13)
                           for j in range(5)] for i in range(4)]] for t in range(length)]
                mask=[float(not(clock%3==0 and t==1)) for t in range(steps)]
                reset=[float(t==2 and subject%2==0) for t in range(steps)];targets=[];valid=[]
                for t in range(steps):
                    active=t<length and bool(mask[t])
                    if active:
                        if reset[t]:state=list(initial)
                        mean=math.fsum(v for row in images[t][0] for v in row)/20
                        features=[mean*a+b for a,b in zip(TEACHER['conv_weight'],TEACHER['conv_bias'])]
                        before=[*features,*state]
                        state=[math.tanh(math.fsum(a*b for a,b in zip(weight,before))+bias)
                               for weight,bias in zip(TEACHER['mix_weight'],TEACHER['mix_bias'])]
                    targets.append(list(state) if active else [0.,0.]);valid.append([active,active])
                rows.append({'id':f's{subject}-t{clock}-r{replicate}', 'values':{
                    'frames':images,'initial':initial,'step_mask':mask,'step_reset':reset,'target':targets,'valid':valid,
                    'target_final':state,'valid_final':any(v[0] for v in valid),'subject':f's{subject}','time':clock*10}})
    return {'version':1,'id':'cnn-sequences','name':'Ragged grouped image sequences','fields':fields,'records':rows,
            'storage':{'version':1,'kind':'inline-records'},'provenance':{
                'procedure':'independent stdlib pixel-mean 1×1 CNN plus tanh feedback teacher; no Torch/GUI/fitting',
                'teacher':copy.deepcopy(TEACHER),'time_semantics':'observation block timestamp; ties across subjects/replicates',
                'state_semantics':'supplied initial state per independent observation; reset only on active steps'}}


def task(*,steps=7):
    return {'version':1,'id':'cnn-sequence-task','name':'Explicit trajectory and final-state objectives',
        'inputs':[{'port':'frames','field':'frames','transforms':[{'kind':'pad','length':steps,'value':0.}]},
                  {'port':'lengths','field':'frames','transforms':[],'derived':'length','length':steps},
                  *[{'port':k,'field':k,'transforms':[]} for k in ('initial','step_mask','step_reset')]],
        'heads':[{'id':'trajectory','port':'prediction','field':'target','loss':'mse','weight':1.,'transforms':[],'mask':'valid'},
                 {'id':'final','port':'final','field':'target_final','loss':'mse','weight':.2,'transforms':[],'mask':'valid_final'}]}


def populate(workspace, *,steps=7,updates=60,split='temporal',materialize=True):
    if split not in ('group','temporal'):raise ValueError('Choose explicit existing group or temporal policy.')
    retained=retain(workspace,family=family(),materialize=materialize)
    refs={'dataset':workspace.put_spec('dataset',observations(steps=steps)),
          'task':workspace.put_spec('task',task(steps=steps))}
    protocol={'method':'temporal','field':'time','fractions':[.5,.25,.25]} if split=='temporal' else {
              'method':'group','field':'subject','seed':5,'fractions':[.5,.25,.25]}
    doc={'version':6,'id':'cnn-recurrent-experiment','name':'CNN/recurrent '+split+' trial',
         'question':'Does explicit padded CNN→recurrent training preserve state and split meaning?',
         'models':[{'id':'cnn-cell','family':retained['family'],'bindings':{},'axes':[], 'execution':'runtime-temporal-v2'}],
         'data_tasks':[{'id':'images','dataset':refs['dataset'],'task':refs['task']}],
         'splits':[{'id':split,'protocol':protocol}],'initialization_seeds':[7],
         'protocols':[{'id':'adam','protocol':training_protocol(epochs=None,updates=updates,batch_size=16,lr=.03,
                     shuffle=True,evaluate_every=10,history_limit=30)}],
         'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
    ref=workspace.put_spec('experiment',doc)
    return {'experiment':ref,'family':retained['family'],'specialization':retained['specialization'],
            'parameters':retained['resolved']['contract']['parameters'],'baseline_T':5,'padded_T':steps,'split':protocol}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('workspace',type=Path)
    parser.add_argument('--split',choices=('group','temporal'),default='temporal')
    parser.add_argument('--time',type=int,default=7);parser.add_argument('--updates',type=int,default=60)
    args=parser.parse_args()
    if args.workspace.exists():parser.error('Refusing to overwrite an existing workspace.')
    w=Workspace()
    try:
        info=populate(w,steps=args.time,updates=args.updates,split=args.split);w.save(args.workspace)
        print(json.dumps({'workspace':str(args.workspace),**info,'status':'retained; not-trained; explicit masks/reset/split'}))
    finally:
        w.db.close()
        if w.history.directory:w.history.directory.cleanup()


if __name__=='__main__':main()
