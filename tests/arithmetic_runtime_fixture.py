"""Bounded explicit Family9 runtime arithmetic; no inference or tensor imports."""
from app.core import Workspace
from app.family_records import retain
from app.runtime_tasks import proof
from app.datasets import make_split
from app.task_execution import task_bundle,training_protocol


def repeated_family(mode='independent',learned=False):
    def inp(k):return {'id':k,'type':'Input','shape':[3,2],'axes':['T','F']}
    def edge(a,b,p=None):return {'from':a,'to':b,**({'from_port':p} if p else {})}
    child={'id':'child','name':'Triple','graph':{'nodes':[inp('x'),
        {'id':'r','type':'Replicator','outputs':['a','b']},
        {'id':'s','type':'Scale','coefficient':2.,'op_version':1},
        {'id':'add','type':'Add','sources':['s','r@b'],'op_version':1},
        {'id':'out','type':'Output'}],
        'edges':[edge('x','r'),edge('r','s','a'),edge('s','add'),edge('r','add','b'),edge('add','out')]}}
    nodes=[inp('x'),inp('other'),{'id':'m','type':'Module','ref':'child'},
           {'id':'sum','type':'Add','sources':['m','other'],'op_version':1}]
    edges=[edge('x','m'),edge('m','sum'),edge('other','sum')]
    if learned:
        nodes.append({'id':'head','type':'Linear','features':2,'bias':False});edges.append(edge('sum','head'))
    nodes.append({'id':'y','type':'Output'});edges.append(edge('head' if learned else 'sum','y'))
    parent={'id':'parent','name':'Repeated arithmetic','graph':{'nodes':nodes,'edges':edges,'input_order':['x','other']}}
    return {'version':9,'id':'repeat-arithmetic','name':'Repeated arithmetic','root':'parent',
        'models':{'parent':parent,'child':child},'symbols':[],'assignments':[],'constructors':[],
        'repeats':[{'id':'repeat','model':'parent','nodes':['m'],'count':3,'mode':mode}],
        'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':{'x':[0],'other':[0]}}}


def retained_records(f):
    w=Workspace()
    try:
        saved=retain(w,family=f,materialize=False)
        def get(ref):return w.get_spec(ref['kind'],ref['id'],ref['revision'])
        ff=get(saved['family']);sp=get(saved['specialization']);fm=get(saved['model'])
        runtime=proof(ff,sp,fm,execution='runtime-temporal-v2')
        assert (ff['version'],sp['version'],fm['version'])==(9,8,5)
        return saved,fm,runtime
    finally:w.db.close()


def make_bundle(fm,runtime,steps):
    rows=[]
    for i in range(8):
        x=[[(i+t)/8.,(i-t)/8.] for t in range(steps)];other=[[.25,-.5] for _ in range(steps)]
        y=[[27*a+c,27*b+d] for (a,b),(c,d) in zip(x,other)]
        rows.append({'id':'r'+str(i),'values':{'x':x,'other':other,'y':y}})
    dataset={'version':1,'id':'data','name':'Arithmetic data','storage':{'kind':'inline-records','version':1},
        'fields':[{'id':k,'dtype':'float32','shape':[steps,2],'axes':['T','F']} for k in ('x','other','y')],'records':rows}
    task={'version':1,'id':'task','name':'Arithmetic MSE','inputs':[{'port':k,'field':k,'transforms':[]} for k in ('x','other')],
        'heads':[{'id':'target','port':'y','field':'y','loss':'mse','weight':1.,'transforms':[]}]}
    manifest=make_split(dataset,{'method':'row','seed':13,'fractions':[.5,.25,.25]})
    return task_bundle(fm,dataset,task,manifest,training_protocol(epochs=None,updates=2,batch_size=2),runtime=runtime)
