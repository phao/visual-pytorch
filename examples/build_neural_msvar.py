"""Build an inspectable full neural MS-VAR workspace; execution is opt-in."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core import Workspace,GraphError
from app.neural_msvar import drawing,dataset,likelihood_task
from app.family_records import retain
from app.task_execution import training_protocol
from app.experiment_objects import execute_candidate,selection_at


def build(path, *, run=False):
    path=Path(path)
    if path.exists():raise GraphError('Example builder never overwrites an existing file.')
    workspace=Workspace()
    try:
        model=drawing();workspace.put(model)
        refs=retain(workspace,family={'version':13,'id':'neural-msvar-family','name':'Neural MS-VAR family',
            'root':model['id'],'models':{model['id']:model},'symbols':[],'assignments':[],'repeats':[],
            'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':{'observations':[0]}}},materialize=False)
        data=workspace.put_spec('dataset',dataset());task=workspace.put_spec('task',likelihood_task())
        experiment={'version':7,'id':'neural-msvar-sanity','name':'Bounded unlabelled sequence-density study',
            'question':'Does a causal five-state trivariate neural MS-VAR learn conditional sequence density?',
            'models':[{'id':'msvar','family':refs['family'],'bindings':{},'axes':[],'execution':'runtime-temporal-v2'}],
            'data_tasks':[{'id':'observed','dataset':data,'task':task}],
            'splits':[{'id':'trajectories','protocol':{'method':'group','field':'trajectory','seed':31,'fractions':[.5,.25,.25]}}],
            'initialization_seeds':[7],
            'protocols':[{'id':'bounded-adam','protocol':training_protocol(epochs=None,updates=12,batch_size=4,lr=.01,evaluate_every=3)}],
            'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
        ref=workspace.put_spec('experiment',experiment)
        if run:
            result=execute_candidate(workspace,ref,selection_at(experiment,0),'sanity-fit','Twelve-update MS-VAR sanity fit',python=sys.executable,timeout=60)
            attempt=result['record']['attempts'][0]
            if attempt['status']!='completed':raise GraphError(str(attempt['error']))
        workspace.save(path)
    finally:
        workspace.db.close()
        if workspace.history.directory:workspace.history.directory.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--run',action='store_true',help='Explicitly execute one bounded twelve-update sanity fit')
    args=parser.parse_args();build(args.output,run=args.run)
