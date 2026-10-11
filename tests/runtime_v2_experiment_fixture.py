"""Pure retained v2 experiment input; no model/data/task inference or fitting."""
import copy

from app.core import Workspace
from app.experiment_objects import object_reference
from runtime_v2_task_fixture import bundle


def experiment(*, times=(3,7), version=6, execution='runtime-temporal-v2', mode='rectangular',
               updates=3, unavailable=False, axes=None):
    w=Workspace();w.set_constrained_mode(False);pairs=[]
    try:
        for steps in times:
            b=bundle(steps=steps,updates=updates,mode=mode,unavailable=unavailable)
            f=w.put_spec('family',b['runtime']['family'])
            refs={kind:w.put_spec(kind,b[kind]) for kind in ('dataset','task')}
            pairs.append({'id':'T'+str(steps),'dataset':refs['dataset'],'task':refs['task']})
        bindings=copy.deepcopy(b['runtime']['specialization']['bindings'])
        for axis in axes or []:bindings.pop(axis['symbol'],None)
        choice={'id':'runtime-lstm','family':f,'bindings':bindings,'axes':copy.deepcopy(axes or [])}
        if execution is not None:choice['execution']=execution
        doc={'version':version,'id':'runtime-lstm-experiment','name':'Explicit runtime-T LSTM trial',
             'question':'Does the same retained LSTM constructor train at explicit dataset times?',
             'models':[choice],'data_tasks':pairs,'initialization_seeds':[7],
             'splits':[{'id':'split','protocol':b['manifest']['protocol']}],
             'protocols':[{'id':'adam','protocol':b['protocol']}],
             'selection':{'kind':'crossed','pairs':[],'exclusions':[]}}
        ref=w.put_spec('experiment',doc)
        return w,w.get_spec('experiment',ref['id'],ref['revision']),ref
    except Exception:
        w.db.close()
        if w.history.directory:w.history.directory.cleanup()
        raise
