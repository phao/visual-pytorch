"""Actual Qt QThread/signals -> isolated v2 worker, no Torch in GUI process.

This checks the direct task adapter, not the still-unadmitted ExperimentSpec6
editor/retention/notebook routes. Use private stdlib fixtures and no user state.
"""
import copy
import json
import sys

from PyQt6.QtCore import QCoreApplication, QTimer
from app.desktop import TaskTrainingJob, find_torch_python
from app.task_result import validate_result
from runtime_v2_task_fixture import bundle

app=QCoreApplication([])
python=find_torch_python()
assert python is not None,'No numerical Python found.'
results=[]
for mode in ('rectangular','pad','standardize'):
    b=bundle(steps=7,mode=mode);before=copy.deepcopy(b);output={'points':[]}
    job=TaskTrainingJob(b,app,python,constrained=False)
    job.point.connect(lambda point:output['points'].append(json.loads(point)))
    job.completed.connect(lambda result:output.update(result=json.loads(result)))
    job.failed.connect(lambda error:output.update(error=error))
    job.finished.connect(app.quit)
    timer=QTimer();timer.setSingleShot(True);timer.timeout.connect(job.stop);timer.start(30000)
    job.start();app.exec();job.wait();timer.stop()
    assert 'result' in output,output
    r=output['result'];assert r['status']=='completed',r['failure']
    assert r['counts']['completed_updates']==3
    assert r['identity']['version']==4 and r['identity']['compiler']=='native-runtime-temporal-v2'
    assert output['points']==r['history'];assert b==before
    validate_result(b,{k:v for k,v in r.items() if k!='bundle'})
    assert not any(n in sys.modules for n in ('torch','numpy'))
    results.append({'mode':mode,'status':r['status'],'identity':r['identity']['compiler'],
                    'updates':r['counts']['completed_updates'],'points':len(output['points']),
                    'lstm_unroll_bytes':r['memory_admission']['components']['lstm_unrolls']})
    job.deleteLater()

bad=copy.deepcopy(b);bad['version']=True;output={}
job=TaskTrainingJob(bad,app,python,constrained=False)
job.completed.connect(lambda result:output.update(result=json.loads(result)))
job.failed.connect(lambda error:output.update(error=error))
job.finished.connect(app.quit);job.start();app.exec();job.wait()
assert 'error' in output and 'result' not in output,output
assert job.process is None and 'torch' not in sys.modules
job.deleteLater()
print(json.dumps({'engine':'native PyQt6 QThread task adapter','host_torch':False,'cases':results}))
