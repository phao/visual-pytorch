"""Private real notebook bundles for native tests, produced by numerical Python."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from app.experiment_objects import object_reference, preview_experiment, execute_candidate
from app.notebooks import build_notebook, create_bundle, execute_bundle
from app.notebook_execution import inspect_bundle
from test_experiment_objects import experiment_workspace


def populate(folder, *, receipt=False):
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=True)
    w,e = experiment_workspace()
    try:
        selected = preview_experiment(e,w.get_spec)['candidates'][0]
        run = execute_candidate(w,object_reference('experiment',e),selected['coordinates'],'saved','Saved trial',timeout=30)
        assert run['record']['attempts'][0]['status']=='completed'
        reports={}
        for mode in ('result-report','rerun'):
            reports[mode]=w.put_spec('report',{'version':1,'id':mode,'name':'Notebook <svg> '+mode,
                'question':'What does exact evidence show?', 'mode':mode,'runs':[run['reference']] if mode=='result-report' else [],
                'experiment':object_reference('experiment',e) if mode=='rerun' else None,
                'sections':['question','training','evaluation','failures','environment','checkpoints'],
                'analysis':{'kind':'descriptive','inferential_unit':'run'}})
        w.save(folder/'workspace.vtw')
        selections={}
        for name in ('saved','rerun','failure','slow'):
            ref=reports['rerun' if name=='rerun' else 'result-report']
            def custom(payload,index):
                nb=build_notebook(payload,index)
                if name in ('failure','slow'):
                    c=next(c for c in nb.cells if c.metadata.visual_pytorch.section=='question')
                    if name=='failure':c.source="raise RuntimeError('private GUI notebook failure <img>')"
                    else:c.source="import os,time\nfrom pathlib import Path\nPath("+repr(str(folder/'kernel.pid'))+").write_text(str(os.getpid()))\nprint('before-stop',flush=True)\ntime.sleep(60)"
                return nb
            with patch('app.notebooks.build_notebook',side_effect=custom):
                create_bundle(w,ref,folder/name)
            selections[name]=inspect_bundle(folder/name)
        output={'selections':selections,'reports':reports,'run':run['reference']}
        if receipt:
            import threading
            cancel=threading.Event();cancel.set()
            output['terminal']=execute_bundle(folder/'saved',cancel=cancel)
        (folder/'fixture.json').write_text(json.dumps(output),encoding='utf-8')
    finally:w.db.close()


if __name__=='__main__':populate(sys.argv[1],receipt='--receipt' in sys.argv[2:])
