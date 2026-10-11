"""Exact old temporal-v1 proof/actual-case/admission comparison, no Torch.

Uses the immutable pre-v2 proof source and unchanged shared graph/recurrent
contracts in the sole checkout. This is not a portable old-runtime deployment.
"""
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

from app.temporal_shapes import specification, shape_contract, admission
from test_temporal_shapes import temporal_fixture, nested_fixture, shapes

ROOT = Path(__file__).resolve().parents[1]
PIN = 'd66a8d6ea9d0a8c914575386ac89fbdfaf26db27'


def pinned(module):
    source = subprocess.check_output(['git','show',PIN+':app/'+module+'.py'],cwd=ROOT,text=True)
    loaded = types.ModuleType('app._temporal_v2_legacy_'+module); loaded.__package__ = 'app'
    exec(compile(source,PIN+':app/'+module+'.py','exec'),loaded.__dict__)
    return loaded


def main():
    old = pinned('runtime_shapes'); temporal = pinned('temporal_shapes')
    temporal.graph_admission = old.admission
    documents = [temporal_fixture(cnn=cnn,controls=controls)
                 for cnn in (False,True) for controls in (False,True)] + [nested_fixture()]
    count = 0
    for models,mid,decl in documents:
        before = old._dependency_specification(models,mid,temporal.declaration(models,mid,decl),temporal=True)
        after = specification(models,mid,decl)
        assert before == after, (mid,'proof')
        for t in (1,4,9):
            actual = shapes(models,mid,t)
            a,b = old.shape_contract(models,mid,before,actual),shape_contract(models,mid,after,actual)
            assert a == b, (mid,t,'actual-shape-plan')
            with patch.object(old,'available_memory',return_value=10**12),\
                 patch('app.runtime_shapes.available_memory',return_value=10**12):
                for training in (False,True):
                    assert temporal.admission(models,mid,a,3,gradients=training) == admission(models,mid,b,3,gradients=training), (mid,t,training,'RAM')
            count += 1
    assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
    print(json.dumps({'ok':True,'pin':PIN,'proofs':len(documents),'actual_shape_cases':count,
                      'admission_cases':2*count,'torch_imported':False}))


if __name__ == '__main__':
    main()
