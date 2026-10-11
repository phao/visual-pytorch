"""Disposable retained family6 -> .vtw15 -> exact native/checked exported model.

Run from the repository root: python examples/temporal_family.py
Only private temporary files are created. This is explicit Python construction,
not yet a variable-T experiment/worker/notebook execution selection.
"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import Workspace, VERSION
from app.family_records import retain, compile_specialization
from app.family_export import stored_source, write_new_family
from examples.temporal_runtime import architectures


def main():
    import torch
    torch.set_num_threads(1);torch.manual_seed(7)
    document={'version':6,'id':'temporal-example','name':'Explicit variable-T family',
              'root':'sequence','models':copy.deepcopy(architectures()),
              'symbols':[],'assignments':[],'repeats':[],
              'runtime_policy':{'version':1,'kind':'runtime-temporal-v1','inputs':{'frames':[0]}}}
    workspace=Workspace();opened=None
    try:
        result=retain(workspace,family=document,materialize=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'example.vtw';workspace.save(path);opened=Workspace.load(path)
            assert opened.db.execute('PRAGMA user_version').fetchone()[0]==VERSION==15
            native=compile_specialization(opened,result['specialization'])
            source=stored_source(opened,result['family']);output=Path(directory)/'checked_family.py'
            write_new_family(source,output)
            spec=importlib.util.spec_from_file_location('checked_temporal_example',output)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            exported=module.ArchitectureFamily();exported.network.load_state_dict(native.state_dict())
            owners=[id(p) for p in native.parameters()]
            for steps in (3,7,11):
                inputs={'frames':torch.randn(3,steps,2),'initial':torch.randn(3,2)}
                actual=native(inputs);candidate=exported(inputs)
                for key in actual:torch.testing.assert_close(actual[key],candidate[key],rtol=0,atol=0)
                assert [id(p) for p in native.parameters()]==owners
                assert actual['trajectory'].shape==(3,steps,2) and actual['final'].shape==(3,2)
                print(f'PASS: retained specialization5, actual T={steps}, exact native/export outputs, one 10-parameter owner')
    finally:
        for item in (opened,workspace):
            if item is not None:
                item.db.close()
                if item.history.directory:item.history.directory.cleanup()


if __name__=='__main__':main()
