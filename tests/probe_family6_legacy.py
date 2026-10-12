"""Compare pure family1-5 receipts/records to the committed pre-family6 resolver.

Not unittest discovery: a bounded manual provenance probe, independent of Torch.
Run PYTHONPATH=.:tests python tests/probe_family6_legacy.py.
"""
import json
import subprocess
import types
from pathlib import Path

from app import families
from app.family_records import resolution_records
from app.revisions import frozen_document

ROOT=Path(__file__).resolve().parents[1]
PIN='492a21706afd65d0d0a9d3b66a3789ff8fecd56f'


def pinned(module):
    source=subprocess.check_output(['git','show',PIN+':app/'+module+'.py'],cwd=ROOT,text=True)
    loaded=types.ModuleType('app._pinned_'+module);loaded.__package__='app'
    exec(compile(source,PIN+':app/'+module+'.py','exec'),loaded.__dict__)
    return loaded


def main():
    old=pinned('families');records=pinned('family_records');count=0
    for filename in ('width-depth.json','width-depth-tied.json','activation-bias.json',
                     'scoped-branches.json','spatial-adaptive.json'):
        doc=json.loads((ROOT/'examples/families'/filename).read_text())
        for bindings in (None,{}):
            before=old.specialize(doc,bindings);after=families.specialize(doc,bindings)
            assert before==after,(filename,'resolution')
            frozen=frozen_document('family',doc)
            assert records.resolution_records(frozen,before)==resolution_records(frozen,after),(filename,'records')
            count+=1
    print(json.dumps({'ok':True,'pinned':PIN,'family_versions':[1,2,3,4,5],
                      'exact_resolution_and_retained_records':count}))


if __name__=='__main__':main()
