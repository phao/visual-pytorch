"""Compare family1–6 resolution/record bytes with accepted pre-Family7 source.

Pure stdlib, committed Git pin, no model allocation or workspace writes.
"""
import json
import subprocess
import types
from pathlib import Path

from app.families import specialize
from app.family_records import resolution_records
from app.revisions import frozen_document
from test_temporal_families import family

ROOT = Path(__file__).resolve().parents[1]
PIN = '254e16348c45986ea586070ff60ceb22ac63b34f'


def pinned(module):
    source = subprocess.check_output(['git', 'show', PIN+':app/'+module+'.py'], cwd=ROOT, text=True)
    loaded = types.ModuleType('app._lstm_legacy_'+module); loaded.__package__ = 'app'
    exec(compile(source, PIN+':app/'+module+'.py', 'exec'), loaded.__dict__)
    return loaded


def main():
    old = pinned('families'); records = pinned('family_records'); count = 0
    documents = [json.loads((ROOT/'examples/families'/filename).read_text()) for filename in
                 ('width-depth.json', 'width-depth-tied.json', 'activation-bias.json',
                  'scoped-branches.json', 'spatial-adaptive.json')]
    documents += [family(cnn=cnn, scoped=scoped, dynamic=dynamic)
                  for cnn in (False, True) for scoped in (False, True) for dynamic in (False, True)]
    for doc in documents:
        for bindings in (None, {}):
            before = old.specialize(doc, bindings); after = specialize(doc, bindings)
            assert before == after, (doc['version'], doc['id'], 'resolution')
            frozen = frozen_document('family', doc)
            assert records.resolution_records(frozen, before) == resolution_records(frozen, after), (doc['version'], 'records')
            count += 1
    print(json.dumps({'ok': True, 'pinned': PIN, 'family_versions': [1, 2, 3, 4, 5, 6],
                      'exact_resolution_and_retained_records': count}))


if __name__ == '__main__': main()
