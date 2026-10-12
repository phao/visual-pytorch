"""Exact old Family1–7 resolution/records against the pre-Family8 commit.

Pure compatibility comparison, not a full historical runnable-closure replay.
The two committed grammar/record modules reuse unchanged/current dependencies.
"""
import itertools
import json
import sys

from app.families import specialize
from app.family_records import resolution_records
from app.revisions import frozen_document
from examples.lstm_family import definition
from probe_lstm_family_legacy import main as old_six, pinned as prior_loader
import probe_lstm_family_legacy as loader


def main():
    old_six()
    original=loader.PIN;loader.PIN='6394fb83b89741b796648f7247c9dfa3c4abcbed'
    try:old=prior_loader('families');records=prior_loader('family_records')
    finally:loader.PIN=original
    count=0;doc=definition();frozen=frozen_document('family',doc)
    for time,layers,directions,projection,bias in itertools.product((1,9),(1,2),(False,True),(0,2),(False,True)):
        bindings={'time':time,'width':2,'cell-width':3,'layers':layers,'directions':directions,'projection':projection,'bias':bias}
        before=old.specialize(doc,bindings);after=specialize(doc,bindings)
        assert before==after,('family7','resolution',bindings)
        assert records.resolution_records(frozen,before)==resolution_records(frozen,after),('family7','records',bindings)
        count+=1
    assert not any(k in sys.modules for k in ('torch','numpy','PyQt6'))
    print(json.dumps({'ok':True,'pinned':'6394fb83b89741b796648f7247c9dfa3c4abcbed',
                      'family7_exact_resolution_records':count,'heavy_imports':False}))


if __name__=='__main__':main()
