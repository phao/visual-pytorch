"""Actual CLI check/replay, separate worker, pinned bytes and no automatic fit."""
import io
import importlib.util
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import vpw_cli

ROOT=Path(__file__).parents[1]


class ExamplePackCLITests(unittest.TestCase):
    def call(self,*args):
        out=io.StringIO()
        with redirect_stdout(out):code=vpw_cli.main(list(map(str,args)))
        return code,json.loads(out.getvalue())

    def test_readonly_check_never_reads_config_or_executes_a_worker(self):
        path=ROOT/'examples/neural-msvar.vtw';before=path.read_bytes()
        with (patch('vpw_cli.ram_percentage',side_effect=AssertionError('config')),
              patch('app.example_packs.execute_candidate',side_effect=AssertionError('automatic fit'))):
            code,result=self.call(path,'example-check',ROOT/'examples/neural-msvar-pack.json')
        self.assertEqual(code,0,result);self.assertFalse(result['value']['executed']);self.assertEqual(path.read_bytes(),before)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Torch runtime needed for explicit replay')
    def test_explicit_actual_replay_matches_checkpoint_and_never_saves_source(self):
        path=ROOT/'examples/neural-msvar.vtw';before=path.read_bytes()
        code,result=self.call(path,'example-check',ROOT/'examples/neural-msvar-pack.json','--run',
            '--torch-python',sys.executable,'--timeout','60','--memory-limit-percent','5')
        self.assertEqual(code,0,result.get('error',result.get('value',{}).get('error')))
        self.assertTrue(result['value']['verified']);self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
