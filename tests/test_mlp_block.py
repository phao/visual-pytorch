"""The MLP building block (webview graph-tools) produces valid, PyTorch-compilable graphs."""
import json
import subprocess
import unittest
from pathlib import Path

from app.core import default_model, validate_model

ROOT = Path(__file__).resolve().parents[1]


def built(after, text, activation):
    doc = default_model('MLP block', inputs=4, outputs=4)
    source = ("let t='';process.stdin.on('data',x=>t+=x);process.stdin.on('end',()=>{const d=JSON.parse(t),"
              "g=require('./app/web/graph-tools.js'),a=JSON.parse(process.argv[1]);"
              "g.addMLP(d,{after:a.after,inputWidth:a.after?4:null,widths:g.parseWidths(a.text),activation:a.activation});"
              "process.stdout.write(JSON.stringify(d));});")
    out = subprocess.run(['node', '-e', source, json.dumps({'after': after, 'text': text, 'activation': activation})],
                         input=json.dumps(doc), text=True, capture_output=True, check=True, cwd=ROOT, timeout=10)
    doc = json.loads(out.stdout)
    return {doc['id']: doc}, doc['id']


class MLPBlockTests(unittest.TestCase):
    def test_inserted_after_selection_validates_with_exact_parameter_count(self):
        models, mid = built('linear1', '4, 3*8, 2', 'GELU')
        contract = validate_model(models, mid)
        # Linear 4->4 (base) + 4->8, 8->8, 8->8, 8->2.
        self.assertEqual(contract['parameters'], (4 + 1) * 4 + (4 + 1) * 8 + 2 * (8 + 1) * 8 + (8 + 1) * 2)
        self.assertEqual(contract['output_shape'], [2])

    def test_new_branch_validates_for_every_activation(self):
        for activation in ('GELU', 'ReLU', 'Sigmoid', 'Softplus', 'Tanh', 'Softmax', 'LogSoftmax'):
            models, mid = built(None, '2, 10*3, 2', activation)
            contract = validate_model(models, mid)
            widths = [2] + [3] * 10 + [2]
            branch = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
            self.assertEqual(contract['parameters'], (4 + 1) * 4 + branch, activation)


if __name__ == '__main__':
    unittest.main()
