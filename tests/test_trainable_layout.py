import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from app.core import Workspace, default_model
from app.lora import definitions
from app.policy_authoring import trainable_block_counts


class TrainableLayoutTests(unittest.TestCase):
    def test_lora_freezes_base_without_discarding_stored_counts(self):
        models = definitions(); counts = trainable_block_counts(models, 'lora-r2')
        self.assertEqual(counts, {'x':'0', 'base':'0', 'A':'32', 'B':'32', 'scale':'0', 'sum':'0', 'y':'0'})
        self.assertEqual(trainable_block_counts(models, 'lora-reusable')['layer-1'], '64')
        mlp = trainable_block_counts(models, 'mlp-adapter')
        self.assertEqual((mlp['layer-1'], mlp['layer-2']), ('64', '64'))
        self.assertEqual(sum(map(int, trainable_block_counts(models, 'mlp-base').values())), 0)

    def test_shared_canonical_owner_counted_once_and_mixed_bias_remains_learnable(self):
        doc = default_model(inputs=2, outputs=2)
        doc['graph']['nodes'].insert(2, {'id':'alias', 'type':'Linear', 'features':2, 'share_with':'linear1'})
        doc['graph']['edges'] = [{'from':'input','to':'linear1'}, {'from':'linear1','to':'alias'}, {'from':'alias','to':'output'}]
        doc['graph']['parameter_policy'] = {'version':1, 'parameters': {'["linear1","weight"]': {'trainable':False}}}
        counts = trainable_block_counts({doc['id']: doc}, doc['id'])
        self.assertEqual(counts['linear1'], '2'); self.assertEqual(counts['alias'], '0')
        self.assertEqual(sum(map(int, counts.values())), 2)

    def test_cli_trainable_projection_is_read_only_and_default_stays_stored(self):
        import vpw_cli
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'counts.vtw'; w = Workspace()
            try:
                for doc in definitions().values(): w.put(doc)
                w.save(path)
            finally: w.db.close()
            before = path.read_bytes(); values = []
            for args in ([], ['--trainable']):
                stream = io.StringIO()
                with redirect_stdout(stream): code = vpw_cli.main([str(path), 'parameter-counts', 'lora-r2', *args])
                self.assertEqual(code, 0); values.append(json.loads(stream.getvalue())['value'])
            self.assertEqual(values[0]['counts']['base'], '256')
            self.assertEqual(values[1]['counts']['base'], '0')
            self.assertEqual(values[1]['scope'], 'unique-trainable-owners')
            self.assertEqual(before, path.read_bytes())


if __name__ == '__main__': unittest.main()
