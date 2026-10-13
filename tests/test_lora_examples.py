import copy
import unittest
from app.core import GraphError, Workspace, validate_model
from app.lora import definitions, linear, populate, dataset, ALPHA
from app.policy_graph import plan
from app.experiment_objects import preview_experiment


class LoRAExampleTests(unittest.TestCase):
    def test_actual_related_graphs_are_valid_and_counts_roles_rank_scaling_match(self):
        models = definitions()
        for mid in models: validate_model(models, mid)
        for rank in (1, 2, 4):
            mid = 'lora-r' + str(rank); c = plan(models, mid)['resolved']['counts']
            self.assertEqual(c, {'stored': 256 + 32 * rank, 'trainable': 32 * rank, 'frozen': 256})
            self.assertEqual(next(n for n in models[mid]['graph']['nodes'] if n['type'] == 'Scale')['coefficient'], ALPHA / rank)
            self.assertFalse(any(n['type'] == 'Activation' for n in models[mid]['graph']['nodes']))
            labels = {node['id']: node.get('label') for node in models[mid]['graph']['nodes']}
            self.assertEqual([labels[key] for key in ('base', 'A', 'B')], ['Base W₀', 'Adapter A', 'Adapter B'])
        self.assertEqual(plan(models, 'lora-reusable')['resolved']['counts'], plan(models, 'lora-r2')['resolved']['counts'])
        self.assertEqual(plan(models, 'mlp-adapter')['resolved']['counts'], {'stored': 640, 'trainable': 128, 'frozen': 512})

    def test_comparison_selection_holds_dataset_task_split_seeds_and_budget_fixed(self):
        w = Workspace()
        try:
            refs = populate(w, updates=3)
            for sid, ref in refs.items():
                exp = w.get_spec('experiment', ref['id'], ref['revision']); rows = preview_experiment(exp, w.get_spec)['candidates']
                self.assertTrue(all(row['status'] == 'compatible' for row in rows))
                self.assertEqual(len({str(row['manifest']) for row in rows}), 1)
                self.assertEqual(len({str(row['protocol']) for row in rows}), 1)
                self.assertEqual(len(rows), 2 if 'mlp' in sid else 4)
        finally: w.db.close()

    def test_rank_invalid_and_synthetic_data_reproducible(self):
        for rank in (0, -1, True, 17, 2.5):
            with self.assertRaises(GraphError): linear('invalid', rank=rank)
        self.assertEqual(dataset(), dataset()); self.assertNotEqual(dataset()['records'], dataset(mlp=True)['records'])


if __name__ == '__main__': unittest.main()
