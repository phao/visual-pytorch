"""Family10/Spec9 policy admission, repetition mapping and pinned reproduction."""
import copy
import unittest
from app.core import GraphError, Workspace
from app.families import specialize, validate_family
from app.family_records import retain, compile_specialization
from app.policy_graph import owner_key, plan
from app.runtime_tasks import proof
from policy_runtime_fixture import family


def repeated(mode='independent'):
    from test_shared_parameters import serial
    models = serial(depth=1, shared=False, kind='Module')
    models['shared']['graph']['parameter_policy'] = {'version': 1, 'parameters': {
        owner_key(['use0', 'use0', 'weight']): {'trainable': False, 'initializer': {'kind': 'normal', 'seed': 11, 'mean': 0., 'std': .1}}}}
    return {'version': 10, 'id': 'repeat-policy', 'name': 'Repeated scoped policies',
            'root': 'shared', 'models': models, 'symbols': [], 'assignments': [], 'constructors': [],
            'repeats': [{'id': 'r', 'model': 'shared', 'nodes': ['use0'], 'count': 3, 'mode': mode}], 'runtime_policy': None}


class PolicyFamilyTests(unittest.TestCase):
    def test_family10_required_policy_and_older_versions_cannot_ignore_it(self):
        f = family(); validate_family(f)
        for version in range(1, 10):
            bad = copy.deepcopy(f); bad['version'] = version
            with self.subTest(version=version), self.assertRaises(GraphError): validate_family(bad)
        bad = copy.deepcopy(f)
        for model in bad['models'].values(): model['graph'].pop('parameter_policy', None)
        with self.assertRaisesRegex(GraphError, 'explicit parameter policy'): validate_family(bad)

    def test_repeat_declarations_apply_to_all_copies_with_once_only_tied_owners(self):
        for mode in ('independent', 'tied'):
            f = repeated(mode); before = copy.deepcopy(f); value = specialize(f)
            self.assertEqual(before, f)
            resolved = plan(value['models'], value['root'])['resolved']
            self.assertEqual(resolved['counts'], {'stored': 18, 'trainable': 6, 'frozen': 12} if mode == 'independent'
                             else {'stored': 6, 'trainable': 2, 'frozen': 4})
            params = value['models'][value['root']]['graph']['parameter_policy']['parameters']
            self.assertEqual(set(params), {owner_key(['r/' + str(i) + '/use0', 'use0', 'weight']) for i in range(3)})

    def test_retained_model6_spec9_exact_reproduction_and_runtime_dispatch(self):
        for temporal in (False, True):
            workspace = Workspace()
            try:
                result = retain(workspace, family=family(temporal), materialize=False)
                model = workspace.get_spec('model', result['model']['id'], result['model']['revision'])
                spec = workspace.get_spec('specialization', result['specialization']['id'], result['specialization']['revision'])
                self.assertEqual(model['version'], 6); self.assertEqual(spec['version'], 9)
                if temporal:
                    f = workspace.get_spec('family', result['family']['id'], result['family']['revision'])
                    self.assertEqual(spec['runtime']['kind'], 'runtime-temporal-v2')
                    witness = proof(f, spec, model, execution='runtime-temporal-v2')
                    self.assertEqual(witness['family']['revision'], result['family']['revision'])
            finally: workspace.db.close()


if __name__ == '__main__': unittest.main()
