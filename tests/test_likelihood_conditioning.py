"""Density interpretation needs causal parameters, not only normalized rows."""
import copy
import unittest
from app.core import GraphError,validate_model
from app.datasets import validate_task
from app.neural_msvar import drawing,dataset,likelihood_task,controlled_drawing


class LikelihoodConditioningTests(unittest.TestCase):
    def checked(self,doc,task=None):
        contract=validate_model({doc['id']:doc},doc['id'])
        validate_task(task or likelihood_task(),dataset(steps=3,trajectories=3),contract)
        return contract

    def test_requested_full_causal_model_and_controls_have_conditional_proof(self):
        for constructor in (drawing,controlled_drawing):
            doc=constructor(steps=3)
            contract=validate_model({doc['id']:doc},doc['id'])
            proof=contract['density_conditioning']['out_sequence_evidence']
            self.assertEqual(proof['status'],'conditional')
            self.assertEqual(proof['observation_inputs'],['observations'])
            self.assertIn('history',proof['conditioning_inputs'])
            self.assertEqual(proof['findings'],[])

    def test_direct_current_observation_mean_is_rejected_before_worker(self):
        doc=drawing(steps=3);nodes={n['id']:n for n in doc['graph']['nodes']}
        nodes['observations']['shape']=[3,3]
        nodes['mean_linear']['features']=15
        # Preserve the transition's causal lag branch, but bypass it for means.
        for edge in doc['graph']['edges']:
            if edge['to']=='mean_linear':edge['from']='observations'
        contract=validate_model({doc['id']:doc},doc['id'])
        proof=contract['density_conditioning']['out_sequence_evidence']
        self.assertEqual(proof['status'],'unproved')
        self.assertEqual(proof['findings'][0]['path'][-1],'means')
        with self.assertRaisesRegex(GraphError,'conditioning refused'):self.checked(doc)

    def test_time_axis_normalization_of_past_features_is_not_a_causal_parameter(self):
        doc=drawing(steps=3)
        node=next(n for n in doc['graph']['nodes'] if n['id']=='transition_gelu')
        node.update(activation='Softmax',dim=0)
        with self.assertRaisesRegex(GraphError,'current/future|temporal mixing'):self.checked(doc)

    def test_known_probabilities_are_not_silently_treated_as_class_logits(self):
        from app.construction_recipes import build
        doc=build('finite-window-regime-estimator',model_id='approx',options={'window':3,'D':3,'K':5})[0]['document']
        contract=validate_model({doc['id']:doc},doc['id'])
        data={'version':1,'id':'d','name':'Classes','storage':{'kind':'inline-records','version':1},
            'fields':[{'id':'x','dtype':'float32','shape':[3,3]},
                      {'id':'class','dtype':'category','shape':[],'vocabulary':{'version':1,'values':list(range(5))}}],
            'records':[{'id':str(i),'values':{'x':[[0.]*3]*3,'class':i}} for i in range(2)]}
        task={'version':1,'id':'t','name':'Classes','inputs':[{'port':'window','field':'x','transforms':[]}],
              'heads':[{'id':'class','port':'estimate','field':'class','transforms':[],'loss':'cross-entropy','weight':1.}]}
        with self.assertRaisesRegex(GraphError,'scores/logits'):validate_task(task,data,contract)
        next(n for n in doc['graph']['nodes'] if n['id']=='probabilities').update(activation='LogSoftmax')
        validate_task(task,data,validate_model({doc['id']:doc},doc['id']))  # normalized log scores are valid logits

    def test_observed_history_cannot_alias_the_current_trajectory_field(self):
        doc=drawing(steps=3);next(n for n in doc['graph']['nodes'] if n['id']=='history').pop('axes')
        task=likelihood_task();next(b for b in task['inputs'] if b['port']=='history')['field']='observations'
        with self.assertRaisesRegex(GraphError,'aliases the scored'):self.checked(doc,task)

    def test_single_input_module_preserves_the_causal_parameter_path(self):
        doc=drawing(steps=3);node=next(n for n in doc['graph']['nodes'] if n['id']=='mean_linear')
        node.pop('features');node.pop('bias');node.update(type='Module',ref='mean-body')
        child={'id':'mean-body','name':'Pointwise mean body','graph':{'nodes':[
            {'id':'x','type':'Input','shape':[3,9]}, {'id':'linear','type':'Linear','features':15,'bias':True},
            {'id':'out','type':'Output'}],'edges':[{'from':'x','to':'linear'},{'from':'linear','to':'out'}]}}
        contract=validate_model({doc['id']:doc,child['id']:child},doc['id'])
        validate_task(likelihood_task(),dataset(steps=3,trajectories=3),contract)
        self.assertEqual(contract['density_conditioning']['out_sequence_evidence']['status'],'conditional')

    def test_adaptive_observation_transform_needs_its_own_density_objective(self):
        doc=drawing(steps=3)
        doc['graph']['nodes'].append({'id':'adaptive-y','type':'Linear','features':3,'bias':True})
        for edge in doc['graph']['edges']:
            if edge['from']=='observations':edge['from']='adaptive-y'
        doc['graph']['edges'].append({'from':'observations','to':'adaptive-y'})
        for node in doc['graph']['nodes']:
            if node['type'] in ('CausalLagWindow','GaussianFilter'):node['inputs']['observations']='adaptive-y'
        contract=validate_model({doc['id']:doc},doc['id'])
        self.assertEqual(contract['density_conditioning']['out_sequence_evidence']['findings'][0]['code'],'ADAPTIVE_OBSERVATION_LIKELIHOOD')
        with self.assertRaisesRegex(GraphError,'Jacobian'):self.checked(doc)


if __name__=='__main__':unittest.main()
