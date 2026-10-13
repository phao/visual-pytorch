"""Predictive moments and densities have independent distribution/path checks."""
import copy
import math
import unittest
from unittest.mock import patch
import torch
from app.sequence_diagnostics import summarize,inspect_model
from test_sequence_controls import args


class SequenceDiagnosticsTests(unittest.TestCase):
    def test_actual_controlled_msvar_intermediates_match_forward_evidence_without_mutation(self):
        from app.neural_msvar import controlled_drawing,controlled_dataset
        from app.core import compile_model
        from app.task_runtime import forward_ports
        doc=controlled_drawing(steps=6);model=compile_model({doc['id']:doc},doc['id'])
        row=controlled_dataset(steps=6,trajectories=1)['records'][0]['values']
        inputs={g['id']:torch.tensor([row[g['id']]],dtype=torch.float32) for g in model.contract['input_groups']}
        weights={key:value.clone() for key,value in model.state_dict().items()}
        report=inspect_model(model,inputs,'filter');outputs=forward_ports(model,inputs,model.contract)
        self.assertEqual(report['conditioning_proofs'][0]['status'],'conditional')
        self.assertAlmostEqual(report['trajectories'][0]['sequence_log_evidence'],float(outputs['out_sequence_evidence'][0,0]),places=5)
        self.assertAlmostEqual(sum(s['mixture_log_density'] or 0. for s in report['trajectories'][0]['steps']),
                               report['trajectories'][0]['sequence_log_evidence'],places=5)
        for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,weights[key]))

    def test_predictive_density_matches_torch_mixture_and_independent_moments(self):
        values=args();summary=summarize(**values);first=summary['trajectories'][0]['steps'][0]
        weights=torch.tensor([.78,.22],dtype=torch.float64)  # [.8,.2] times asymmetric transition
        mu=values['means'][0,0];variance=values['positive_variances'][0,0]
        mixture=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(probs=weights),
            torch.distributions.Independent(torch.distributions.Normal(mu,variance.sqrt()),1))
        self.assertAlmostEqual(first['mixture_log_density'],float(mixture.log_prob(values['observations'][0,0])),places=13)
        expected_mean=sum(float(w*m) for w,m in zip(weights,mu[:,0]))
        expected_variance=sum(float(w*(v+(m-expected_mean)**2)) for w,v,m in zip(weights,variance[:,0],mu[:,0]))
        self.assertAlmostEqual(first['predictive_mean'][0],expected_mean,places=13)
        self.assertAlmostEqual(first['predictive_covariance'][0][0],expected_variance,places=13)
        for step in summary['trajectories'][0]['steps']:
            self.assertAlmostEqual(step['mixture_log_density'],step['filter_step_log_evidence'],places=12)

    def test_equal_mixture_means_do_not_imply_equal_predictive_density(self):
        values=args();values['observations'].zero_();values['means'].zero_()
        values['positive_variances'].fill_(.01);narrow=summarize(**values)
        values['positive_variances'].fill_(1.);wide=summarize(**values)
        a=narrow['trajectories'][0]['steps'][0];b=wide['trajectories'][0]['steps'][0]
        self.assertEqual(a['predictive_mean'],b['predictive_mean']);self.assertNotEqual(a['mixture_log_density'],b['mixture_log_density'])
        self.assertGreater(a['mixture_log_density'],0.)  # negative NLL is legitimate

    def test_execution_reset_and_emission_masks_keep_distinct_diagnostics(self):
        values=args();values['execution_mask']=torch.tensor([[True,False,True,True,False]])
        values['reset_mask']=torch.tensor([[False,True,False,True,True]])
        values['emission_mask']=torch.tensor([[True,True,False,True,True]])
        values['observations'][:,[1,2,4]]=math.nan;values['means'][:,[1,2,4]]=math.nan
        row=summarize(**values)['trajectories'][0]
        self.assertEqual((row['executed_steps'],row['scored_steps']),(3,2))
        self.assertFalse(row['steps'][1]['reset_applied']);self.assertTrue(row['steps'][3]['reset_applied'])
        self.assertIsNone(row['steps'][2]['mixture_log_density']);self.assertIsNotNone(row['steps'][2]['predictive_state_probabilities'])
        self.assertAlmostEqual(sum(s['mixture_log_density'] or 0. for s in row['steps']),row['sequence_log_evidence'],places=12)

    def test_empty_batch_time_and_immutable_inputs(self):
        values=args();before={key:value.clone() for key,value in values.items()}
        summarize(**values)
        for key,value in values.items():self.assertTrue(torch.equal(value,before[key]))
        for key in ('observations','log_transition','means','positive_variances'):values[key]=values[key][:,:0]
        values['lengths'].zero_();row=summarize(**values)['trajectories'][0]
        self.assertEqual(row['steps'],[]);self.assertEqual(row['sequence_log_evidence'],0.)
        for key in values:values[key]=values[key][:0]
        self.assertEqual(summarize(**values)['trajectories'],[])

    def test_covariance_budget_precedes_filter_and_high_dynamic_range_is_finite(self):
        with patch('app.sequence_controls.filter_run',side_effect=AssertionError('premature filter')):
            with self.assertRaisesRegex(ValueError,'budget'):summarize(**args(),max_diagnostic_elements=1)
        values={key:value.float() if value.is_floating_point() else value for key,value in args().items()}
        values['observations'].zero_();values['means'].fill_(1e20);values['positive_variances'].fill_(1e38)
        row=summarize(**values)['trajectories'][0]
        self.assertTrue(all(math.isfinite(s['mixture_log_density']) for s in row['steps']))


if __name__=='__main__':unittest.main()
