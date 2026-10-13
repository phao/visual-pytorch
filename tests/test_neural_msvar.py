"""Full requested K5/p3/D3 composition, independent paths and real training."""
import itertools
import math
import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from app.core import GraphError, Workspace, compile_model, generate_code, validate_model
from app.neural_msvar import drawing, dataset, likelihood_task
from app.datasets import make_split
from app.revisions import model_document
from app.task_execution import task_bundle, training_protocol, restore_task_predictor, generate_task_predictor
from app.task_training import train_task
from app.task_result import validate_result
from app.task_runtime import forward_ports
from app.family_records import retain, compile_specialization
from app.family_export import family_source

torch.set_num_threads(1)


def invoke(model, inputs):
    contract=model.contract if hasattr(model,'contract') else model.network.contract
    return forward_ports(model,inputs,contract)


def full_bundle(updates=12):
    doc=drawing(steps=8);data=dataset(steps=8)
    split=make_split(data,{'method':'group','field':'trajectory','seed':31,'fractions':[.5,.25,.25]})
    return task_bundle(model_document({doc['id']:doc},doc['id']),data,likelihood_task(),split,
        training_protocol(epochs=None,updates=updates,batch_size=4,lr=.01,evaluate_every=3))


class NeuralMSVARTests(unittest.TestCase):
    def test_requested_dimensions_independent_path_likelihood_causality_and_export_gradients(self):
        doc=drawing(steps=3);models={doc['id']:doc};contract=validate_model(models,doc['id'])
        self.assertEqual(contract['parameters'],1315)
        self.assertEqual(contract['shapes']['lags']['output_shape'],[3,9])
        native=compile_model(models,doc['id']);namespace={};exec(generate_code(models,doc['id']),namespace)
        exported=namespace['model'];exported.load_state_dict(native.state_dict(),strict=True)
        values=dataset(trajectories=3,steps=3)['records'][0]['values']
        x={key:torch.tensor([values[key]],dtype=torch.float32) for key in ('observations','history','lengths')}
        outputs=invoke(native,x)
        layers={item['id']:getattr(native,item['layer']) for item in native.contract['plan'] if 'layer' in item}
        history=values['history'];observations=values['observations']
        # Independent list construction, not the CausalLagWindow implementation.
        features=torch.tensor([[sum((history+observations[:t])[-3:],[]) for t in range(3)]],dtype=torch.float32)
        hidden=F.gelu(F.linear(features,layers['transition_hidden'].weight,layers['transition_hidden'].bias))
        probabilities=F.softmax(F.linear(hidden,layers['transition_logits'].weight,layers['transition_logits'].bias).reshape(1,3,5,5),dim=-1)
        means=F.linear(features,layers['mean_linear'].weight,layers['mean_linear'].bias).reshape(1,3,5,3)
        variances=F.softplus(layers['raw_variances'].value)+layers['variances'].positive_floor
        initial=F.softmax(layers['initial_logits'].value,dim=-1)
        torch.testing.assert_close(outputs['out_transition'].exp(),probabilities)
        torch.testing.assert_close(outputs['out_means'],means)
        torch.testing.assert_close(outputs['out_variances'][0],variances)
        p=probabilities[0].detach().tolist();mu=means[0].detach().tolist();var=variances.detach().tolist();q=initial.detach().tolist()
        # Enumerate the initial state AND all three latent transitions (5^4 paths).
        weights=[];ends=[]
        for path in itertools.product(range(5),repeat=4):
            log_weight=math.log(q[path[0]])
            for t in range(3):
                k=path[t+1];log_weight+=math.log(p[t][path[t]][k])
                log_weight-=.5*sum(math.log(2*math.pi*var[k][d])+(observations[t][d]-mu[t][k][d])**2/var[k][d] for d in range(3))
            weights.append(log_weight);ends.append(path[-1])
        high=max(weights);evidence=high+math.log(math.fsum(math.exp(w-high) for w in weights))
        self.assertAlmostEqual(float(outputs['out_sequence_evidence'][0,0].detach()),evidence,places=5)
        for k in range(5):
            expected=math.fsum(math.exp(w-evidence) for w,end in zip(weights,ends) if end==k)
            self.assertAlmostEqual(float(outputs['out_final'][0,k].exp().detach()),expected,places=6)
        altered={k:v.clone() for k,v in x.items()};altered['observations'][0,1]+=100.
        changed=invoke(native,altered)
        torch.testing.assert_close(outputs['out_transition'][:,:2],changed['out_transition'][:,:2],rtol=0,atol=0)
        torch.testing.assert_close(outputs['out_means'][:,:2],changed['out_means'][:,:2],rtol=0,atol=0)
        for key,value in outputs.items():torch.testing.assert_close(invoke(exported,x)[key],value,rtol=0,atol=0)
        (-outputs['out_sequence_evidence'].mean()).backward()
        (-invoke(exported,x)['out_sequence_evidence'].mean()).backward()
        for key,parameter in native.named_parameters():
            self.assertIsNotNone(parameter.grad,key);self.assertTrue(torch.isfinite(parameter.grad).all(),key)
            torch.testing.assert_close(dict(exported.named_parameters())[key].grad,parameter.grad,rtol=0,atol=0)

    def test_full_unlabelled_fit_group_split_checkpoint_predictor_and_true_sequence_density(self):
        frozen=full_bundle();result=train_task(frozen,constrained=False)
        self.assertEqual(result['status'],'completed',result['failure'])
        self.assertEqual(result['counts']['completed_updates'],12)
        self.assertEqual(set(frozen['dataset']['fields'][i]['id'] for i in range(4)),{'observations','history','lengths','trajectory'})
        groups={role:{next(r['values']['trajectory'] for r in frozen['dataset']['records'] if r['id']==rid) for rid in ids}
                for role,ids in frozen['manifest']['roles'].items()}
        self.assertFalse(groups['train']&groups['test']);self.assertFalse(groups['train']&groups['validation'])
        initial=result['history'][0]['metrics']['train']['objective'];final=result['final_metrics']['train']['objective']
        self.assertLess(final,initial)
        validate_result(frozen,{k:v for k,v in result.items() if k!='bundle'})
        predictor=restore_task_predictor(frozen,result['checkpoint'],constrained=False)
        namespace={};exec(generate_task_predictor(frozen,result['checkpoint'],constrained=False),namespace)
        self.assertEqual(namespace['target_fields'],[])
        rows=[r['values'] for r in frozen['dataset']['records'][-2:]]
        expected=predictor(rows);actual=namespace['predict_records'](rows)
        for key,value in expected.items():torch.testing.assert_close(actual[key],value,rtol=0,atol=0)
        torch.testing.assert_close(actual['out_sequence_evidence'][:,0],actual['out_step_evidence'].sum(-1))
        torch.testing.assert_close(actual['out_filtered'].exp().sum(-1),torch.ones(2,8))

    def test_family13_runtime_changes_time_not_parameter_owners(self):
        doc=drawing(steps=4);family={'version':13,'id':'msvar-family','name':'Neural MS-VAR family',
            'root':doc['id'],'models':{doc['id']:doc},'symbols':[],'assignments':[],'repeats':[],
            'runtime_policy':{'version':2,'kind':'runtime-temporal-v2','inputs':{'observations':[0]}}}
        workspace=Workspace();self.addCleanup(workspace.db.close)
        refs=retain(workspace,family=family,materialize=False)
        self.assertEqual(workspace.get_spec('specialization',refs['specialization']['id'],refs['specialization']['revision'])['version'],12)
        native=compile_specialization(workspace,refs['specialization'],constrained=False)
        namespace={};exec(family_source(family)['code'],namespace);exported=namespace['ArchitectureFamily']()
        exported.network.load_state_dict(native.state_dict(),strict=True)
        for steps in (2,7):
            x={'observations':torch.zeros(2,steps,3),'history':torch.zeros(2,3,3),'lengths':torch.full((2,1),float(steps))}
            for key,value in invoke(native,x).items():torch.testing.assert_close(invoke(exported,x)[key],value,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in native.parameters()),1315)

    def test_old_headers_and_frozen_model_versions_refuse_causal_graphs(self):
        doc=drawing();workspace=Workspace();self.addCleanup(workspace.db.close);workspace.put(doc)
        frozen=model_document({doc['id']:doc},doc['id']);self.assertEqual(frozen['version'],9)
        from app.revisions import validate_model_document
        frozen['version']=8
        with self.assertRaisesRegex(GraphError,'ModelRevision9'):validate_model_document(frozen)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'msvar.vtw';workspace.save(path)
            with sqlite3.connect(path) as database:database.execute('PRAGMA user_version=30')
            with self.assertRaisesRegex(GraphError,'version-31'):Workspace.load(path)


if __name__=='__main__':unittest.main()
