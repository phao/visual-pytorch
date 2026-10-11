"""Allocation-free v2 proofs, explicit version fences and whole-tree RAM gates."""
import copy
import json
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

from app.core import GraphError, MAX_INTEGER, compile_model, default_model, validate_model
from app.lstm_nodes import lstm_memory_report
from app.temporal_shapes_v2 import specification, shape_contract, admission, compile_temporal_model
from app.temporal_export_v2 import temporal_source
from examples.projected_lstm import models as fixture


def runtime_fixture(**options):
    models, mid = fixture(**options)
    return models, mid, {'sequence':[0]}


def shapes(models, mid, steps):
    return {g['id']:[steps, *g['shape'][1:]] if g['id'] == 'sequence' else list(g['shape'])
            for g in validate_model(models, mid)['input_groups']}


def nested_fixture():
    models, child, _ = runtime_fixture(steps=2, width=2, hidden=3, projection=2, layers=1, lengths=False)
    graph = models[child]['graph']
    for node in graph['nodes']:
        if node['id']=='sequence':
            node.pop('axes')
        if node['id'] in ('hidden','cell'):
            node['axes'] = ['LD','Hout' if node['id']=='hidden' else 'Hcell']
    graph['nodes'] = [n for n in graph['nodes'] if n['id'] != 'out_sequence']
    graph['edges'] = [e for e in graph['edges'] if e['to'] != 'out_sequence']
    root = default_model('Nested LSTM transition'); root['id'] = 'outer'
    root['graph'] = {'nodes':[
        {'id':'sequence','type':'Input','shape':[4,2,2],'axes':['T','D0','D1']},
        {'id':'hidden','type':'Input','shape':[2,2],'axes':['LD','Hout']},
        {'id':'cell','type':'Input','shape':[2,3],'axes':['LD','Hcell']},
        {'id':'scan','type':'Recurrent','ref':child,
         'sequence_inputs':[{'input':'sequence','source':'sequence'}],
         'states':[{'input':role,'output':'out_'+role,'source':role,'port':role} for role in ('hidden','cell')],
         'sequence_outputs':[{'output':'out_hidden','port':'trajectory'}]},
        *[{'id':'out_'+port,'type':'Output'} for port in ('trajectory','hidden','cell')]],
        'edges':[{'from':role,'to':'scan'} for role in ('sequence','hidden','cell')] +
                [{'from':'scan','from_port':port,'to':'out_'+port} for port in ('trajectory','hidden','cell')],
        'input_order':['sequence','hidden','cell']}
    models[root['id']] = root
    return models, root['id'], {'sequence':[0]}


def module_fixture():
    models,child,_ = runtime_fixture(steps=2,hidden=3,projection=2,layers=1,lengths=False)
    c = validate_model(models,child)
    root = default_model('Fixed packed LSTM module beside runtime frames'); root['id'] = 'module-parent'
    root['graph'] = {'nodes':[
        {'id':'packed','type':'Input','shape':[c['inputs']]},
        {'id':'nested','type':'Module','ref':child}, {'id':'nested-output','type':'Output'},
        {'id':'sequence','type':'Input','shape':[4,1],'axes':['T','F']},
        {'id':'variable-output','type':'Output'}],
        'edges':[{'from':'packed','to':'nested'},{'from':'nested','to':'nested-output'},
                 {'from':'sequence','to':'variable-output'}], 'input_order':['packed','sequence']}
    models[root['id']] = root
    return models,root['id'],{'sequence':[0]}


class TemporalV2ProofTests(unittest.TestCase):
    def test_versions_roles_and_parameter_dependency_proof_are_explicit_and_pure(self):
        models, mid, decl = runtime_fixture()
        before = copy.deepcopy((models,decl)); rng = random.getstate()
        spec = specification(models,mid,decl)
        self.assertEqual(spec['version'],2)
        self.assertEqual(spec['scope'],'explicit-eager-runtime-temporal-v2-contract')
        self.assertEqual(spec['output_dependencies'],[['sequence.T']])
        guard = next(g for g in spec['guards'] if g['path'].endswith('/lstm'))
        self.assertEqual(guard['ports'],['sequence','hidden','cell'])
        self.assertEqual(guard['inputs'],{role:role for role in guard['ports']})
        self.assertEqual(guard['lengths'],{'source':'lengths','adapter':'exact-int64-v1'})
        self.assertEqual((models,decl),before); self.assertEqual(random.getstate(),rng)

    def test_many_times_and_constructor_shapes_do_not_change_parameter_counts(self):
        rng = random.Random(49)
        for _ in range(64):
            h = rng.randrange(2,9); p = rng.choice([0,h-1]); layers = rng.randrange(1,4)
            directions = rng.randrange(1,3); width = rng.randrange(1,5); bias = rng.choice([True,False])
            models,mid,decl = runtime_fixture(hidden=h,projection=p,layers=layers,
                                             bidirectional=directions==2,width=width,bias=bias)
            spec = specification(models,mid,decl); q = p or h
            expected = directions * sum(4*h*((width if i == 0 else directions*q)+q) +
                                       (8*h if bias else 0) + (p*h if p else 0) for i in range(layers))
            for t in (1,3,17):
                c = shape_contract(models,mid,spec,shapes(models,mid,t))
                self.assertEqual(c['parameters'],expected)
                self.assertEqual(c['shapes']['lstm']['ports'],
                                 {'sequence':[t,directions*q],'hidden':[layers*directions,q],'cell':[layers*directions,h]})

    def test_old_temporal_and_spatial_and_families_stay_closed(self):
        from app.temporal_shapes import specification as old
        from app.runtime_shapes import specification as spatial
        from app.families import validate_family
        from examples.lstm_family import definition
        models,mid,decl = runtime_fixture()
        with self.assertRaisesRegex(GraphError,'do not support LSTM'): old(models,mid,decl)
        graph = models[mid]['graph']
        graph['nodes'] += [{'id':'image','type':'Input','shape':[1,2,2],'axes':['C','H','W']},
                          {'id':'image-out','type':'Output'}]
        graph['edges'].append({'from':'image','to':'image-out'}); graph['input_order'].append('image')
        with self.assertRaisesRegex(GraphError,'do not support LSTM'): spatial(models,mid,{'image':[1,2]})
        f = definition(); f['runtime_policy'] = {'version':2,'kind':'runtime-temporal-v2','inputs':decl}
        with self.assertRaises(GraphError): validate_family(f)

    def test_declaration_is_exact_axis_zero_not_state_width_or_control_shape(self):
        models,mid,_ = runtime_fixture()
        for decl in (None,{},[],{'unknown':[0]},{'sequence':[]},{'sequence':[True]},
                     {'sequence':[0.]},{'sequence':[0,0]},{'sequence':[1]},{'hidden':[0]},{'lengths':[0]}):
            with self.subTest(decl=decl), self.assertRaises(GraphError): specification(models,mid,decl)
        models[mid]['graph']['nodes'][0].pop('axes')
        with self.assertRaisesRegex(GraphError,'semantic T'): specification(models,mid,{'sequence':[0]})

    def test_actual_shapes_are_all_inputs_exact_fixed_states_frame_and_positive_time(self):
        models,mid,decl = runtime_fixture(); spec = specification(models,mid,decl); good = shapes(models,mid,7)
        for bad in (None,{},dict(good,extra=[1]),dict(good,sequence=[0,3]),dict(good,sequence=[7,4]),
                    dict(good,sequence=[True,3]),dict(good,sequence=[7.,3]),dict(good,sequence=[MAX_INTEGER,3]),
                    dict(good,hidden=[4,7]),dict(good,cell=[4,4]),dict(good,lengths=[7])):
            with self.subTest(bad=bad), self.assertRaises(GraphError): shape_contract(models,mid,spec,bad)

    def test_learned_width_from_time_fails_before_allocation(self):
        models,mid,decl = runtime_fixture(steps=3,width=3,hidden=3,projection=2,layers=1)
        graph = models[mid]['graph']; graph['nodes'].append({'id':'transpose','type':'Permute','dims':[1,0]})
        next(e for e in graph['edges'] if e['from']=='sequence')['from'] = 'transpose'
        graph['edges'].append({'from':'sequence','to':'transpose'})
        # Clear semantic axes explicitly after the permutation.
        graph['nodes'].append({'id':'adapter','type':'Reshape','shape':[3,3]})
        next(e for e in graph['edges'] if e['from']=='transpose')['from'] = 'adapter'
        graph['edges'].append({'from':'transpose','to':'adapter'})
        graph['nodes'][3]['inputs']['sequence'] = 'adapter'
        # A fixed reshape rejects unequal T at runtime rather than proving width dependence.
        spec = specification(models,mid,decl)
        with self.assertRaises(GraphError): shape_contract(models,mid,spec,shapes(models,mid,4))
        graph['nodes'][-1]['shape'] = [3,-1]
        with patch('app.core.compile_model',side_effect=AssertionError('Do not allocate')):
            with self.assertRaisesRegex(GraphError,'learned input width.*runtime T'):
                compile_temporal_model(models,mid,decl,constrained=False)

    def test_time_dependent_hidden_and_cell_sources_are_refused_statically(self):
        for role,width in (('hidden',2),('cell',3)):
            models,mid,decl = runtime_fixture(steps=2,hidden=3,projection=2,layers=1)
            graph = models[mid]['graph']
            graph['nodes'].extend([{'id':'state-source','type':'Input','shape':[2,width],'axes':['T','D0']},
                                  {'id':'state-adapter','type':'Reshape','shape':[-1,width]}])
            graph['input_order'].append('state-source'); decl['state-source'] = [0]
            graph['nodes'][3]['inputs'][role] = 'state-adapter'
            next(e for e in graph['edges'] if e['from']==role)['from'] = 'state-adapter'
            graph['edges'].append({'from':'state-source','to':'state-adapter'})
            graph['nodes'].append({'id':'old-'+role,'type':'Output'})
            graph['edges'].append({'from':role,'to':'old-'+role})
            with self.subTest(role=role),self.assertRaisesRegex(GraphError,'supplied '+role+' state dimensions'):
                specification(models,mid,decl)

    def test_flatten_to_learned_head_is_refused_but_last_axis_head_works(self):
        models,mid,decl = runtime_fixture()
        graph = models[mid]['graph']; graph['nodes'].append({'id':'head','type':'Linear','features':2,'bias':False})
        next(e for e in graph['edges'] if e['to']=='out_sequence')['to'] = 'head'
        graph['edges'].append({'from':'head','to':'out_sequence'})
        spec = specification(models,mid,decl); c = shape_contract(models,mid,spec,shapes(models,mid,11))
        self.assertEqual(c['shapes']['head']['output_shape'],[11,2])
        graph['nodes'].append({'id':'flat','type':'Flatten'})
        next(e for e in graph['edges'] if e['to']=='head')['to'] = 'flat'
        graph['edges'].append({'from':'flat','to':'head'})
        with self.assertRaisesRegex(GraphError,'Linear weight input width.*sequence.T'): specification(models,mid,decl)

    def test_explicit_output_and_length_float32_range_remain_actual_case_guards(self):
        models,mid,decl = runtime_fixture()
        graph = models[mid]['graph']; next(n for n in graph['nodes'] if n['id']=='out_sequence')['shape'] = [5,8]
        spec = specification(models,mid,decl)
        with self.assertRaisesRegex(GraphError,'Output out_sequence declares'):
            shape_contract(models,mid,spec,shapes(models,mid,6))
        next(n for n in graph['nodes'] if n['id']=='out_sequence').pop('shape')
        spec = specification(models,mid,decl)
        with self.assertRaisesRegex(GraphError,r'T <= 2\*\*24'):
            shape_contract(models,mid,spec,shapes(models,mid,2**24+1))

    def test_current_ram_exact_boundary_and_unknown_ram(self):
        models,mid,decl = runtime_fixture(); c = shape_contract(models,mid,specification(models,mid,decl),shapes(models,mid,19))
        with patch('app.runtime_shapes.available_memory',return_value=10**12):
            memory = admission(models,mid,c,3)['memory']
        expected = 32*3*c['shapes']['lstm']['lstm']['activation_elements'] + 16*3*(
            19*3 + 19*8 + 4*4 + 4*7)
        self.assertEqual(memory['components']['lstm_unrolls'],expected)
        inference = admission(models,mid,c,3,gradients=False,constrained=False)['memory']
        self.assertEqual(inference['components']['parameters'],128*c['parameters'])
        self.assertEqual(inference['components']['lstm_unrolls'],expected)
        amount = memory['estimated_bytes']
        with patch('app.runtime_shapes.available_memory',return_value=10*amount):
            self.assertTrue(admission(models,mid,c,3)['allowed'])
        for available in (10*amount-1,1,None):
            with patch('app.runtime_shapes.available_memory',return_value=available):
                with self.assertRaisesRegex(GraphError,'whole current-B/T graph'): admission(models,mid,c,3)
                self.assertTrue(admission(models,mid,c,3,constrained=False)['allowed'])

    def test_memory_grows_with_time_batch_and_nested_lstm_unrolls(self):
        models,mid,decl = nested_fixture(); spec = specification(models,mid,decl)
        reports = []
        for t,b in ((2,1),(9,4)):
            c = shape_contract(models,mid,spec,shapes(models,mid,t))
            value = admission(models,mid,c,b,constrained=False)['memory']
            child = validate_model(models,models[mid]['graph']['nodes'][3]['ref'])
            r = lstm_memory_report(child['shapes']['lstm']['lstm'],b,constrained=False)
            self.assertEqual(value['components']['lstm_unrolls'],t*(
                r['components']['gates_states_backward']+r['components']['boundary_copies']))
            reports.append(value)
        self.assertGreater(reports[1]['estimated_bytes'],reports[0]['estimated_bytes'])
        self.assertEqual(reports[0]['version'],2)

    def test_bad_admission_flags_and_policy_combinations_refuse(self):
        models,mid,decl = runtime_fixture(); c = validate_model(models,mid)
        for kw in ({'batch_size':True},{'batch_size':0},{'constrained':0},{'gradients':1},{'dtype_bytes':8},{'dtype_bytes':True}):
            args = dict(batch_size=2,constrained=False); args.update(kw)
            with self.subTest(kw=kw),self.assertRaises(GraphError): admission(models,mid,c,**args)
        for kw in ({'runtime_inputs':{}},{'runtime_temporal':decl}):
            with self.assertRaisesRegex(GraphError,'select exactly one'):
                compile_model(models,mid,False,runtime_temporal_v2=decl,**kw)

    def test_fixed_module_lstm_charge_is_not_hidden_by_an_unrelated_dynamic_input(self):
        models,mid,decl = module_fixture(); c = shape_contract(models,mid,specification(models,mid,decl),shapes(models,mid,7))
        memory = admission(models,mid,c,3,constrained=False)['memory']
        child = validate_model(models,models[mid]['graph']['nodes'][1]['ref'])
        value = lstm_memory_report(child['shapes']['lstm']['lstm'],3,constrained=False)
        self.assertEqual(memory['components']['lstm_unrolls'],
                         value['components']['gates_states_backward']+value['components']['boundary_copies'])

    def test_alias_owners_count_once_but_each_lstm_call_is_charged(self):
        models,mid,decl = runtime_fixture()
        baseline = admission(models,mid,validate_model(models,mid),2,constrained=False)['memory']
        graph = models[mid]['graph']; shared = copy.deepcopy(graph['nodes'][3]); shared.update(id='shared',share_with='lstm')
        graph['nodes'].extend([shared,{'id':'shared-output','type':'Output'}])
        graph['edges'] += [dict(e,to='shared') for e in list(graph['edges']) if e['to']=='lstm']
        graph['edges'].append({'from':'shared','from_port':'sequence','to':'shared-output'})
        c = shape_contract(models,mid,specification(models,mid,decl),shapes(models,mid,5))
        memory = admission(models,mid,c,2,constrained=False)['memory']
        self.assertEqual(memory['components']['parameters'],baseline['components']['parameters'])
        self.assertEqual(memory['components']['lstm_unrolls'],2*baseline['components']['lstm_unrolls'])

    def test_initial_actual_case_whole_graph_ram_precedes_any_model_construction(self):
        models,mid,decl = runtime_fixture()
        for initial in (None,shapes(models,mid,7)):
            with patch('app.runtime_shapes.available_memory',return_value=1),\
                 patch('app.core.compile_model',side_effect=AssertionError('No layers')):
                with self.assertRaisesRegex(GraphError,'whole current-B/T graph'):
                    compile_temporal_model(models,mid,decl,initial_shapes=initial)

    def test_export_is_pure_and_captures_v2_while_old_export_refuses_lstm(self):
        from app.temporal_export import temporal_source as old
        from app.temporal_export import _temporal_source
        models,mid,decl = runtime_fixture(); before = copy.deepcopy(models)
        for invalid in (False,True,3,None,'2',2.):
            with self.subTest(version=invalid),self.assertRaisesRegex(GraphError,'explicit integer version'):
                _temporal_source(models,mid,decl,constrained=False,memory_limit_percent=10.,version=invalid)
        result = temporal_source(models,mid,decl,constrained=False)
        self.assertEqual(result['version'],2); self.assertEqual(result['provenance']['policy'],'runtime-temporal-v2')
        self.assertIn('temporal_shapes_v2',result['provenance']['runtime_files'])
        scope = {}; exec(result['code'],scope)
        self.assertEqual(scope['runtime_specification'](),specification(models,mid,decl))
        self.assertEqual(models,before)
        with self.assertRaisesRegex(GraphError,'do not support LSTM'): old(models,mid,decl)
        for key in ('_MODEL','_EXPORT','_SOURCES'):
            namespace = {}; exec(result['code'],namespace)
            if key == '_MODEL': namespace[key]['models'][mid]['graph']['nodes'][3]['hidden_size'] = 8
            elif key == '_EXPORT': namespace[key]['runtime_inputs']['sequence'] = [1]
            else: namespace[key]['temporal_shapes_v2'] += '\n# tampered\n'
            with self.subTest(key=key),self.assertRaises((ValueError,GraphError)):
                namespace['runtime_specification']()

    def test_stdlib_probe_import_proof_preview_and_export_allocate_no_tensors(self):
        code = """
import sys
from examples.projected_lstm import models
from app.temporal_shapes_v2 import specification, shape_contract, admission
from app.temporal_export_v2 import temporal_source
collection,mid=models();decl={'sequence':[0]}
spec=specification(collection,mid,decl)
shapes={g['id']:g['shape'] for g in spec['baseline_input_groups']}
c=shape_contract(collection,mid,spec,shapes)
admission(collection,mid,c,2,constrained=False)
s={};exec(temporal_source(collection,mid,decl,constrained=False)['code'],s)
s['runtime_specification']()
assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
print('pure-v2-verified')
"""
        result = subprocess.run(['/usr/bin/python3','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(result.stdout.strip(),'pure-v2-verified')


if __name__ == '__main__':
    unittest.main()
