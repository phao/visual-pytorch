"""Pure temporal-domain proofs, guards and pre-allocation safety checks."""
import copy
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.core import GraphError, MAX_INTEGER, compile_model, validate_model
from app.temporal_shapes import declaration, specification, shape_contract, admission, compile_temporal_model
from app.temporal_export import temporal_source
from test_recurrent_graph_contract import graph_fixture


def temporal_fixture(*, cnn=False, steps=4, controls=True):
    models, mid = graph_fixture(cnn=cnn, steps=steps, controls=controls)
    nodes = {n['id']:n for n in models[mid]['graph']['nodes']}
    nodes['sequence']['axes'] = ['T', 'C', 'H', 'W'] if cnn else ['T', 'D0']
    runtime = {'sequence':[0]}
    if controls:
        for role in ('mask', 'reset'):
            nodes[role]['axes'] = ['T']
            runtime[role] = [0]
    return models, mid, runtime


def shapes(models, mid, steps):
    return {g['id']:[steps, *g['shape'][1:]] if g.get('axes', [None])[0] == 'T' else g['shape']
            for g in validate_model(models, mid)['input_groups']}


def nested_fixture():
    models, inner = graph_fixture(steps=2, controls=False)
    body = models.pop(inner); body['id'] = 'inner-transition'
    body['graph']['nodes'] = [n for n in body['graph']['nodes'] if n['id'] != 'prediction']
    body['graph']['edges'] = [e for e in body['graph']['edges'] if e['to'] != 'prediction']
    next(n for n in body['graph']['nodes'] if n['type'] == 'Recurrent')['sequence_outputs'] = []
    models['inner-transition'] = body
    outer, mid, decl = temporal_fixture()
    outer[mid]['graph']['nodes'][0].update(shape=[4,2,2], axes=['T','D0','D1'])
    scan = outer[mid]['graph']['nodes'][2]
    scan.update(ref='inner-transition', sequence_inputs=[{'input':'sequence','source':'sequence'}],
                states=[{'input':'initial','output':'final','port':'hidden_final','source':'initial'}],
                sequence_outputs=[{'output':'final','port':'trajectory'}])
    models[mid] = outer[mid]
    return models, mid, decl


class TemporalShapeTests(unittest.TestCase):
    def test_proof_tracks_time_and_exact_bindings_without_changing_source_or_rng(self):
        models, mid, decl = temporal_fixture(); before = copy.deepcopy((models, decl)); rng = random.getstate()
        spec = specification(models, mid, decl)
        self.assertEqual(spec['scope'], 'explicit-eager-runtime-temporal-contract')
        self.assertEqual(spec['version'], 1); self.assertEqual(spec['status'], 'conditional')
        self.assertEqual(spec['output_dependencies'], [['sequence.T']])
        guard = next(g for g in spec['guards'] if g['path'].endswith('/scan'))
        self.assertEqual(guard['bindings']['states'][0], {'input':'h', 'output':'next', 'port':'hidden_final'})
        self.assertEqual(guard['sources'], ['sequence', 'initial', 'lengths', 'mask', 'reset'])
        self.assertEqual(guard['controls']['mask']['adapter'], 'exact-bool-v1')
        self.assertEqual((models, decl), before); self.assertEqual(random.getstate(), rng)

    def test_many_actual_times_keep_parameters_and_fixed_returned_state(self):
        for cnn in (False, True):
            models, mid, decl = temporal_fixture(cnn=cnn); spec = specification(models, mid, decl)
            for steps in (1, 2, 3, 7, 19, 101):
                c = shape_contract(models, mid, spec, shapes(models, mid, steps))
                self.assertEqual(c['parameters'], 14 if cnn else 10)
                self.assertEqual(c['shapes']['scan']['ports'], {'trajectory':[steps,2], 'hidden_final':[2]})
                self.assertEqual(c['shapes']['scan']['recurrent']['steps'], steps)
                self.assertEqual(c['shapes']['prediction']['output_axes'], ['T','D0'])

    def test_declarations_are_closed_exact_axis_zero_and_require_semantic_time(self):
        models, mid, decl = temporal_fixture()
        for value in (None, {}, [], {'unknown':[0]}, {'sequence':[]}, {'sequence':[True]},
                      {'sequence':[0.0]}, {'sequence':[0,0]}, {'sequence':[1]}, {'sequence':0},
                      {'sequence':[0], 'initial':[0]}, {'lengths':[0]}):
            with self.subTest(value=value), self.assertRaises(GraphError): declaration(models, mid, value)
        next(n for n in models[mid]['graph']['nodes'] if n['id']=='sequence').pop('axes')
        with self.assertRaisesRegex(GraphError, 'semantic T'): declaration(models, mid, {'sequence':[0]})

    def test_undeclared_control_time_does_not_silently_become_dynamic(self):
        models, mid, _ = temporal_fixture(); spec = specification(models, mid, {'sequence':[0]})
        with self.assertRaisesRegex(GraphError, 'only declared T'): shape_contract(models, mid, spec, shapes(models, mid, 7))
        supplied = shapes(models, mid, 4); supplied['sequence'] = [7,2]
        with self.assertRaisesRegex(GraphError, 'mask.*sample shape'): shape_contract(models, mid, spec, supplied)

    def test_actual_shapes_require_every_input_and_fixed_rank_frame_and_state(self):
        models, mid, decl = temporal_fixture(); spec = specification(models, mid, decl); baseline = shapes(models, mid, 7)
        for value in (None, {}, dict(baseline, unknown=[1]), dict(baseline, sequence=[7,3]),
                      dict(baseline, sequence=[7]), dict(baseline, initial=[3]), dict(baseline, lengths=[7]),
                      dict(baseline, sequence=[0,2]), dict(baseline, sequence=[True,2]),
                      dict(baseline, sequence=[7.0,2]), dict(baseline, sequence=[MAX_INTEGER,2])):
            with self.subTest(value=value), self.assertRaises(GraphError): shape_contract(models, mid, spec, value)

    def test_time_dependent_linear_width_fails_before_construction(self):
        models, mid, decl = temporal_fixture(); g = models[mid]['graph']
        g['nodes'].extend([{'id':'flat', 'type':'Flatten'}, {'id':'bad', 'type':'Linear', 'features':2}])
        e = next(e for e in g['edges'] if e['to']=='prediction'); e['to'] = 'flat'
        g['edges'].extend([{'from':'flat','to':'bad'}, {'from':'bad','to':'prediction'}])
        with patch('app.core.compile_model', side_effect=AssertionError('Do not allocate')):
            with self.assertRaisesRegex(GraphError, '/bad.*Linear.*sequence.T.*Bind T'):
                compile_temporal_model(models, mid, decl, constrained=False)

    def test_last_axis_linear_after_sequence_port_preserves_time_and_fixed_weights(self):
        models, mid, decl = temporal_fixture(); g = models[mid]['graph']
        g['nodes'].append({'id':'head', 'type':'Linear', 'features':3, 'bias':False})
        next(e for e in g['edges'] if e['to']=='prediction')['to'] = 'head'
        g['edges'].append({'from':'head','to':'prediction'})
        spec = specification(models, mid, decl)
        c = shape_contract(models, mid, spec, shapes(models, mid, 11))
        self.assertEqual(c['shapes']['head']['output_shape'], [11,3]); self.assertEqual(c['parameters'],16)

    def test_fixed_output_shape_is_a_conditional_per_call_guard(self):
        models, mid, decl = temporal_fixture()
        next(n for n in models[mid]['graph']['nodes'] if n['id']=='prediction')['shape'] = [4,2]
        spec = specification(models, mid, decl)
        self.assertTrue(any(g['rule']=='explicit-output-shape-equality' for g in spec['guards']))
        with self.assertRaisesRegex(GraphError, 'Output prediction declares'):
            shape_contract(models, mid, spec, shapes(models, mid, 5))

    def test_spatial_policy_and_old_family_grammars_still_refuse_recurrence(self):
        from app.runtime_shapes import specification as spatial_specification
        from app.families import validate_family
        models, mid, _ = temporal_fixture(cnn=True)
        # A valid CHW independent root must not hide recurrence elsewhere.
        g = models[mid]['graph']; g['nodes'].extend([
            {'id':'image','type':'Input','shape':[1,4,5],'axes':['C','H','W']}, {'id':'image-out','type':'Output'}])
        g['input_order'].append('image'); g['edges'].append({'from':'image','to':'image-out'})
        with self.assertRaisesRegex(GraphError,'Runtime-spatial-v1'): spatial_specification(models, mid, {'image':[1,2]})
        f = {'id':'f','name':'f','root':mid,'models':{k:{p:m[p] for p in ('id','name','graph')} for k,m in models.items()},
             'symbols':[],'assignments':[],'repeats':[]}
        for version in range(1,6):
            with self.subTest(version=version),self.assertRaisesRegex(GraphError,'recurrent family grammar'):
                validate_family(dict(f,version=version))

    def test_mutually_exclusive_policy_selection_is_rejected_before_import_or_allocation(self):
        models, mid, decl = temporal_fixture()
        with self.assertRaisesRegex(GraphError,'select exactly one'):
            compile_model(models, mid, False, runtime_inputs={'sequence':[1]}, runtime_temporal=decl)

    def test_current_batch_time_memory_and_exact_budget_boundary(self):
        models, mid, decl = temporal_fixture(cnn=True); spec = specification(models, mid, decl)
        a = shape_contract(models, mid, spec, shapes(models, mid, 2))
        b = shape_contract(models, mid, spec, shapes(models, mid, 17))
        with patch('app.runtime_shapes.available_memory', return_value=10**12):
            small = admission(models, mid, a, 1); large = admission(models, mid, b, 11)
        self.assertGreater(large['memory']['components']['recurrent_unrolls'], small['memory']['components']['recurrent_unrolls'])
        amount = large['memory']['estimated_bytes']
        with patch('app.runtime_shapes.available_memory', return_value=10*amount):
            self.assertTrue(admission(models, mid, b, 11)['allowed'])
        for available in (10*amount-1, None, 0):
            with patch('app.runtime_shapes.available_memory', return_value=available), self.assertRaisesRegex(GraphError,'runtime temporal'):
                admission(models, mid, b, 11)
        with patch('app.runtime_shapes.available_memory', return_value=None):
            self.assertTrue(admission(models, mid, b, 11, constrained=False)['allowed'])
        for kwargs in ({'dtype_bytes':8}, {'dtype_bytes':True}, {'gradients':1}, {'constrained':1}, {'memory_limit_percent':True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): admission(models, mid, a, 1, **kwargs)

    def test_nested_concrete_recurrence_proof_and_memory_charge_outer_times_inner_unroll(self):
        models,mid,decl = nested_fixture(); spec = specification(models,mid,decl)
        self.assertEqual(sum('equal-positive-T' in g['rule'] for g in spec['guards']),2)
        small = shape_contract(models,mid,spec,shapes(models,mid,2))
        large = shape_contract(models,mid,spec,shapes(models,mid,9))
        with patch('app.runtime_shapes.available_memory',return_value=10**12):
            a = admission(models,mid,small,3); b = admission(models,mid,large,3)
        self.assertEqual(small['parameters'],10); self.assertEqual(large['parameters'],10)
        self.assertGreater(b['memory']['components']['recurrent_unrolls'],a['memory']['components']['recurrent_unrolls'])
        # Independent sum: each outer unroll, plus outer T copies of the inner unroll.
        from app.recurrent import recurrent_memory_report
        outer = next(p['recurrent'] for p in large['plan'] if p['type']=='Recurrent')
        child = validate_model(models,'inner-transition')
        inner = next(p['recurrent'] for p in child['plan'] if p['type']=='Recurrent')
        def unroll(value):
            r = recurrent_memory_report(value['contract'],3,value['steps'],False)
            return sum(v for k,v in r['components'].items() if k not in {'runtime_headroom','parameters_optimizer_evidence'})
        self.assertEqual(b['memory']['components']['recurrent_unrolls'],unroll(outer)+9*unroll(inner))

    def test_multi_sequence_state_contract_checks_equal_time_but_preserves_stable_roles(self):
        from test_recurrent_contract import fixture
        from app.recurrent_authoring import prepare, build
        from app.core import default_model
        models,body,_ = fixture(); graph=models[body]['graph']; cloned=copy.deepcopy(graph['nodes'])
        for node in cloned:
            node['id']+='2'
            if 'label' in node: node['label']+=' second'
            if node['type']=='Concat': node['sources']=[s+'2' for s in node['sources']]
        graph['nodes'].extend(cloned)
        graph['edges'].extend({'from':e['from']+'2','to':e['to']+'2'} for e in list(graph['edges']))
        graph['input_order'].extend(['x2','h2'])
        root=default_model();models[root['id']]=root
        token=prepare(models,root['id'],body)
        options={'version':1,'steps':4,'inputs':[
            {'input':nid,'role':'sequence' if nid.startswith('x') else 'state','output':None if nid.startswith('x') else 'next'+('2' if nid.endswith('2') else '')}
            for nid in ('x','h','x2','h2')], 'sequence_outputs':['next','next2'], 'controls':['mask','reset'],'label':'Two streams'}
        doc,scan=build(models,{k:token[k] for k in ('id','ref','root_graph','transition')},options);models[root['id']]=doc
        c=validate_model(models,root['id']); added=next(n for n in doc['graph']['nodes'] if n['id']==scan)
        variable=[r['source'] for r in added['sequence_inputs']]+[r['source'] for r in added['controls'].values()]
        # The concrete authoring form does not opt controls into a temporal domain.
        for node in doc['graph']['nodes']:
            if node['id'] in [r['source'] for r in added['controls'].values()]: node['axes']=['T']
        decl={nid:[0] for nid in variable};spec=specification(models,root['id'],decl)
        actual={g['id']:[7,*g['shape'][1:]] if g['id'] in variable else g['shape'] for g in c['input_groups']}
        current=shape_contract(models,root['id'],spec,actual)
        self.assertEqual(current['shapes'][scan]['ports'][added['states'][1]['port']],[2])
        actual[added['sequence_inputs'][1]['source']][0]=8
        with self.assertRaisesRegex(GraphError,'equal T'):shape_contract(models,root['id'],spec,actual)

    def test_final_state_only_has_fixed_output_dependencies_despite_variable_unroll(self):
        models,mid,decl=temporal_fixture();g=models[mid]['graph']
        g['nodes']=[n for n in g['nodes'] if n['id']!='prediction'];g['edges']=[e for e in g['edges'] if e['to']!='prediction']
        next(n for n in g['nodes'] if n['type']=='Recurrent')['sequence_outputs']=[]
        spec=specification(models,mid,decl);self.assertEqual(spec['output_dependencies'],[[]])
        self.assertEqual(shape_contract(models,mid,spec,shapes(models,mid,13))['output_shape'],[2])

    def test_reshape_can_preserve_time_but_cannot_make_state_dimensions_runtime_dependent(self):
        for role in ('sequence','initial'):
            models,mid,decl=temporal_fixture();g=models[mid]['graph'];scan=g['nodes'][2]
            g['nodes'].append({'id':'adapter','type':'Reshape','shape':[-1,2] if role=='sequence' else [-1]})
            # Keep a valid concrete baseline while a symbolic inferred axis carries T.
            if role=='initial':
                # Reshape removes the source's semantic axes while retaining its
                # symbolic extent; baseline state shape/axes still match exactly.
                source=next(n for n in g['nodes'] if n['id']=='initial');source['axes']=['T'];decl['initial']=[0]
                scan['states'][0]['source']='adapter'
                next(e for e in g['edges'] if e['from']=='initial' and e['to']=='scan')['to']='adapter'
                g['edges'].append({'from':'adapter','to':'scan'})
                with self.assertRaisesRegex(GraphError,'state dimensions.*runtime T'):specification(models,mid,decl)
            else:
                # [-1,2] is conservatively marked T-dependent on its inferred axis
                # only, which remains a valid sequence time rather than frame size.
                scan['sequence_inputs'][0]['source']='adapter'
                next(e for e in g['edges'] if e['from']=='sequence' and e['to']=='scan')['to']='adapter'
                g['edges'].append({'from':'adapter','to':'scan'})
                self.assertEqual(specification(models,mid,decl)['output_dependencies'],[['sequence.T']])

    def test_reshape_cannot_move_runtime_time_into_the_fixed_transition_frame(self):
        models,mid,decl=temporal_fixture(controls=False);g=models[mid]['graph'];scan=g['nodes'][2]
        next(n for n in models['transition']['graph']['nodes'] if n['id']=='x')['features']=4
        g['nodes'].append({'id':'reshape','type':'Reshape','shape':[2,-1]})
        scan['sequence_inputs'][0]['source']='reshape'
        next(e for e in g['edges'] if e['from']=='sequence' and e['to']=='scan')['to']='reshape'
        g['edges'].append({'from':'reshape','to':'scan'})
        self.assertEqual(validate_model(models,mid)['shapes']['scan']['recurrent']['steps'],2)
        with patch('app.core.compile_model',side_effect=AssertionError('No allocation')):
            with self.assertRaisesRegex(GraphError,'transition frame dimensions.*runtime T'):
                compile_temporal_model(models,mid,decl,constrained=False)

    def test_lengths_remain_fixed_even_if_a_source_is_named_semantic_time(self):
        models,mid,decl=temporal_fixture()
        next(n for n in models[mid]['graph']['nodes'] if n['id']=='lengths')['axes']=['T'];decl['lengths']=[0]
        with self.assertRaisesRegex(GraphError,'lengths must retain.*fixed sample shape'):
            specification(models,mid,decl)

    def test_fresh_process_proof_imports_no_torch_numpy_or_qt_and_preserves_rng(self):
        code = '''import json,random,sys
from test_temporal_shapes import temporal_fixture,shapes
from app.temporal_shapes import specification,shape_contract
models,mid,decl=temporal_fixture(); rng=random.getstate()
spec=specification(models,mid,decl); c=shape_contract(models,mid,spec,shapes(models,mid,13))
assert random.getstate()==rng
assert not any(k.startswith(('torch','numpy','PyQt')) for k in sys.modules)
print(json.dumps(c['shapes']['scan']['ports']))
'''
        result = subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout), {'trajectory':[13,2], 'hidden_final':[2]})

    def test_portable_source_captures_exact_reachable_domain_without_allocation(self):
        from app.core import default_model
        from app.evidence import digest
        models,mid,decl=temporal_fixture();models['unrelated']=dict(default_model('not-exported-secret'),id='unrelated')
        before=copy.deepcopy(models);rng=random.getstate()
        source=temporal_source(models,mid,decl,constrained=False);namespace={};exec(source['code'],namespace)
        self.assertEqual(source['code_bytes'],len(source['code'].encode('utf-8')))
        self.assertEqual(source['provenance']['policy'],'runtime-temporal-v1')
        self.assertIn('temporal_shapes',source['provenance']['runtime_files'])
        frozen=namespace['model_definition']();self.assertEqual(set(frozen['models']),{mid,'transition'})
        self.assertNotIn('not-exported-secret',source['code']);self.assertEqual(digest(frozen),source['provenance']['model_sha256'])
        self.assertEqual(namespace['runtime_specification']()['runtime_inputs'],decl)
        self.assertEqual(models,before);self.assertEqual(random.getstate(),rng)

    def test_portable_source_tampered_model_policy_or_runtime_bytes_fail_closed(self):
        models,mid,decl=temporal_fixture();source=temporal_source(models,mid,decl,constrained=False)
        for which in ('model','policy','runtime'):
            namespace={};exec(source['code'],namespace)
            if which=='model':namespace['_MODEL']['models'][mid]['graph']['nodes'][0]['shape']=[5,2]
            elif which=='policy':namespace['_EXPORT']['runtime_inputs']['sequence']=[1]
            else:namespace['_SOURCES']['temporal_shapes']+='\n# changed bytes'
            with self.subTest(which=which),self.assertRaises(ValueError):namespace['runtime_specification']()
