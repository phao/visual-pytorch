"""Exact counts and additive layout profiles without allocating tensors."""
import copy,json,tempfile,unittest,subprocess
from pathlib import Path
from app.core import Workspace,default_model,default_mlp,validate_model
from app.presentation import LAYOUT_DEFAULTS,validate_layout,block_parameter_counts
from app.evidence import identity

class ParameterLayoutTests(unittest.TestCase):
    def counts(self,models,mid):
        c=validate_model(models,mid)
        refs={n['id']:validate_model(models,n['ref']) for n in models[mid]['graph']['nodes'] if n['type']=='Module'}
        result=block_parameter_counts(models[mid],c,refs)
        self.assertEqual(sum(map(int,result.values())),c['parameters']);return result

    def test_affine_and_multidimensional_share_last_axis_counts(self):
        d=default_model(inputs=3,outputs=4);d['graph']['nodes'][0]['shape']=[10,3]
        self.assertEqual(self.counts({d['id']:d},d['id']),{'input':'0','linear1':'16','output':'0'})
        d['graph']['nodes'][1]['bias']=False
        self.assertEqual(self.counts({d['id']:d},d['id'])['linear1'],'12')

    def test_mlp_activations_are_zero_and_independent_layers_count_bias(self):
        d=default_mlp(inputs=2,outputs=1,hidden=[3,5],activation='Softmax')
        result=self.counts({d['id']:d},d['id'])
        self.assertEqual(sorted(int(result[n['id']]) for n in d['graph']['nodes'] if n['type']=='Linear'),[6,9,20])
        self.assertTrue(all(result[n['id']]=='0' for n in d['graph']['nodes'] if n['type']!='Linear'))

    def test_grouped_convolution_includes_only_learnable_kernel_and_optional_bias(self):
        d=default_model();d['graph']['nodes']=[{'id':'i','type':'Input','shape':[4,8,8]},
          {'id':'c','type':'Conv2d','out_channels':6,'kernel_size':3,'padding':1,'groups':2,'bias':True},
          {'id':'o','type':'Output'}];d['graph']['edges']=[{'from':'i','to':'c'},{'from':'c','to':'o'}]
        self.assertEqual(self.counts({d['id']:d},d['id']),{'i':'0','c':'114','o':'0'})
        d['graph']['nodes'][1]['bias']=False
        self.assertEqual(self.counts({d['id']:d},d['id'])['c'],'108')

    def test_each_reusable_module_instance_includes_nested_parameters(self):
        ref=default_model(inputs=2,outputs=2);outer=default_model(inputs=2,outputs=2)
        outer['graph']['nodes'][1].update(type='Module',ref=ref['id'])
        outer['graph']['nodes'].insert(2,{'id':'m2','type':'Module','ref':ref['id']})
        outer['graph']['edges']=[{'from':'input','to':'linear1'},{'from':'linear1','to':'m2'},{'from':'m2','to':'output'}]
        self.assertEqual(self.counts({ref['id']:ref,outer['id']:outer},outer['id']),{'input':'0','linear1':'6','m2':'6','output':'0'})

    def test_optional_expanded_boolean_roundtrip_history_identity_and_legacy_read(self):
        w=Workspace();self.addCleanup(w.db.close);self.addCleanup(lambda:w.history.directory.cleanup() if w.history.directory else None)
        d=default_model();w.put(d);before=identity(w.models(),d['id']);value=copy.deepcopy(LAYOUT_DEFAULTS)
        legacy=copy.deepcopy(value);legacy['expanded'].pop('parameter_sizing')
        self.assertEqual(validate_layout(legacy),legacy)
        w.set_layout_preferences(legacy)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.vtw';w.save(path);raw=path.read_bytes();old=Workspace.load(path)
            try:self.assertEqual(old.layout_preferences,legacy);self.assertEqual(path.read_bytes(),raw)
            finally:old.db.close()
            snap=w.history.capture(w.db);value['expanded']['parameter_sizing']=True
            w.set_layout_preferences(value);w.history.record(snap,w.db);w.save(path)
            loaded=Workspace.load(path)
            try:self.assertTrue(loaded.layout_preferences['expanded']['parameter_sizing']);self.assertEqual(identity(loaded.models(),d['id']),before)
            finally:loaded.db.close()
            self.assertTrue(w.history.restore(w));self.assertEqual(w.layout_preferences,legacy)
            self.assertTrue(w.history.restore(w,redo=True));self.assertEqual(w.layout_preferences,value)
        for bad in (0,1,'true',None,[],{}):
            wrong=copy.deepcopy(value);wrong['expanded']['parameter_sizing']=bad
            with self.assertRaises(ValueError):validate_layout(wrong)
        wrong=copy.deepcopy(value);wrong['collapsed']['parameter_sizing']=True
        with self.assertRaises(ValueError):validate_layout(wrong)

    def test_inconsistent_or_unknown_operator_counts_are_never_silently_guessed(self):
        d=default_model();c=validate_model({d['id']:d},d['id']);c['parameters']+=1
        with self.assertRaisesRegex(ValueError,'disagree'):block_parameter_counts(d,c,{})
        d['graph']['nodes'][1]['type']='FutureOperator'
        with self.assertRaisesRegex(ValueError,'not implemented'):block_parameter_counts(d,c,{})

    def test_actual_js_layout_preserves_pytorch_outputs_weights_and_gradients(self):
        import torch
        from app.core import compile_model,generate_code
        torch.set_num_threads(1)
        d=default_mlp(inputs=2,outputs=1,hidden=[3,5]);models={d['id']:d};counts=self.counts(models,d['id'])
        script="const fs=require('fs'),L=require('./app/web/layout.js'),G=require('./app/web/graph-tools.js');const v=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(L.apply(v.doc,false,{...L.defaults().expanded,parameter_sizing:true},G,v.counts)));"
        run=subprocess.run(['node','-e',script],input=json.dumps({'doc':d,'counts':counts}),capture_output=True,text=True,
                           cwd=Path(__file__).resolve().parents[1],timeout=10,check=True)
        moved=json.loads(run.stdout);after={d['id']:moved}
        self.assertEqual(identity(models,d['id']),identity(after,d['id']))
        self.assertEqual(generate_code(models,d['id']),generate_code(after,d['id']))
        torch.manual_seed(17);a=compile_model(models,d['id'],constrained=False)
        b=compile_model(after,d['id'],constrained=False);b.load_state_dict(a.state_dict())
        x=torch.randn(4,2);ya=a(x);yb=b(x);self.assertTrue(torch.equal(ya,yb))
        ya.square().sum().backward();yb.square().sum().backward()
        for pa,pb in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.equal(pa,pb));self.assertTrue(torch.equal(pa.grad,pb.grad))

    def test_cli_exact_counts_are_readonly_and_use_the_same_contract(self):
        import io,vpw_cli
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.vtw';w=Workspace();d=default_mlp(inputs=2,outputs=1,hidden=[3,5]);w.put(d);w.save(path)
            expected=self.counts(w.models(),d['id']);w.db.close();raw=path.read_bytes();output=io.StringIO()
            with redirect_stdout(output):code=vpw_cli.main([str(path),'parameter-counts',d['id']])
            self.assertEqual(code,0);self.assertEqual(json.loads(output.getvalue())['value'],{'id':d['id'],'counts':expected})
            self.assertEqual(path.read_bytes(),raw)
