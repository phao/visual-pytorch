const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');
const P=require('../app/web/parameter-policy.js');
test('initializer browsing is alphabetical and the grammar matches the backend',()=>{
  assert.deepEqual(P.kinds,[...P.kinds].sort());assert.deepEqual(P.parse('zeros',{}),{kind:'zeros'});
  assert.deepEqual(P.parse('constructor',{}),{kind:'constructor'});
  assert.deepEqual(P.parse('normal',{seed:'7',mean:'0',std:'.05'}),{kind:'normal',seed:7,mean:0,std:.05});
  assert.deepEqual(P.parse('values',{values:'[[1,2],[3,4]]'}),{kind:'values',dtype:'float32',values:[[1,2],[3,4]]});
});
test('empty, nonfinite and invalid random controls refuse',()=>{
  for(const fields of [{seed:'',mean:'0',std:'1'},{seed:'-1',mean:'0',std:'1'},{seed:'1.5',mean:'0',std:'1'},
    {seed:'7',mean:'Infinity',std:'1'},{seed:'7',mean:'0',std:'0'}])assert.throws(()=>P.parse('normal',fields));
  assert.throws(()=>P.parse('uniform',{seed:'7',low:'1',high:'1'}));assert.throws(()=>P.parse('values',{values:'1'}));
  assert.throws(()=>P.parse('eval',{}));
});
test('actual editor callbacks serialize delayed policy acknowledgments and preserve concurrent moves',async()=>{
  const source=fs.readFileSync(require.resolve('../app/web/editor.js'),'utf8');
  const persist=source.slice(source.indexOf('  function persist()'),source.indexOf('  function svgElement'));
  const edit=source.slice(source.indexOf('        function policyEdit('),source.indexOf('        window.LoomParameterPolicy.mount'));
  for(const failedStateRead of [false,true]){
    const clone=x=>JSON.parse(JSON.stringify(x));let release;
    const original={id:'m',graph:{nodes:[{id:'linear1',x:0}],edges:[]},view:{x:0,y:0,zoom:1}};
    let server=clone(original);const updates=[];
    const context={doc:clone(original),state:{models:{m:clone(original)},panel_scope:'scope'},syncQueue:Promise.resolve(),
      mid:'m',nid:'linear1',selected:'linear1',policyRequest:1,policyGeneration:1,policyMutation:null,
      context:{panel_scope:'scope'},info:{architecture:'exact'},clone,window:{LoomParameterPolicy:P},
      busy:()=>false,header:()=>{},renderInspector:()=>{},showDescription:()=>{},description:null,
      call:async(action)=>{
        if(action==='editParameterPolicy'){
          server.graph.parameter_policy={version:1,parameters:{'["linear1","weight"]':{trainable:false}}};
          const answer={document:clone(server)};await new Promise(resolve=>{release=resolve;});return answer;
        }
        if(action==='state'){if(failedStateRead)throw Error('State transport failed');return {models:{m:clone(server)},panel_scope:'scope'};}
        throw Error(action);
      },
      safe:async(action,args)=>{assert.equal(action,'updateModel');server=clone(args.document);updates.push(clone(server));return {valid:true};},
      hydrate:value=>{context.state=value;context.doc=clone(value.models.m);},
    };
    vm.createContext(context);vm.runInContext(persist+'\n'+edit,context);
    const operation=context.policyEdit('editParameterPolicy',{selector:'["linear1","weight"]',changes_json:'{"trainable":false}'});
    // Ensure the scientific edit is accepted, but its acknowledgment is held.
    await new Promise(resolve=>setImmediate(resolve));assert.equal(typeof release,'function');
    context.doc.graph.nodes[0].x=99;const pendingSave=context.persist();
    assert.equal(updates.length,0);release();
    if(failedStateRead)await assert.rejects(operation,/transport/);else await operation;
    await pendingSave;
    assert.equal(server.graph.nodes[0].x,99);
    assert.equal(server.graph.parameter_policy.parameters['["linear1","weight"]'].trainable,false);
    assert.equal(updates.length,1);assert.equal(context.policyMutation,null);
  }
});
