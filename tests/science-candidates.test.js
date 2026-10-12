'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture');
const ref=id=>({kind:'experiment',id,revision:'a'.repeat(64)});
const row=i=>({id:'candidate-'+i,status:'compatible',coordinates:{model:{kind:'model',id:'model',revision:'b'.repeat(64)},data_task:'pair',split:'row',initialization:7,protocol:'adam'},counts:{train:400,validation:100,test:100,excluded:0},reason:null});
function fixture(){
  const nodes=new Map(),requests=[],toasts=[],state={workspace:'w1',busy:false,blocked:false},window={};
  const document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},createElement:tag=>new Element(tag)};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...args)=>toasts.push(args),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/science-candidates.js','utf8'),{window,Promise,JSON,Error});window.LoomScienceCandidates.mount(document,api,()=>state.blocked);
  return {nodes,requests,toasts,state,window,controller:window.LoomScienceCandidates};
}
const reply=(r,changes={})=>({version:1,scope:'stored-experiment-preview',workspace_id:r.args.workspace_id,experiment:r.args.reference,offset:r.args.offset,limit:25,total_combinations_exact:'9007199254740993',execution:'no-fitting-or-training',candidates:Array.from({length:7},(_,i)=>row(i+r.args.offset)),next_offset:r.args.offset===0?7:null,...changes});
const buttons=f=>f.nodes.get('science-candidates').querySelectorAll('[data-science-run]');
async function initial(f){f.controller.show(ref('first'));await tick();f.requests[0].resolve(reply(f.requests[0]));await tick();}
test('actual stored browser uses bounded preview, literal decimal totals and role counts',async()=>{
  const f=fixture();await initial(f);assert.equal(f.requests[0].action,'sciencePreviewPage');assert.equal(f.requests[0].args.workspace_id,'w1');assert.equal(f.requests[0].args.limit,25);
  assert.match(f.nodes.get('science-page').textContent,/9007199254740993/);assert.match(f.nodes.get('science-candidates').children[0].children[1].textContent,/train: 400/);assert.equal(buttons(f).length,7);
});
test('actual shortened cursors and pending guards, failed Back and Retry preserve history',async()=>{
  const f=fixture();await initial(f);const n=f.nodes;n.get('science-next').onclick();n.get('science-next').onclick();await tick();assert.equal(f.requests.length,2);assert.equal(f.requests[1].args.offset,7);assert.equal(n.get('science-next').disabled,true);
  f.requests[1].resolve(reply(f.requests[1]));await tick();n.get('science-previous').onclick();await tick();f.requests[2].reject(Error('Failed Back'));await tick();assert.equal(n.get('science-previous').disabled,false);
  n.get('science-candidate-refresh').onclick();await tick();assert.equal(f.requests[3].args.offset,7);f.requests[3].resolve(reply(f.requests[3]));await tick();n.get('science-previous').onclick();await tick();assert.equal(f.requests[4].args.offset,0);
  f.requests[4].resolve(reply(f.requests[4]));await tick();assert.equal(n.get('science-previous').disabled,true);
});
test('failed Next does not advance cursor or create false Back history',async()=>{
  const f=fixture();await initial(f);f.nodes.get('science-next').onclick();await tick();f.requests[1].reject(Error('Failed Next'));await tick();assert.equal(f.nodes.get('science-previous').disabled,true);
  f.nodes.get('science-candidate-refresh').onclick();await tick();assert.equal(f.requests[2].args.offset,0);
});
test('wrong scope, exact reference, workspace, offset, limit and execution cannot install candidates',async()=>{
  for(const changes of [{version:2},{scope:'full-preview'},{experiment:ref('foreign')},{workspace_id:'foreign'},{offset:1},{limit:100},{execution:'training'},{total_combinations_exact:9007199254740992},{next_offset:25}]){
    const f=fixture();f.controller.show(ref('first'));await tick();f.requests[0].resolve(reply(f.requests[0],changes));await tick();assert.equal(buttons(f).length,0);assert.equal(f.toasts.length,1);assert.equal(f.nodes.get('science-next').disabled,true);
  }
});
test('new selections fence old pages, errors and navigation continuations',async()=>{
  for(const fail of [true,false]){const f=fixture();await initial(f);f.nodes.get('science-next').onclick();await tick();const old=f.requests[1];f.controller.show(ref('new'));await tick();f.requests[2].resolve(reply(f.requests[2]));await tick();
    if(fail)old.reject(Error('obsolete error'));else old.resolve(reply(old));await tick();assert.equal(f.toasts.length,0);assert.equal(f.nodes.get('science-previous').disabled,true);f.nodes.get('science-next').onclick();await tick();assert.equal(f.requests[3].args.offset,7);assert.equal(f.requests[3].args.reference.id,'new');}
});
test('workspace replacement and explicit same-UUID reload clear fence pending replies',async()=>{
  for(const reload of [false,true]){const f=fixture();f.controller.show(ref('old'));await tick();if(reload)f.controller.clear();else f.state.workspace='w2';f.requests[0].reject(Error('obsolete'));await tick();assert.equal(f.toasts.length,0);assert.equal(buttons(f).length,0);}
});
test('partial reasons remain literal text and non-compatible rows cannot run',async()=>{
  const f=fixture();f.controller.show(ref('first'));await tick();const r=row(0);r.status='excluded';r.counts=null;r.reason={text:'<script>literal</script>',next_offset:128};const bad={...row(1),status:'incompatible'};
  f.requests[0].resolve(reply(f.requests[0],{candidates:[r,bad],next_offset:null}));await tick();assert.match(f.nodes.get('science-candidates').children[0].children[1].textContent,/<script>literal<\/script>.*partial reason/);
  assert.ok(buttons(f).every(b=>b.disabled));buttons(f)[0].onclick();buttons(f)[1].onclick();await tick();assert.equal(f.requests.length,1);
});
test('explicit run sends only exact retained selection and workspace, duplicate/busy/blocked guarded',async()=>{
  const f=fixture();await initial(f);const b=buttons(f)[0];for(const key of ['busy','blocked']){f.state[key]=true;f.controller.stateChanged();assert.equal(b.disabled,true);b.onclick();await tick();assert.equal(f.requests.length,1);f.state[key]=false;}
  b.onclick();b.onclick();await tick();assert.equal(f.requests.length,2);const run=f.requests[1];assert.equal(run.action,'scienceTrain');assert.equal(run.args.workspace_id,'w1');assert.equal(JSON.stringify(run.args.reference),JSON.stringify(ref('first')));assert.equal('manifest' in run.args,false);
  assert.ok(buttons(f).every(b=>b.disabled));run.resolve({});await tick();assert.equal(f.toasts.length,1);assert.equal(b.disabled,false);
});
test('detached old row buttons and foreign run acknowledgements cannot affect new selection',async()=>{
  const f=fixture();await initial(f);const old=buttons(f)[0];old.onclick();await tick();const run=f.requests[1];f.controller.show(ref('new'));await tick();f.requests[2].resolve(reply(f.requests[2]));await tick();run.reject(Error('old run'));await tick();old.onclick();await tick();assert.equal(f.toasts.length,0);assert.equal(f.requests.length,3);assert.equal(buttons(f)[0].disabled,false);
});
test('an unsafe unrelated cursor cannot be followed as a rounded request',async()=>{
  const f=fixture();f.controller.show(ref('edge'));await tick();const req=f.requests[0];
  req.resolve(reply(req,{next_offset:9007199254740992}));await tick();assert.equal(f.nodes.get('science-next').disabled,true);
  assert.equal(f.toasts.length,1);f.nodes.get('science-next').onclick();await tick();assert.equal(f.requests.length,1);
});
test('family candidates display explicit bindings and preserve the exact run coordinate',async()=>{
  const f=fixture();f.controller.show(ref('families'));await tick();const r=row(0);
  r.coordinates.model={id:'width-depth',bindings:{width:4,depth:3}};r.parameters_exact='9007199254740993';
  f.requests[0].resolve(reply(f.requests[0],{candidates:[r],next_offset:null}));await tick();
  const card=f.nodes.get('science-candidates').children[0];assert.match(card.children[0].textContent,/width=4.*depth=3/);
  assert.equal(card.children[1].textContent,'9007199254740993 trainable parameters');buttons(f)[0].onclick();await tick();
  assert.equal(JSON.stringify(f.requests[1].args.coordinates),JSON.stringify(r.coordinates));f.requests[1].resolve({});await tick();
});
test('conditional family point has no run action despite an explicit coordinate',async()=>{
  const f=fixture();f.controller.show(ref('families'));await tick();const r=row(0);r.status='conditional';r.counts=null;
  r.coordinates.model={id:'unbound-family',bindings:{width:4}};r.reason={text:'Bind missing depth',next_offset:null};
  f.requests[0].resolve(reply(f.requests[0],{candidates:[r],next_offset:null}));await tick();assert.equal(buttons(f)[0].disabled,true);
  buttons(f)[0].onclick();await tick();assert.equal(f.requests.length,1);
});
