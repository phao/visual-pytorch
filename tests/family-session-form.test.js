'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture.js');
const h=(sequence=0)=>({scope:'owned-authoring-session',version:1,kind:'family',id:'family',session:'owned',sequence,workspace_id:'one',base:null,name:'Family'});
const ref={kind:'family',id:'family',revision:'a'.repeat(64)};
const descriptor=(key,value,path)=>({key,path,present:true,text:JSON.stringify(value),summary:null});
function page(r,changes={}){
  const values={templates:{id:'seed',name:'Seed',graph:{}},symbols:{id:'width',scope:'seed',name:'width',default:3,minimum:1},
    assignments:{model:'seed',node:'linear1',field:'features',value:{symbol:'width'}},repeats:{id:'repeat',model:'seed',nodes:['linear1'],count:2,mode:'independent'}}[r.args.collection];
  return {version:1,scope:'owned-family-page',session:'owned',sequence:r.args.sequence,workspace_id:'one',id:'family',
    collection:r.args.collection,offset:r.args.offset,limit:6,total:8,next_offset:r.args.offset+1,path:[2],
    roots:[descriptor('name','Family',[1]),descriptor('root','seed',[2])],
    rows:[{ordinal:r.args.offset,template:r.args.collection==='templates'?'seed':null,path:[3,r.args.offset],
      properties:Object.entries(values).map(([key,value],i)=>descriptor(key,value,[3,r.args.offset,i]))}],...changes};
}
function fixture(){
  const body=new Element('body'),nodes=new Map(),get=id=>{
    if(!nodes.has(id)){const n=new Element();n.addEventListener=(type,fn)=>{const old=n['on'+type];n['on'+type]=e=>{old?.(e);fn(e);};};nodes.set(id,n);body.append(n);}return nodes.get(id);};
  const editor=get('science-family-editor');editor.append(get('science-family-roots'),get('science-family-cards'));
  get('science-kind').value='family';get('science-object').value=JSON.stringify(ref);get('science-family-bindings').value='{}';get('science-family-instance-bindings').value='{}';
  const root={getElementById:get,createElement:tag=>new Element(tag),activeElement:null},requests=[],toasts=[],saved=[],staged=[];
  const state={workspace:'one',busy:false,other:false,value:false};
  const window={LoomScienceAuthor:{active:()=>state.other||Boolean(window.LoomFamilySessionForm?.active()),legacyActive:()=>state.other,stateChanged(){}},
    LoomDraftValueEditor:{active:()=>state.value,openOwned:async target=>staged.push(target)}};
  const api={model:()=> 'seed',workspace:()=>state.workspace,busy:()=>state.busy,toast:(...v)=>toasts.push(v),
    call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/family-session-form.js','utf8'),{window,Event:class{constructor(type){this.type=type;}},crypto:{randomUUID:()=> 'new-symbol'}});
  const form=window.LoomFamilySessionForm;form.mount(root,api,async r=>saved.push(r));
  const field=(collection,key)=>get('science-family-cards').querySelectorAll('[data-family-card]').find(n=>n.dataset.familyCard===collection)?.querySelectorAll('[data-key]').find(n=>n.dataset.key===key);
  return {form,root,get,field,state,requests,toasts,saved,staged,click:id=>get(id).onclick()};
}
async function opened(f){
  const done=f.form.open();await tick();assert.equal(f.requests[0].action,'scienceDraftOpen');f.requests[0].resolve(h());await tick();
  f.requests.slice(1,5).forEach(r=>r.resolve(page(r)));await done;return f;
}
function edit(n,value,type='change'){n.value=value;n.dispatchEvent({type});}

test('new family opens despite its own pending flag and uses a concrete seed',async()=>{
  const f=await opened(fixture());assert.equal(f.requests[0].args.model,'seed');assert.equal(f.form.active(),true);
  assert.equal(f.get('science-family-check').disabled,false);assert.equal(f.requests.length,5);
  assert.ok(f.requests.slice(1).every(r=>r.args.limit===6&&!('draft' in r.args)));
});
test('focused exact integer text is flushed before validation without Number coercion',async()=>{
  const f=await opened(fixture()),n=f.field('symbols','default');edit(n,'9007199254740993','input');f.root.activeElement=n;
  const done=f.click('science-family-check');await tick();assert.equal(f.requests[5].args.operations[0].text,'9007199254740993');
  f.requests[5].resolve(h(1));await tick();assert.equal(f.requests[6].args.sequence,1);f.requests[6].resolve(ref);await done;
});
test('invalid expression text stays visible and blocks all admission requests',async()=>{
  const f=await opened(fixture()),n=f.field('assignments','value');edit(n,'not JSON');
  await f.click('science-family-save');assert.equal(f.requests.length,5);assert.equal(n.value,'not JSON');assert.ok('invalid' in n.dataset);
});
test('wrong edit sequence acknowledgement keeps queued text and forbids Check',async()=>{
  const f=await opened(fixture());edit(f.field('symbols','default'),'4');const done=f.click('science-family-check');await tick();
  f.requests[5].resolve(h(9));await done;assert.equal(f.requests.length,6);assert.match(f.toasts.at(-1)[0],/acknowledgement/);
});
test('workspace switches invalidate late preview results and retain visible draft',async()=>{
  const f=await opened(fixture()),done=f.click('science-family-preview');await tick();
  f.state.workspace='two';f.form.stateChanged();f.requests[5].resolve({version:1,scope:'owned-family-preview',session:'owned',sequence:0,id:'family',workspace_id:'one',status:'verified',parameters:21});
  await done;assert.equal(f.get('science-family-preview-result').textContent,'');assert.equal(f.get('science-family-save').disabled,true);assert.equal(f.form.active(),true);
});
test('retention closes only the acknowledged exact family draft',async()=>{
  const f=await opened(fixture()),done=f.click('science-family-save');await tick();f.requests[5].resolve(ref);await done;
  assert.equal(f.form.active(),false);assert.equal(f.saved.length,1);assert.equal(f.saved[0].revision,ref.revision);
});
test('staged graph editing retains parent sequence and complete identity',async()=>{
  const f=await opened(fixture());await f.click('science-family-raw');
  assert.equal(f.staged[0].header.session,'owned');assert.deepEqual(Array.from(f.staged[0].path),[]);
  const callback=f.staged[0].onClosed(h(1));await tick();f.requests.slice(5,9).forEach(r=>r.resolve(page(r)));
  await callback;assert.ok(f.requests.slice(5,9).every(r=>r.args.sequence===1));
});
test('creating a retained instance pins exact family and leaves materialization explicit',async()=>{
  const f=fixture();f.get('science-family-materialize').checked=false;const done=f.click('science-family-create');await tick();
  assert.equal(f.requests[0].action,'scienceFamilyResolve');assert.equal(f.requests[0].args.family.revision,ref.revision);assert.equal(f.requests[0].args.materialize,false);
  f.requests[0].resolve({specialization:{revision:'b'.repeat(64)},parameters:21,materialized:false});await done;
  assert.match(f.get('science-family-instance-status').textContent,/registry only/);
});
test('external reload with same workspace UUID prevents further draft mutation',async()=>{
  const f=await opened(fixture());f.form.event({kind:'externalReload'});assert.equal(f.get('science-family-save').disabled,true);assert.equal(f.form.active(),true);
});
test('another draft blocks opening without a native session request',async()=>{
  const f=fixture();f.state.other=true;await f.form.open();assert.equal(f.requests.length,0);assert.match(f.toasts[0][0],/current draft/);
});
test('failed card read preserves a raw repair path while typed admission remains disabled',async()=>{
  const f=fixture(),done=f.form.open();await tick();f.requests[0].resolve(h());await tick();
  f.requests.slice(1,5).forEach(r=>r.args.collection==='symbols'?r.reject(Error('temporary read failure')):r.resolve(page(r)));
  await done;assert.equal(f.form.active(),true);assert.equal(f.get('science-family-save').disabled,true);
  assert.equal(f.get('science-family-raw').disabled,false);assert.equal(f.get('science-family-retry').disabled,false);
  assert.equal(f.field('assignments','node').disabled,true);
});

test('tied repeat mode explicitly upgrades family grammar before mode edit',async()=>{
  const f=await opened(fixture()),n=f.field('repeats','mode');assert.equal(n.tagName,'select');
  edit(n,'tied');const done=f.click('science-family-check');await tick();
  const operations=f.requests[5].args.operations;
  assert.deepEqual(JSON.parse(JSON.stringify(operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'2'},
    {op:'property',target:{kind:'entry',collection:'repeats',ordinal:0},key:'mode',text:'"tied"'}]);
  f.requests[5].resolve(h(1));await tick();f.requests[6].resolve(ref);await done;
});

test('independent repeat mode does not infer a downgrade or share equal values',async()=>{
  const f=await opened(fixture());edit(f.field('repeats','mode'),'independent');
  const done=f.click('science-family-check');await tick();
  assert.equal(f.requests[5].args.operations.length,1);assert.equal(f.requests[5].args.operations[0].key,'mode');
  f.requests[5].resolve(h(1));await tick();f.requests[6].resolve(ref);await done;
});
