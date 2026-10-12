'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture.js');
const h=(sequence=0)=>({scope:'owned-authoring-session',version:1,kind:'family',id:'family',session:'owned',sequence,workspace_id:'one',base:null,name:'Family'});
const ref={kind:'family',id:'family',revision:'a'.repeat(64)};
const descriptor=(key,value,path)=>({key,path,present:true,text:JSON.stringify(value),summary:null});
function page(r,changes={}){
  const values={templates:{id:'seed',name:'Seed',graph:{}},symbols:{id:'width',scope:'seed',name:'width',default:3,minimum:1},
    assignments:{model:'seed',node:'linear1',field:'features',value:{symbol:'width'}},repeats:{id:'repeat',model:'seed',nodes:['linear1'],count:2,mode:'independent'},constructors:{model:'seed',node:'module',symbol:'child-width',value:{symbol:'width'}}}[r.args.collection];
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
async function opened(f,changes={}){
  const done=f.form.open();await tick();assert.equal(f.requests[0].action,'scienceDraftOpen');f.requests[0].resolve(h());await tick();
  f.requests.slice(1,6).forEach(r=>r.resolve(page(r,changes[r.args.collection]||{})));await done;return f;
}
function edit(n,value,type='change'){n.value=value;n.dispatchEvent({type});}

test('an omitted scalar type displays integer without silently inserting a draft field',async()=>{
  const f=await opened(fixture(),{symbols:{rows:[{ordinal:0,path:[3,0],properties:[
    descriptor('id','width',[3,0,0]),{key:'type',present:false,text:null,path:null,summary:null}]}]}});
  assert.equal(f.field('symbols','type').value,'integer');
  assert.equal(f.requests.length,6);
});

async function runtimeOpened(version=4){
  const roots=[descriptor('version',version,[0]),{key:'runtime_inputs',present:false,text:null,path:null,summary:null}];
  return opened(fixture(),{templates:{roots}});
}
const runtimeField=f=>f.get('science-family-roots').querySelectorAll('[data-key]').find(n=>n.dataset.key==='runtime_inputs');
async function temporalOpened(version=4){
  return opened(fixture(),{templates:{roots:[descriptor('version',version,[0]),
    descriptor('runtime_policy',null,[1]),{key:'runtime_inputs',present:false,text:null,path:null,summary:null}]}});
}
const temporalField=f=>f.get('science-family-roots').querySelectorAll('[data-key]').find(n=>n.dataset.key==='runtime_policy');
test('temporal-v2 policy explicitly upgrades version-seven before exact policy edit',async()=>{
  const f=await temporalOpened(7),n=temporalField(f);
  const text='{"version":2,"kind":"runtime-temporal-v2","inputs":{"sequence":[0]}}';
  edit(n,text);const done=f.click('science-family-save');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'8'},
    {op:'property',target:{kind:'document'},key:'runtime_policy',text}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});
test('temporal-v2 preserves its exact unsupported version for backend refusal',async()=>{
  const f=await temporalOpened(8),n=temporalField(f);
  const text='{"version":1,"kind":"runtime-temporal-v2","inputs":{"sequence":[0]}}';
  edit(n,text);const done=f.click('science-family-save');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[{op:'property',target:{kind:'document'},key:'runtime_policy',text}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].reject(Error('Wrong runtime-v2 version'));
  await done;assert.equal(n.value,text);assert.equal(f.saved.length,0);
});
test('temporal JSON queues explicit version-six acknowledgement before policy and keeps stable IDs',async()=>{
  const f=await temporalOpened(),n=temporalField(f);assert.equal(n.tagName,'textarea');
  const text='{"version":1,"kind":"runtime-temporal-v1","inputs":{"sequence":[0],"mask":[0]}}';edit(n,text);
  const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'6'},
    {op:'property',target:{kind:'document'},key:'runtime_policy',text}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
  assert.equal(f.get('science-family-roots').querySelector('[data-key="version"]').value,'6');
});
test('blank temporal policy retains required explicit null rather than removing grammar field',async()=>{
  const f=await temporalOpened(6);edit(temporalField(f),'');const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'runtime_policy',text:'null'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});
test('fixed LSTM family7 policy edits preserve its newer grammar acknowledgement',async()=>{
  const f=await temporalOpened(7),n=temporalField(f);
  assert.ok(n.title.includes('required for LSTM in family7'));
  edit(n,'');const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'runtime_policy',text:'null'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
  assert.equal(f.get('science-family-roots').querySelector('[data-key="version"]').value,'7');
});
test('invalid temporal text stays editable and blocks preview or retention',async()=>{
  const f=await temporalOpened(6),n=temporalField(f);edit(n,'{not json');
  await f.click('science-family-preview');await f.click('science-family-save');
  assert.equal(f.requests.length,6);assert.equal(n.value,'{not json');assert.ok('invalid' in n.dataset);
});
test('temporal editing never silently removes an older spatial declaration',async()=>{
  const f=await temporalOpened(5);edit(temporalField(f),'{"version":1,"kind":"runtime-temporal-v1","inputs":{"sequence":[0]}}');
  const done=f.click('science-family-preview');await tick();
  assert.equal(f.requests[6].args.operations.length,2);
  assert.ok(f.requests[6].args.operations.every(o=>o.key!=='runtime_inputs'));
  f.requests[6].resolve(h(1));await tick();f.requests[7].reject(Error('Remove old spatial declaration explicitly'));await done;
  assert.ok(f.toasts.some(t=>String(t[0]).includes('explicitly')));
});
test('runtime declaration queues explicit grammar upgrade before stable Input ID JSON',async()=>{
  const f=await runtimeOpened(),n=runtimeField(f);assert.equal(n.tagName,'textarea');
  edit(n,'{"input":[1,2]}','input');f.root.activeElement=n;
  const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'5'},
    {op:'property',target:{kind:'document'},key:'runtime_inputs',text:'{"input":[1,2]}'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
  assert.equal(f.get('science-family-roots').querySelector('[data-key="version"]').value,'5');
  edit(f.field('repeats','mode'),'tied');const again=f.click('science-family-check');await tick();
  assert.ok(f.requests[8].args.operations.every(o=>o.key!=='version'));
  f.requests[8].resolve(h(2));await tick();f.requests[9].resolve(ref);await again;
});
test('blank runtime declaration removes optional field without downgrading version five',async()=>{
  const f=await runtimeOpened(5);edit(runtimeField(f),'');const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'runtime_inputs',text:null}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});
test('invalid runtime JSON stays visible and prevents preview or retention',async()=>{
  const f=await runtimeOpened(),n=runtimeField(f);edit(n,'{wrong');
  await f.click('science-family-preview');await f.click('science-family-save');
  assert.equal(f.requests.length,6);assert.equal(n.value,'{wrong');assert.ok('invalid' in n.dataset);
});
test('stale runtime edit acknowledgement cannot trigger preview in another workspace',async()=>{
  const f=await runtimeOpened();edit(runtimeField(f),'{"input":[1]}');
  const done=f.click('science-family-preview');await tick();f.state.workspace='two';f.form.stateChanged();
  f.requests[6].resolve(h(1));await done;assert.equal(f.requests.length,7);
  assert.equal(f.form.active(),true);assert.equal(f.get('science-family-save').disabled,true);
});

test('new family opens despite its own pending flag and uses a concrete seed',async()=>{
  const f=await opened(fixture());assert.equal(f.requests[0].args.model,'seed');assert.equal(f.form.active(),true);
  assert.equal(f.get('science-family-check').disabled,false);assert.equal(f.requests.length,6);
  assert.ok(f.requests.slice(1).every(r=>r.args.limit===6&&!('draft' in r.args)));
});
test('focused exact integer text is flushed before validation without Number coercion',async()=>{
  const f=await opened(fixture()),n=f.field('symbols','default');edit(n,'9007199254740993','input');f.root.activeElement=n;
  const done=f.click('science-family-check');await tick();assert.equal(f.requests[6].args.operations[0].text,'9007199254740993');
  f.requests[6].resolve(h(1));await tick();assert.equal(f.requests[7].args.sequence,1);f.requests[7].resolve(ref);await done;
});
test('invalid expression text stays visible and blocks all admission requests',async()=>{
  const f=await opened(fixture()),n=f.field('assignments','value');edit(n,'not JSON');
  await f.click('science-family-save');assert.equal(f.requests.length,6);assert.equal(n.value,'not JSON');assert.ok('invalid' in n.dataset);
});
test('wrong edit sequence acknowledgement keeps queued text and forbids Check',async()=>{
  const f=await opened(fixture());edit(f.field('symbols','default'),'4');const done=f.click('science-family-check');await tick();
  f.requests[6].resolve(h(9));await done;assert.equal(f.requests.length,7);assert.match(f.toasts.at(-1)[0],/acknowledgement/);
});
test('workspace switches invalidate late preview results and retain visible draft',async()=>{
  const f=await opened(fixture()),done=f.click('science-family-preview');await tick();
  f.state.workspace='two';f.form.stateChanged();f.requests[6].resolve({version:1,scope:'owned-family-preview',session:'owned',sequence:0,id:'family',workspace_id:'one',status:'verified',parameters:21});
  await done;assert.equal(f.get('science-family-preview-result').textContent,'');assert.equal(f.get('science-family-save').disabled,true);assert.equal(f.form.active(),true);
});
test('retention closes only the acknowledged exact family draft',async()=>{
  const f=await opened(fixture()),done=f.click('science-family-save');await tick();f.requests[6].resolve(ref);await done;
  assert.equal(f.form.active(),false);assert.equal(f.saved.length,1);assert.equal(f.saved[0].revision,ref.revision);
});
test('staged graph editing retains parent sequence and complete identity',async()=>{
  const f=await opened(fixture());await f.click('science-family-raw');
  assert.equal(f.staged[0].header.session,'owned');assert.deepEqual(Array.from(f.staged[0].path),[]);
  const callback=f.staged[0].onClosed(h(1));await tick();f.requests.slice(6,11).forEach(r=>r.resolve(page(r)));
  await callback;assert.ok(f.requests.slice(6,11).every(r=>r.args.sequence===1));
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
  f.requests.slice(1,6).forEach(r=>r.args.collection==='symbols'?r.reject(Error('temporary read failure')):r.resolve(page(r)));
  await done;assert.equal(f.form.active(),true);assert.equal(f.get('science-family-save').disabled,true);
  assert.equal(f.get('science-family-raw').disabled,false);assert.equal(f.get('science-family-retry').disabled,false);
  assert.equal(f.field('assignments','node').disabled,true);
});

test('tied repeat mode explicitly upgrades family grammar before mode edit',async()=>{
  const f=await opened(fixture()),n=f.field('repeats','mode');assert.equal(n.tagName,'select');
  edit(n,'tied');const done=f.click('science-family-check');await tick();
  const operations=f.requests[6].args.operations;
  assert.deepEqual(JSON.parse(JSON.stringify(operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'2'},
    {op:'property',target:{kind:'entry',collection:'repeats',ordinal:0},key:'mode',text:'"tied"'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});

test('independent repeat mode does not infer a downgrade or share equal values',async()=>{
  const f=await opened(fixture());edit(f.field('repeats','mode'),'independent');
  const done=f.click('science-family-check');await tick();
  assert.equal(f.requests[6].args.operations.length,1);assert.equal(f.requests[6].args.operations[0].key,'mode');
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});

test('constructor type queues version 3 before its type and leaves explicit repairs to the user',async()=>{
  const f=fixture(),r={args:{collection:'symbols',sequence:0,offset:0}},p=page(r);
  p.rows[0].properties.push({key:'type',present:false,text:null,path:null,summary:null});
  await opened(f,{symbols:p});const n=f.field('symbols','type');
  assert.equal(n.tagName,'select');assert.deepEqual(n.children.map(o=>o.value),['boolean','enum','integer']);
  edit(n,'enum');const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'3'},
    {op:'property',target:{kind:'entry',collection:'symbols',ordinal:0},key:'type',text:'"enum"'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].reject(Error('Remove arithmetic bounds from enum'));await done;
  assert.equal(f.form.active(),true);assert.equal(f.field('symbols','minimum').value,'1');
});

test('typed scalar defaults and enum domains preserve exact JSON strings and booleans',async()=>{
  const f=fixture(),p=page({args:{collection:'symbols',sequence:0,offset:0}});
  p.rows[0].properties=[descriptor('type','enum',[0]),descriptor('default','Tanh',[1]),descriptor('values',['Tanh','ReLU'],[2])];
  await opened(f,{symbols:p});assert.equal(f.field('symbols','default').value,'"Tanh"');
  edit(f.field('symbols','default'),'"GELU"');edit(f.field('symbols','values'),'["ReLU","GELU"]');
  const done=f.click('science-family-check');await tick();const operations=f.requests[6].args.operations;
  assert.equal(operations.find(o=>o.key==='default').text,'"GELU"');
  assert.equal(operations.find(o=>o.key==='values').text,'["ReLU","GELU"]');
  assert.equal(operations.find(o=>o.key==='version').text,'3');
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
  edit(f.field('symbols','default'),'false');const again=f.click('science-family-check');await tick();
  assert.equal(f.requests[8].args.operations[0].text,'false');f.requests[8].resolve(h(2));await tick();f.requests[9].resolve(ref);await again;
});

test('tied mode preserves an already-read or pending version 3 upgrade',async()=>{
  const f=fixture(),roots=[descriptor('version',3,[0])];await opened(f,{repeats:{roots}});
  edit(f.field('repeats','mode'),'tied');const done=f.click('science-family-check');await tick();
  assert.equal(f.requests[6].args.operations.length,1);assert.equal(f.requests[6].args.operations[0].key,'mode');
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
});

test('invalid unquoted enum default stays visible and blocks preview without native mutation',async()=>{
  const f=await opened(fixture());edit(f.field('symbols','default'),'Tanh');
  await f.click('science-family-preview');assert.equal(f.requests.length,6);
  assert.equal(f.field('symbols','default').value,'Tanh');assert.ok('invalid' in f.field('symbols','default').dataset);
});

test('scoped constructor cards preserve typed AST, boolean and enum JSON without a downgrade',async()=>{
  const symbols=page({args:{collection:'symbols',sequence:0,offset:0}});
  symbols.rows[0].properties.push(descriptor('type','integer',[4]));
  const f=await opened(fixture(),{symbols,constructors:{roots:[descriptor('version',4,[0])]}});
  edit(f.field('constructors','value'),'false');const done=f.click('science-family-check');await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[6].args.operations)),[
    {op:'property',target:{kind:'entry',collection:'constructors',ordinal:0},key:'value',text:'false'}]);
  f.requests[6].resolve(h(1));await tick();f.requests[7].resolve(ref);await done;
  edit(f.field('repeats','mode'),'tied');edit(f.field('symbols','type'),'enum');
  const again=f.click('science-family-check');await tick();assert.ok(f.requests[8].args.operations.every(o=>o.key!=='version'));
  f.requests[8].resolve(h(2));await tick();f.requests[9].resolve(ref);await again;
});

test('Add constructor explicitly upgrades the fresh grammar and inserts in one owned batch',async()=>{
  const f=await opened(fixture(),{constructors:{rows:[],total:0,next_offset:null,path:null}});
  const add=f.get('science-family-cards').querySelectorAll('button').find(n=>n.textContent==='Add constructor');
  const done=add.onclick();await tick();assert.equal(f.requests[6].action,'scienceDraftFamilyPage');
  f.requests[6].resolve(page(f.requests[6],{rows:[],total:0,next_offset:null,path:null,roots:[descriptor('version',3,[0])]}));await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[7].args.operations)),[
    {op:'property',target:{kind:'document'},key:'version',text:'4'},
    {op:'insert',collection:'constructors',index:0,value:{model:'seed',node:'',symbol:'',value:1}}]);
  f.requests[7].resolve(h(1));await tick();f.requests.slice(8,13).forEach(r=>r.resolve(page(r,{roots:[descriptor('version',4,[0])]})));
  await done;assert.equal(f.get('science-family-check').disabled,false);
});

test('late insertion page after workspace switch cannot append a constructor',async()=>{
  const f=await opened(fixture()),add=f.get('science-family-cards').querySelectorAll('button').find(n=>n.textContent==='Add constructor');
  const done=add.onclick();await tick();f.state.workspace='two';f.form.stateChanged();f.requests[6].resolve(page(f.requests[6]));await done;
  assert.equal(f.requests.length,7);assert.equal(f.form.active(),true);assert.equal(f.get('science-family-check').disabled,true);
});

test('foreign insertion page is rejected without mutating the owned draft',async()=>{
  const f=await opened(fixture()),add=f.get('science-family-cards').querySelectorAll('button').find(n=>n.textContent==='Add constructor');
  const done=add.onclick();await tick();f.requests[6].resolve(page(f.requests[6],{session:'foreign'}));await done;
  assert.equal(f.requests.length,7);assert.match(f.toasts.at(-1)[0],/insertion page/);
});

test('invalid constructor JSON is retained and cannot issue preview requests',async()=>{
  const f=await opened(fixture());edit(f.field('constructors','value'),'__import__("torch")');
  await f.click('science-family-preview');assert.equal(f.requests.length,6);
  assert.ok('invalid' in f.field('constructors','value').dataset);
});

test('checked family export pins exact retained selection without bindings or creating an instance',async()=>{
  const f=fixture(),done=f.click('science-family-export');await tick();
  assert.equal(f.requests[0].action,'scienceExportFamily');
  assert.deepEqual(JSON.parse(JSON.stringify(f.requests[0].args)),{workspace_id:'one',family:ref});
  f.requests[0].resolve({path:'/private/family.py',status:'checked-family-source-exported; not-instantiated-or-trained'});await done;
  assert.match(f.toasts.at(-1)[0],/not instantiated or trained/);assert.equal(f.requests.length,1);
});

test('cancelled or stale family export cannot claim source completion in another workspace',async()=>{
  const f=fixture(),cancel=f.click('science-family-export');await tick();f.requests[0].resolve(null);await cancel;
  assert.equal(f.toasts.length,0);const done=f.click('science-family-export');await tick();
  f.state.workspace='two';f.form.stateChanged();f.requests[1].resolve({path:'/private/family.py'});await done;
  assert.equal(f.toasts.length,0);
});

test('checked source export remains disabled while any scientific draft is open',async()=>{
  const f=await opened(fixture());assert.equal(f.get('science-family-export').disabled,true);
  const cancel=f.click('science-family-cancel');await tick();f.requests[6].resolve({closed:true});await cancel;
  assert.equal(f.get('science-family-export').disabled,false);
});
