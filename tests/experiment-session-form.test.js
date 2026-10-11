'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture.js');
const header=(sequence=0)=>({scope:'owned-authoring-session',version:1,kind:'experiment',id:'experiment',session:'owned',sequence,workspace_id:'one',base:null,name:'Experiment',expected_head:null});
const reference={kind:'experiment',id:'experiment',revision:'a'.repeat(64)};
const model={kind:'model',id:'model',revision:'b'.repeat(64)},dataset={kind:'dataset',id:'data',revision:'c'.repeat(64)},task={kind:'task',id:'task',revision:'d'.repeat(64)};
const descriptor=(key,value,path,present=true)=>({key,path,present,text:present?JSON.stringify(value):null,summary:null});
const collections=['models','data_tasks','splits','initialization_seeds','protocols','pairs','exclusions'];
function page(r,changes={}){const a=r.args,path=[collections.indexOf(a.collection)+2],values={
  models:{reference:model},data_tasks:{id:'pair-0',dataset,task},splits:{id:'s','method':'row',seed:3,'fractions.0':.5,'fractions.1':.25,'fractions.2':.25},
  initialization_seeds:{value:7},protocols:{id:'adam',batch_size:2,evaluate_every:1,'history.max_points':20,'budget.epochs':null,'budget.updates':3,'budget.examples':null,'budget.seconds':null,
    'optimizer.lr':.03,'optimizer.eps':1e-8,'optimizer.weight_decay':0,'optimizer.betas.0':.9,'optimizer.betas.1':.999,'seeds.initialization':7,'seeds.shuffle':11,shuffle:true},
  pairs:{model,data_task:'pair-0'},exclusions:{model,data_task:'pair-0',reason:'negative control'}}[a.collection];
  return {scope:'owned-experiment-page',version:1,...a,workspace_id:'one',id:'experiment',total:20,path,next_offset:a.offset+1,
    roots:[descriptor('name','Experiment',[0]),descriptor('question','Question',[1]),descriptor('selection.kind','crossed',[9,0])],
    rows:[{ordinal:a.offset,path:[...path,a.offset],properties:Object.entries(values).map(([k,v],i)=>descriptor(k,v,[...path,a.offset,i]))}],...changes};
}
function preview(r,changes={}){return {scope:'owned-experiment-preview',version:1,...r.args,workspace_id:'one',id:'experiment',experiment:reference,
  total_combinations_exact:'1152921504606846977',execution:'no-fitting-or-training',next_offset:r.args.offset+1,
  candidates:[{id:'candidate',coordinates:{model,data_task:'pair-0',initialization:7,split:'s',protocol:'adam'},status:'excluded',
    reason:{text:'Partial λ',offset:0,total:20000,next_offset:128},counts:null}],...changes};}
function fixture(){const body=new Element('body'),nodes=new Map(),get=id=>{if(!nodes.has(id)){const n=new Element();nodes.set(id,n);body.append(n);}return nodes.get(id);};
  get('science-author').append(get('science-author-name'),get('science-experiment-bindings'));
  const root={getElementById:get,createElement:tag=>new Element(tag),activeElement:null},requests=[],saved=[],toasts=[],children=[];
  const state={workspace:'one',busy:false,dataset:false,task:false,legacy:false,value:false,refreshes:0};
  const choice=()=>({controls(){},dispose(){},refresh:async()=>{state.refreshes++;}});
  const window={LoomExperimentAuthoring:require('../app/web/experiment-authoring.js'),LoomScienceAuthor:{stateChanged(){},ownActive:()=>state.legacy},
    LoomDatasetSessionForm:{active:()=>state.dataset},LoomTaskSessionForm:{active:()=>state.task},LoomScienceDocument:{stateChanged(){}},
    LoomDraftValueEditor:{active:()=>state.value,openOwned:async target=>children.push(target)},LoomRevisionChoice:{attach:choice},LoomExperimentLocalChoice:{attach:choice}};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...args)=>toasts.push(args),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/experiment-session-form.js','utf8'),{window,Event:class{constructor(type){this.type=type;}},crypto:{randomUUID:()=> 'entry'}});
  const form=window.LoomExperimentSessionForm;form.mount(root,api,async r=>saved.push(r));
  const section=c=>get('science-experiment-bindings').querySelector('[data-experiment-collection='+c+']');
  const card=(kind='protocol',ordinal=0)=>get('science-experiment-bindings').querySelectorAll('[data-experiment-axis]').find(n=>n.dataset.experimentAxis===kind&&n.dataset.axisIndex===ordinal);
  const field=(label,kind='protocol',ordinal=0)=>card(kind,ordinal).querySelectorAll('label').find(n=>n.textContent===label)?.querySelector('input,textarea,select');
  const button=(name,c)=>section(c).querySelectorAll('button').find(n=>n.textContent===name);
  return {get,root,state,requests,saved,toasts,children,form,field,card,section,button,click:id=>get(id).onclick()};
}
function replyPages(f,start,changes={}){for(const r of f.requests.slice(start,start+7))r.resolve(page(r,changes[r.args.collection]||{}));}
async function opened(f,changes={}){const p=f.form.open();await tick();f.requests[0].resolve(header());await tick();replyPages(f,1,changes);await p;return f;}
function edit(n,value,event='change'){n.value=value;n['on'+event]();}
test('actual experiment opens a header and seven bounded axes, never a complete draft',async()=>{
  const f=await opened(fixture());assert.equal(f.requests.length,8);assert.equal(f.requests[0].action,'scienceDraftOpen');
  assert.deepEqual(f.requests.slice(1).map(r=>r.args.collection),collections);assert.ok(f.requests.slice(1).every(r=>r.args.limit===6&&!('draft' in r.args)));
  assert.equal(f.field('Maximum updates').value,'3');assert.equal(f.get('science-author-check').disabled,false);
});
test('focused unsafe integer literals survive native flush before Check',async()=>{
  const f=await opened(fixture()),n=f.field('Maximum updates');edit(n,'9007199254740993','input');f.root.activeElement=n;
  const done=f.click('science-author-check');await tick();assert.equal(n.disabled,true);assert.equal(f.requests[8].args.operations[0].text,'9007199254740993');
  assert.equal(f.requests[8].args.operations[0].key,'budget.updates');f.requests[8].resolve(header(1));await tick();
  assert.equal(f.requests[9].args.sequence,1);f.requests[9].resolve(reference);await done;assert.match(f.get('science-author-status').textContent,/Valid ExperimentSpec/);
});
test('blank budgets set null while shuffled keys remain named and unchanged targets are retained',async()=>{
  const f=await opened(fixture());edit(f.field('Maximum epochs'),'');edit(f.field('Adam epsilon'),'1e-7');
  const done=f.click('science-author-check');await tick();const ops=f.requests[8].args.operations;
  assert.equal(ops[0].text,'null');assert.equal(ops[1].key,'optimizer.eps');assert.ok(ops.every(o=>!('path' in o)));
  f.requests[8].resolve(header(1));await tick();f.requests[9].resolve(reference);await done;
});
test('invalid text survives metadata refresh and cannot reach mutation until corrected',async()=>{
  const f=await opened(fixture()),n=f.field('Maximum updates');edit(n,'1e400');await f.click('science-experiment-refresh');
  assert.equal(n.value,'1e400');assert.ok('invalid' in n.dataset);await f.click('science-author-check');assert.equal(f.requests.length,8);
  edit(n,'3');const done=f.click('science-author-check');await tick();assert.equal(f.requests[8].args.operations.length,1);
  f.requests[8].resolve(header(1));await tick();f.requests[9].resolve(reference);await done;
});
test('failed native edit batches retain fields and support correcting the failed entry',async()=>{
  const f=await opened(fixture()),n=f.field('Configuration ID');edit(n,'keep');const done=f.click('science-author-check');await tick();
  f.requests[8].reject(Error('Experiment edit #0: invalid'));await done;assert.equal(n.value,'keep');assert.ok('invalid' in n.dataset);assert.equal(f.form.active(),true);
});
test('queues over 25 split without rounding or admitting intermediate specifications',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  const r=f.requests.find(r=>r.args.collection==='protocols'),p=page(r);p.rows=Array.from({length:3},(_,i)=>({...p.rows[0],ordinal:i,path:[...p.path,i]}));
  for(const q of f.requests.slice(1))q.resolve(q===r?p:page(q));await opening;
  const inputs=f.section('protocols').querySelectorAll('input').filter(n=>n.type!=='checkbox').slice(0,32);inputs.forEach(n=>n.onchange());
  const done=f.click('science-author-check');await tick();assert.equal(f.requests[8].args.operations.length,25);f.requests[8].resolve(header(1));await tick();
  assert.equal(f.requests[9].args.operations.length,7);assert.equal(f.requests[9].args.sequence,1);f.requests[9].resolve(header(2));await tick();
  assert.equal(f.requests[10].action,'scienceDraftCheck');f.requests[10].resolve(reference);await done;
});
test('axes navigate actual cursors and retry failed Back without dropping its history',async()=>{
  const f=await opened(fixture());let done=f.button('Entries ›','initialization_seeds').onclick();await tick();assert.equal(f.requests[8].args.offset,1);
  f.requests[8].resolve(page(f.requests[8]));await done;done=f.button('‹ Entries','initialization_seeds').onclick();await tick();f.requests[9].reject(Error('temporary'));await done;
  assert.equal(f.button('‹ Entries','initialization_seeds').disabled,false);done=f.button('‹ Entries','initialization_seeds').onclick();await tick();
  assert.equal(f.requests[10].args.offset,0);f.requests[10].resolve(page(f.requests[10]));await done;assert.equal(f.button('‹ Entries','initialization_seeds').disabled,true);
});
test('wrong owner, sequence, scope and cursor pages cannot replace the old card',async()=>{
  for(const changes of [{scope:'full-experiment'},{version:8},{session:'foreign'},{sequence:99},{workspace_id:'two'},{id:'wrong'},{collection:'models'},{offset:99},{limit:1}]){
    const f=await opened(fixture()),old=f.card('seed'),done=f.button('Entries ›','initialization_seeds').onclick();await tick();f.requests[8].resolve(page(f.requests[8],changes));await done;
    assert.equal(f.card('seed'),old);assert.equal(f.toasts.length,1);
  }
});
test('parallel initial read failure waits for outstanding pages and retries the same session',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();f.requests[1].reject(Error('initial failure'));await tick();
  assert.equal(f.get('science-author-cancel').disabled,true);for(const r of f.requests.slice(2))r.resolve(page(r));await opening;
  assert.equal(f.get('science-author-save').disabled,true);assert.equal(f.get('science-author-retry-page').hidden,false);
  const done=f.click('science-author-retry-page');await tick();replyPages(f,8);await done;assert.equal(f.get('science-author-save').disabled,false);
});
test('structure flushes literal edits and resolves fresh array paths before removal',async()=>{
  const f=await opened(fixture());edit(f.get('science-author-name'),'Changed');const done=f.button('Remove entry','models').onclick();await tick();
  f.requests[8].resolve(header(1));await tick();f.requests[9].resolve(page(f.requests[9],{path:[11]}));await tick();
  assert.equal(JSON.stringify(f.requests[10].args.operations),JSON.stringify([{op:'remove',path:[11,0]}]));f.requests[10].resolve(header(2));await tick();
  assert.equal(f.get('science-author-save').disabled,true);f.requests[11].reject(Error('reload failure'));for(const r of f.requests.slice(12))r.resolve(page(r));await done;
  assert.equal(f.get('science-author-save').disabled,true);const retry=f.click('science-author-retry-page');await tick();replyPages(f,18);await retry;assert.equal(f.get('science-author-save').disabled,false);
});
test('large child resolves after typed flush and accepts only the same parent sequence',async()=>{
  const f=fixture(),base=page({args:{collection:'models',offset:0,limit:6}}).roots;base[1]={...base[1],text:null,summary:{type:'string',size:20000,preview:'old'}};
  await opened(f,{models:{roots:base}});edit(f.get('science-author-name'),'Changed');
  const label=f.get('science-experiment-bindings').querySelectorAll('label').find(n=>n.textContent==='Research question'),done=label.querySelector('button').onclick();await tick();
  f.requests[8].resolve(header(1));await tick();const r=f.requests[9];assert.equal(r.action,'scienceDraftExperimentValue');assert.equal(r.args.sequence,1);
  r.resolve({scope:'owned-experiment-value',version:1,...r.args,workspace_id:'one',id:'experiment',property:{present:true,path:[14],text:null}});await done;
  assert.equal(JSON.stringify(f.children[0].path),'[14]');const returning=f.children[0].onClosed(header(2));await tick();replyPages(f,10);await returning;
  assert.equal(f.form.active(),true);assert.equal(f.saved.length,0);assert.equal(f.get('science-author-check').disabled,false);
});
test('preview uses exact decimal totals, explicit partial reasons and actual next/back cursors',async()=>{
  const f=await opened(fixture());let done=f.click('science-experiment-preview');await tick();f.requests[8].resolve(preview(f.requests[8]));await done;
  assert.match(f.get('science-experiment-preview-status').textContent,/1152921504606846977/);assert.match(f.get('science-experiment-preview-rows').querySelector('small').textContent,/partial reason/);
  done=f.click('science-experiment-next');await tick();assert.equal(f.requests[9].args.offset,1);f.requests[9].resolve(preview(f.requests[9]));await done;
  done=f.click('science-experiment-prev');await tick();assert.equal(f.requests[10].args.offset,0);f.requests[10].resolve(preview(f.requests[10]));await done;
  assert.equal(f.get('science-experiment-prev').disabled,true);
});
test('workspace/reload changes reject late previews, errors and validation acknowledgements',async()=>{
  for(const action of ['science-experiment-preview','science-author-check']){const f=await opened(fixture()),done=f.click(action);await tick();f.state.workspace='two';f.form.stateChanged();
    f.requests[8].resolve(action==='science-author-check'?reference:preview(f.requests[8]));await done;
    assert.equal(f.get('science-experiment-preview-rows').children.length,0);assert.doesNotMatch(f.get('science-author-status').textContent,/Valid ExperimentSpec/);
  }
  const f=await opened(fixture()),done=f.click('science-author-check');await tick();f.form.event({kind:'externalReload'});f.requests[8].reject(Error('obsolete'));await done;assert.equal(f.toasts.length,0);
});
test('native origin proof, not a returning UUID, governs round-trip resumption',async()=>{
  const f=await opened(fixture());f.state.workspace='two';f.form.stateChanged();f.state.workspace='one';f.form.stateChanged();await tick();
  assert.equal(f.requests[8].action,'scienceDraftOrigin');assert.equal(f.get('science-author-save').disabled,true);f.requests[8].resolve(header());await tick();assert.equal(f.get('science-author-save').disabled,false);
  f.form.event({kind:'externalReload'});f.state.workspace='two';f.form.stateChanged();f.state.workspace='one';f.form.stateChanged();await tick();assert.equal(f.requests.length,9);assert.equal(f.get('science-author-save').disabled,true);
});
test('only exact successful Commit closes the form and stale/wrong results remain unsaved',async()=>{
  const f=await opened(fixture());let done=f.click('science-author-save');await tick();f.requests[8].resolve({...reference,id:'wrong'});await done;
  assert.equal(f.form.active(),true);assert.equal(f.saved.length,0);done=f.click('science-author-save');await tick();f.requests[9].resolve(reference);await done;
  assert.equal(f.form.active(),false);assert.equal(f.saved.length,1);
});
test('other editors, a live job and duplicate pending commands cannot open another owner',async()=>{
  for(const flag of ['busy','dataset','task','legacy','value']){const f=fixture();f.state[flag]=true;await f.form.open();assert.equal(f.requests.length,0);}
  const f=fixture(),first=f.form.open(),second=f.form.open();await tick();assert.equal(f.requests.length,1);f.requests[0].resolve(header());await tick();replyPages(f,1);await Promise.all([first,second]);
});
test('comparison mode queues a single named edit without silently deleting whitelist entries',async()=>{
  const f=await opened(fixture()),n=f.get('science-experiment-bindings').querySelectorAll('label').find(n=>n.textContent==='Comparison mode').querySelector('select');edit(n,'paired');
  const done=f.click('science-author-check');await tick();assert.equal(f.requests[8].args.operations.length,1);assert.equal(f.requests[8].args.operations[0].key,'selection.kind');
  f.requests[8].resolve(header(1));await tick();f.requests[9].resolve(reference);await done;
});
test('external role IDs keep literal spaces, commas, order and duplicates for native validation',async()=>{
  const f=fixture(),r={args:{collection:'splits',offset:0,limit:6}},p=page(r);p.rows[0].properties=[descriptor('id','ext',[4,0,0]),descriptor('method','external',[4,0,1]),
    ...['train','validation','test'].map(role=>descriptor('roles.'+role,[],[4,0,2])),descriptor('excluded',[],[4,0,3])];
  await opened(f,{splits:{rows:p.rows}});edit(f.field('train observation IDs','split'),'r,1\n r 2 \nr,1');const done=f.click('science-author-check');await tick();
  assert.equal(f.requests[8].args.operations[0].text,JSON.stringify(['r,1',' r 2 ','r,1']));f.requests[8].resolve(header(1));await tick();f.requests[9].reject(Error('duplicate IDs'));await done;
  assert.equal(f.field('train observation IDs','split').value,'r,1\n r 2 \nr,1');
});
