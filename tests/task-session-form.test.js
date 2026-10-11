'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(tag='div',text=''){this.tagName=tag;this.textContent=text;this.children=[];this.dataset={};this.attributes={};this.disabled=false;this.value='';this.hidden=false;}
  append(...n){this.children.push(...n);}replaceChildren(...n){this.children=[...n];}
  setAttribute(k,v){this.attributes[k]=v;}removeAttribute(k){delete this.attributes[k];}
  matches(s){return s.split(',').some(k=>k.startsWith('[data-')?k.slice(6,-1).replace(/-([a-z])/g,(_,c)=>c.toUpperCase()) in this.dataset:k===this.tagName);}
  querySelectorAll(s){return this.children.flatMap(n=>[...(n.matches(s)?[n]:[]),...n.querySelectorAll(s)]);}
  querySelector(s){return this.querySelectorAll(s)[0]||null;}contains(n){return this===n||this.children.some(c=>c.contains(n));}
  dispatchEvent(e){this['on'+e.type]?.(e);}
}
const tick=()=>new Promise(r=>setImmediate(r));
const header=(sequence=0)=>({scope:'owned-authoring-session',version:1,kind:'task',id:'task',session:'owned',sequence,workspace_id:'one',base:null,name:'Task',expected_head:null});
const desc=(key,text,path,present=true)=>({key,text,path,present,summary:text===null?{type:'object',size:20000}:null});
const taskRef={kind:'task',id:'task',revision:'a'.repeat(64)};
const refs={model:{kind:'model',id:'model',revision:'b'.repeat(64)},dataset:{kind:'dataset',id:'data',revision:'c'.repeat(64)}};
function page(r,changes={}){
  const a=r.args,inputs=a.collection==='inputs',path=inputs?[3]:[1];
  const values=inputs?{port:'in',field:'x'}:{id:'h',port:'out',field:'y',loss:'mse',weight:1};
  const keys=inputs?['port','field','derived','length']:['id','port','field','loss','weight','mask','sample_weight'];
  return {scope:'owned-task-page',version:1,...a,workspace_id:'one',id:'task',total:20,path,next_offset:7,
    description:desc('description','"A description"',[0]),rows:[{ordinal:a.offset,path:[...path,a.offset],
      properties:keys.map((key,i)=>desc(key,Object.hasOwn(values,key)?JSON.stringify(values[key]):null,[...path,a.offset,i],Object.hasOwn(values,key))),
      transforms:{path:[...path,a.offset,2],total:0}}],...changes};
}
function pipeline(r,changes={}){
  const a=r.args,path=[3,a.binding,2];
  return {scope:'owned-task-transform-page',version:1,...a,workspace_id:'one',id:'task',total:30,path,next_offset:9,
    rows:[{ordinal:a.offset,path:[...path,a.offset],properties:['kind','scale','offset','length','value'].map((key,i)=>
      desc(key,key==='kind'?'"affine"':key==='scale'?'1':key==='offset'?'0':null,[...path,a.offset,i],i<3))}],...changes};
}
function fixture(){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  const root={getElementById:get,createElement:tag=>new Element(tag),activeElement:null};
  get('science-author').append(get('science-author-name'),get('science-task-bindings'),get('science-task-model'),get('science-task-dataset'));
  const requests=[],toasts=[],saved=[],children=[],state={workspace:'one',busy:false,dataset:false,legacy:false,value:false};
  const window={LoomTaskAuthoring:require('../app/web/task-authoring.js'),LoomScienceAuthor:{stateChanged(){},ownActive:()=>state.legacy},
    LoomDatasetSessionForm:{active:()=>state.dataset},LoomScienceDocument:{stateChanged(){}},
    LoomTaskContext:{controls(){},clear(){},show:async()=>{}},LoomDraftValueEditor:{active:()=>state.value,openOwned:async t=>children.push(t)},
    LoomRevisionChoice:{attach(){return {refresh:async()=>{},controls(){},dispose(){}};}}};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...a)=>toasts.push(a),
    call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/task-session-form.js','utf8'),{window,JSON,Array,Object,Map,Promise,Error,Number,
    Event:class{constructor(type){this.type=type;}},crypto:{randomUUID:()=> 'new-head'}});
  const form=window.LoomTaskSessionForm;form.mount(root,api,async result=>saved.push(result));
  const card=(kind='input')=>get('science-task-bindings').querySelector('[data-task-'+kind+']');
  const control=(label,kind='input')=>card(kind).querySelectorAll('label').find(n=>n.textContent===label)?.querySelector('input,textarea,select');
  const group=(kind='inputs')=>{const c=get('science-task-bindings'),offset=kind==='inputs'?0:4;return {prev:c.children[offset+1].children[0],next:c.children[offset+1].children[2]};};
  return {nodes,get,root,state,requests,toasts,saved,children,form,window,card,control,group,click:id=>get(id).onclick()};
}
async function opened(f,extra={}){
  const promise=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  f.requests[1].resolve(page(f.requests[1],extra));f.requests[2].resolve(page(f.requests[2]));await promise;return f;
}
test('actual task controller opens only header/two bounded pages and never sends a complete draft',async()=>{
  const f=await opened(fixture());assert.deepEqual(f.requests.map(r=>r.action),['scienceDraftOpen','scienceDraftTaskPage','scienceDraftTaskPage']);
  assert.equal(f.control('Input port ID').value,'in');assert.equal(f.control('Objective weight','head').value,'1');
  assert.equal(f.get('science-author-check').disabled,false);assert.ok(f.requests.every(r=>!('draft' in r.args)));
  assert.equal(f.card().querySelectorAll('button').some(n=>n.textContent==='Edit value…'),false);
});
test('focused literal numeric tokens survive capture and bounded native flush before Check',async()=>{
  const f=await opened(fixture()),n=f.control('Objective weight','head');n.value='9007199254740993';n.oninput();f.root.activeElement=n;
  const promise=f.click('science-author-check');await tick();assert.equal(n.disabled,true);
  assert.equal(f.requests[3].action,'scienceDraftTaskEdit');assert.equal(f.requests[3].args.operations[0].text,'9007199254740993');
  assert.equal(f.requests[3].args.operations[0].target.collection,'heads');f.requests[3].resolve(header(1));await tick();
  assert.equal(f.requests[4].action,'scienceDraftCheck');assert.equal(f.requests[4].args.sequence,1);
  f.requests[4].resolve(taskRef);await promise;assert.match(f.get('science-author-status').textContent,/Valid TaskSpec/);
});
test('optional removals and unrelated edits use named targets rather than old dictionary paths',async()=>{
  const f=await opened(fixture()),n=f.get('science-task-bindings').querySelectorAll('label').find(n=>n.textContent==='Task description').querySelector('textarea');
  n.value='';n.onchange();f.control('Input port ID').value='changed';f.control('Input port ID').onchange();
  const promise=f.click('science-author-check');await tick();const ops=f.requests[3].args.operations;
  assert.equal(ops[0].text,null);assert.equal(ops[0].target.kind,'document');assert.equal(ops[1].target.kind,'binding');
  assert.equal('path' in ops[0],false);f.requests[3].resolve(header(1));await tick();f.requests[4].resolve(taskRef);await promise;
});
test('invalid numbers preserve text and block mutation; correction replaces only the latest value',async()=>{
  const f=await opened(fixture()),n=f.control('Objective weight','head');n.value='not-a-number';n.onchange();
  await f.click('science-author-check');assert.equal(f.requests.length,3);assert.equal(n.value,'not-a-number');
  n.value='2';n.onchange();const promise=f.click('science-author-check');await tick();assert.equal(f.requests[3].args.operations.length,1);
  assert.equal(f.requests[3].args.operations[0].text,'2');f.requests[3].resolve(header(1));await tick();f.requests[4].resolve(taskRef);await promise;
});
test('failed native property batches retain text and identify the failed field for repair',async()=>{
  const f=await opened(fixture()),n=f.control('Input port ID');n.value='retained';n.onchange();
  const promise=f.click('science-author-check');await tick();f.requests[3].reject(Error('Task edit #0: bad value'));await promise;
  assert.equal(n.value,'retained');assert.equal(n.dataset.invalid,'');assert.equal(f.form.active(),true);assert.equal(f.saved.length,0);
});
test('binding navigation uses actual cursors, and failed previous navigation retains retry history',async()=>{
  const f=await opened(fixture());let promise=f.group().next.onclick();await tick();assert.equal(f.requests[3].args.offset,7);
  f.requests[3].resolve(page(f.requests[3]));await promise;promise=f.group().prev.onclick();await tick();
  f.requests[4].reject(Error('transient'));await promise;assert.equal(f.group().prev.disabled,false);
  promise=f.group().prev.onclick();await tick();assert.equal(f.requests[5].args.offset,0);f.requests[5].resolve(page(f.requests[5]));await promise;
  assert.equal(f.group().prev.disabled,true);
});
test('wrong scope/owner/sequence/collection/cursor pages cannot replace current controls',async()=>{
  for(const changes of [{scope:'full-task'},{version:9},{session:'foreign'},{sequence:99},{workspace_id:'two'},{id:'wrong'},{collection:'heads'},{offset:88},{limit:1}]){
    const f=await opened(fixture()),old=f.control('Input port ID'),promise=f.group().next.onclick();await tick();
    f.requests[3].resolve(page(f.requests[3],changes));await promise;assert.equal(f.control('Input port ID'),old);
    assert.match(f.get('science-author-status').textContent,/does not match/);
  }
});
test('late page successes/errors after workspace switch or same-UUID reload do not overwrite text/notices',async()=>{
  for(const transition of ['switch','reload'])for(const fail of [false,true]){
    const f=await opened(fixture()),old=f.control('Input port ID'),promise=f.group().next.onclick();await tick();
    if(transition==='switch'){f.state.workspace='two';f.form.stateChanged();}else f.form.event({kind:'externalReload'});
    const notice=f.get('science-author-status').textContent;if(fail)f.requests[3].reject(Error('obsolete'));else f.requests[3].resolve(page(f.requests[3]));await promise;
    assert.equal(f.get('science-author-status').textContent,notice);assert.equal(f.control('Input port ID'),old);assert.equal(f.toasts.length,0);
    assert.equal(f.get('science-author-save').disabled,true);
  }
});
test('initial read failure allows Retry on the same session, never an implicit second Open',async()=>{
  const f=fixture(),promise=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  f.requests[1].reject(Error('temporary'));f.requests[2].resolve(page(f.requests[2]));await promise;
  assert.equal(f.get('science-author-check').disabled,true);assert.equal(f.get('science-author-retry-page').hidden,false);
  const retry=f.click('science-author-retry-page');await tick();assert.equal(f.requests[3].action,'scienceDraftTaskPage');
  f.requests[3].resolve(page(f.requests[3]));f.requests[4].resolve(page(f.requests[4]));await retry;assert.equal(f.get('science-author-check').disabled,false);
});
test('only successful exact Commit closes; wrong or stale results do not select another origin',async()=>{
  for(const stale of [false,true]){
    const f=await opened(fixture()),promise=f.click('science-author-save');await tick();
    if(stale)f.form.event({kind:'externalReload'});f.requests[3].resolve(stale?taskRef:{...taskRef,id:'wrong'});await promise;
    assert.equal(f.saved.length,0);assert.equal(f.form.active(),true);
  }
  const f=await opened(fixture()),promise=f.click('science-author-save');await tick();f.requests[3].resolve(taskRef);await promise;
  assert.equal(f.form.active(),false);assert.equal(f.saved.length,1);
});
test('duplicate pending Open/navigation and busy/other-owner Open cannot issue extra requests',async()=>{
  const f=fixture(),first=f.form.open(),second=f.form.open();await tick();assert.equal(f.requests.length,1);f.requests[0].resolve(header());await tick();
  f.requests[1].resolve(page(f.requests[1]));f.requests[2].resolve(page(f.requests[2]));await Promise.all([first,second]);
  const next=f.group().next.onclick(),duplicate=f.group().next.onclick();await tick();assert.equal(f.requests.length,4);f.requests[3].resolve(page(f.requests[3]));await Promise.all([next,duplicate]);
  for(const key of ['busy','dataset','legacy','value']){const f=fixture();f.state[key]=true;await f.form.open();assert.equal(f.requests.length,0);}
});
test('returning UUID alone cannot resume: exact native origin and unchanged sequence must be proved',async()=>{
  const f=await opened(fixture());f.state.workspace='two';f.form.stateChanged();f.state.workspace='one';f.form.stateChanged();await tick();
  assert.equal(f.requests[3].action,'scienceDraftOrigin');assert.equal(f.get('science-author-save').disabled,true);
  f.requests[3].reject(Error('origin changed'));await tick();assert.equal(f.get('science-author-save').disabled,true);
  f.state.workspace='two';f.form.stateChanged();f.state.workspace='one';f.form.stateChanged();await tick();f.requests[4].resolve(header());await tick();
  assert.equal(f.get('science-author-save').disabled,false);
  f.form.event({kind:'externalReload'});f.state.workspace='two';f.form.stateChanged();f.state.workspace='one';f.form.stateChanged();await tick();
  assert.equal(f.requests.length,5);assert.equal(f.get('science-author-save').disabled,true);
});
test('large property editor resolves the current value after typed flush and returns to the same parent',async()=>{
  const f=await opened(fixture(),{description:desc('description',null,[0])});
  f.get('science-author-name').value='Changed';f.get('science-author-name').onchange();
  const raw=f.get('science-task-bindings').querySelectorAll('label').find(n=>n.textContent==='Task description').querySelector('button');
  const promise=raw.onclick();await tick();f.requests[3].resolve(header(1));await tick();
  assert.equal(f.requests[4].action,'scienceDraftTaskPage');assert.equal(f.requests[4].args.sequence,1);
  f.requests[4].resolve(page(f.requests[4],{description:desc('description',null,[8])}));await promise;
  assert.equal(f.children.length,1);assert.equal(JSON.stringify(f.children[0].path),'[8]');assert.equal(f.children[0].header.session,'owned');
  const returning=f.children[0].onClosed(header(2));await tick();f.requests[5].resolve(page(f.requests[5]));f.requests[6].resolve(page(f.requests[6]));await returning;
  assert.equal(f.form.active(),true);assert.equal(f.saved.length,0);
});
test('pipelines request independent bounded pages, preserve ordered ordinals and retry failed cursors',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  const cards=page(f.requests[1]);cards.rows[0].transforms.total=30;f.requests[1].resolve(cards);f.requests[2].resolve(page(f.requests[2]));await tick();
  assert.equal(f.requests[3].action,'scienceDraftTaskTransforms');f.requests[3].resolve(pipeline(f.requests[3]));await opening;
  const box=f.card().querySelector('details'),next=box.querySelectorAll('button').find(b=>b.textContent==='Transforms ›');
  let promise=next.onclick();await tick();assert.equal(f.requests[4].args.offset,9);f.requests[4].resolve(pipeline(f.requests[4],{next_offset:null}));await promise;
  const prev=box.querySelectorAll('button').find(b=>b.textContent==='‹ Transforms');promise=prev.onclick();await tick();f.requests[5].reject(Error('retry'));await promise;
  assert.equal(prev.disabled,false);assert.equal(box.querySelectorAll('button').find(b=>b.textContent==='Retry transforms').hidden,false);
  promise=prev.onclick();await tick();assert.equal(f.requests[6].args.offset,0);f.requests[6].resolve(pipeline(f.requests[6]));await promise;
  assert.equal(prev.disabled,true);assert.equal(next.disabled,false);
  assert.equal(box.querySelectorAll('button').find(b=>b.textContent==='Retry transforms').hidden,true);
});
test('exact compatibility rejects mismatched context and late check does not restore an obsolete validity label',async()=>{
  const f=await opened(fixture());f.get('science-task-model').value=JSON.stringify(refs.model);f.get('science-task-dataset').value=JSON.stringify(refs.dataset);
  const promise=f.click('science-task-compatibility');await tick();f.requests[3].resolve({scope:'interface-only-no-fitting-or-execution',version:1,
    task:taskRef,status:'compatible',model:{...refs.model,id:'wrong'},dataset:refs.dataset});await promise;
  assert.equal(f.get('science-task-compatibility-status').textContent,'');assert.match(f.get('science-author-status').textContent,/different exact context/);
  const check=f.click('science-author-check');await tick();f.control('Input port ID').oninput();f.requests[4].resolve(taskRef);await check;
  assert.doesNotMatch(f.get('science-author-status').textContent,/Valid TaskSpec/);
});
test('32 distinct queued edits are ordered and split at 25 operations without Number conversion',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  for(const r of f.requests.slice(1)){
    const value=page(r),first=value.rows[0];value.rows=Array.from({length:6},(_,i)=>({...first,ordinal:i,path:[...value.path,i]}));r.resolve(value);
  }
  await opening;const controls=f.get('science-task-bindings').querySelectorAll('input').filter(n=>n.dataset.large!=='true').slice(0,32);
  assert.equal(controls.length,32);controls.forEach(n=>{n.value='9007199254740993';n.onchange();});
  const checking=f.click('science-author-check');await tick();assert.equal(f.requests[3].args.operations.length,25);
  assert.ok(f.requests[3].args.operations.every(o=>o.text==='9007199254740993'||o.text==='"9007199254740993"'));
  const signatures=f.requests[3].args.operations.map(o=>JSON.stringify([o.target,o.key]));
  f.requests[3].resolve(header(1));await tick();assert.equal(f.requests[4].args.operations.length,7);
  assert.equal(f.requests[4].args.sequence,1);signatures.push(...f.requests[4].args.operations.map(o=>JSON.stringify([o.target,o.key])));
  assert.equal(new Set(signatures).size,32);f.requests[4].resolve(header(2));await tick();
  assert.equal(f.requests[5].action,'scienceDraftCheck');assert.equal(f.requests[5].args.sequence,2);f.requests[5].resolve(taskRef);await checking;
});
test('wrong mutation acknowledgement retains queued text and never issues Check',async()=>{
  const f=await opened(fixture()),n=f.control('Input port ID');n.value='retained';n.onchange();
  const checking=f.click('science-author-check');await tick();f.requests[3].resolve(header(9));await checking;
  assert.equal(f.requests.length,4);assert.equal(n.value,'retained');assert.match(f.get('science-author-status').textContent,/acknowledgement/);
  const retry=f.click('science-author-check');await tick();assert.equal(f.requests[4].args.sequence,0);
  assert.equal(f.requests[4].args.operations[0].text,'"retained"');f.requests[4].resolve(header(1));await tick();f.requests[5].resolve(taskRef);await retry;
});
test('structural edits flush named properties then resolve fresh paths; failed reload fences admission',async()=>{
  const f=await opened(fixture()),name=f.get('science-author-name');name.value='queued';name.onchange();
  const add=f.get('science-task-bindings').querySelectorAll('button').find(n=>n.textContent==='Add input binding');
  const changing=add.onclick();await tick();assert.equal(f.requests[3].action,'scienceDraftTaskEdit');f.requests[3].resolve(header(1));await tick();
  assert.equal(f.requests[4].args.sequence,1);f.requests[4].resolve(page(f.requests[4],{path:[9]}));await tick();
  assert.equal(JSON.stringify(f.requests[5].args.operations[0].path),'[9]');f.requests[5].resolve(header(2));await tick();
  f.requests[6].reject(Error('reload failed'));await tick();assert.equal(f.get('science-author-cancel').disabled,true);
  f.requests[7].resolve(page(f.requests[7]));await changing;assert.equal(f.get('science-author-check').disabled,true);
  assert.equal(f.get('science-author-retry-page').hidden,false);assert.equal(add.disabled,true);
  const retry=f.click('science-author-retry-page');await tick();f.requests[8].resolve(page(f.requests[8]));f.requests[9].resolve(page(f.requests[9]));await retry;
  assert.equal(f.get('science-author-check').disabled,false);
});
test('late initial-read errors cannot change retained range or workspace notification',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  f.state.workspace='two';f.form.stateChanged();const before=f.get('science-author-status').textContent;
  f.requests[1].reject(Error('late failure'));f.requests[2].resolve(page(f.requests[2]));await opening;
  assert.equal(f.get('science-author-status').textContent,before);assert.equal(f.group().prev.disabled,true);
  assert.ok(!f.get('science-task-bindings').children.some(n=>n.textContent.includes('late failure')));assert.equal(f.toasts.length,0);
});
test('initial pipeline failure keeps navigation disabled and explicit Retry reads the same owner',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  const cards=page(f.requests[1]);cards.rows[0].transforms.total=30;f.requests[1].resolve(cards);f.requests[2].resolve(page(f.requests[2]));await tick();
  f.requests[3].reject(Error('initial pipeline'));await opening;const box=f.card().querySelector('details'),buttons=box.querySelectorAll('button');
  assert.equal(buttons.find(b=>b.textContent==='Transforms ›').disabled,true);const retry=buttons.find(b=>b.textContent==='Retry transforms');assert.equal(retry.hidden,false);
  const trying=retry.onclick();await tick();assert.equal(f.requests[4].action,'scienceDraftTaskTransforms');assert.equal(f.requests[4].args.session,'owned');
  f.requests[4].resolve(pipeline(f.requests[4]));await trying;assert.equal(retry.hidden,true);assert.equal(buttons.find(b=>b.textContent==='Transforms ›').disabled,false);
});
