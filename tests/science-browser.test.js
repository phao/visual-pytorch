'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element {
  constructor(tag='div'){this.tagName=tag;this.children=[];this.dataset={};this.listeners={};this.textContent='';this.disabled=false;this.hidden=false;this.open=false;this._value='';}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];this._value='';}
  addEventListener(name,fn){this.listeners[name]=fn;}
  querySelectorAll(selector){return this.children.flatMap(c=>[...((selector==='button'&&c.tagName==='button'||selector==='[data-science-run]'&&'scienceRun' in c.dataset)?[c]:[]),...c.querySelectorAll(selector)]);}
  get options(){return this.children.filter(c=>c.tagName==='option');}
  get value(){return this.tagName==='select'?(this.options.find(o=>o.value===this._value)?.value||this.options[0]?.value||''):this._value;}
  set value(value){this._value=value;}
  get selectedOptions(){return this.options.filter(o=>o.value===this.value);}
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const ref=(id='first')=>({kind:'dataset',id,revision:'a'.repeat(64)});
function fixture(whole=false){
  const source=fs.readFileSync('app/web/'+(whole?'science.js':'science-document.js'),'utf8');
  const ids=new Set([...source.matchAll(/\$\('([^']+)'\)/g)].map(m=>m[1]));
  const candidates=fs.readFileSync('app/web/science-candidates.js','utf8');
  if(whole)for(const m of candidates.matchAll(/\$\('([^']+)'\)/g))ids.add(m[1]);
  if(whole)for(const id of ['science-selection','science-run','science-report','science-import','science-freeze','science-retry','science-make-report','science-generate','science-recovery-export','science-document','science-details'])ids.add(id);
  const nodes=new Map([...ids].map(id=>[id,new Element(id==='science-object'||id==='science-kind'?'select':'div')]));
  const panel=new Element(),document={getElementById:id=>{assert.ok(nodes.has(id),'missing '+id);return nodes.get(id);},createElement:tag=>new Element(tag),querySelector:()=>panel};
  const requests=[],toasts=[],window={},state={workspace:'workspace-1'};
  for(const name of ['LoomScienceResults','LoomScienceNotebooks','LoomScienceAuthor','LoomScienceDocument'])window[name]={mount(){},clear(){},select(){},stateChanged(){},selectionChanged(){},event(){}};
  const api={workspace:()=>state.workspace,busy:()=>false,model:()=>null,notebookJob:()=>null,toast:(...args)=>toasts.push(args),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  if(whole)vm.runInNewContext(candidates,{window,Promise,JSON,Array,Error});
  vm.runInNewContext(source,{window,Promise,JSON,Array,Error});
  if(whole){const kind=nodes.get('science-kind'),o=new Element('option');o.value='dataset';o.textContent='Datasets';kind.append(o);window.LoomScience.mount(document,api);}
  else window.LoomScienceDocument.mount(document,api);
  return {nodes,window,api,state,requests,toasts};
}
const inspection=(request,changes={})=>({scope:'exact-spec-inspection-page',reference:request.args.reference,path:request.args.path,part:request.args.part,offset:request.args.offset,type:'array',total:30,entries:[{ordinal:request.args.offset,preview:{type:'number',value:{text:'0',next_offset:null}}}],scalar:null,next_offset:7,...changes});
const metadata=(reference)=>({offset:0,total:1,next_offset:null,rows:[{...reference,name:reference.id,selected:true}],pinned:null});
const summary=(reference)=>({scope:'exact-spec-summary',reference,id:reference.id,name:reference.id,counts:{observations:20}});

test('inspection ignores older same-workspace responses and late errors',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;
  f.window.LoomScienceDocument.select(ref('old'));await tick();const old=f.requests[0];
  f.window.LoomScienceDocument.select(ref('new'));await tick();const latest=f.requests[1];
  latest.resolve(inspection(latest,{total:99}));await tick();const status=f.nodes.get('science-tree-status').textContent;
  old.reject(Error('obsolete failure'));await tick();assert.equal(f.nodes.get('science-tree-status').textContent,status);assert.equal(f.toasts.length,0);
  f.window.LoomScienceDocument.select(ref('third'));await tick();const third=f.requests[2];
  f.window.LoomScienceDocument.select(ref('fourth'));await tick();const fourth=f.requests[3];
  third.resolve(inspection(third,{total:888}));await tick();assert.match(f.nodes.get('science-tree-status').textContent,/Reading/);
  fourth.resolve(inspection(fourth));await tick();assert.doesNotMatch(f.nodes.get('science-tree-status').textContent,/888/);
});
test('workspace replacement and explicit clear invalidate a pending inspection',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.select(ref());await tick();
  f.state.workspace='workspace-2';f.requests[0].resolve(inspection(f.requests[0]));await tick();assert.equal(f.nodes.get('science-tree-rows').children.length,0);
  f.window.LoomScienceDocument.select(ref('next'));await tick();f.window.LoomScienceDocument.clear();
  f.requests[1].reject(Error('old workspace error'));await tick();assert.equal(f.toasts.length,0);assert.match(f.nodes.get('science-tree-status').textContent,/Open a retained/);
});
test('inspection validates exact identity, path, part and offset before rendering',async()=>{
  for(const changes of [{reference:ref('foreign')},{path:[99]},{part:'key'},{offset:99},{scope:'full-document'}]){
    const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.select(ref());await tick();
    f.requests[0].resolve(inspection(f.requests[0],changes));await tick();assert.equal(f.nodes.get('science-tree-rows').children.length,0);assert.equal(f.toasts.length,1);assert.match(f.toasts[0][0],/does not match/);
  }
});
test('inspection uses actual next offsets, blocks duplicate pending navigation and restores cursor history',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.select(ref());await tick();
  f.requests[0].resolve(inspection(f.requests[0]));await tick();f.nodes.get('science-tree-next').onclick();f.nodes.get('science-tree-next').onclick();await tick();
  assert.equal(f.requests.length,2);assert.equal(f.requests[1].args.offset,7);assert.equal(f.nodes.get('science-tree-rows').children.length,0);assert.equal(f.nodes.get('science-tree-next').disabled,true);
  f.requests[1].resolve(inspection(f.requests[1],{next_offset:null}));await tick();f.nodes.get('science-tree-prev').onclick();await tick();assert.equal(f.requests[2].args.offset,0);
});
test('scalar content is literal text and full-key parent returns to the same value',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.select(ref());await tick();
  f.requests[0].resolve(inspection(f.requests[0],{type:'object',entries:[{ordinal:3,key:{text:'long',next_offset:128},preview:{type:'string',value:{text:'<script>not HTML</script>',next_offset:null}}}]}));await tick();
  const buttons=f.nodes.get('science-tree-rows').querySelectorAll('button');buttons[1].onclick();await tick();assert.equal(f.requests[1].args.part,'key');
  f.requests[1].resolve(inspection(f.requests[1],{type:'string',entries:[],scalar:{text:'<img src=x onerror=alert(1)>',representation:'native-string'},total:26,next_offset:null}));await tick();
  assert.equal(f.nodes.get('science-tree-rows').children[0].textContent,'<img src=x onerror=alert(1)>');
  f.nodes.get('science-tree-parent').onclick();await tick();assert.equal(JSON.stringify(f.requests[2].args.path),'[3]');assert.equal(f.requests[2].args.part,'value');
});
test('current inspection errors remain visible without fabricating a page',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.select(ref());await tick();f.requests[0].reject(Error('Missing revision'));await tick();assert.equal(f.toasts.length,1);assert.equal(f.nodes.get('science-tree-next').disabled,true);assert.equal(f.nodes.get('science-tree-status').textContent,'Missing revision');
});
test('metadata and summary reject stale delivery after external reload with unchanged UUID',async()=>{
  const f=fixture(true);f.nodes.get('science-refresh').onclick();await tick();const old=f.requests[0];
  f.window.LoomScience.event({kind:'externalReload'});await tick();const current=f.requests[1];assert.equal(current.action,'scienceBrowse');
  current.resolve(metadata(ref('new')));await tick();const currentSummary=f.requests[2];
  old.resolve(metadata(ref('old')));await tick();assert.equal(f.nodes.get('science-object').options.length,1);assert.equal(JSON.parse(f.nodes.get('science-object').value).id,'new');
  f.window.LoomScience.event({kind:'externalReload'});await tick();currentSummary.reject(Error('obsolete summary'));await tick();assert.equal(f.toasts.length,0);assert.equal(f.nodes.get('science-document').textContent,'');
  f.requests[3].resolve(metadata(ref('latest')));await tick();f.requests[4].resolve(summary(ref('latest')));await tick();assert.match(f.nodes.get('science-document').textContent,/latest/);
});
test('metadata navigation waits for the read and follows server offsets, not assumed page size',async()=>{
  const f=fixture(true);f.nodes.get('science-refresh').onclick();await tick();assert.equal(f.nodes.get('science-object-next').disabled,true);
  f.requests[0].resolve({...metadata(ref()),next_offset:13,total:100});await tick();f.requests[1].resolve(summary(ref()));await tick();
  f.nodes.get('science-object-next').onclick();f.nodes.get('science-object-next').onclick();await tick();assert.equal(f.requests.length,3);assert.equal(f.requests[2].args.offset,13);
  f.requests[2].resolve({...metadata(ref()),offset:13,next_offset:20,total:100});await tick();f.requests[3].resolve(summary(ref()));await tick();f.nodes.get('science-object-prev').onclick();await tick();assert.equal(f.requests[4].args.offset,0);
});
test('an empty search page is not presented as an empty workspace',async()=>{
  const f=fixture(true);f.nodes.get('science-object-query').value='not found';f.nodes.get('science-refresh').onclick();await tick();
  f.requests[0].resolve({offset:0,total:0,next_offset:null,rows:[],pinned:null});await tick();
  assert.match(f.nodes.get('science-status').textContent,/No matching.*Clear the search/);assert.doesNotMatch(f.nodes.get('science-status').textContent,/stored yet/);assert.equal(f.requests.length,1);
});

test('actual science mount routes experiment preview through bounded controller and reload clears pending candidates',async()=>{
  const f=fixture(true),r={...ref('experiment'),kind:'experiment'},kind=f.nodes.get('science-kind'),o=new Element('option');o.value='experiment';o.textContent='Experiments';kind.append(o);kind.value='experiment';
  f.nodes.get('science-refresh').onclick();await tick();f.requests[0].resolve(metadata(r));await tick();
  f.requests[1].resolve({...summary(r),question:{text:'Compare models',next_offset:null}});await tick();
  const pending=f.requests[2];assert.equal(pending.action,'sciencePreviewPage');assert.equal(pending.args.workspace_id,'workspace-1');
  f.window.LoomScience.event({kind:'externalReload'});await tick();pending.reject(Error('old preview'));await tick();
  assert.equal(f.toasts.length,0);assert.equal(f.nodes.get('science-candidates').children.length,0);assert.equal(f.nodes.get('science-candidate-refresh').disabled,true);
  const browse=f.requests[3];browse.resolve(metadata(r));await tick();f.requests[4].resolve({...summary(r),question:{text:'Current comparison',next_offset:null}});await tick();
  const latest=f.requests[5];latest.resolve({version:1,scope:'stored-experiment-preview',workspace_id:'workspace-1',experiment:r,offset:0,limit:25,total_combinations_exact:'0',candidates:[],next_offset:null,execution:'no-fitting-or-training'});await tick();
  assert.match(f.nodes.get('science-page').textContent,/0–0 of 0/);assert.equal(f.nodes.get('science-candidate-refresh').disabled,false);
  assert.equal(f.requests.some(r=>r.action==='sciencePreview'),false);
});

const recovery=(request,changes={})=>({...inspection(request),version:1,scope:'experiment-recovery-inspection',
  scope_id:request.args.scope_id,sequence:request.args.sequence,id:request.args.id,limit:25,...changes});
const recoveryIdentity=()=>({scope_id:'a'.repeat(32),sequence:1,id:'b'.repeat(32)});
test('recovery tree uses exact window identity and cannot edit captured values',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
  assert.equal(f.requests[0].action,'scienceRecoveryInspect');assert.equal('reference' in f.requests[0].args,false);
  f.requests[0].resolve(recovery(f.requests[0]));await tick();assert.match(f.nodes.get('science-tree-status').textContent,/session recovery evidence/);
  assert.equal(f.nodes.get('science-tree-edit').disabled,true);f.nodes.get('science-tree-rows').querySelectorAll('button')[0].onclick();await tick();
  assert.equal(JSON.stringify(f.requests[1].args.path),'[0]');assert.equal(f.requests[1].action,'scienceRecoveryInspect');
  assert.equal(f.nodes.get('science-tree-edit').disabled,true);
});
test('recovery inspection rejects foreign identity/sequence and preserves exact literal scalar text',async()=>{
  for(const changes of [{scope:'exact-spec-inspection-page'},{scope_id:'foreign'},{sequence:2},{id:'foreign'},{limit:99}]){
    const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
    f.requests[0].resolve(recovery(f.requests[0],changes));await tick();assert.equal(f.toasts.length,1);assert.equal(f.nodes.get('science-tree-rows').children.length,0);
  }
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
  f.requests[0].resolve(recovery(f.requests[0],{type:'number',entries:[],scalar:{text:'9007199254740993'},total:16,next_offset:null}));await tick();
  assert.equal(f.nodes.get('science-tree-rows').children[0].textContent,'9007199254740993');
});
test('failed tree Back keeps history, Retry restores current cursor and Back can be retried',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
  f.requests[0].resolve(recovery(f.requests[0]));await tick();f.nodes.get('science-tree-next').onclick();await tick();
  f.requests[1].resolve(recovery(f.requests[1],{next_offset:null}));await tick();f.nodes.get('science-tree-prev').onclick();await tick();
  f.requests[2].reject(Error('failed Back'));await tick();assert.equal(f.nodes.get('science-tree-prev').disabled,false);
  f.nodes.get('science-tree-retry').onclick();await tick();assert.equal(f.requests[3].args.offset,7);
  f.requests[3].resolve(recovery(f.requests[3],{next_offset:null}));await tick();f.nodes.get('science-tree-prev').onclick();await tick();assert.equal(f.requests[4].args.offset,0);
});
test('stale navigation completion cannot change a newer selection cursor or history',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
  f.requests[0].resolve(recovery(f.requests[0]));await tick();f.nodes.get('science-tree-next').onclick();await tick();const old=f.requests[1];
  f.window.LoomScienceDocument.select(ref('new'));await tick();f.requests[2].resolve(inspection(f.requests[2]));await tick();
  old.reject(Error('obsolete Back'));await tick();assert.equal(f.toasts.length,0);
  assert.equal(f.nodes.get('science-tree-prev').disabled,true);f.nodes.get('science-tree-next').onclick();await tick();assert.equal(f.requests[3].args.offset,7);
});
test('recovery invalidation clears only recovery views and fences late errors',async()=>{
  const f=fixture();f.nodes.get('science-details').open=true;f.window.LoomScienceDocument.selectRecovery(recoveryIdentity());await tick();
  f.window.LoomScienceDocument.clearRecovery();f.requests[0].reject(Error('old recovery'));await tick();assert.equal(f.toasts.length,0);
  f.window.LoomScienceDocument.select(ref());await tick();f.window.LoomScienceDocument.clearRecovery();f.requests[1].resolve(inspection(f.requests[1]));await tick();
  assert.ok(f.nodes.get('science-tree-rows').children.length);
});
