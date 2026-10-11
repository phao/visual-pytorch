'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
class Element{constructor(tag='div',text=''){this.tagName=tag;this.textContent=text;this.children=[];this.dataset={};this.attributes={};this.disabled=false;this.value='';}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=[...nodes];}setAttribute(k,v){this.attributes[k]=v;}}
const refs={model:{kind:'model',id:'model',revision:'a'.repeat(64)},dataset:{kind:'dataset',id:'data',revision:'b'.repeat(64)}};
const row=(id='x')=>({id,name:id,dtype:'float32',ordinal:0,shape:[2],shape_rank:1,shape_complete:true,axes:null,axes_complete:true,vocabulary_count:null});
const page=(request,extra={})=>({scope:'task-context-page',version:1,...request.args,requested_pin:request.args.pin,
  rows:[row()],pinned:request.args.pin?row(request.args.pin):null,missing_pin:null,record_count:1000,total:80,source_total:80,next_offset:7,...extra});
function fixture(){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},requests=[],toasts=[],state={workspace:'one',valid:true,busy:false,blocked:false};
  const root={getElementById:get,createElement:tag=>new Element(tag)},window={};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,blocked:()=>state.blocked,updateControls(){},toast:(...a)=>toasts.push(a),
    call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/task-context.js','utf8'),{window,JSON,Array,Promise,Error});
  const context=window.LoomTaskContext;context.mount(root,api);
  const group=kind=>{const box=get('science-task-context-pages').children.find(n=>n.dataset.contextCollection===kind),grid=box.children[1],actions=box.children[2];
    return {box,query:grid.children[0].children[0],pin:grid.children[1].children[0],search:actions.children[0],prev:actions.children[1],next:actions.children[2],range:box.children[3]};};
  return {get,state,requests,toasts,context,group,show:()=>context.show(refs,()=>state.valid)};
}
async function opened(f){const promise=f.show();await tick();assert.equal(f.requests.length,3);f.requests.forEach(r=>r.resolve(page(r)));await promise;return f;}
test('actual controller requests three independent bounded exact pages and does not fetch complete context',async()=>{
  const f=await opened(fixture());assert.deepEqual(f.requests.map(r=>r.args.collection),['fields','input_ports','output_ports']);
  assert.ok(f.requests.every(r=>r.action==='scienceTaskContextPage'&&r.args.limit===25&&r.args.workspace_id==='one'));
  assert.equal(f.get('science-task-fields').children[0].value,'x');assert.match(f.get('science-task-context-status').textContent,/1000 observations/);
  assert.equal(f.group('fields').pin.maxLength,400,'Allow 200 supplementary-character ID code points, not only 100');
});
test('search and exact pin update one collection only; missing pins are visible and never replaced',async()=>{
  const f=await opened(fixture()),g=f.group('fields');g.query.value='missing query';g.pin.value='manual';const promise=g.search.onclick();await tick();
  assert.equal(f.requests[3].args.collection,'fields');assert.equal(f.requests[3].args.query,'missing query');assert.equal(f.requests[3].args.pin,'manual');
  f.requests[3].resolve(page(f.requests[3],{rows:[],total:0,next_offset:null,pinned:null,missing_pin:'manual'}));await promise;
  assert.match(g.range.textContent,/ID unavailable: manual/);assert.equal(f.get('science-task-fields').children.length,0);
  assert.equal(f.get('science-task-input-ports').children.length,1);
});
test('actual forward cursors and failed previous retry do not consume successful navigation history',async()=>{
  const f=await opened(fixture()),g=f.group('input_ports');let promise=g.next.onclick();await tick();assert.equal(f.requests[3].args.offset,7);
  f.requests[3].resolve(page(f.requests[3],{next_offset:null}));await promise;
  promise=g.prev.onclick();await tick();f.requests[4].reject(Error('temporary'));await promise;assert.equal(g.prev.disabled,false);
  promise=g.prev.onclick();await tick();assert.equal(f.requests[5].args.offset,0);f.requests[5].resolve(page(f.requests[5]));await promise;assert.equal(g.prev.disabled,true);
});
test('wrong context refs/query/collection/cursor/pin/owner cannot overwrite displayed options',async()=>{
  for(const changes of [{scope:'full-context'},{version:2},{workspace_id:'wrong'},{model:{...refs.model,revision:'c'.repeat(64)}},
    {dataset:{...refs.dataset,id:'wrong'}},{collection:'fields'},{query:'wrong'},{offset:999},{requested_pin:'wrong'}]){
    const f=await opened(fixture()),g=f.group('input_ports'),list=f.get('science-task-input-ports'),prior=list.children[0],promise=g.next.onclick();await tick();
    f.requests[3].resolve(page(f.requests[3],changes));await promise;assert.equal(list.children[0],prior);assert.match(g.range.textContent,/do not match/);
  }
});
test('clear, invalidated same-UUID origin and workspace switch fence both stale replies and errors',async()=>{
  for(const transition of ['clear','invalidate','workspace'])for(const failure of [false,true]){
    const f=await opened(fixture()),g=f.group('fields'),promise=g.next.onclick();await tick();
    if(transition==='clear')f.context.clear();else if(transition==='workspace')f.state.workspace='two';else f.state.valid=false;
    const prior=f.get('science-task-fields').children[0],notice=g.range.textContent;
    if(failure)f.requests[3].reject(Error('obsolete'));else f.requests[3].resolve(page(f.requests[3],{rows:[row('obsolete')]}));await promise;
    assert.equal(f.get('science-task-fields').children[0],prior);assert.equal(g.range.textContent,notice);assert.equal(f.toasts.length,0);
  }
});
test('duplicate pending page/search and parent busy controls cannot issue extra reads',async()=>{
  const f=await opened(fixture()),g=f.group('fields'),first=g.next.onclick(),second=g.next.onclick();await tick();assert.equal(f.requests.length,4);
  assert.equal(g.search.disabled,true);f.requests[3].resolve(page(f.requests[3]));await Promise.all([first,second]);
  f.state.blocked=true;f.context.controls();await g.next.onclick();assert.equal(f.requests.length,4);
  f.state.blocked=false;f.state.busy=true;await g.search.onclick();assert.equal(f.requests.length,4);
});
test('same exact-context refresh preserves search/pin/cursor but new refs reset them without binding DOM edits',async()=>{
  const f=await opened(fixture()),g=f.group('fields');g.query.value='x';g.pin.value='historical';let promise=g.search.onclick();await tick();f.requests[3].resolve(page(f.requests[3]));await promise;
  f.get('manual-binding').value='unfinished stable ID';promise=f.show();await tick();assert.equal(f.requests[4].args.query,'x');assert.equal(f.requests[4].args.pin,'historical');
  f.requests.slice(4).forEach(r=>r.resolve(page(r)));await promise;assert.equal(f.get('manual-binding').value,'unfinished stable ID');
  promise=f.context.show({...refs,dataset:{...refs.dataset,revision:'c'.repeat(64)}},()=>true);await tick();assert.equal(f.requests[7].args.query,'');assert.equal(f.requests[7].args.pin,null);
  f.requests.slice(7).forEach(r=>r.resolve(page(r)));await promise;
});
test('partial shape previews and category counts are explicitly labelled, with exact IDs kept out of innerHTML',async()=>{
  const f=fixture(),promise=f.show();await tick();for(const r of f.requests)r.resolve(page(r,{rows:[{...row('<script>literal</script>'),name_complete:false,shape:[1,2,3,4],shape_rank:8,shape_complete:false,vocabulary_count:20000}]}));await promise;
  const option=f.get('science-task-fields').children[0];assert.equal(option.value,'<script>literal</script>');assert.match(option.textContent,/preview, rank 8/);assert.match(option.textContent,/name preview/);assert.match(option.title,/20000 fixed categories/);
});
