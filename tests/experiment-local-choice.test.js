'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture.js');
const ref=(id='model')=>({kind:'model',id,revision:'a'.repeat(64)});
function fixture(collection='models',chosen=ref()){
  const body=new Element('body'),label=new Element('label'),select=new Element('select');body.append(label);label.append(select);
  const state={workspace:'one',sequence:0,alive:true,busy:false,chosen},requests=[],toasts=[],window={};
  const root={createElement:tag=>new Element(tag)};
  for(const name of ['revision-choice','experiment-local-choice'])vm.runInNewContext(fs.readFileSync('app/web/'+name+'.js','utf8'),{window});
  const api={arguments:()=>({session:'owned',sequence:state.sequence}),workspace:()=>state.workspace,id:()=> 'experiment',busy:()=>state.busy,
    toast:(...a)=>toasts.push(a),run:fn=>Promise.resolve().then(fn),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  const choice=window.LoomExperimentLocalChoice.attach(root,select,api,{collection,getValue:()=>state.chosen,valid:()=>state.alive});
  const box=label.parent.children.find(n=>n.dataset.localChoice),find=name=>box.children.find(n=>n.dataset.choiceAction===name);
  return {body,label,select,state,requests,toasts,choice,box,find};
}
function page(r,changes={}){const a=r.args,value=a.collection==='models'?ref('other'):'p-other';
  return {scope:'owned-experiment-choices',version:1,...a,workspace_id:'one',id:'experiment',total:80,next_offset:a.offset+1,
    rows:[{ordinal:7,value,label:'Other',valid:true}],pinned:a.pin===null?null:{ordinal:31,value:a.pin,label:'Selected',valid:true},missing_pin:null,ambiguous_pin:false,...changes};
}
async function opened(f){const done=f.choice.refresh();f.requests[0].resolve(page(f.requests[0]));await done;return f;}
test('local model choices normalize key order without changing exact references or sending drafts',async()=>{
  const value={id:'model',revision:'a'.repeat(64),kind:'model'},f=await opened(fixture('models',value));
  assert.equal(f.select.value,JSON.stringify(ref()));assert.equal(f.requests[0].action,'scienceDraftExperimentChoices');
  assert.equal(f.requests[0].args.limit,25);assert.equal(f.state.chosen,value);assert.equal('draft' in f.requests[0].args,false);
});
test('pairing options expose literal IDs including spaces, commas and quotes, not encoded JSON',async()=>{
  const id='pair, " λ ',f=await opened(fixture('data_tasks',id));assert.equal(f.select.value,id);assert.equal(f.requests[0].args.pin,id);
});
test('independent search retains an outside exact pin and never advances to a browsing head',async()=>{
  const f=await opened(fixture());f.find('query').value='different';const done=f.find('search').onclick();await tick();
  assert.equal(f.requests[1].args.query,'different');f.requests[1].resolve(page(f.requests[1],{rows:[],total:0,next_offset:null}));await done;
  assert.equal(f.select.value,JSON.stringify(ref()));assert.equal(f.state.chosen.id,'model');
});
test('missing, duplicate and incomplete identities stay visible without coercion',async()=>{
  const f=fixture(),done=f.choice.refresh();f.requests[0].resolve(page(f.requests[0],{pinned:null,missing_pin:ref(),rows:[{ordinal:4,value:null,label:'Incomplete selection 5',valid:false}]}));await done;
  assert.equal(f.select.value,JSON.stringify(ref()));assert.equal(f.select.options.find(o=>o.value==='incomplete/4').disabled,true);assert.match(f.box.children.at(-1).textContent,/unavailable/);
  const next=f.choice.refresh();f.requests[1].resolve(page(f.requests[1],{ambiguous_pin:true}));await next;assert.match(f.box.children.at(-1).textContent,/duplicate/);
});
test('wrong-kind selected references are not pinned or silently replaced, allowing explicit repair',async()=>{
  const value={kind:'task',id:'wrong-kind',revision:'b'.repeat(64)},f=fixture('models',value),done=f.choice.refresh();
  assert.equal(f.requests[0].args.pin,null);f.requests[0].resolve(page(f.requests[0]));await done;
  assert.equal(f.select.value,JSON.stringify(value));assert.equal(f.select.options.some(o=>o.value===JSON.stringify(ref('other'))),true);assert.equal(f.state.chosen,value);
});
test('actual next cursors and failed Back retries preserve navigation history',async()=>{
  const f=await opened(fixture());let done=f.find('next').onclick();await tick();assert.equal(f.requests[1].args.offset,1);f.requests[1].resolve(page(f.requests[1]));await done;
  done=f.find('prev').onclick();await tick();f.requests[2].reject(Error('temporary'));await done;assert.equal(f.find('prev').disabled,false);
  done=f.find('prev').onclick();await tick();assert.equal(f.requests[3].args.offset,0);f.requests[3].resolve(page(f.requests[3]));await done;assert.equal(f.find('prev').disabled,true);
});
test('closed response identity and valid row kinds reject foreign choices',async()=>{
  for(const changes of [{session:'foreign'},{sequence:2},{workspace_id:'two'},{id:'foreign'},{collection:'data_tasks'},{offset:8},{query:'wrong'},{pin:ref('foreign')},
    {rows:[{ordinal:1,value:{...ref(),kind:'task'},label:'Wrong',valid:true}]}]){
    const f=fixture(),done=f.choice.refresh();f.requests[0].resolve(page(f.requests[0],changes));await done;assert.equal(f.toasts.length,1);assert.equal(f.find('next').disabled,true);
    assert.equal(f.select.value,JSON.stringify(ref()));
  }
});
test('late sequence, selection and workspace replies cannot overwrite current selections',async()=>{
  for(const change of [f=>f.state.sequence++,f=>f.state.chosen=ref('changed'),f=>f.state.workspace='two',f=>f.state.alive=false,f=>f.choice.dispose()]){
    const f=fixture(),done=f.choice.refresh();change(f);f.requests[0].resolve(page(f.requests[0]));await done;assert.equal(f.toasts.length,0);assert.equal(f.find('next').disabled,true);
  }
});
test('newer reads suppress old success/errors and busy or duplicate page gestures issue no request',async()=>{
  const f=fixture(),old=f.choice.refresh(),fresh=f.choice.refresh();f.requests[1].resolve(page(f.requests[1],{total:777}));await fresh;f.requests[0].reject(Error('obsolete'));await old;
  assert.equal(f.toasts.length,0);assert.match(f.box.children.at(-1).textContent,/777/);
  f.state.busy=true;await f.find('next').onclick();assert.equal(f.requests.length,2);f.state.busy=false;
  const next=f.find('next').onclick(),duplicate=f.find('next').onclick();await tick();assert.equal(f.requests.length,3);f.requests[2].resolve(page(f.requests[2]));await Promise.all([next,duplicate]);
});
test('tools are outside the label, accessible, and disposal restores the original hierarchy',()=>{
  const f=fixture();assert.equal(f.find('query').closest('label'),null);assert.equal(f.select.closest('label'),f.label);
  assert.equal(f.find('next').attributes['aria-label'],'Next selected models');f.choice.dispose();assert.equal(f.body.children.length,1);assert.equal(f.body.children[0],f.label);
});
