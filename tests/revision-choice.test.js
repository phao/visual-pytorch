'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element {
  constructor(tag){this.tagName=tag;this.children=[];this.parent=null;this.dataset={};this.attributes={};this.textContent='';this.disabled=false;this._value='';}
  append(...nodes){for(const n of nodes){n.remove();n.parent=this;this.children.push(n);}}
  remove(){if(this.parent){this.parent.children=this.parent.children.filter(c=>c!==this);this.parent=null;}}
  before(node){const p=this.parent,i=p.children.indexOf(this);node.remove();node.parent=p;p.children.splice(i,0,node);}
  after(node){const p=this.parent,i=p.children.indexOf(this);node.remove();node.parent=p;p.children.splice(i+1,0,node);}
  closest(tag){return this.tagName===tag?this:this.parent?.closest(tag)||null;}
  replaceChildren(...nodes){this.children.forEach(c=>c.parent=null);this.children=[];this.append(...nodes);}
  setAttribute(k,v){this.attributes[k]=v;}
  get options(){return this.children.filter(c=>c.tagName==='option');}
  set value(v){this._value=v;}
  get value(){return this.tagName==='select'?(this.options.find(c=>c.value===this._value)?.value||this.options[0]?.value||''):this._value;}
  get isConnected(){return this.tagName==='body'||Boolean(this.parent?.isConnected);}
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const reference=(id='historical')=>({kind:'model',id,revision:'a'.repeat(64)});
const exactKey=ref=>JSON.stringify(ref);
function fixture(opts={}){
  const body=new Element('body'),label=new Element('label'),select=new Element('select');body.append(label);label.append(select);
  const state={workspace:'workspace-1',busy:false,alive:true,chosen:opts.chosen===undefined?reference():opts.chosen};
  const requests=[],toasts=[],window={},root={createElement:tag=>new Element(tag)};
  vm.runInNewContext(fs.readFileSync('app/web/revision-choice.js','utf8'),{window,Promise,JSON,Error});
  const services={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...args)=>toasts.push(args),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  const controller=window.LoomRevisionChoice.attach(root,select,services,{kind:'model',getReference:opts.liveSelect?()=>select.value?JSON.parse(select.value):null:()=>state.chosen,defaultHead:Boolean(opts.defaultHead),valid:()=>state.alive});
  const box=label.parent.children.find(n=>n.className==='revision-choice'),find=action=>box.children.find(c=>c.dataset.choiceAction===action);
  return {controller,state,requests,toasts,select,label,body,box,find};
}
function page(request,changes={}){return {scope:'scientific-metadata-page',kind:'model',query:request.args.query,offset:request.args.offset,total:100,next_offset:request.args.offset+13,rows:[{...reference('other'),name:'Other <img>',selected:true}],pinned:request.args.preferred?{...request.args.preferred,name:'Old architecture',selected:false}:null,missing_pin:null,...changes};}

test('bounded request keeps an exact pin, disables pending actions and never transmits a draft',async()=>{
  const f=fixture();const done=f.controller.refresh();assert.equal(f.requests.length,1);assert.equal(f.requests[0].action,'scienceBrowse');
  assert.equal(f.requests[0].args.limit,25);assert.equal(f.requests[0].args.allow_missing_pin,true);assert.equal('draft' in f.requests[0].args,false);
  assert.equal(f.select.disabled,true);assert.equal(f.find('query').disabled,true);assert.equal(f.select.value,exactKey(reference()));
  f.requests[0].resolve(page(f.requests[0]));await done;assert.equal(f.select.disabled,false);assert.equal(f.select.options.length,3);assert.equal(f.select.value,exactKey(reference()));
});
test('search/page refresh retains a historical pin without modifying its reference',async()=>{
  const f=fixture();const before=JSON.stringify(f.state.chosen);let done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0]));await done;
  f.find('query').value='different';f.find('search').onclick();await tick();assert.equal(f.requests[1].args.query,'different');
  f.requests[1].resolve(page(f.requests[1],{rows:[],total:0,next_offset:null}));await tick();assert.equal(f.select.value,before);assert.equal(f.select.options.length,2);assert.match(f.box.children.at(-1).textContent,/pinned/);assert.equal(JSON.stringify(f.state.chosen),before);
});
test('unavailable exact pin is labelled, not dropped or replaced by a head',async()=>{
  const f=fixture({defaultHead:true});const done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0],{pinned:null,missing_pin:reference()}));await done;
  assert.equal(f.select.value,exactKey(reference()));assert.match(f.select.options.at(-1).textContent,/Unavailable exact model/);assert.match(f.box.children.at(-1).textContent,/unavailable/);
});
test('task context defaults to a current head only once, never guessing experiment selections',async()=>{
  const f=fixture({chosen:null,defaultHead:true,liveSelect:true});let done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0]));await done;assert.equal(f.select.value,exactKey(reference('other')));
  f.select.value='';done=f.controller.refresh();f.requests[1].resolve(page(f.requests[1]));await done;assert.equal(f.select.value,'');
  const g=fixture({chosen:null});done=g.controller.refresh();g.requests[0].resolve(page(g.requests[0]));await done;assert.equal(g.select.value,'');
});
test('rapid next/previous navigation follows returned offsets and guards duplicate pending actions',async()=>{
  const f=fixture();let done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0]));await done;
  f.find('next').onclick();f.find('next').onclick();await tick();assert.equal(f.requests.length,2);assert.equal(f.requests[1].args.offset,13);
  f.requests[1].resolve(page(f.requests[1],{next_offset:null}));await tick();f.find('prev').onclick();await tick();assert.equal(f.requests[2].args.offset,0);
});
test('newer same-workspace refresh rejects old success and late errors',async()=>{
  const f=fixture();const first=f.controller.refresh(),second=f.controller.refresh();f.requests[1].resolve(page(f.requests[1],{total:777}));await second;const status=f.box.children.at(-1).textContent;
  f.requests[0].reject(Error('obsolete'));await first;assert.equal(f.toasts.length,0);assert.equal(f.box.children.at(-1).textContent,status);
  const third=f.controller.refresh(),fourth=f.controller.refresh();f.requests[3].resolve(page(f.requests[3],{total:333}));await fourth;f.requests[2].resolve(page(f.requests[2],{total:999}));await third;assert.match(f.box.children.at(-1).textContent,/333/);assert.doesNotMatch(f.box.children.at(-1).textContent,/999/);
});
test('workspace/owner replacement, detached form and disposal suppress stale delivery',async()=>{
  for(const change of [f=>f.state.workspace='workspace-2',f=>f.state.alive=false,f=>f.label.remove(),f=>f.controller.dispose()]){
    const f=fixture(),done=f.controller.refresh();change(f);f.requests[0].reject(Error('old scope'));await done;assert.equal(f.toasts.length,0);
  }
});
test('kind, offset, query and pin mismatch are visible errors, not foreign choices',async()=>{
  for(const changes of [{kind:'task'},{offset:999},{query:'other'},{pinned:{...reference('foreign'),name:'Wrong'}},{missing_pin:reference('foreign')},{rows:[{...reference(),kind:'task',name:'Wrong'}]}]){
    const f=fixture(),done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0],changes));await done;assert.equal(f.toasts.length,1);assert.equal(f.select.value,exactKey(reference()));assert.equal(f.find('next').disabled,true);
  }
});
test('search tools are outside the select label and dispose restores the original structure',()=>{
  const f=fixture();assert.equal(f.select.closest('label'),f.label);assert.equal(f.label.children.length,1);assert.equal(f.find('query').closest('label'),null);
  assert.equal(f.find('query').attributes['aria-label'],'Search model revisions');assert.equal(f.find('next').attributes['aria-label'],'Next model revisions');
  f.controller.dispose();assert.equal(f.body.children[0],f.label);assert.equal(f.body.children.length,1);
});
test('busy state prevents user page requests; literal metadata names stay text',async()=>{
  const f=fixture(),done=f.controller.refresh();f.requests[0].resolve(page(f.requests[0]));await done;assert.equal(f.select.options[1].textContent.includes('<img>'),true);
  f.state.busy=true;f.controller.controls();f.find('next').onclick();await tick();assert.equal(f.requests.length,1);assert.equal(f.select.disabled,true);
});
