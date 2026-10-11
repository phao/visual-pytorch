'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(tag='div',text=''){this.tagName=tag;this.textContent=text;this.children=[];this.dataset={};this.attributes={};this.disabled=false;this.value='';this.hidden=false;}
  append(...nodes){this.children.push(...nodes);}
  replaceChildren(...nodes){this.children=[...nodes];}
  setAttribute(key,value){this.attributes[key]=value;}
  removeAttribute(key){delete this.attributes[key];}
  matches(selector){return selector.split(',').some(s=>s.startsWith('[data-')?s.slice(6,-1).replace(/-([a-z])/g,(_,c)=>c.toUpperCase()) in this.dataset:s===this.tagName);}
  querySelectorAll(selector){return this.children.flatMap(n=>[...(n.matches(selector)?[n]:[]),...n.querySelectorAll(selector)]);}
  querySelector(selector){return this.querySelectorAll(selector)[0]||null;}
  contains(node){return this===node||this.children.some(n=>n.contains(node));}
  dispatchEvent(event){this['on'+event.type]?.(event);}
}
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const header=(sequence=0)=>({scope:'owned-authoring-session',version:1,kind:'dataset',id:'data',session:'owned',sequence,workspace_id:'one',base:null,name:'Measurements',expected_head:null});
const desc=(key,text,path=[0],present=true)=>({key,text,path,present,summary:text===null?{type:'array',size:20000}:null});
function page(request,changes={}){
  return {scope:'owned-dataset-page',session:'owned',sequence:request.args.sequence,workspace_id:'one',id:'data',name:'Measurements',
    field_offset:request.args.field_offset,row_offset:request.args.row_offset,field_total:10,row_total:50,next_field_offset:3,next_row_offset:7,
    fields:[{id:'x',ordinal:request.args.field_offset,path:[0,0],properties:['name','dtype','shape','axes','nullable','vocabulary','coordinates','modality','unit'].map(key=>
      desc(key,key==='dtype'?'"float32"':key==='shape'?'[1]':null,[0,0,1],key==='dtype'||key==='shape'))}],
    rows:[{id:'r0',ordinal:request.args.row_offset,path:[1,0],cells:[desc('x','[1]',[1,0,0,0])]}],
    metadata:[desc('provenance',null,[],false),desc('usage',null,[3])],...changes};
}
function fixture(){
  const nodes=new Map(),get=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  const root={getElementById:get,createElement:tag=>new Element(tag),activeElement:null};
  get('science-author').append(get('science-author-name'),get('science-author-fields'),get('science-author-records'),get('science-author-metadata'));
  const state={workspace:'one',busy:false,author:false,value:false},requests=[],toasts=[],saved=[],children=[];
  const window={LoomDatasetAuthoring:require('../app/web/dataset-authoring.js'),LoomScienceAuthor:{stateChanged(){},ownActive:()=>state.author},
    LoomScienceDocument:{stateChanged(){}},LoomDraftValueEditor:{active:()=>state.value,openOwned:async target=>children.push(target)}};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...a)=>toasts.push(a),call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/dataset-session-form.js','utf8'),{window,Promise,JSON,Array,Error,Event:class{constructor(type){this.type=type;}},crypto:{randomUUID:()=> 'new-record'}});
  const form=window.LoomDatasetSessionForm;form.mount(root,api,async result=>saved.push(result));
  return {nodes,get,root,state,requests,toasts,saved,children,form,window,click:id=>get(id).onclick()};
}
async function opened(f){const promise=f.form.open();await tick();f.requests[0].resolve(header());await tick();f.requests[1].resolve(page(f.requests[1]));await promise;return f;}
function cell(f){return f.get('science-author-records').querySelector('textarea');}
function metadataButton(f){return f.get('science-author-metadata').querySelector('button');}

test('actual typed controller opens only a header/page and shows explicit raw-value targets',async()=>{
  const f=await opened(fixture());assert.deepEqual(f.requests.map(r=>r.action),['scienceDraftOpen','scienceDraftDatasetPage']);
  assert.equal(cell(f).value,'[1]');assert.equal(f.get('science-author-metadata').querySelectorAll('textarea')[1].dataset.large,'true');
  assert.equal(f.get('science-author-check').disabled,false);assert.match(f.get('science-author-row-page').textContent,/1–1 \/ 50/);
});
test('focused input is captured before pending disables it; native edits carry raw exact tokens and no whole document',async()=>{
  const f=await opened(fixture()),input=cell(f);input.value='[9007199254740993]';input.oninput();f.root.activeElement=input;
  const promise=f.click('science-author-check');await tick();const request=f.requests[2];assert.equal(request.action,'scienceDraftDatasetEdit');
  assert.equal(request.args.operations[0].text,'[9007199254740993]');assert.equal('draft' in request.args,false);assert.equal(input.disabled,true);
  request.resolve(header(1));await tick();assert.equal(f.requests[3].action,'scienceDraftCheck');assert.equal(f.requests[3].args.sequence,1);
  f.requests[3].resolve({revision:'a'.repeat(64)});await promise;assert.match(f.get('science-author-status').textContent,/Valid DatasetSpec/);
});
test('malformed typed text survives failed batch/paging and corrected edits replace that control rather than replaying the invalid value',async()=>{
  const f=await opened(fixture()),input=cell(f);input.value='[broken';input.onchange();let promise=f.click('science-author-rows-next');await tick();
  f.requests[2].reject(Error('Dataset edit #0: Invalid revision JSON.'));await promise;
  assert.equal(input.value,'[broken');assert.equal(input.dataset.invalid,'');assert.match(f.get('science-author-row-page').textContent,/1–1/);
  input.value='[9]';input.onchange();promise=f.click('science-author-rows-next');await tick();assert.equal(f.requests[3].args.operations.length,1);assert.equal(f.requests[3].args.operations[0].text,'[9]');
  f.requests[3].resolve(header(1));await tick();assert.equal(f.requests[4].args.row_offset,7);f.requests[4].resolve(page(f.requests[4]));await promise;
});
test('field/row navigation uses actual cursors and failed backward reads keep retry history',async()=>{
  const f=await opened(fixture());let promise=f.click('science-author-fields-next');await tick();assert.equal(f.requests[2].args.field_offset,3);f.requests[2].resolve(page(f.requests[2]));await promise;
  promise=f.click('science-author-fields-prev');await tick();f.requests[3].reject(Error('read failed'));await promise;assert.equal(f.get('science-author-fields-prev').disabled,false);
  promise=f.click('science-author-fields-prev');await tick();assert.equal(f.requests[4].args.field_offset,0);f.requests[4].resolve(page(f.requests[4]));await promise;
  assert.equal(f.get('science-author-fields-prev').disabled,true);
});
test('staged large values resolve stable targets after pending typed edits and reuse the same native session',async()=>{
  const f=await opened(fixture());f.get('science-author-name').value='Changed';f.get('science-author-name').onchange();const promise=metadataButton(f).onclick();await tick();
  f.requests[2].resolve(header(1));await tick();assert.equal(f.requests[3].action,'scienceDraftDatasetValue');assert.equal(f.requests[3].args.sequence,1);
  f.requests[3].resolve({scope:'owned-dataset-value',session:'owned',sequence:1,target:f.requests[3].args.target,path:[2]});await promise;
  assert.equal(f.children.length,1);assert.equal(f.children[0].header.session,'owned');assert.equal(f.children[0].header.sequence,1);assert.equal(JSON.stringify(f.children[0].path),'[2]');
  const returned=f.children[0].onClosed(header(2));await tick();assert.equal(f.requests[4].args.sequence,2);f.requests[4].resolve(page(f.requests[4]));await returned;
});
test('duplicate pending Open and navigation do not issue extra requests',async()=>{
  const f=fixture(),first=f.form.open(),second=f.form.open();await tick();assert.equal(f.requests.length,1);f.requests[0].resolve(header());await tick();f.requests[1].resolve(page(f.requests[1]));await Promise.all([first,second]);
  const next=f.click('science-author-rows-next'),duplicate=f.click('science-author-rows-next');await tick();assert.equal(f.requests.length,3);f.requests[2].resolve(page(f.requests[2]));await Promise.all([next,duplicate]);
});
test('page identities/cursors are checked before replacing current controls',async()=>{
  for(const changes of [{session:'foreign'},{sequence:9},{row_offset:88},{field_offset:55},{workspace_id:'foreign'},{scope:'full-document'}]){
    const f=await opened(fixture()),input=cell(f),promise=f.click('science-author-rows-next');await tick();f.requests[2].resolve(page(f.requests[2],changes));await promise;
    assert.equal(cell(f),input);assert.match(f.get('science-author-status').textContent,/does not match/);
  }
});
test('workspace replacement and same-UUID reload fence late success/error without discarding visible text',async()=>{
  for(const replace of [true,false])for(const failure of [true,false]){
    const f=await opened(fixture()),input=cell(f),promise=f.click('science-author-rows-next');await tick();
    if(replace){f.state.workspace='two';f.form.stateChanged();}else f.form.event({kind:'externalReload'});
    const notice=f.get('science-author-status').textContent;if(failure)f.requests[2].reject(Error('obsolete'));else f.requests[2].resolve(page(f.requests[2]));await promise;
    assert.equal(f.get('science-author-status').textContent,notice);assert.equal(cell(f),input);assert.equal(f.toasts.length,0);assert.equal(f.get('science-author-save').disabled,true);
  }
});
test('check is pure transport and only explicit commit closes; late commit cannot select a new origin',async()=>{
  const f=await opened(fixture());let promise=f.click('science-author-check');await tick();f.requests[2].resolve({revision:'a'.repeat(64)});await promise;assert.equal(f.form.active(),true);assert.equal(f.saved.length,0);
  promise=f.click('science-author-save');await tick();f.form.event({kind:'externalReload'});const notice=f.get('science-author-status').textContent;
  f.requests[3].resolve({kind:'dataset',id:'data',revision:'a'.repeat(64)});await promise;
  assert.equal(f.saved.length,0);assert.equal(f.form.active(),true);assert.equal(f.get('science-author-status').textContent,notice);
});
test('repeated edits of one control retain its latest raw value without replaying obsolete text',async()=>{
  const f=await opened(fixture());
  for(let i=0;i<31;i++){cell(f).value='['+i+']';cell(f).onchange();}
  const promise=f.click('science-author-check');await tick();assert.equal(f.requests[2].args.operations.length,1);assert.equal(f.requests[2].args.operations[0].text,'[30]');
  f.requests[2].resolve(header(1));await tick();f.requests[3].resolve({revision:'a'.repeat(64)});await promise;
});
test('32 separate cell edits retain order across bounded operation/ASCII-byte batches',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();
  const template=page(f.requests[1]),fields=Array.from({length:4},(_,i)=>({...template.fields[0],id:'f'+i}));
  const rows=Array.from({length:8},(_,i)=>({id:'r'+i,ordinal:i,path:[1,i],cells:fields.map((field,j)=>desc(field.id,'[1]',[1,i,j]))}));
  f.requests[1].resolve({...template,fields,rows,field_total:4,row_total:8});await opening;
  const inputs=f.get('science-author-records').querySelectorAll('textarea');assert.equal(inputs.length,32);
  for(let i=0;i<inputs.length;i++){inputs[i].value=JSON.stringify('😀'.repeat(500)+i);inputs[i].onchange();}
  const promise=f.click('science-author-check');await tick();let index=2,sequence=0;const actual=[];
  while(f.requests[index].action==='scienceDraftDatasetEdit'){
    const request=f.requests[index];assert.ok(request.args.operations.length<=25);
    const asciiBytes=[...JSON.stringify({...request.args,action:request.action}).split('')].reduce((n,c)=>n+(c.charCodeAt(0)>127?6:1),0);
    assert.ok(asciiBytes<=65536);actual.push(...request.args.operations.map(op=>op.record+'/'+op.field));request.resolve(header(++sequence));await tick();index++;
  }
  assert.equal(actual.length,32);assert.deepEqual(actual,Array.from({length:32},(_,i)=>'r'+Math.floor(i/4)+'/f'+(i%4)));
  assert.ok(sequence>1);assert.equal(f.requests[index].action,'scienceDraftCheck');assert.equal(f.requests[index].args.sequence,sequence);
  f.requests[index].resolve({revision:'a'.repeat(64)});await promise;
});
test('failed initial page has a visible retry without opening a second session or enabling page-dependent actions',async()=>{
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve(header());await tick();f.requests[1].reject(Error('temporary page read failure'));await opening;
  assert.equal(f.get('science-author').hidden,false);assert.equal(f.form.active(),true);
  for(const id of ['science-author-add-field','science-author-add-record','science-author-save','science-author-check'])assert.equal(f.get(id).disabled,true,id);
  assert.equal(f.get('science-author-cancel').disabled,false);assert.equal(f.get('science-author-retry-page').hidden,false);
  const retry=f.click('science-author-retry-page');await tick();assert.equal(f.requests.length,3);assert.equal(f.requests[2].action,'scienceDraftDatasetPage');
  assert.equal(f.requests[2].args.sequence,0);f.requests[2].resolve(page(f.requests[2]));await retry;
  assert.equal(cell(f).value,'[1]');assert.equal(f.get('science-author-save').disabled,false);assert.equal(f.get('science-author-retry-page').hidden,true);
});
test('new/historical draft headers must match the requested base kind and ID before any page read',async()=>{
  for(const changes of [{id:'other',base:{kind:'dataset',id:'data',revision:'a'.repeat(64)}},
    {base:{kind:'task',id:'data',revision:'a'.repeat(64)}}, {version:9},{sequence:false}]){
    const f=fixture(),opening=f.form.open({kind:'dataset',id:'data',revision:'a'.repeat(64)});await tick();
    f.requests[0].resolve({...header(),base:{kind:'dataset',id:'data',revision:'a'.repeat(64)},...changes});await opening;
    assert.equal(f.requests.length,1);assert.equal(f.form.active(),false);assert.equal(f.toasts.length,1);
  }
  const f=fixture(),opening=f.form.open();await tick();f.requests[0].resolve({...header(),base:{kind:'dataset',id:'data',revision:'a'.repeat(64)}});await opening;
  assert.equal(f.requests.length,1);assert.equal(f.form.active(),false);
});
test('stable value target matching does not depend on JSON object key insertion order',async()=>{
  const f=await opened(fixture()),button=f.get('science-author-records').querySelector('button'),opening=button.onclick();await tick();
  assert.equal(f.requests[2].action,'scienceDraftDatasetValue');
  f.requests[2].resolve({scope:'owned-dataset-value',session:'owned',sequence:0,target:{field:'x',record:'r0',kind:'cell'},path:[1,0,0,0]});await opening;
  assert.equal(f.children.length,1);assert.equal(f.toasts.length,0);
});
test('hidden unrelated invalid controls do not block dataset edits, and explicit cancel cleans discarded dataset markers',async()=>{
  const f=await opened(fixture()),unrelated=new Element('input');unrelated.dataset.invalid='';f.get('science-author').append(unrelated);
  const check=f.click('science-author-check');await tick();assert.equal(f.requests[2].action,'scienceDraftCheck');f.requests[2].resolve({revision:'a'.repeat(64)});await check;
  const input=cell(f);input.dataset.invalid='';f.get('science-author-name').dataset.invalid='';
  const cancel=f.click('science-author-cancel');await tick();f.requests[3].resolve({session:'owned',status:'discarded-unsaved-draft'});await cancel;
  assert.equal(f.get('science-author-records').children.length,0);assert.equal('invalid' in f.get('science-author-name').dataset,false);
  assert.equal(unrelated.dataset.invalid,'');assert.equal(f.form.active(),false);
});
