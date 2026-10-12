'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),L=require('../app/web/lstm-authoring.js');
const base=()=>({label:'LSTM',steps:'4',input_size:'3',hidden_size:'7',num_layers:'2',proj_size:'4',
  bias:true,bidirectional:true,lengths:true});
test('LSTM options and state shapes are explicit and projection does not narrow cell',()=>{
  const source=base(),before=JSON.stringify(source),v=L.options(source);
  assert.equal(v.version,1);assert.equal(v.steps,4);assert.equal(v.input_size,3);
  assert.deepEqual(L.stateShapes(source),{hidden:[4,4],cell:[4,7],sequence_features:8});
  assert.equal(JSON.stringify(source),before);
  assert.deepEqual(L.stateShapes({...source,proj_size:'0',bidirectional:false}),{hidden:[2,7],cell:[2,7],sequence_features:7});
});
test('LSTM integer parser rejects empty, decimal, exponent, unsafe and boolean forms',()=>{
  for(const value of ['',true,false,'1e3','1.5','-1','9007199254740992'])assert.throws(()=>L.integer(value,'width'));
  assert.equal(L.integer('0','projection',0),0);assert.equal(L.integer('9007199254740991','width'),Number.MAX_SAFE_INTEGER);
});
test('LSTM constructor and length failures do not silently coerce or clamp',()=>{
  for(const [key,value] of [['label',''],['label','x'.repeat(61)],['hidden_size','0'],['num_layers','65'],
    ['proj_size','7'],['bias',1],['bidirectional','false'],['lengths',1],['steps',String(2**24+1)]])
    assert.throws(()=>L.options({...base(),[key]:value}));
});
test('LSTM editing keeps stable role and named-port identities',()=>{
  const value={...base(),inputs:{cell:'c',sequence:'x@frames',hidden:'h'},lengths:{source:'len',adapter:'exact-int64-v1'},share_with:'owner'};
  const v=L.options(value,true);assert.deepEqual(Object.keys(v.inputs),['sequence','hidden','cell']);
  assert.equal(v.inputs.sequence,'x@frames');assert.equal(v.share_with,'owner');assert.ok(!('steps' in v));
  for(const patch of [{inputs:{sequence:'x'}},{lengths:{source:'l',adapter:'guess'}},{share_with:false}])assert.throws(()=>L.options({...value,...patch},true));
});
test('LSTM response guards reject wrong workspace, instance, node and digest',()=>{
  const context={id:'p',workspace_id:'w',panel_scope:'s'},info={...context,node:'n',root_graph:'a'.repeat(64),root_model:'b'.repeat(64)};
  assert.deepEqual(L.checkToken(info,context,'n'),{id:'p',root_graph:info.root_graph,root_model:info.root_model,node:'n'});
  for(const patch of [{id:'q'},{workspace_id:'q'},{panel_scope:'q'},{node:'q'},{root_graph:'no'},{root_model:true}])
    assert.throws(()=>L.checkToken({...info,...patch},context,'n'));
});

function harness(){
  class Element{
    constructor(id=''){this.id=id;this.value='';this.checked=false;this.disabled=false;this.hidden=false;this.open=false;this.textContent='';this.children=[];this.events={};}
    addEventListener(name,callback){(this.events[name]??=[]).push(callback);}
    replaceChildren(){this.children=[];}
    append(child){this.children.push(child);}
    showModal(){this.open=true;}
    close(){this.open=false;for(const callback of this.events.close||[])callback();}
  }
  const ids=['dialog','form','cancel','label','hidden-size','layers','projection','bias','bidirectional','steps','input-size',
    'lengths','title','apply','error','shapes','create-fields','edit-fields','owner',...['sequence','hidden','cell','lengths'].map(r=>'source-'+r)];
  const elements=Object.fromEntries(ids.map(id=>[id,new Element('lstm-'+id)]));
  elements.form.querySelectorAll=()=>Object.values(elements).filter(el=>!['dialog','form'].includes(el.id.slice(5)));
  const document={getElementById:id=>elements[id.slice(5)],createElement:()=>new Element()};
  let context={id:'p',workspace_id:'w',panel_scope:'scope'},busy=false;
  const calls=[],accepted=[],messages=[],info={...context,root_graph:'a'.repeat(64),root_model:'b'.repeat(64),options:L.options(base())};
  const services={context:()=>({...context}),busy:()=>busy,sync:async()=>{},refresh:()=>{},owners:()=>[],
    call:async(action,args)=>{calls.push({action,args});return action.startsWith('lstm')?info:{...context,node:'new'};},
    accept:async(result,current)=>{if(current())accepted.push(result);},toast:(message,error)=>messages.push({message,error})};
  const ui=L.mount(document,services);
  return {elements,ui,services,calls,accepted,messages,info,setContext:value=>context=value,setBusy:value=>busy=value,
    submit:()=>elements.form.onsubmit({preventDefault(){}})};
}
function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}

test('actual mounted form prepares read-only, then writes one explicit native request and accepts same owner',async()=>{
  const h=harness();h.elements.dialog.scrollTop=500;await h.ui.open();
  assert.equal(h.elements.dialog.scrollTop,0);assert.equal(h.elements.dialog.open,true);assert.equal(h.ui.pending(),false);
  assert.deepEqual(h.calls.map(c=>c.action),['lstmAuthoring']);assert.equal(h.accepted.length,0);
  await h.submit();assert.equal(h.elements.dialog.open,false);assert.equal(h.accepted.length,1);
  const request=h.calls[1];assert.equal(request.action,'addLSTM');
  assert.deepEqual(JSON.parse(request.args.options_json),h.info.options);
  assert.equal(request.args.root_graph,h.info.root_graph);assert.ok(!('node' in request.args));
});
test('mounted form blocks duplicate pending opens and rejects a late preparation after workspace reload',async()=>{
  const h=harness(),wait=deferred();h.services.call=async()=>wait.promise;
  const opening=h.ui.open();await Promise.resolve();assert.equal(h.ui.pending(),true);
  await h.ui.open();h.setContext({id:'p',workspace_id:'w',panel_scope:'new-instance'});wait.resolve(h.info);await opening;
  assert.equal(h.elements.dialog.open,false);assert.equal(h.accepted.length,0);assert.equal(h.ui.pending(),false);
});
test('mounted preparation refuses wrong owner response without showing or mutating',async()=>{
  const h=harness();h.info.workspace_id='wrong';await h.ui.open();
  assert.equal(h.elements.dialog.open,false);assert.equal(h.accepted.length,0);assert.equal(h.messages.length,1);
  assert.equal(h.ui.pending(),false);
});
test('native refusal retains the mounted draft and re-enables correction',async()=>{
  const h=harness();await h.ui.open();h.elements.label.value='Retained draft';h.elements['hidden-size'].value='8';
  h.services.call=async()=>{throw Error('State shape incompatible');};await h.submit();
  assert.equal(h.elements.dialog.open,true);assert.equal(h.elements.label.value,'Retained draft');
  assert.equal(h.elements['hidden-size'].value,'8');assert.equal(h.elements.error.textContent,'State shape incompatible');
  assert.equal(h.elements.apply.disabled,false);assert.equal(h.accepted.length,0);
});
test('post-acknowledgement refresh refusal keeps the draft rather than claiming success',async()=>{
  const h=harness();await h.ui.open();h.elements.label.value='Retain after refresh failure';
  h.services.accept=async()=>{throw Error('Workspace changed; reopen LSTM authoring.');};
  await h.submit();assert.equal(h.elements.dialog.open,true);assert.equal(h.accepted.length,0);
  assert.equal(h.elements.label.value,'Retain after refresh failure');assert.match(h.elements.error.textContent,/Workspace changed/);
  assert.equal(h.elements.apply.disabled,false);assert.equal(h.messages.length,0);
});
test('mounted pending Apply is single-flight and Cancel fences its later acknowledgement',async()=>{
  const h=harness();await h.ui.open();const wait=deferred();let writes=0;
  h.services.call=async()=>{writes++;return wait.promise;};const applying=h.submit();await Promise.resolve();
  await h.submit();assert.equal(writes,1);assert.equal(h.elements.apply.disabled,true);
  assert.equal(h.elements.cancel.disabled,false);h.elements.cancel.onclick();wait.resolve({id:'p',workspace_id:'w',panel_scope:'scope',node:'n'});await applying;
  assert.equal(h.accepted.length,0);assert.equal(h.elements.dialog.open,false);assert.equal(h.ui.pending(),false);
  // Cancel fences this view; it does not promise to undo an already authorized native write.
});
test('mounted wrong acknowledgement and post-open stale scope never close or accept the draft',async()=>{
  const h=harness();await h.ui.open();h.services.call=async()=>({id:'p',workspace_id:'w',panel_scope:'wrong',node:'n'});
  await h.submit();assert.equal(h.elements.dialog.open,true);assert.match(h.elements.error.textContent,/another block or workspace/);
  h.setContext({id:'p',workspace_id:'w',panel_scope:'reloaded'});await h.submit();
  assert.equal(h.accepted.length,0);assert.match(h.elements.error.textContent,/Workspace changed/);
});
test('mounted editing selects literal source IDs, explicit lengths and owner; label text is not HTML',async()=>{
  const h=harness();h.info.node='existing';h.info.options=L.options({...base(),inputs:{sequence:'x@frames',hidden:'h',cell:'c'},lengths:null,share_with:null},true);
  h.info.sources=['x@frames','h','c'].map(key=>({key,name:'<script>'+key+'</script>',shape:[2,3]}));
  await h.ui.open('existing');assert.equal(h.elements['edit-fields'].hidden,false);assert.equal(h.elements['create-fields'].hidden,true);
  assert.equal(h.elements['source-sequence'].value,'x@frames');assert.match(h.elements['source-sequence'].children[0].textContent,/<script>/);
  h.services.call=async(action,args)=>{h.calls.push({action,args});return {id:'p',workspace_id:'w',panel_scope:'scope',node:'existing'};};
  await h.submit();assert.equal(h.calls.at(-1).action,'editLSTM');assert.equal(h.calls.at(-1).args.node,'existing');
  assert.equal(JSON.parse(h.calls.at(-1).args.options_json).lengths,null);
});
test('busy or missing architecture cannot open the mounted LSTM authoring form',async()=>{
  const h=harness();h.setBusy(true);await h.ui.open();assert.equal(h.calls.length,0);
  h.setBusy(false);h.setContext({id:null,workspace_id:'w',panel_scope:'scope'});await h.ui.open();assert.equal(h.calls.length,0);
});
