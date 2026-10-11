'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function fixture(){
  const ids=['editor','text','identity','status','start','apply','finish','check','save','prev','next','discard'];
  const nodes=new Map(ids.map(id=>['science-value-'+id,{value:'',textContent:'',hidden:id==='editor',disabled:false}]));
  const requests=[],toasts=[],saved=[],state={workspace:'one',busy:false,author:false};
  const window={LoomScienceAuthor:{active:()=>state.author,stateChanged(){}},LoomScienceDocument:{stateChanged(){}}};
  const api={workspace:()=>state.workspace,busy:()=>state.busy,toast:(...args)=>toasts.push(args),
    call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(fs.readFileSync('app/web/draft-value-editor.js','utf8'),{window,Promise,JSON,Array,Error,TextEncoder});
  const editor=window.LoomDraftValueEditor;
  editor.mount({getElementById:id=>{assert.ok(nodes.has(id),id);return nodes.get(id);}},api,async result=>saved.push(result));
  return {nodes,requests,toasts,saved,state,editor,window,api,node:id=>nodes.get('science-value-'+id),click:id=>nodes.get('science-value-'+id).onclick()};
}
const target=()=>({workspace:'one',reference:{kind:'dataset',id:'measurements',revision:'a'.repeat(64)},path:[7,0]});
const header=(changes={})=>({version:1,scope:'owned-authoring-session',session:'owned',sequence:0,kind:'dataset',id:'measurements',workspace_id:'one',
  base:{revision:'a'.repeat(64),id:'measurements',kind:'dataset'},...changes});
const buffer=(changes={})=>({version:1,scope:'staged-draft-value',session:'owned',sequence:0,value:'buffer',value_revision:0,
  source_path:[7,0],from_base:true,path:[6,0],total_bytes:10000,offset_units:'utf8-bytes',byte_limit:65536,...changes});
const fragment=(request,changes={})=>({...buffer(),offset:request.args.offset,text:'"λ😀<script>literal</script>"',next_offset:97,...changes});
async function opened(f){
  const promise=f.editor.open(target());await tick();assert.equal(f.requests[0].action,'scienceDraftOpen');
  f.requests[0].resolve(header());await tick();assert.equal(f.requests[1].action,'scienceDraftValueStart');
  f.requests[1].resolve(buffer());await tick();assert.equal(f.requests[2].action,'scienceDraftValueRead');
  f.requests[2].resolve(fragment(f.requests[2]));await promise;return f;
}
test('opens an exact base without full-document transport; normalizes stored root ordinal and preserves literal JSON',async()=>{
  const f=await opened(fixture());assert.equal(JSON.stringify(f.requests[1].args.path),'[7,0]');assert.equal(f.requests[1].args.from_base,true);
  assert.equal(f.node('text').value,'"λ😀<script>literal</script>"');assert.match(f.node('identity').textContent,/stored path \[7,0\]/);
  assert.equal(f.node('editor').hidden,false);assert.equal(f.node('finish').disabled,false);assert.equal(f.node('save').disabled,true);
  assert.equal(f.toasts.length,0);assert.ok(f.requests.every(r=>!['scienceShow','scienceIndex','scienceAuthorBegin'].includes(r.action)));
});
test('dirty navigation keeps unfinished text and causes no transport',async()=>{
  const f=await opened(fixture());f.node('text').value='[broken';f.node('text').oninput();assert.equal(f.node('next').disabled,true);
  await f.click('next');assert.equal(f.requests.length,3);assert.equal(f.node('text').value,'[broken');assert.match(f.node('status').textContent,/Apply.*before navigating/);
});
test('fragment splice uses UTF-8 delete length, acknowledges buffer revision, then reads only that cursor',async()=>{
  const f=await opened(fixture()),original=f.node('text').value;f.node('text').value='[1,';f.node('text').oninput();
  const action=f.click('apply');await tick();const splice=f.requests[3];assert.equal(splice.args.delete_bytes,new TextEncoder().encode(original).length);
  assert.equal(splice.args.text,'[1,');assert.equal(splice.args.offset,0);splice.resolve(buffer({value_revision:1}));await tick();
  assert.equal(f.requests[4].args.value_revision,1);f.requests[4].resolve(fragment(f.requests[4],{value_revision:1,text:'[1,',next_offset:null}));await action;
  assert.equal(f.node('text').value,'[1,');assert.equal(f.node('finish').disabled,false);assert.equal(f.node('save').disabled,true);
});
test('Unicode code point edit limit allows 2000 supplementary characters but rejects 2001 without changing text',async()=>{
  const f=await opened(fixture());f.node('text').value='😀'.repeat(2001);await f.click('apply');assert.equal(f.requests.length,3);assert.match(f.node('status').textContent,/2000 Unicode/);
  f.node('text').value='😀'.repeat(2000);const promise=f.click('apply');await tick();assert.equal(f.requests[3].args.text.length,4000);
  f.requests[3].reject(Error('staging disk full'));await promise;assert.equal(f.node('text').value,'😀'.repeat(2000));
});
test('failed whole-value admission retains staged text and permits repair; successful acceptance still requires explicit retention',async()=>{
  const f=await opened(fixture()),original=f.node('text').value;let promise=f.click('finish');await tick();f.requests[3].reject(Error('Constrained memory budget exceeded'));await promise;
  assert.equal(f.node('text').value,original);assert.equal(f.node('save').disabled,true);assert.equal(f.editor.active(),true);
  promise=f.click('finish');await tick();f.requests[4].resolve(header({sequence:1}));await promise;
  assert.equal(f.node('save').disabled,false);assert.equal(f.node('check').disabled,false);assert.equal(f.saved.length,0);
  promise=f.click('check');await tick();assert.equal(f.requests[5].args.sequence,1);f.requests[5].resolve({revision:'b'.repeat(64)});await promise;
  promise=f.click('save');await tick();f.requests[6].resolve({kind:'dataset',id:'measurements',revision:'b'.repeat(64)});await promise;
  assert.equal(f.editor.active(),false);assert.equal(f.node('editor').hidden,true);assert.equal(f.saved.length,1);
});
test('actual server cursors and successful backward navigation are used, with duplicate pending clicks suppressed',async()=>{
  const f=await opened(fixture());const first=f.click('next'),second=f.click('next');await tick();assert.equal(f.requests.length,4);assert.equal(f.requests[3].args.offset,97);
  f.requests[3].resolve(fragment(f.requests[3],{text:'tail',next_offset:null}));await Promise.all([first,second]);assert.equal(f.node('next').disabled,true);
  const back=f.click('prev');await tick();assert.equal(f.requests[4].args.offset,0);f.requests[4].resolve(fragment(f.requests[4]));await back;
});
test('opening failure can retry the original stored path with from_base true',async()=>{
  const f=fixture(),promise=f.editor.open(target());await tick();f.requests[0].resolve(header());await tick();f.requests[1].reject(Error('RAM changed'));await promise;
  assert.equal(f.editor.active(),true);assert.equal(f.node('start').disabled,false);const retry=f.click('start');await tick();
  assert.equal(JSON.stringify(f.requests[2].args.path),'[7,0]');assert.equal(f.requests[2].args.from_base,true);
  f.requests[2].resolve(buffer());await tick();f.requests[3].resolve(fragment(f.requests[3]));await retry;
});
test('wrong base, staged source path, and fragment identity/cursor are refused before replacement',async()=>{
  for(const changes of [{base:{kind:'dataset',id:'wrong',revision:'a'.repeat(64)}},{workspace_id:'wrong'},{scope:'full-document'}]){
    const f=fixture(),promise=f.editor.open(target());await tick();f.requests[0].resolve(header(changes));await promise;assert.equal(f.requests.length,1);assert.equal(f.toasts.length,1);
  }
  const f=fixture(),promise=f.editor.open(target());await tick();f.requests[0].resolve(header());await tick();f.requests[1].resolve(buffer({source_path:[8,0]}));await promise;
  assert.equal(f.requests.length,2);assert.equal(f.node('text').value,'');assert.match(f.node('status').textContent,/does not match/);
  for(const changes of [{path:[999]},{offset:98},{value:'foreign'},{value_revision:3},{offset_units:'characters'}]){
    const g=await opened(fixture()),text=g.node('text').value,next=g.click('next');await tick();g.requests[3].resolve(fragment(g.requests[3],changes));await next;
    assert.equal(g.node('text').value,text);assert.match(g.node('status').textContent,/does not match/);
  }
});
test('pending open prevents a duplicate request and retained draft conflicts with typed authoring',async()=>{
  const f=fixture(),first=f.editor.open(target()),second=f.editor.open(target());await tick();assert.equal(f.requests.length,1);assert.equal(f.editor.active(),true);
  f.requests[0].reject(Error('no revision'));await Promise.all([first,second]);assert.equal(f.editor.active(),false);
  f.state.author=true;await f.editor.open(target());assert.equal(f.requests.length,1);assert.match(f.node('status').textContent,/Finish or discard/);
});
test('same-UUID reload during open retains the exact draft but does not start a value or overwrite the reload notice',async()=>{
  const f=fixture(),promise=f.editor.open(target());await tick();f.editor.event({kind:'externalReload'});f.requests[0].resolve(header());await promise;
  assert.equal(f.requests.length,1);assert.equal(f.editor.active(),true);assert.equal(f.node('start').disabled,true);assert.match(f.node('status').textContent,/Workspace changed/);
});
test('workspace switch and same-UUID reload suppress late read successes/errors while preserving the displayed fragment',async()=>{
  for(const replacement of [true,false])for(const failure of [true,false]){
    const f=await opened(fixture()),original=f.node('text').value,promise=f.click('next');await tick();
    if(replacement){f.state.workspace='two';f.editor.stateChanged();}else f.editor.event({kind:'externalReload'});
    const notice=f.node('status').textContent;if(failure)f.requests[3].reject(Error('obsolete'));else f.requests[3].resolve(fragment(f.requests[3],{text:'obsolete'}));await promise;
    assert.equal(f.node('status').textContent,notice);assert.equal(f.node('text').value,original);assert.equal(f.toasts.length,0);assert.equal(f.node('finish').disabled,true);
  }
});
test('same-UUID reload suppresses admitted splice/finish acknowledgements without discarding the current editor',async()=>{
  for(const action of ['apply','finish']){
    const f=await opened(fixture());if(action==='apply'){f.node('text').value='[unfinished';f.node('text').oninput();}
    const promise=f.click(action);await tick();f.editor.event({kind:'externalReload'});const notice=f.node('status').textContent,text=f.node('text').value;
    f.requests[3].resolve(action==='apply'?buffer({value_revision:1}):header({sequence:1}));await promise;
    assert.equal(f.node('status').textContent,notice);assert.equal(f.node('text').value,text);assert.equal(f.requests.length,4);assert.equal(f.node('save').disabled,true);assert.equal(f.editor.active(),true);
  }
});
test('stale drafts can be explicitly discarded; busy execution disables mutation but not discard',async()=>{
  const f=await opened(fixture());f.state.busy=true;f.editor.stateChanged();assert.equal(f.node('apply').disabled,true);assert.equal(f.node('finish').disabled,true);assert.equal(f.node('discard').disabled,false);
  f.editor.event({kind:'externalReload'});const promise=f.click('discard');await tick();assert.equal(f.requests[3].action,'scienceDraftClose');f.requests[3].resolve({session:'owned',status:'discarded-unsaved-draft'});await promise;
  assert.equal(f.node('editor').hidden,true);assert.equal(f.editor.active(),false);assert.equal(f.saved.length,0);
});
test('failed backward navigation retains its cursor so retry returns to the actual prior fragment',async()=>{
  const f=await opened(fixture());let promise=f.click('next');await tick();f.requests[3].resolve(fragment(f.requests[3],{text:'tail',next_offset:null}));await promise;
  promise=f.click('prev');await tick();f.requests[4].reject(Error('transient read failure'));await promise;
  assert.equal(f.node('prev').disabled,false);assert.equal(f.node('text').value,'tail');
  promise=f.click('prev');await tick();assert.equal(f.requests[5].args.offset,0);f.requests[5].resolve(fragment(f.requests[5]));await promise;
  assert.equal(f.node('prev').disabled,true);
});
test('late commit after a same-UUID reload does not select old results or discard the visible stale editor',async()=>{
  const f=await opened(fixture());let promise=f.click('finish');await tick();f.requests[3].resolve(header({sequence:1}));await promise;
  promise=f.click('save');await tick();f.editor.event({kind:'externalReload'});const notice=f.node('status').textContent;
  f.requests[4].resolve({kind:'dataset',id:'measurements',revision:'b'.repeat(64)});await promise;
  assert.equal(f.saved.length,0);assert.equal(f.node('editor').hidden,false);assert.equal(f.node('status').textContent,notice);assert.equal(f.node('save').disabled,true);
});
test('actual typed authoring is already active during pending Begin, excluding a concurrent value draft',async()=>{
  const f=fixture(),document={getElementById:id=>{
    if(!f.nodes.has(id))f.nodes.set(id,{value:'',textContent:'',disabled:false,dataset:{},querySelectorAll:()=>[]});
    return f.nodes.get(id);
  }};
  vm.runInNewContext(fs.readFileSync('app/web/science-author.js','utf8'),{window:f.window,Promise,JSON,Array,Error});
  f.window.LoomScienceAuthor.mount(document,f.api,async()=>{});
  document.getElementById('science-author-new').onclick();await tick();
  assert.equal(f.window.LoomScienceAuthor.active(),true);assert.equal(f.requests[0].action,'scienceAuthorBegin');
  await f.editor.open(target());assert.equal(f.requests.length,1);assert.match(f.node('status').textContent,/Finish or discard/);
  f.requests[0].reject(Error('missing base'));await tick();assert.equal(f.window.LoomScienceAuthor.active(),false);
});
