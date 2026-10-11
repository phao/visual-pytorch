'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {Element,tick}=require('./science-dom-fixture');
const sid=i=>i.toString(16).padStart(32,'0');
const row=i=>({id:sid(i),run_id:{text:'run '+i,next_offset:null},error:{text:'failure '+i,next_offset:null},exported:false,has_result:true});
function fixture(){
  const source=fs.readFileSync('app/web/science-recoveries.js','utf8'),nodes=new Map(),requests=[],toasts=[],selections=[];
  const document={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id==='science-recovery-object'?'select':'div'));return nodes.get(id);},createElement:tag=>new Element(tag)};
  const state={h:{version:1,scope:'experiment-recovery-header',scope_id:'a'.repeat(32),sequence:1,total:80},busy:false,clears:0};
  const window={LoomScienceDocument:{clearRecovery(){state.clears++;},selectRecovery(r){selections.push(r);}}};
  const api={recoveryHeader:()=>state.h,busy:()=>state.busy,toast:(...args)=>toasts.push(args),
    call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  vm.runInNewContext(source,{window,JSON,Promise,Error});window.LoomScienceRecoveries.mount(document,api);
  return {nodes,requests,toasts,selections,state,window,api};
}
const reply=(r,changes={})=>({version:1,scope:'experiment-recovery-page',...r.args,total:80,rows:[row(r.args.offset)],pinned:null,missing_pin:null,next_offset:r.args.offset===0?7:null,...changes});
async function initial(f){await tick();f.requests[0].resolve(reply(f.requests[0]));await tick();}
test('actual recovery browser opens bounded metadata, never complete recovery contents',async()=>{
  const f=fixture();await initial(f);assert.equal(f.requests[0].action,'scienceRecoveryPage');
  assert.equal(f.requests[0].args.limit,25);assert.equal(f.nodes.get('science-recovery-object').options.length,1);
  assert.match(f.nodes.get('science-recovery-page').textContent,/session evidence, not durable jobs/);
  assert.equal(f.nodes.get('science-recovery-next').disabled,false);
});
test('actual cursors, duplicate gesture guards and failed Back retain navigation history',async()=>{
  const f=fixture();await initial(f);const n=f.nodes; n.get('science-recovery-next').onclick();n.get('science-recovery-next').onclick();await tick();
  assert.equal(f.requests.length,2);assert.equal(f.requests[1].args.offset,7);
  f.requests[1].resolve(reply(f.requests[1]));await tick();n.get('science-recovery-prev').onclick();await tick();
  f.requests[2].reject(Error('failed Back'));await tick();assert.equal(n.get('science-recovery-prev').disabled,false);
  n.get('science-recovery-refresh').onclick();await tick();assert.equal(f.requests[3].args.offset,7);
  f.requests[3].resolve(reply(f.requests[3]));await tick();n.get('science-recovery-prev').onclick();await tick();
  assert.equal(f.requests[4].args.offset,0);f.requests[4].resolve(reply(f.requests[4]));await tick();
  assert.equal(n.get('science-recovery-prev').disabled,true);
});
test('search keeps exact outside pins and missing identities without switching evidence',async()=>{
  const f=fixture();await initial(f);f.nodes.get('science-recovery-query').value='other';f.nodes.get('science-recovery-search').onclick();await tick();
  assert.equal(f.requests[1].args.pin,sid(0));f.requests[1].resolve(reply(f.requests[1],{rows:[row(79)],pinned:row(0)}));await tick();
  assert.equal(f.nodes.get('science-recovery-object').value,sid(0));assert.equal(f.nodes.get('science-recovery-object').options.length,2);
  f.nodes.get('science-recovery-refresh').onclick();await tick();f.requests[2].resolve(reply(f.requests[2],{rows:[],total:0,missing_pin:sid(0),next_offset:null}));await tick();
  f.nodes.get('science-recovery-show').onclick();assert.equal(f.selections.length,0);
  assert.equal(f.nodes.get('science-recovery-object').value,sid(0));
});
test('wrong scope, sequence, identity or cursor cannot install a page',async()=>{
  for(const changes of [{scope:'full-recovery'},{scope_id:'foreign'},{sequence:2},{offset:99},{limit:99},{pin:sid(1)},{missing_pin:sid(1)}]){
    const f=fixture();await tick();f.requests[0].resolve(reply(f.requests[0],changes));await tick();
    assert.equal(f.nodes.get('science-recovery-object').options.length,0);assert.equal(f.toasts.length,1);
  }
});
test('header changes fence late pages and errors while workspace changes alone do not discard window evidence',async()=>{
  const f=fixture();await tick();const old=f.requests[0];f.state.h={...f.state.h,sequence:2};
  f.window.LoomScienceRecoveries.stateChanged();await tick();f.requests[1].resolve(reply(f.requests[1],{rows:[row(9)]}));await tick();
  old.reject(Error('old failure'));await tick();assert.equal(f.toasts.length,0);
  assert.equal(f.nodes.get('science-recovery-object').value,sid(9));assert.equal(f.state.clears,1);
  f.state.workspace='replacement';f.window.LoomScienceRecoveries.stateChanged();await tick();assert.equal(f.requests.length,2);
});
test('inspection chooses the exact recovery identity, not a registry revision or a full result',async()=>{
  const f=fixture();await initial(f);f.nodes.get('science-recovery-show').onclick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.selections)),[{scope_id:f.state.h.scope_id,sequence:1,id:sid(0)}]);
  assert.equal(f.nodes.get('science-details').open,true);assert.equal(f.requests.length,1);
  const summary=JSON.parse(f.nodes.get('science-document').textContent);assert.equal(summary.scope,'session-recovery-summary');
  assert.equal('snapshot' in summary,false);
});
test('export is explicit, sequence-scoped, disabled while busy and duplicate guarded',async()=>{
  const f=fixture();await initial(f);f.state.busy=true;f.window.LoomScienceRecoveries.stateChanged();
  assert.equal(f.nodes.get('science-recovery-export').disabled,true);f.nodes.get('science-recovery-export').onclick();await tick();assert.equal(f.requests.length,1);
  f.state.busy=false;f.nodes.get('science-recovery-export').onclick();f.nodes.get('science-recovery-export').onclick();await tick();assert.equal(f.requests.length,2);
  assert.equal(f.requests[1].action,'scienceExportRecovery');assert.deepEqual(JSON.parse(JSON.stringify(f.requests[1].args)),{id:sid(0),scope_id:f.state.h.scope_id,sequence:1});
  f.requests[1].reject(Error('export refused'));await tick();assert.equal(f.nodes.get('science-recovery-export').disabled,false);assert.equal(f.toasts.length,1);
});
test('partial labels are marked and literal hostile text is not HTML',async()=>{
  const f=fixture();await tick();const r=row(0);r.error={text:'<script>not executable</script>',next_offset:128};f.requests[0].resolve(reply(f.requests[0],{rows:[r]}));await tick();
  const label=f.nodes.get('science-recovery-object').options[0].textContent;
  assert.match(label,/<script>not executable<\/script>/);assert.match(label,/partial; inspect full evidence/);
});
test('stale navigation and search completion cannot restore a previous session cursor or history',async()=>{
  for(const gesture of ['science-recovery-prev','science-recovery-search']){
    const f=fixture();await initial(f);f.nodes.get('science-recovery-next').onclick();await tick();f.requests[1].resolve(reply(f.requests[1]));await tick();
    f.nodes.get(gesture).onclick();await tick();const old=f.requests[2];f.state.h={...f.state.h,sequence:2};
    f.window.LoomScienceRecoveries.stateChanged();await tick();f.requests[3].resolve(reply(f.requests[3]));await tick();
    old.reject(Error('obsolete navigation'));await tick();assert.equal(f.toasts.length,0);assert.equal(f.nodes.get('science-recovery-prev').disabled,true);
    f.nodes.get('science-recovery-next').onclick();await tick();assert.equal(f.requests[4].args.offset,7);
  }
});
