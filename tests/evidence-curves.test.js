'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const C=require('../app/web/evidence-curves.js'),{Element,tick}=require('./science-dom-fixture.js');
const ref={kind:'run',id:'saved-run',revision:'a'.repeat(64)};
function reply(request,changes={}){
  const total=210,head=request.head||'head0',headTotal=30;
  const rows=Array.from({length:Math.max(0,Math.min(request.limit,total-request.offset))},(_,i)=>{
    const ordinal=request.offset+i,checkpoint='c'.repeat(64);
    return {ordinal,update:ordinal,checkpoint_id:checkpoint,timing:ordinal?'post-update':'initial',mode:'eval',attempted_batches:ordinal,examples_processed:ordinal*2,
      metrics:Object.fromEntries(['train','validation'].map((role,i)=>[role,{kind:'mse',checkpoint_id:checkpoint,population:role,status:'available',value:(210-ordinal)/(i+1),denominator:3.5,numerator:7,evaluated_records:4}]))};
  });
  return {version:1,scope:'read-saved-curve-only',workspace_id:request.workspace_id,reference:{...request.reference},attempt:{number:request.attempt,status:'completed'},
    head:{id:head,loss:'mse',transforms:[]},source_unit:'m',execution:'no-fitting-or-training',loss_space:'transformed-target-space',scale:'linear',x_range:[0,209],y_range:[0,221],
    retention:{observed_points:250,discarded_points:40,kind:'latest-window'},final_checkpoint:'c'.repeat(64),
    heads:{offset:request.head_offset,limit:request.head_limit,total:headTotal,rows:Array.from({length:Math.max(0,Math.min(request.head_limit,headTotal-request.head_offset))},(_,i)=>({id:'head'+(request.head_offset+i),loss:'mse'})),next_offset:request.head_offset+request.head_limit<headTotal?request.head_offset+request.head_limit:null},
    points:{offset:request.offset,limit:request.limit,total,rows,next_offset:request.offset+rows.length<total?request.offset+rows.length:null},...changes};
}
function fixture(){
  const document={createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)},container=new Element('body');
  const state={workspace:'one'},requests=[];
  const api={workspace:()=>state.workspace,call:(action,args)=>new Promise((resolve,reject)=>requests.push({action,args,resolve,reject}))};
  const view={reference:{...ref},attempt:{number:2}},details=C.attach(document,container,view,api);
  const buttons=details.querySelectorAll('button'),choice=details.querySelector('select'),status=details.querySelector('[role="status"]')||details.children[2];
  const open=()=>{details.open=true;details.ontoggle();};
  return {document,container,state,api,view,requests,details,buttons,choice,status,open};
}
async function loaded(){const f=fixture();f.open();f.requests[0].resolve(reply(f.requests[0].args));await tick();return f;}
const text=n=>[n.textContent,...n.children.map(text)].join(' ');
test('curve opening is explicit, pins the displayed attempt, and never sends a training action',async()=>{
  const f=fixture();assert.equal(f.requests.length,0);f.open();assert.equal(f.requests.length,1);
  assert.equal(f.requests[0].action,'scienceRunCurve');assert.deepEqual(f.requests[0].args.reference,ref);assert.equal(f.requests[0].args.attempt,2);
  f.requests[0].resolve(reply(f.requests[0].args));await tick();assert.equal(f.details.querySelectorAll('svg').length,1);assert.match(text(f.details),/no fitting, inference or training/);
});
test('geometry uses numeric saved coordinates without averaging, removes no zero and breaks null spans',()=>{
  const request={reference:ref,workspace_id:'one',attempt:1,head:null,offset:0,limit:100,head_offset:0,head_limit:25};
  const page=reply(request);page.points.rows=page.points.rows.slice(0,4);
  page.points.rows[0].metrics.train.value=0;page.points.rows[1].metrics.train.value=null;page.points.rows[1].metrics.train.status='unavailable';
  const geometry=C.geometry(page);assert.equal(geometry.train.length,2);assert.equal(geometry.train[0][0].y,244);
  assert.equal(geometry.train[1][0].x,80+650*(2/209));assert.equal(geometry.train[1][0].y,244-210*(208/221));
  assert.equal(geometry.train[1][0].point,page.points.rows[2]);assert.equal(geometry.validation.length,1);
});
test('shape and line style redundantly distinguish train and validation; exact numbers remain in a table',async()=>{
  const f=await loaded();const svg=f.details.querySelector('svg');assert.equal(svg.attributes.role,'img');
  assert.equal(svg.querySelectorAll('circle').length,100);assert.equal(svg.querySelectorAll('rect').length,100);
  assert.ok(svg.querySelectorAll('polyline').some(n=>n.attributes['stroke-dasharray']==='6 4'));
  assert.equal(f.details.querySelector('tbody').children.length,200);assert.match(text(f.details),/Effective weight/);assert.match(text(f.details),/3.5/);
  assert.match(text(f.details),/Test metrics are final-only/);assert.match(text(f.details),/discarded 40/);
});
test('point paging acknowledges actual cursors and blocks duplicate pending requests',async()=>{
  const f=await loaded();f.buttons[4].onclick();f.buttons[4].onclick();assert.equal(f.requests.length,2);assert.equal(f.requests[1].args.offset,100);
  f.requests[1].resolve(reply(f.requests[1].args));await tick();f.buttons[3].onclick();assert.equal(f.requests[2].args.offset,0);
  f.requests[2].resolve(reply(f.requests[2].args));await tick();assert.equal(f.buttons[3].disabled,true);
});
test('failed Next and Back preserve acknowledged history; refresh retries the visible cursor',async()=>{
  const f=await loaded();f.buttons[4].onclick();f.requests[1].reject(Error('Next failure'));await tick();assert.match(f.status.textContent,/preserved/);
  assert.equal(f.buttons[3].disabled,true);f.buttons[2].onclick();assert.equal(f.requests[2].args.offset,0);f.requests[2].resolve(reply(f.requests[2].args));await tick();
  f.buttons[4].onclick();f.requests[3].resolve(reply(f.requests[3].args));await tick();f.buttons[3].onclick();f.requests[4].reject(Error('Back failure'));await tick();
  f.buttons[2].onclick();assert.equal(f.requests[5].args.offset,100);f.requests[5].resolve(reply(f.requests[5].args));await tick();
  f.buttons[3].onclick();assert.equal(f.requests[6].args.offset,0);
});
test('head navigation preserves selection as a literal outside-page pin, separate from point navigation',async()=>{
  const f=await loaded();f.buttons[1].onclick();assert.equal(f.requests[1].args.head_offset,25);assert.equal(f.requests[1].args.offset,0);
  f.requests[1].resolve(reply(f.requests[1].args));await tick();assert.equal(f.choice.value,'head0');assert.match(text(f.choice),/selected outside this page/);
  f.choice.value='head29';f.choice.onchange();assert.equal(f.requests[2].args.head,'head29');assert.equal(f.requests[2].args.offset,0);
  f.requests[2].resolve(reply(f.requests[2].args));await tick();assert.equal(f.buttons[3].disabled,true);
  f.buttons[0].onclick();assert.equal(f.requests[3].args.head_offset,0);assert.equal(f.requests[3].args.head,'head29');
});
test('failed head change restores accepted selection instead of displaying a foreign head under old axes',async()=>{
  const f=await loaded();f.choice.value='head1';f.choice.onchange();f.requests[1].reject(Error('Head failure'));await tick();
  assert.equal(f.choice.value,'head0');assert.match(text(f.details),/head head0/);f.buttons[2].onclick();assert.equal(f.requests[2].args.head,'head0');
});
test('detached selections and replaced workspaces cannot render late data/errors or start more requests',async()=>{
  for(const detach of [false,true]){
    const f=fixture();f.open();if(detach)f.details.remove();else f.state.workspace='two';
    f.requests[0].resolve(reply(f.requests[0].args));await tick();assert.equal(f.details.querySelectorAll('svg').length,0);
    f.buttons[2].onclick();assert.equal(f.requests.length,1);
  }
  const f=fixture();f.open();f.details.remove();f.requests[0].reject(Error('obsolete'));await tick();assert.doesNotMatch(f.status.textContent,/obsolete/);
});
test('identity, attempt, workspace, head, page and scientific behavior mismatches are rejected',()=>{
  const request={reference:ref,workspace_id:'one',attempt:1,head:'head0',offset:0,limit:100,head_offset:0,head_limit:25};
  for(const changes of [{scope:'rerun'},{reference:{...ref,revision:'b'.repeat(64)}},{attempt:{number:2}},{workspace_id:'other'},
    {head:{id:'other',loss:'mse'}},{execution:'training'},{scale:'log'},{loss_space:'physical'},{x_range:[0,Infinity]},{y_range:[0,0]}])assert.throws(()=>C.check(reply(request,changes),request));
  const page=reply(request);page.points.offset=3;assert.throws(()=>C.check(page,request));
  const unsafe=reply(request);unsafe.points.rows[0].update=2**53;assert.throws(()=>C.check(unsafe,request));
});
test('foreign checkpoint/population/loss, unavailable-as-zero, duplicate heads and nonfinite values are rejected',()=>{
  const request={reference:ref,workspace_id:'one',attempt:1,head:null,offset:0,limit:100,head_offset:0,head_limit:25};
  for(const change of [{checkpoint_id:'b'.repeat(64)},{population:'test'},{kind:'mae'},{status:'unavailable',value:0},{value:NaN},{value:Infinity},{value:-1}]){
    const p=reply(request);Object.assign(p.points.rows[0].metrics.train,change);assert.throws(()=>C.check(p,request));
  }
  const p=reply(request);p.heads.rows[1].id='head0';assert.throws(()=>C.check(p,request));
});
test('missing and failed attempts show explicit absence without a zero curve or earlier checkpoint',async()=>{
  const f=fixture();f.open();const page=reply(f.requests[0].args);page.attempt.status='failed';page.retention=null;page.final_checkpoint=null;
  page.points={offset:0,limit:100,total:0,rows:[],next_offset:null};f.requests[0].resolve(page);await tick();
  assert.equal(f.details.querySelectorAll('svg').length,0);assert.match(text(f.details),/Earlier attempts are not substituted/);assert.match(text(f.details),/not a zero-valued curve/);
});
test('malicious head, unit and transform labels remain literal text, not markup',async()=>{
  const f=fixture();f.open();const page=reply(f.requests[0].args),bad='<img src=x onerror=alert(1)>';
  page.head.id=bad;page.heads.rows[0].id=bad;page.source_unit=bad;page.head.transforms=[{kind:bad}];f.requests[0].resolve(page);await tick();
  assert.match(text(f.details),/<img src=x/);assert.equal(f.details.querySelectorAll('img,script').length,0);
});
test('initial failed read can retry without accepting fabricated page/history',async()=>{
  const f=fixture();f.open();f.requests[0].reject(Error('Missing retained source'));await tick();assert.equal(f.buttons[3].disabled,true);
  f.buttons[2].onclick();assert.equal(f.requests[1].args.offset,0);f.requests[1].resolve(reply(f.requests[1].args));await tick();assert.equal(f.details.querySelectorAll('svg').length,1);
});
test('actual evidence renderer attaches the curve viewer to the saved attempt with no automatic read',()=>{
  const f=fixture(),window={LoomEvidenceCurves:C,LoomRunEvidence:require('../app/web/run-evidence.js')};
  vm.runInNewContext(fs.readFileSync('app/web/evidence-view.js','utf8'),{window});
  const view={...f.view,name:'Saved',attempt_count:2,execution:null,objects:{},populations:[],metrics:{offset:0,rows:[],total:0,next_offset:null}};
  window.LoomEvidenceView.render(f.document,f.container,view,null,f.api);
  assert.equal(f.container.querySelectorAll('.science-curve').length,1);assert.equal(f.requests.length,0);
  f.container.querySelector('.science-curve').open=true;f.container.querySelector('.science-curve').ontoggle();assert.equal(f.requests[0].args.attempt,2);
});
