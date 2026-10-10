'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
test('Offline MathJax adapter filters unsafe TeX attributes and queues dynamic rendering',async()=>{
  const window={},document={body:{}};vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../app/web/math.js'),'utf8'),{window,document,Promise});
  assert.equal(window.MathJax.showMathMenu,false);
  assert.equal(window.MathJax.tex2jax.ignoreClass,'.*');
  assert.equal(window.MathJax.tex2jax.processClass,'loom-math');
  for(const attr of ['URLs','classes','cssIDs','styles','require'])assert.equal(window.MathJax.Safe.allow[attr],'none');
  let startup;const calls=[],element={textContent:'',classList:{add:value=>calls.push(value)}};
  const hub={Register:{StartupHook:(event,fn)=>{assert.equal(event,'End');startup=fn;}},
    getAllJax:el=>[{Remove:()=>calls.push('clear')}],Queue:(prepare,command,done)=>{prepare();assert.equal(command[0],'Typeset');calls.push(command[2]);done();}};
  window.MathJax.Hub=hub;window.MathJax.AuthorInit();
  const p=window.LoomMath.set(element,'y=Ax+b');startup();await p;
  assert.equal(element.textContent,'\\(y=Ax+b\\)');assert.deepEqual(calls.slice(0,2),['clear','loom-math']);
  await window.LoomMath.set(element,'\\frac{1}{2}',{display:true});
  assert.equal(element.textContent,'\\[\\frac{1}{2}\\]');
  await window.LoomMath.typeset(element);assert.equal(calls.at(-1),element);
});
