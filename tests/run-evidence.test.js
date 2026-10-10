"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),H=require('../app/web/run-evidence.js');
const r=(id='run',revision='a'.repeat(64))=>({kind:'run',id,revision});
test('selection pins exact identities and preserves semantic order',()=>{
  const first=H.add([],r('z')),next=H.add(first,r('a'));
  assert.deepEqual(next.map(v=>v.id),['z','a']);next[0].id='changed';assert.equal(first[0].id,'z');
  assert.equal(H.add(first,r('z')).length,1);
});
test('different revisions/retries of one run are rejected, never silently replaced',()=>{
  const first=H.add([],r());assert.throws(()=>H.add(first,r('run','b'.repeat(64))),/Remove it explicitly/);
  assert.equal(first[0].revision,'a'.repeat(64));
});
test('implicit heads, non-run references and invalid hashes cannot enter selection',()=>{
  for(const ref of [null,{kind:'run',id:'r',revision:null},{...r(),kind:'report'},r('',null),r('r','A'.repeat(64))])assert.throws(()=>H.add([],ref));
});
test('metric formatting preserves true zero and labels unavailable without arithmetic',()=>{
  assert.equal(H.value(0),'0.00000');assert.equal(H.value(.123456789),'0.123457');
  for(const value of [null,undefined,NaN,Infinity,'0'])assert.equal(H.value(value),'Unavailable');
});
test('exact ledger formatting preserves numbers and literal untrusted labels',()=>{
  assert.equal(H.literal(.123456789),'.123456789'.replace(/^\./,'0.'));
  assert.equal(H.literal(null),'Not retained');assert.equal(H.literal('<img onerror=x>'),'<img onerror=x>');
  assert.equal(H.literal({weight:0}),JSON.stringify({weight:0}));
});
test('predictor target pins the displayed retained attempt, not a mutable head or unread input',()=>{
  const page={scope:'read-saved-evidence-only',reference:r(),attempt:{number:1},execution:{checkpoint_id:'c'.repeat(64)}};
  const request=H.predictorRequest(page,'workspace');assert.deepEqual(request,{reference:r(),attempt:1,workspace_id:'workspace'});
  request.reference.id='other';assert.equal(page.reference.id,'run');
  page.attempt.number=2;assert.equal(H.predictorRequest(page,'workspace').attempt,2);
});
test('missing numerical checkpoint, implicit references or nonexact attempt cannot export',()=>{
  const page={scope:'read-saved-evidence-only',reference:r(),attempt:{number:1},execution:{checkpoint_id:'c'.repeat(64)}};
  for(const bad of [null,{...page,execution:null},{...page,execution:{}},{...page,reference:{...r(),revision:null}},
    {...page,attempt:{number:true}},{...page,attempt:{number:0}},{...page,scope:'rerun'}])assert.throws(()=>H.predictorRequest(bad,'workspace'));
  assert.throws(()=>H.predictorRequest(page,''));
});
