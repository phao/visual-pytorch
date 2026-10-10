"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const T=require('../app/web/task-authoring.js');

test('transform browsing is alphabetical; computation is never sorted',()=>{
  assert.deepEqual(T.kinds,[...T.kinds].sort());
  const input=T.input('port','field');input.transforms=[T.transform('standardize'),T.transform('affine')];
  assert.deepEqual(input.transforms.map(t=>t.kind),['standardize','affine']);
});
test('explicit finite and exact integer parsing, without empty or boolean coercion',()=>{
  assert.equal(T.number('-2.5'),-2.5);assert.equal(T.number('1e3',{integer:true,positive:true}),1000);
  for(const value of ['', ' ', 'Infinity','NaN','1e400',true,3,null])assert.throws(()=>T.number(value));
  for(const value of ['1.5','9007199254740992'])assert.throws(()=>T.number(value,{integer:true}));
  for(const value of ['0','-1'])assert.throws(()=>T.number(value,{positive:true}));
});
test('all five deterministic transform defaults are independent closed objects',()=>{
  assert.deepEqual(T.transform('affine'),{kind:'affine',scale:1,offset:0});
  assert.deepEqual(T.transform('pad'),{kind:'pad',length:1,value:0});
  for(const kind of ['impute-mean','onehot','standardize'])assert.deepEqual(T.transform(kind),{kind});
  const pad=T.transform('pad');pad.length=20;assert.equal(T.transform('pad').length,1);
  for(const kind of ['Normalize',null,{},'script'])assert.throws(()=>T.transform(kind));
});
test('input/head factories do not invent bindings or share transform arrays',()=>{
  assert.deepEqual(T.input(),{port:'',field:'',transforms:[]});
  assert.deepEqual(T.head('stable'),{id:'stable',port:'',field:'',loss:'mse',weight:1,transforms:[]});
  const a=T.input(),b=T.input();a.transforms.push(T.transform('pad'));assert.deepEqual(b.transforms,[]);
});
test('derived switching preserves explicit transforms for visible validation',()=>{
  const a=T.input('sequence','seq');a.transforms=[T.transform('pad')];
  const b=T.derived(a,'length');assert.equal(b.length,1);assert.deepEqual(b.transforms,a.transforms);
  b.length=3;const mask=T.derived(b,'mask');assert.equal(mask.length,3);
  const plain=T.derived(mask,'none');assert.equal(Object.hasOwn(plain,'derived'),false);assert.equal(Object.hasOwn(plain,'length'),false);
  assert.deepEqual(plain.transforms,a.transforms);assert.equal(Object.hasOwn(a,'derived'),false);
  assert.throws(()=>T.derived(a,'automatic'));
});
test('ordered move swaps neighboring transforms without mutation or sorting',()=>{
  const a=[T.transform('standardize'),T.transform('affine'),T.transform('impute-mean')];
  const b=T.move(a,1,-1);assert.deepEqual(b.map(t=>t.kind),['affine','standardize','impute-mean']);
  assert.deepEqual(a.map(t=>t.kind),['standardize','affine','impute-mean']);
  b[0].scale=20;assert.equal(a[1].scale,1);
  assert.deepEqual(T.move(a,0,-1),a);assert.deepEqual(T.move(a,2,1),a);
});
test('binding reorder keeps stable identities and validates move arguments',()=>{
  const a=[T.input('z','left'),T.input('a','right')];
  assert.deepEqual(T.move(a,0,1).map(b=>[b.port,b.field]),[['a','right'],['z','left']]);
  for(const [index,delta] of [[-1,1],[2,1],[.5,1],[true,1],[0,0],[0,2]])assert.throws(()=>T.move(a,index,delta));
  assert.throws(()=>T.move(null,0,1));
});
