'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const T=require('../app/web/tensor-operators.js'),I=require('../app/web/type-icons.js');
test('Structured palette sorted, icon-covered, explicit operator version and safe defaults',()=>{
  assert.deepEqual(T.types,[...T.types].sort());
  for(const type of T.types){
    assert.ok(I.types.includes(type));const n=T.defaults(type,[1,4,5]);
    assert.equal(n.op_version,1);assert.ok(T.definitions[type].explanation.length>50);
    n.op_version=9;assert.equal(T.defaults(type,[1,4,5]).op_version,1);
  }
  assert.deepEqual(T.defaults('Permute',[3,4]).dims,[0,1]);
  assert.deepEqual(T.defaults('Permute',[3,4,2]).dims,[2,0,1]);
});
test('Pairs normalize scalars and preserve unequal dimensions; no numeric coercion',()=>{
  assert.deepEqual(T.parse('Conv2d','kernel_size','3'),[3,3]);
  assert.deepEqual(T.parse('Conv2d','kernel_size','2, 3'),[2,3]);
  assert.deepEqual(T.parse('Conv2d','padding','0 1'),[0,1]);
  assert.equal(T.parse('Conv2d','bias',false),false);
  for(const text of ['', '1.5', 'true', '1e3', 'NaN', 'Infinity', '9007199254740992', '1,2,3', '0', '-1']){
    assert.throws(()=>T.parse('Conv2d','kernel_size',text));
  }
  assert.throws(()=>T.parse('Conv2d','bias','true'));
  assert.throws(()=>T.parse('Conv2d','groups','1,2'));
});
test('Sample axis edits omit batch and reject duplicate/out-of-range permutations',()=>{
  assert.deepEqual(T.parse('Permute','dims','2,0,1'),[2,0,1]);
  assert.equal(T.parse('Flatten','start_dim','0'),0);
  assert.equal(T.parse('Flatten','end_dim','-1'),-1);
  for(const text of ['0,0,1','1,2','-1,0','0,1,3','true,0']){
    assert.throws(()=>T.parse('Permute','dims',text));
  }
  assert.throws(()=>T.parse('Flatten','start_dim','8'));
  assert.throws(()=>T.parse('Flatten','end_dim','-9'));
  assert.throws(()=>T.parse('Conv2d','unknown','1'));
});
