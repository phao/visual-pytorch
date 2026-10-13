"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const A=require('../app/web/activations.js');

test('Seven alphabetized activations explain LogSoftmax and fixed Softplus semantics',()=>{
  assert.deepEqual(Object.keys(A.info),['GELU','LogSoftmax','ReLU','Sigmoid','Softmax','Softplus','Tanh']);
  for(const key of Object.keys(A.info)){
    assert.equal(A.info[key].length,2);
    assert.ok(A.info[key][0]);assert.ok(A.info[key][1]);assert.ok(A.tex[key]);
  }
  const logs=A.info.LogSoftmax[1];
  assert.ok(logs.includes('log probabilities'));
  assert.ok(logs.includes('sample axis'));
  assert.ok(logs.includes('exp(h) sums to 1'));
  assert.ok(logs.includes('Batch B is excluded'));
  assert.ok(logs.includes('no trainable parameters'));
  const softplus=A.info.Softplus[1];
  for(const fragment of ['beta = 1','threshold = 20','No positive floor','round to zero','no trainable parameters'])
    assert.ok(softplus.includes(fragment),fragment);
  assert.ok(A.tex.LogSoftmax.includes('\\log'));
  assert.ok(A.tex.Softplus.includes('\\beta=1'));
});

test('Shared sample-axis grammar covers every axis at ranks one to eight',()=>{
  for(let rank=1;rank<=8;rank++)for(let dim=-rank;dim<rank;dim++)
    assert.equal(A.axis(String(dim),rank),dim);
  assert.equal(A.axis(' -1 '),-1);
  for(const token of ['', ' ', '1.0', '1e0', 'false', 'NaN', 'Infinity', '1,2', '8', '-9', '9007199254740992'])
    assert.throws(()=>A.axis(token));
  for(const [token,rank] of [['1',1],['-2',1],['2',2],['-3',2],['0',0],['0',9],['0',true]])
    assert.throws(()=>A.axis(token,rank));
});
