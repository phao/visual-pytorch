"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const A=require('../app/web/activations.js'),H=require('../app/web/building-block-help.js');
test('Five alphabetical choices retain elementwise meanings and explicit Softmax normalization',()=>{
  assert.deepEqual(Object.keys(A.info),['GELU','ReLU','Sigmoid','Softmax','Tanh']);
  for(const key of Object.keys(A.info)){assert.ok(A.info[key][0]);assert.ok(A.info[key][1]);assert.ok(A.tex[key]);}
  const body=H.describe('add-activation',{activation:'Softmax'}).paragraphs.join(' ');
  assert.ok(body.includes('[B, N]'));assert.ok(body.includes('each row sums to 1'));
  assert.ok(body.includes('batch B is excluded'));assert.ok(body.includes('cross-entropy'));
  assert.ok(!body.includes('independently to each tensor value'));
  assert.ok(H.describe('add-activation',{activation:'ReLU'}).paragraphs[0].includes('independently'));
});
test('Sample axis permits positive/negative exact indices, never blank/fraction/exponent',()=>{
  for(let rank=1;rank<=8;rank++)for(let dim=-rank;dim<rank;dim++)assert.equal(A.axis(String(dim),rank),dim);
  assert.equal(A.axis(' -1 '),-1);
  for(const token of ['', ' ', '1.0', '1e0', 'false', 'NaN', 'Infinity', '1,2', '8','-9','9007199254740992'])
    assert.throws(()=>A.axis(token));
  for(const [token,rank] of [['1',1],['-2',1],['2',2],['-3',2],['0',0],['0',9],['0',true]])
    assert.throws(()=>A.axis(token,rank));
});
