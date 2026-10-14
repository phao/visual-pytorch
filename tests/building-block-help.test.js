"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const H=require('../app/web/building-block-help.js'),T=require('../app/web/tensor-operators.js');
test('All 25 About pages have concrete non-executing teaching descriptions',()=>{
  // 23 palette entries (MLP added), plus About pages for legacy Replicator (no longer in the palette)
  // and Causal lag window (Inspector About only).
  const keys=Object.keys(H.names),before=JSON.stringify(T.definitions);
  assert.equal(keys.length+T.types.length,25);
  assert.ok(keys.includes('add-replicator')&&keys.includes('block-causal-lag'));
  for(const id of keys){const h=H.describe(id);assert.equal(h.name,H.names[id]);assert.ok(h.paragraphs.length>=3);assert.ok(h.paragraphs.every(p=>typeof p==='string' && p.length>30));}
  for(const type of T.types){const h=H.describe('add-tensor-'+type,{tensorType:type});assert.equal(h.name,type);assert.equal(h.tex,T.definitions[type].tex);assert.ok(h.paragraphs[0].includes(T.definitions[type].explanation));}
  assert.equal(JSON.stringify(T.definitions),before);
  assert.equal(H.describe('unknown'),null);
});
test('Linear identifies the last axis, learned weights, bias and affine composition',()=>{
  const h=H.describe('add-linear'),body=h.paragraphs.join(' ');
  for(const value of ['[B, 10, 3]','[B, 10, k]','k × n','bias','affine'])assert.ok(body.includes(value),value);
  assert.equal(h.equation,'y = x Aᵀ + b');
});
test('Activation explanation tracks all selectable functions without choosing a loss',()=>{
  for(const activation of ['ReLU','Tanh','Sigmoid','GELU','Softmax']){
    const context={activation},before=JSON.stringify(context),h=H.describe('add-activation',context);
    assert.ok(h.paragraphs[0].includes(activation));assert.equal(JSON.stringify(context),before);
  }
  assert.ok(H.describe('add-activation',{activation:'Sigmoid'}).paragraphs.join(' ').includes('does not choose a classification loss'));
});
test('Split, fan-out, reshape and axis permutation remain distinct operations',()=>{
  assert.ok(H.describe('add-splitter').paragraphs.join(' ').includes('sum to the incoming'));
  assert.ok(H.describe('add-replicator').paragraphs.join(' ').includes('not a physical tensor copy'));
  assert.ok(H.describe('add-reshape').paragraphs.join(' ').includes('use Permute'));
  assert.ok(H.describe('add-concat').paragraphs.join(' ').includes('[B, 10, 5]'));
});
test('Architecture help describes independent instance weights and current packed interface',()=>{
  const name='<script>not executable</script>',body=H.describe('add-reference',{referenceName:name}).paragraphs.join(' ');
  assert.ok(body.includes(name));assert.ok(body.includes('not weight sharing'));assert.ok(body.includes('packed tensors'));
});
test('LSTM help separates supplied state, projection, execution lengths and task heads',()=>{
  const body=H.describe('add-lstm').paragraphs.join(' ');
  for(const phrase of ['supplied initial','cell width','[B,layers × directions,width]','not loss masks','Family7','runtime-T LSTM is not yet supported'])
    assert.ok(body.includes(phrase),phrase);
});
