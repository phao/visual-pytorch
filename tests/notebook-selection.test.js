'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),H=require('../app/web/notebook-selection.js');
const selected=()=>({version:1,kind:'visual-pytorch-notebook-selection',path:'/a bundle',mode:'rerun',recipe:{kind:'report',id:'<svg>report',revision:'a'.repeat(64)},input_index_hash:'b'.repeat(64),requested_sha256:'c'.repeat(64)});
test('request captures exact inspected recipe and input bytes, independently of mutable heads',()=>{
  const s=selected(),r=H.request(s,'workspace',' /path/python ',600);assert.deepEqual(r,{selection:s,workspace_id:'workspace',python:'/path/python',timeout:600});
  r.selection.recipe.id='other';assert.equal(s.recipe.id,'<svg>report');
});
test('uninspected, implicit hashes, wrong modes and non-report references cannot execute',()=>{
  const s=selected();for(const bad of [null,{...s,mode:'train'},{...s,input_index_hash:null},{...s,requested_sha256:''},{...s,recipe:{...s.recipe,revision:null}},
    {...s,recipe:{...s.recipe,kind:'run'}},{...s,version:2}])assert.throws(()=>H.request(bad,'workspace','python',600));
});
test('execution requires explicit valid environment, workspace and cell timeout',()=>{
  for(const value of [0,-1,NaN,Infinity,1.5,true,Number.MAX_SAFE_INTEGER+1])assert.throws(()=>H.request(selected(),'w','python',value));
  assert.throws(()=>H.request(selected(),'','python',1));assert.throws(()=>H.request(selected(),'w',' ',1));
});
test('the two notebook modes have different honest behavior labels',()=>{
  assert.match(H.behavior('rerun'),/fresh CPU fits/);assert.match(H.behavior('result-report'),/no training/);
});
