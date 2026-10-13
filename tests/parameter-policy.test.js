const {test}=require('node:test');const assert=require('node:assert/strict');
const P=require('../app/web/parameter-policy.js');
test('initializer browsing is alphabetical and the grammar matches the backend',()=>{
  assert.deepEqual(P.kinds,[...P.kinds].sort());assert.deepEqual(P.parse('zeros',{}),{kind:'zeros'});
  assert.deepEqual(P.parse('constructor',{}),{kind:'constructor'});
  assert.deepEqual(P.parse('normal',{seed:'7',mean:'0',std:'.05'}),{kind:'normal',seed:7,mean:0,std:.05});
  assert.deepEqual(P.parse('values',{values:'[[1,2],[3,4]]'}),{kind:'values',dtype:'float32',values:[[1,2],[3,4]]});
});
test('empty, nonfinite and invalid random controls refuse',()=>{
  for(const fields of [{seed:'',mean:'0',std:'1'},{seed:'-1',mean:'0',std:'1'},{seed:'1.5',mean:'0',std:'1'},
    {seed:'7',mean:'Infinity',std:'1'},{seed:'7',mean:'0',std:'0'}])assert.throws(()=>P.parse('normal',fields));
  assert.throws(()=>P.parse('uniform',{seed:'7',low:'1',high:'1'}));assert.throws(()=>P.parse('values',{values:'1'}));
  assert.throws(()=>P.parse('eval',{}));
});
