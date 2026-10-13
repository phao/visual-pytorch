const {test}=require('node:test');const assert=require('node:assert/strict');
const B=require('../app/web/parameter-binding.js');
test('tensor compatibility is exact shape and float32, never broadcasting',()=>{
  assert.equal(B.compatible({shape:[2,3]},{shape:[2,3],dtype:'float32'}),true);
  for(const row of [{shape:[3,2],dtype:'float32'},{shape:[6],dtype:'float32'},{shape:[2,3],dtype:'float64'}])
    assert.equal(B.compatible({shape:[2,3]},row),false);
});
test('selection pins exact attempt/checkpoint/native path and owner, without values',()=>{
  const row={path:'layer_0.weight',id:'["source","weight"]',shape:[2,3],dtype:'float32'};
  const page={run:{kind:'run',id:'r',revision:'a'.repeat(64)},attempt:1,checkpoint_id:'b'.repeat(64),parameters:[row]};
  assert.deepEqual(B.selection(page,row),{run:page.run,attempt:1,checkpoint_id:page.checkpoint_id,parameter:{path:row.path,id:row.id}});
  for(const changed of [{...page,attempt:0},{...page,checkpoint_id:null},{...page,parameters:[]}])assert.throws(()=>B.selection(changed,row));
  assert.throws(()=>B.selection(page,{...row,id:'another'}));
});
