const {test}=require('node:test');const assert=require('node:assert/strict');const J=require('../app/web/json-codec.js');
test('scientific clone and transport preserve signed zeros without sentinel collisions',()=>{
  const doc={values:[[0,-0,.6]],text:'-0.0 "marker"',none:null,truth:false};const value=J.clone(doc);
  assert.equal(Object.is(value.values[0][1],-0),true);assert.equal(Object.is(value.values[0][0],0),true);
  assert.equal(Object.is(JSON.parse(J.stringify(doc)).values[0][1],-0),true);assert.equal(value.text,doc.text);
  assert.deepEqual(J.clone({text:'\n',empty:{},array:[],missing:undefined}),{text:'\n',empty:{},array:[]});
  assert.equal(J.stringify([undefined]),'[null]');
});
test('nonfinite, cyclic and unsupported numbers refuse rather than corrupting scientific JSON',()=>{
  for(const value of [NaN,Infinity,-Infinity,1n])assert.throws(()=>J.stringify({value}));
  const cyclic={};cyclic.self=cyclic;assert.throws(()=>J.clone(cyclic));
});
