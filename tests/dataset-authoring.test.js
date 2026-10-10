const test=require('node:test'),assert=require('node:assert/strict');
const a=require('../app/web/dataset-authoring.js');
const fixture=()=>({fields:[{id:'x',dtype:'float32',shape:[1]},{id:'y',dtype:'float32',shape:[]}],records:[{id:'one',values:{x:[1],y:2}},{id:'two',values:{x:[3],y:4}}]});
test('sample shape accepts scalar, tensor and first-axis ragged only',()=>{
  assert.deepEqual(a.dimensions(''),[]);assert.deepEqual(a.dimensions('3, 32 32'),[3,32,32]);assert.deepEqual(a.dimensions('-1,2'),[-1,2]);
  for(const value of ['0','2,-1','true','NaN','Infinity','1.5','-2','9007199254740992',Array(9).fill('1').join(',')])assert.throws(()=>a.dimensions(value));
});
test('cells preserve JSON types and missing values, reject nonfinite and unsafe integers',()=>{
  assert.deepEqual(a.cell('[[null,2],[3,4]]','float32'),[[null,2],[3,4]]);assert.equal(a.cell('"1"','category'),'1');assert.equal(a.cell('1','category'),1);
  assert.equal(a.cell('true','bool'),true);assert.equal(a.cell('null','float32'),null);assert.equal(a.cell('1e20','float32'),1e20);
  for(const value of ['1e400','NaN','{"tensor":1}'])assert.throws(()=>a.cell(value,'float32'));
  for(const dtype of ['int64','category'])for(const value of ['9007199254740993','1.2','[1,1e20]'])assert.throws(()=>a.cell(value,dtype));
});
test('adding fields preserves identities/order/data and marks new values explicitly missing',()=>{
  const d=fixture(),before=structuredClone(d),result=a.addField(d,{id:'new',dtype:'bool',shape:[]});
  assert.deepEqual(d,before);assert.deepEqual(result.fields.map(f=>f.id),['x','y','new']);assert.deepEqual(result.records[0].values,{x:[1],y:2,new:null});
  for(const id of ['x','','a\n'])assert.throws(()=>a.addField(d,{id}));
});
test('arbitrary legal stable IDs cannot trigger object prototype setters',()=>{
  let d=fixture();for(const id of ['__proto__','constructor','toString'])d=a.addField(d,{id,dtype:'float32',shape:[]});
  for(const row of d.records)for(const id of ['__proto__','constructor','toString']){assert.equal(Object.hasOwn(row.values,id),true);assert.equal(row.values[id],null);}
  d=a.addRecord(d,'three');assert.equal(Object.hasOwn(d.records[2].values,'__proto__'),true);
  assert.equal(Object.hasOwn(a.removeField(d,'__proto__').records[0].values,'__proto__'),false);
});
test('explicit field removal does not affect other values or source',()=>{
  const d=fixture(),result=a.removeField(d,'x');assert.deepEqual(result.fields.map(f=>f.id),['y']);assert.deepEqual(result.records.map(r=>r.values),[{y:2},{y:4}]);assert.equal(d.fields.length,2);
});
test('new observations have stable explicit IDs without inventing measurements',()=>{
  const d=fixture(),result=a.addRecord(d,'third');assert.deepEqual(result.records[2],{id:'third',values:{x:null,y:null}});assert.equal(d.records.length,2);
  for(const id of ['one','','a\t'])assert.throws(()=>a.addRecord(d,id));
});
test('metadata parsing rejects repeated keys, including escaped and nested aliases',()=>{
  assert.deepEqual(a.json('{"procedure":{"unit":"m"},"labels":["x",2,true,null],"e":1e-5}'),{procedure:{unit:'m'},labels:['x',2,true,null],e:1e-5});
  for(const value of ['{"x":1,"x":2}','{"x":1,"\\u0078":2}','[{"x":1,"x":2}]','{"nested":{"x":1,"x":2}}'])assert.throws(()=>a.json(value),/Duplicate/);
  assert.deepEqual(a.json('{"x":{},"y":{"x":1},"z":[],"quoted":"a\\\"b"}'),{x:{},y:{x:1},z:[],quoted:'a"b'});
});
test('paging bounds both axes, preserves order and does not copy hidden cells',()=>{
  const d=fixture();for(let i=0;i<30;i++)d.records.push({id:'more'+i,values:{x:[i],y:i}});
  const page=a.page(d,20,1,5,1);assert.equal(page.rowTotal,32);assert.equal(page.fieldTotal,2);assert.equal(page.rows.length,5);assert.equal(page.fields.length,1);
  assert.equal(page.rows[0].id,'more18');assert.deepEqual(page.rows[0].values,{y:18});page.rows[0].values.y=-1;assert.equal(d.records[20].values.y,18);
  for(const args of [[-1],[0,-1],[0,0,0],[0,0,101],[0,0,20,21],[0.5]])assert.throws(()=>a.page(d,...args));
});
