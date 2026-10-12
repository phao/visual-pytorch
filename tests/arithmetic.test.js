'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const G=require('../app/web/graph-tools.js'),T=require('../app/web/tensor-operators.js'),H=require('../app/web/building-block-help.js');
const fixture=()=>({graph:{nodes:[{id:'x',type:'Input',features:2,x:0,y:0},{id:'other',type:'Input',features:2,x:0,y:160},
  {id:'sum',type:'Add',sources:['x'],op_version:1,x:200,y:0},{id:'y',type:'Output',x:420,y:0}],
  edges:[{from:'x',to:'sum'},{from:'sum',to:'y'}]}});
test('Insertion anchor accepts only the selected block and declared multioutput port',()=>{
  const n={id:'split',type:'Splitter',parts:[{output:'a',features:2},{output:'b',features:3}]};
  assert.equal(G.appendTarget(n,'split','b'),true);
  for(const [id,port] of [['other','b'],['split','missing'],['split',null],['split',undefined]])assert.equal(G.appendTarget(n,id,port),false);
  assert.equal(G.appendTarget({id:'r',type:'Replicator',outputs:['first','second']},'r','second'),true);
  assert.equal(G.appendTarget({id:'fc',type:'Linear'},'fc','b'),false);
});
test('Add appends ordered input sockets and source wires without replacing existing contributors',()=>{
  const d=fixture();G.connect(d,'other','sum');
  assert.deepEqual(d.graph.nodes[2].sources,['x','other']);assert.deepEqual(G.slots(d.graph.nodes[2],false),['x','other',null]);
  assert.ok(d.graph.edges.some(e=>e.from==='x'&&e.to==='sum'));
  G.removeWire(d,{from:'x',to:'sum'});assert.deepEqual(d.graph.nodes[2].sources,['other']);
});
test('Destination dragging into Add preserves source/port, permits empty socket and refuses occupied socket atomically',()=>{
  const d=fixture();d.graph.nodes.push({id:'old',type:'Output'});d.graph.edges.push({from:'other',to:'old'});
  const edge={from:'other',to:'old'};const before=JSON.stringify(d);
  assert.throws(()=>G.retargetWire(d,edge,'sum','x'));assert.equal(JSON.stringify(d),before);
  assert.equal(G.retargetWire(d,edge,'sum',null),true);assert.deepEqual(d.graph.nodes[2].sources,['x','other']);
  assert.ok(d.graph.edges.some(e=>e.from==='other'&&e.to==='sum'));
});
test('Removing Add never pretends a remaining input is an equivalent bypass; Scale removal updates downstream source keys',()=>{
  const d=fixture();G.removeBlock(d,'sum');assert.ok(!d.graph.edges.some(e=>e.to==='y'));
  const q=fixture();q.graph.nodes.push({id:'scale',type:'Scale',coefficient:2,x:120,y:160});
  q.graph.edges.push({from:'other',to:'scale'});G.connect(q,'scale','sum');
  G.removeBlock(q,'scale');assert.deepEqual(q.graph.nodes.find(n=>n.id==='sum').sources,['x','other']);
  assert.ok(q.graph.edges.some(e=>e.from==='other'&&e.to==='sum'));
});
test('Scale parses fixed finite signed/decimal coefficients and refuses coercion, infinities, unsafe integers and unsupported fields',()=>{
  assert.equal(T.defaults('Scale',[3,2]).coefficient,2);
  for(const [text,value] of [['0',0],['-2',-2],['.25',.25],['1.2e3',1200],['9007199254740991',9007199254740991],['-9007199254740991',-9007199254740991]])assert.equal(T.parse('Scale','coefficient',text),value);
  for(const value of ['',true,'false','NaN','Infinity','1e100','9007199254740992','-9007199254740992','9007199254740992.0','9.007199254740992e15','1e20','0x10','1,2'])assert.throws(()=>T.parse('Scale','coefficient',value));
  assert.throws(()=>T.parse('Scale','trainable','false'));
});
test('Add teaching is distinct from concatenation and explains derivative paths; Scale is explicitly fixed',()=>{
  const body=H.describe('add-add').paragraphs.join(' ');assert.match(body,/identical sample shape/);assert.match(body,/No broadcasting/);assert.match(body,/never detaches/);
  const scale=H.describe('add-tensor-Scale',{tensorType:'Scale'});assert.match(scale.paragraphs.join(' '),/fixed coefficient/);assert.match(scale.paragraphs.join(' '),/no learned weights/);
});
