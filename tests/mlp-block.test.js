"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const G=require('../app/web/graph-tools.js'),A=require('../app/web/activations.js');
test('Width syntax expands count*width terms; the first width is the input',()=>{
  assert.deepEqual(G.parseWidths('2, 3, 2, 4'),[2,3,2,4]);
  assert.deepEqual(G.parseWidths('2, 10*3, 2'),[2,...Array(10).fill(3),2]);
  assert.deepEqual(G.parseWidths('2, 4*3, 2*2, 4*3, 2'),[2,3,3,3,3,2,2,3,3,3,3,2]);
  assert.deepEqual(G.parseWidths(' 5 ,1 * 7 '),[5,7]);
  for(const [bad,why] of [['4','at least two'],['2,,3','separated by commas'],['2, x','not a width'],['2, 0','positive'],['2, 3*0','positive'],['2, -3','not a width'],['2, 2.5','not a width'],['2, 70*3','at most 64']])
    assert.throws(()=>G.parseWidths(bad),new RegExp(why),bad);
});
const base=()=>({graph:{nodes:[{id:'input',type:'Input',features:4,x:90,y:170},{id:'linear1',type:'Linear',features:4,bias:true,x:340,y:170},{id:'output',type:'Output',x:600,y:170}],
  edges:[{from:'input',to:'linear1'},{from:'linear1',to:'output'}]}});
test('After a selection the chain is spliced in with one activation per hidden layer and none after the readout',()=>{
  const doc=base(),r=G.addMLP(doc,{after:'linear1',inputWidth:4,widths:G.parseWidths('4, 3*8, 2'),activation:'GELU'});
  const order=[];let at='linear1';
  while(at!=='output'){at=doc.graph.edges.find(e=>e.from===at).to;order.push(doc.graph.nodes.find(n=>n.id===at));}
  assert.deepEqual(order.map(n=>n.type==='Linear'?'L'+n.features:n.type==='Activation'?n.activation:n.type),['L8','GELU','L8','GELU','L8','GELU','L2','Output']);
  assert.equal(r.blocks,7);assert.equal(doc.graph.edges.filter(e=>e.to==='output').length,1);
  assert.ok(order.every((n,i)=>i===0||n.x>order[i-1].x||n.type==='Output'));
});
test('With nothing selected a new Input -> chain -> Output branch is added; Softmax gets an explicit axis',()=>{
  const doc=base(),r=G.addMLP(doc,{widths:[3,5,1],activation:'Softmax'});
  const input=doc.graph.nodes.find(n=>n.label==='MLP input 1'),output=doc.graph.nodes.find(n=>n.label==='MLP output 1');
  assert.equal(input.features,3);assert.ok(output);assert.equal(r.blocks,5);
  assert.equal(doc.graph.nodes.find(n=>n.type==='Activation').dim,-1);
});
test('Width mismatch and the 64-block limit are refused without changing the drawing',()=>{
  const doc=base(),before=JSON.stringify(doc);
  assert.throws(()=>G.addMLP(doc,{after:'linear1',inputWidth:4,widths:[3,2]}),/selected block outputs 4, not 3/);
  assert.throws(()=>G.addMLP(doc,{after:'linear1',inputWidth:4,widths:[4,...Array(40).fill(2)]}),/adds 79 blocks; the limit is 64/);
  assert.equal(JSON.stringify(doc),before);
});
test('Building blocks and the Inspector share one grouped list of all seven activations',()=>{
  assert.deepEqual(A.groups.flatMap(g=>g.names).sort(),Object.keys(A.info).sort());
  assert.deepEqual(A.groups.map(g=>g.label),['Elementwise','Normalizing along an axis']);
  assert.match(A.options('Tanh'),/<optgroup label="Elementwise"><option>GELU<\/option>.*<option selected>Tanh<\/option><\/optgroup><optgroup label="Normalizing along an axis"><option>LogSoftmax/);
});
