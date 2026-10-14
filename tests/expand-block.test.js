"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const G=require('../app/web/graph-tools.js');
const doc=(block,more={})=>({graph:{nodes:[{id:'input',type:'Input',shape:[3,8,8],axes:['C','H','W'],x:90,y:170},{id:'b',x:340,y:170,...block},
  {id:'add',type:'Add',sources:['b','input'],x:600,y:170},{id:'output',type:'Output',x:850,y:170}],
  edges:[{from:'input',to:'b'},{from:'b',to:'add'},{from:'input',to:'add'},{from:'add',to:'output'}]},...more});
// The blocks between Input and Add along the main path.
const chain=d=>{const order=[];for(let at='input';;){const e=d.graph.edges.find(e=>e.from===at&&e.to!=='add'&&e.to!=='output');if(!e)return order;at=e.to;order.push(d.graph.nodes.find(n=>n.id===at));}};
test('Conv2d expands into Patches -> Linear -> Permute, rewired in place, downstream shifted',()=>{
  const d=doc({type:'Conv2d',out_channels:4,kernel_size:[3,3],padding:[1,1],bias:false}),r=G.expandBlock(d,'b',{input_shape:[3,8,8],output_shape:[4,8,8]});
  const order=chain(d);
  assert.deepEqual(order.map(n=>n.type),['Patches','Linear','Permute']);
  assert.deepEqual([order[0].kernel_size,order[0].stride,order[0].padding,order[0].dilation],[[3,3],[1,1],[1,1],[1,1]]);
  assert.deepEqual([order[1].features,order[1].bias,order[2].dims],[4,false,[2,0,1]]);
  assert.equal(r.blocks,3);assert.ok(!d.graph.nodes.some(n=>n.id==='b'));
  const add=d.graph.nodes.find(n=>n.id==='add');
  assert.deepEqual(add.sources,[r.last,'input']);assert.ok(d.graph.edges.some(e=>e.from===r.last&&e.to==='add'));
  assert.equal(add.x,600+2*224);assert.ok(order.every((n,i)=>!i||n.x>order[i-1].x));
});
test('Pooling expands through Reshape and Reduce; global average pooling into Reshape -> Reduce -> Reshape',()=>{
  let d=doc({type:'MaxPool2d',kernel_size:[2,2],stride:[2,2],padding:[0,0],dilation:[1,1]});G.expandBlock(d,'b',{input_shape:[3,8,8],output_shape:[3,4,4]});
  assert.deepEqual(chain(d).map(n=>n.type+(n.shape?JSON.stringify(n.shape):'')+(n.mode||'')),['Patches','Reshape[4,4,3,4]','Reducemax','Permute']);
  d=doc({type:'AvgPool2d',kernel_size:[3,3],stride:[1,1],padding:[1,1],count_include_pad:true});G.expandBlock(d,'b',{input_shape:[3,8,8],output_shape:[3,8,8]});
  assert.deepEqual(chain(d).map(n=>n.mode||n.type),['Patches','Reshape','mean','Permute']);
  d=doc({type:'AdaptiveAvgPool2d',output_size:[1,1]});G.expandBlock(d,'b',{input_shape:[3,8,8],output_shape:[3,1,1]});
  assert.deepEqual(chain(d).map(n=>n.shape?JSON.stringify(n.shape):n.mode),['[3,64]','mean','[3,1,1]']);
});
test('Inexact or shared cases are refused without changing the drawing',()=>{
  const s={input_shape:[3,8,8],output_shape:[3,8,8]};
  for(const [block,why,more] of [[{type:'Conv2d',out_channels:3,kernel_size:[1,1],groups:3},'groups'],[{type:'MaxPool2d',kernel_size:[3,3],stride:[1,1],padding:[1,1]},'−∞'],
    [{type:'AvgPool2d',kernel_size:[3,3],stride:[1,1],padding:[1,1],count_include_pad:false},'excludes padded'],[{type:'AdaptiveAvgPool2d',output_size:[2,2]},'global average'],
    [{type:'Conv2d',out_channels:3,kernel_size:[1,1],share_with:'x'},'shares parameters'],[{type:'Conv2d',out_channels:3,kernel_size:[1,1]},'parameter policy',{}],[{type:'Linear',features:2},'Only Conv2d']]){
    const d=doc(block);if(why==='parameter policy')d.graph.parameter_policy={parameters:{'["b","weight"]':{mode:'frozen'}}};
    const before=JSON.stringify(d);assert.throws(()=>G.expandBlock(d,'b',s),new RegExp(why));assert.equal(JSON.stringify(d),before);
  }
  const d=doc({type:'Conv2d',out_channels:3,kernel_size:[1,1]});assert.throws(()=>G.expandBlock(d,'b',undefined),/shapes are known/);
});
