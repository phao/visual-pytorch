"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const L=require('../app/web/layout.js'),G=require('../app/web/graph-tools.js');
const clone=x=>JSON.parse(JSON.stringify(x));
const fixture=()=>({id:'fixture',graph:{input_order:['a','b'],nodes:[
  {id:'a',type:'Input',features:2,x:99,y:17}, {id:'b',type:'Input',features:3,x:99,y:180},
  {id:'la',type:'Linear',features:2,width:250,height:120,x:99,y:17},
  {id:'lb',type:'Activation',activation:'ReLU',x:99,y:17},
  {id:'join',type:'Concat',sources:['la','lb'],x:99,y:17},
  {id:'y',type:'Output',x:99,y:17}],edges:[
    {from:'a',to:'la'},{from:'b',to:'lb'},{from:'la',to:'join'},{from:'lb',to:'join'},{from:'join',to:'y'}]},
  view:{x:1,y:2,zoom:1},dataset:{x:[[1,2,3,4,5]],y:[[0]]},result:{weights:{layer_0:[[1]]}}});
function meaning(doc){const result=clone(doc);for(const n of result.graph.nodes)for(const key of ['x','y','collapsed'])delete n[key];return result;}
function noOverlap(nodes){
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
    const a=nodes[i],b=nodes[j];assert.ok(a.x+G.width(a)<=b.x || b.x+G.width(b)<=a.x ||
      a.y+G.height(a)<=b.y || b.y+G.height(b)<=a.y,`${a.id} overlaps ${b.id}`);
  }
}
test('Independent profiles/defaults and validation reject invalid numbers, modes and booleans',()=>{
  const a=L.defaults();a.collapsed.block_gap=900;assert.equal(L.defaults().collapsed.block_gap,16);
  for(const bad of [null,{},[],{...L.defaults().collapsed,style:'pretend'},
    {...L.defaults().collapsed,direction:'RL'},...['layer_gap','block_gap'].flatMap(key=>
      [0,7,1001,1.5,NaN,Infinity,'16',true].map(v=>({...L.defaults().collapsed,[key]:v}))),
    {...L.defaults().collapsed,fit:1},{...L.defaults().collapsed,rearrange:'yes'}])assert.throws(()=>L.validate(bad));
});
test('Bulk collapse/reflow and expansion retain semantics, preferred sizes and source order',()=>{
  const doc=fixture(),before=clone(doc),small=L.apply(doc,true,L.defaults().collapsed,G);
  assert.deepEqual(doc,before);assert.deepEqual(meaning(small),meaning(doc));
  assert.ok(small.graph.nodes.every(n=>n.collapsed));noOverlap(small.graph.nodes);
  const big=L.apply(small,false,L.defaults().expanded,G);
  assert.deepEqual(meaning(big),meaning(doc));assert.ok(big.graph.nodes.every(n=>!n.collapsed));
  assert.equal(G.width(big.graph.nodes.find(n=>n.id==='la')),250);noOverlap(big.graph.nodes);
  assert.deepEqual(big.graph.nodes.map(n=>n.id),doc.graph.nodes.map(n=>n.id));
  assert.deepEqual(big.graph.edges,doc.graph.edges);
  assert.ok(Math.max(...big.graph.nodes.map(n=>n.x+G.width(n)))>Math.max(...small.graph.nodes.map(n=>n.x+G.width(n))));
});
test('Layers advance in the requested direction, account for actual sizes and configured gaps',()=>{
  for(const collapsed of [false,true])for(const direction of ['LR','TB']){
    const p={...L.defaults()[collapsed?'collapsed':'expanded'],direction,layer_gap:101,block_gap:37};
    const doc=L.apply(fixture(),collapsed,p,G),by=new Map(doc.graph.nodes.map(n=>[n.id,n]));
    noOverlap(doc.graph.nodes);
    for(const e of doc.graph.edges){const a=by.get(e.from),b=by.get(e.to);
      assert.ok(direction==='LR'?b.x-a.x-G.width(a)>=101:b.y-a.y-G.height(a)>=101);
    }
    assert.deepEqual(L.apply(doc,collapsed,p,G),doc,'Relayout is deterministic and idempotent');
  }
});
test('Position-preserving profile folds without rearranging or discarding unrelated state',()=>{
  const doc=fixture(),next=L.apply(doc,true,{...L.defaults().collapsed,rearrange:false},G);
  assert.deepEqual(next.graph.nodes.map(n=>[n.x,n.y]),doc.graph.nodes.map(n=>[n.x,n.y]));
  assert.deepEqual(next.dataset,doc.dataset);assert.deepEqual(next.result,doc.result);
});
test('Cyclic, disconnected and empty drafts remain unchanged computationally and terminate',()=>{
  for(const direction of ['LR','TB']){
    const doc=fixture();doc.graph.edges.push({from:'y',to:'join'});
    doc.graph.nodes.push({id:'alone',type:'Linear',features:3});
    const result=L.apply(doc,true,{...L.defaults().collapsed,direction},G);
    assert.deepEqual(meaning(result),meaning(doc));noOverlap(result.graph.nodes);
  }
  const empty={graph:{nodes:[],edges:[]}};assert.deepEqual(L.apply(empty,true,L.defaults().collapsed,G),empty);
});
test('Router and Concat port counts enlarge compact bodies without overlapping any block',()=>{
  const doc=fixture();doc.graph.nodes.push({id:'s',type:'Splitter',parts:Array.from({length:12},(_,i)=>({output:'p'+i,features:1})),x:0,y:0});
  const result=L.apply(doc,true,L.defaults().collapsed,G),split=result.graph.nodes.find(n=>n.id==='s');
  noOverlap(result.graph.nodes);assert.ok(G.height(split)>180);
  const sockets=G.slots(split,true).map(p=>G.portPoint(split,true,p).y);
  assert.equal(new Set(sockets).size,12);
});
test('Oversized geometry and stale edges fail atomically before replacing the original diagram',()=>{
  const doc=fixture();doc.graph.nodes.forEach(n=>{n.width=100000;});const before=clone(doc);
  assert.throws(()=>L.apply(doc,false,L.defaults().expanded,G),/bounds/);assert.deepEqual(doc,before);
  const missing=fixture();missing.graph.edges.push({from:'missing',to:'a'});
  assert.throws(()=>L.apply(missing,true,L.defaults().collapsed,G),/missing/);
});
test('Many uneven DAGs, independent paths, fanout and both directions have collision-free layouts',()=>{
  let seed=91;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(let run=0;run<25;run++){
    const nodes=Array.from({length:64},(_,i)=>({id:String(i).padStart(2,'0'),type:'Linear',features:3,width:150+random()*200,height:70+random()*100}));
    const edges=[];for(let i=0;i<63;i++)for(let j=i+1;j<64;j++)if(random()<.03)edges.push({from:nodes[i].id,to:nodes[j].id});
    const doc={graph:{nodes,edges}};
    for(const collapsed of [true,false])for(const direction of ['LR','TB']){
      const result=L.apply(doc,collapsed,{...L.defaults().collapsed,direction},G);noOverlap(result.graph.nodes);
      assert.deepEqual(meaning(result),meaning(doc));
    }
  }
});
