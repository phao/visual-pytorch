"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const R=require('../app/web/recurrent-editing.js'),G=require('../app/web/graph-tools.js');
const fixture=()=>({graph:{nodes:[
  ...['x','h','c','len','other'].map(id=>({id,type:'Input'})),
  {id:'lstm',type:'LSTM',hidden_size:7,proj_size:4,num_layers:2,bidirectional:true,
   inputs:{cell:'c',sequence:'x',hidden:'h'},lengths:{source:'len',adapter:'exact-int64-v1'},x:100,y:100},
  {id:'out',type:'Output'},{id:'free',type:'Linear'}],edges:[
    ...['x','h','c','len'].map(from=>({from,to:'lstm'})),
    {from:'lstm',from_port:'sequence',to:'out'},{from:'other',to:'free'}]}});
const node=d=>d.graph.nodes.find(n=>n.id==='lstm');
test('LSTM socket order follows role semantics, never object-key or edge order',()=>{
  const n=node(fixture());assert.equal(R.multiport(n),true);
  assert.deepEqual(R.sources(n),['x','h','c','len']);assert.deepEqual(R.ports(n),['sequence','hidden','cell']);
  assert.match(R.portTitle(n,false,'h'),/Initial hidden/);
  assert.match(R.portTitle(n,true,'hidden'),/Hout/);assert.match(R.portTitle(n,true,'cell'),/Hcell/);
});
test('All LSTM sockets remain distinct when collapsed, resized and moved',()=>{
  const n=node(fixture());
  for(const spacing of [1,50,100])for(const collapsed of [false,true]){
    G.setSpacing(spacing);n.collapsed=collapsed;
    for(const output of [false,true]){
      const points=G.slots(n,output).map(peer=>G.portPoint(n,output,peer));
      assert.equal(new Set(points.map(p=>p.y)).size,points.length);
      assert.ok(points.every(p=>p.y>n.y&&p.y<n.y+G.height(n)));
      n.x+=24;n.y+=48;
      const moved=G.slots(n,output).map(peer=>G.portPoint(n,output,peer));
      moved.forEach((p,index)=>assert.ok(Math.abs(p.x-points[index].x-24)<1e-9&&Math.abs(p.y-points[index].y-48)<1e-9));
    }
  }
  G.setSpacing(100);
});
test('LSTM source replacement leaves other role bindings and returned wires intact',()=>{
  const d=fixture();G.connect(d,'other','lstm',null,'h');
  assert.deepEqual(node(d).inputs,{cell:'c',sequence:'x',hidden:'other'});
  assert.equal(node(d).lengths.source,'len');assert.ok(d.graph.edges.some(e=>e.from==='lstm'&&e.from_port==='sequence'));
});
test('An explicitly shared physical source updates all aliased state roles',()=>{
  const d=fixture();node(d).inputs.cell='h';node(d).lengths.source='h';
  R.replaceSource(node(d),'h','other');assert.equal(node(d).inputs.hidden,'other');
  assert.equal(node(d).inputs.cell,'other');assert.equal(node(d).lengths.source,'other');
  assert.deepEqual(R.sources(node(d)),['x','other']);
});
test('An LSTM Output must choose a real returned port',()=>{
  const d=fixture();const id=G.addOutput(d,'lstm','cell');
  assert.ok(d.graph.edges.some(e=>e.from==='lstm'&&e.from_port==='cell'&&e.to===id));
  for(const peer of [null,'missing'])assert.throws(()=>G.connect(d,'lstm','free',peer,null));
  const shape={ports:{sequence:[5,8],hidden:[4,4],cell:[4,7]}};
  assert.deepEqual(G.tensorShape(node(d),shape,'cell'),[4,7]);
});
test('Retargeting a wire to an empty LSTM role preserves occupied role wires',()=>{
  const d=fixture();G.removeWire(d,{from:'c',to:'lstm'});
  assert.equal(G.retargetWire(d,{from:'other',to:'free'},'lstm','c'),true);
  assert.equal(node(d).inputs.cell,'other');assert.equal(node(d).inputs.hidden,'h');
  assert.ok(d.graph.edges.some(e=>e.from==='x'&&e.to==='lstm'));
});
test('Occupied/unknown LSTM destinations reject atomically',()=>{
  const d=fixture(),before=JSON.stringify(d);
  for(const peer of ['h','c',null,'unknown'])assert.throws(()=>G.retargetWire(d,{from:'other',to:'free'},'lstm',peer));
  assert.equal(JSON.stringify(d),before);
});
test('Deleting LSTM never bypasses computation or silently rewires states',()=>{
  const d=fixture();G.removeBlock(d,'lstm');
  assert.ok(!d.graph.edges.some(e=>e.from==='x'&&e.to==='out'));
  assert.ok(d.graph.nodes.some(n=>n.id==='h'));assert.ok(d.graph.nodes.some(n=>n.id==='out'));
});
