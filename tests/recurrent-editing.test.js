"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const R=require('../app/web/recurrent-editing.js'),G=require('../app/web/graph-tools.js');
function fixture(){return {graph:{nodes:[
  {id:'seq',type:'Input'},{id:'h',type:'Input'},{id:'mask',type:'Input'},{id:'other',type:'Input'},
  {id:'scan',type:'Recurrent',ref:'step',x:100,y:100,
    sequence_inputs:[{input:'x',source:'seq'}],states:[{input:'h',output:'next',port:'final',source:'h'}],
    sequence_outputs:[{output:'next',port:'trajectory'}],controls:{mask:{source:'mask',adapter:'exact-bool-v1'}}},
  {id:'out',type:'Output'},{id:'end',type:'Output'},{id:'free',type:'Linear'}],edges:[
    {from:'seq',to:'scan'},{from:'h',to:'scan'},{from:'mask',to:'scan'},
    {from:'scan',from_port:'trajectory',to:'out'},{from:'scan',from_port:'final',to:'end'},
    {from:'other',to:'free',annotation:'keep'}]}};}
const node=(d,id)=>d.graph.nodes.find(n=>n.id===id);
test('Existing-interface tokens retain the block, candidate and current closure fences',()=>{
  const context={id:'parent',ref:'step',node:'scan',workspace_id:'workspace',panel_scope:'scope'};
  const info={...context,root_graph:'a'.repeat(64),root_model:'b'.repeat(64),transition:'c'.repeat(64),options:{}};
  const before=structuredClone(info),token=R.editResponse(info,context);
  assert.deepEqual(Object.keys(token),['id','ref','node','root_graph','root_model','transition']);
  assert.deepEqual(info,before);assert.equal(token.node,'scan');
});
test('A response for another node, reference, architecture or workspace is refused',()=>{
  const context={id:'parent',ref:'step',node:'scan',workspace_id:'workspace',panel_scope:'scope'};
  const info={...context,root_graph:'a'.repeat(64),root_model:'b'.repeat(64),transition:'c'.repeat(64)};
  for(const key of Object.keys(context))assert.throws(()=>R.editResponse({...info,[key]:'other'},context),/another block or workspace/);
  assert.throws(()=>R.editResponse(null,context));
});
test('Missing or malformed graph/current/candidate closure tokens cannot be submitted',()=>{
  const context={id:'parent',ref:'step',node:'scan',workspace_id:'workspace',panel_scope:'scope'};
  const info={...context,root_graph:'a'.repeat(64),root_model:'b'.repeat(64),transition:'c'.repeat(64)};
  for(const key of ['root_graph','root_model','transition'])for(const value of [null,undefined,7,'wrong','A'.repeat(64)])
    assert.throws(()=>R.editResponse({...info,[key]:value},context),/closure token/);
});
test('Stable recurrent source and returned-port order matches the declared interface',()=>{
  const d=fixture(),n=node(d,'scan');
  assert.deepEqual(G.slots(n,false),['seq','h','mask']);assert.deepEqual(G.slots(n,true),['trajectory','final']);
  n.controls.reset={source:'mask',adapter:'exact-bool-v1'};
  assert.deepEqual(R.sources(n),['seq','h','mask']);
  assert.match(R.portTitle(n,false,'mask'),/mask.*reset/);
  assert.match(R.portTitle(n,true,'final'),/Final state h/);
});
test('All expanded/collapsed recurrent sockets are distinct, visible, and move with the box',()=>{
  const n=node(fixture(),'scan');n.width=240;n.height=250;
  for(const spacing of [1,50,100])for(const collapsed of [false,true]){
    G.setSpacing(spacing);n.collapsed=collapsed;
    for(const output of [false,true]){
      const points=G.slots(n,output).map(peer=>G.portPoint(n,output,peer));
      assert.equal(new Set(points.map(p=>p.y)).size,points.length);
      assert.ok(points.every(p=>p.y>n.y&&p.y<n.y+G.height(n)));
      n.x+=24;n.y+=48;
      assert.deepEqual(G.slots(n,output).map(peer=>G.portPoint(n,output,peer)),points.map(p=>({x:p.x+24,y:p.y+48})));
    }
  }
  G.setSpacing(100);
});
test('Replacing one input binding preserves other roles, outgoing ports and sibling wires',()=>{
  const d=fixture(),before=structuredClone(d.graph.edges);G.connect(d,'other','scan',null,'mask');
  assert.equal(node(d,'scan').controls.mask.source,'other');
  assert.equal(node(d,'scan').states[0].source,'h');assert.equal(node(d,'scan').sequence_inputs[0].source,'seq');
  for(const edge of before.filter(e=>e.from!=='mask'))assert.ok(d.graph.edges.some(e=>G.sameWire(e,edge)));
  assert.ok(!d.graph.edges.some(e=>e.from==='mask'&&e.to==='scan'));
});
test('One source can explicitly serve several roles without duplicate physical wires',()=>{
  const d=fixture();G.connect(d,'h','scan',null,'mask');
  assert.deepEqual(R.sources(node(d,'scan')),['seq','h']);
  assert.equal(node(d,'scan').controls.mask.source,'h');
  assert.equal(d.graph.edges.filter(e=>e.from==='h'&&e.to==='scan').length,1);
});
test('Unknown named sockets are rejected without partially replacing the graph',()=>{
  const d=fixture(),before=JSON.stringify(d);
  for(const args of [['other','scan',null,null],['other','scan',null,'unknown'],['scan','free',null,null],['scan','free','unknown',null],['missing','scan',null,'h']])
    assert.throws(()=>G.connect(d,...args));
  assert.equal(JSON.stringify(d),before);
});
test('Deleting a wire preserves the explicit disconnected argument for repair',()=>{
  const d=fixture(),n=node(d,'scan');G.removeWire(d,{from:'h',to:'scan'});
  assert.equal(n.states[0].source,'h');assert.deepEqual(G.slots(n,false),['seq','h','mask']);
  G.connect(d,'other','scan',null,'h');assert.equal(n.states[0].source,'other');
});
test('Retargeting into an empty recurrent argument preserves all occupied argument wires and metadata',()=>{
  const d=fixture();G.removeWire(d,{from:'h',to:'scan'});
  assert.equal(G.retargetWire(d,{from:'other',to:'free'},'scan','h'),true);
  assert.equal(node(d,'scan').states[0].source,'other');
  assert.deepEqual(d.graph.edges.at(-1),{from:'other',to:'scan',annotation:'keep'});
  assert.ok(d.graph.edges.some(e=>e.from==='seq'&&e.to==='scan'));
});
test('Retargeting occupied/unknown recurrent sockets is atomic',()=>{
  const d=fixture(),before=JSON.stringify(d);
  for(const peer of ['h','mask',null,'missing'])assert.throws(()=>G.retargetWire(d,{from:'other',to:'free'},'scan',peer));
  assert.equal(JSON.stringify(d),before);
});
test('Moving a destination within one recurrent box leaves the previous role visibly disconnected',()=>{
  const d=fixture();G.removeWire(d,{from:'h',to:'scan'});
  assert.equal(G.retargetWire(d,{from:'seq',to:'scan'},'scan','h'),true);
  assert.equal(node(d,'scan').states[0].source,'seq');
  assert.match(node(d,'scan').sequence_inputs[0].source,/^disconnected\//);
  assert.equal(d.graph.edges.filter(e=>e.from==='seq'&&e.to==='scan').length,1);
});
test('Adding an Output preserves the selected recurrent port, not an aggregate tensor',()=>{
  const d=fixture(),id=G.addOutput(d,'scan','final');
  assert.ok(d.graph.edges.some(e=>e.from==='scan'&&e.from_port==='final'&&e.to===id));
  const before=JSON.stringify(d);assert.throws(()=>G.addOutput(d,'scan','unknown'));assert.equal(JSON.stringify(d),before);
});
test('Insertion geometry uses the selected tensor port shape, never packed aggregate width',()=>{
  const n=node(fixture(),'scan');
  const c={ports:{trajectory:[4,2],final:[2]},output_shape:[10]};
  assert.deepEqual(G.tensorShape(n,c,'final'),[2]);assert.deepEqual(G.tensorShape(n,c,'trajectory'),[4,2]);
  assert.equal(G.tensorShape(n,c,'missing'),undefined);
  assert.deepEqual(G.tensorShape({type:'Splitter',parts:[{output:'a',features:2}]},{input_shape:[7,5]},'a'),[7,2]);
  assert.deepEqual(G.tensorShape({type:'Replicator'},{output_shape:[7,5]},'a'),[7,5]);
});
test('Router insertion on a recurrent output changes only the selected port path',()=>{
  const d=fixture();node(d,'scan').x=100;node(d,'scan').y=100;node(d,'out').x=500;node(d,'end').x=500;
  const id=G.addRouter(d,'Splitter',2,'scan','final');
  assert.ok(d.graph.edges.some(e=>e.from==='scan'&&e.from_port==='trajectory'&&e.to==='out'));
  assert.ok(d.graph.edges.some(e=>e.from==='scan'&&e.from_port==='final'&&e.to===id));
  G.removeBoundary(d,id);
  assert.ok(d.graph.edges.some(e=>e.from==='scan'&&e.from_port==='final'&&e.to==='end'));
});
test('Unary insertion/removal updates recurrent source bindings without guessing recurrence bypass',()=>{
  const d=fixture(),n=node(d,'scan');G.replaceSource(n,'h','free');
  d.graph.edges=d.graph.edges.filter(e=>e.from!=='h'&&e.from!=='other');
  d.graph.edges.push({from:'h',to:'free'},{from:'free',to:'scan'});
  G.removeBlock(d,'free');assert.equal(n.states[0].source,'h');
  assert.ok(d.graph.edges.some(e=>e.from==='h'&&e.to==='scan'));
  G.removeBlock(d,'scan');assert.ok(!d.graph.edges.some(e=>e.to==='out'||e.to==='end'));
});
