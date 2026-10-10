"use strict";
const outputGraph=require('../app/web/graph-tools.js');
const outputTest=require('node:test'),outputAssert=require('node:assert/strict');
outputTest('Deleted final Output is repairable without a valid contract',()=>{
  const doc={graph:{nodes:[{id:'x',type:'Input',x:0,y:0,features:2},{id:'fc',type:'Linear',x:224,y:0,features:1}],edges:[{from:'x',to:'fc'}]}};
  const id=outputGraph.addOutput(doc,'fc');
  outputAssert.equal(doc.graph.nodes.at(-1).type,'Output');
  outputAssert.deepEqual(doc.graph.edges.at(-1),{from:'fc',to:id});
});
const test = require("node:test"), assert = require("node:assert/strict");
const G = require("../app/web/graph-tools.js");
const seed = () => ({graph: {nodes: [
  {id: "input", type: "Input", features: 2, x: 90, y: 170},
  {id: "linear", type: "Linear", features: 4, x: 340, y: 170},
  {id: "output", type: "Output", x: 600, y: 170},
], edges: [{from: "input", to: "linear"}, {from: "linear", to: "output"}]}});
const get = (d, id) => d.graph.nodes.find(n => n.id === id);
test('Compact boxes retain distinct ports and expanded preferences through movement',()=>{
  for(const type of ['Linear','Concat','Splitter','Replicator']){
    const n={id:'n',type,x:48,y:72,width:300,height:220,sources:['a','b','c'],
      parts:[{output:'a',features:1},{output:'b',features:1},{output:'c',features:1}],outputs:['a','b','c']};
    n.collapsed=true;assert.equal(G.width(n),72);
    for(const output of [false,true]){
      const points=G.slots(n,output).map(peer=>G.portPoint(n,output,peer));
      assert.equal(new Set(points.map(p=>p.y)).size,points.length);
      assert.ok(points.every(p=>p.x===n.x+(output?72:0) && p.y>n.y && p.y<n.y+G.height(n)));
      const before=points.map(p=>({...p}));n.x+=24;n.y-=48;
      assert.deepEqual(G.slots(n,output).map(peer=>G.portPoint(n,output,peer)),before.map(p=>({x:p.x+24,y:p.y-48})));
    }
    n.collapsed=false;assert.equal(G.width(n),300);assert.equal(G.height(n),220);
  }
});
test('Destination rewiring preserves source, splitter port, siblings and stored geometry',()=>{
  const d={graph:{nodes:[{id:'r',type:'Splitter',parts:[{output:'a',features:1},{output:'b',features:1}]},
    {id:'old',type:'Linear',x:50,y:70},{id:'new',type:'Linear'},{id:'other',type:'Output'}],
    edges:[{from:'r',from_port:'a',to:'old',annotation:'Keep me'},{from:'r',from_port:'b',to:'other'}]}};
  assert.equal(G.retargetWire(d,d.graph.edges[0],'new'),true);
  assert.deepEqual(d.graph.edges,[{from:'r',to:'new',from_port:'a',annotation:'Keep me'},{from:'r',from_port:'b',to:'other'}]);
  assert.equal(get(d,'old').x,50);
});
test('Failed and unchanged wire drops are atomic; occupied sockets and cycles do not overwrite work',()=>{
  const d=seed(),before=JSON.stringify(d),edge=d.graph.edges[0];
  assert.equal(G.retargetWire(d,edge,'linear'),false);
  for(const destination of ['output','input','missing'])assert.throws(()=>G.retargetWire(d,edge,destination));
  assert.equal(JSON.stringify(d),before);
  d.graph.nodes.push({id:'a',type:'Linear'},{id:'b',type:'Linear'});
  d.graph.edges.push({from:'a',to:'b'});
  assert.throws(()=>G.retargetWire(d,{from:'a',to:'b'},'a'),/valid input/);
  d.graph.nodes.push({id:'free',type:'Linear'});d.graph.edges.push({from:'free',to:'a'});
  const withCycle=JSON.stringify(d);
  assert.throws(()=>G.retargetWire(d,{from:'a',to:'b'},'free'),/cycle/);
  assert.equal(JSON.stringify(d),withCycle);
});
test('Moving out of and into Concat removes only the old feature and appends in explicit order',()=>{
  const d={graph:{nodes:[{id:'r',type:'Replicator',outputs:['a','b']},{id:'x',type:'Input'},
    {id:'old',type:'Concat',sources:['r@a','x','r@b']},{id:'new',type:'Concat',sources:['x']}],
    edges:[{from:'r',from_port:'a',to:'old'},{from:'x',to:'old'},{from:'r',from_port:'b',to:'old'},{from:'x',to:'new'}]}};
  const edge=d.graph.edges[0],before=JSON.stringify(d);
  assert.throws(()=>G.retargetWire(d,edge,'new','x'),/occupied/);
  assert.equal(JSON.stringify(d),before);
  G.retargetWire(d,edge,'new');
  assert.deepEqual(get(d,'old').sources,['x','r@b']);
  assert.deepEqual(get(d,'new').sources,['x','r@a']);
  G.removeWire(d,{from:'r',from_port:'b',to:'old'});
  assert.deepEqual(get(d,'old').sources,['x']);
  assert.ok(d.graph.edges.some(e=>e.from==='r' && e.from_port==='a' && e.to==='new'));
});
test('Wire deletion normalizes legacy Concat keys and refuses stale edge identities',()=>{
  const d={graph:{nodes:[{id:'r',type:'Replicator',outputs:['part']},{id:'join',type:'Concat',sources:['r','x']}],
    edges:[{from:'r',from_port:'part',to:'join'},{from:'x',to:'join'}]}};
  G.removeWire(d,d.graph.edges[0]);assert.deepEqual(get(d,'join').sources,['x']);
  const before=JSON.stringify(d);assert.throws(()=>G.removeWire(d,{from:'r',to:'join'}),/no longer exists/);
  assert.equal(JSON.stringify(d),before);
});
test('Retargeting to a Concat does not silently merge duplicate fanout or feature keys',()=>{
  const d={graph:{nodes:[{id:'x',type:'Input'},{id:'old',type:'Output'},
    {id:'join',type:'Concat',sources:['x']}],edges:[{from:'x',to:'old'},{from:'x',to:'join'}]}};
  const before=JSON.stringify(d);
  assert.throws(()=>G.retargetWire(d,d.graph.edges[0],'join'),/already connected/);
  assert.equal(JSON.stringify(d),before);
});
test('Removing a branched unary block preserves fanout and ordered join identity',()=>{
  const d={graph:{nodes:[{id:'r',type:'Replicator',outputs:['a','b']},{id:'fc',type:'Linear'},
    {id:'join',type:'Concat',sources:['fc','other']},{id:'out',type:'Output'}],
    edges:[{from:'r',from_port:'a',to:'fc'},{from:'fc',to:'join'},{from:'fc',to:'out'}]}};
  G.removeBlock(d,'fc');
  assert.ok(d.graph.edges.some(e=>e.from==='r' && e.from_port==='a' && e.to==='join'));
  assert.ok(d.graph.edges.some(e=>e.from==='r' && e.from_port==='a' && e.to==='out'));
  assert.deepEqual(get(d,'join').sources,['r@a','other']);
});
test('Removing a splitter never deletes its downstream learned layers or guesses slice semantics',()=>{
  const d={graph:{nodes:[{id:'x',type:'Input'},{id:'split',type:'Splitter',parts:[{output:'left',features:1},{output:'right',features:1}]},
    {id:'left',type:'Linear'},{id:'right',type:'Linear'}],edges:[{from:'x',to:'split'},{from:'split',from_port:'left',to:'left'},{from:'split',from_port:'right',to:'right'}]}};
  G.removeBlock(d,'split');assert.deepEqual(d.graph.nodes.map(n=>n.id),['x','left','right']);assert.equal(d.graph.edges.length,0);
  const joined={graph:{nodes:[{id:'x',type:'Input'},{id:'r',type:'Replicator',outputs:['a','b']},{id:'c',type:'Concat',sources:['r@a','r@b']}],
    edges:[{from:'x',to:'r'},{from:'r',from_port:'a',to:'c'},{from:'r',from_port:'b',to:'c'}]}};
  G.removeBlock(joined,'r');assert.deepEqual(get(joined,'c').sources,[]);
  assert.equal(joined.graph.edges.length,0); // Do not silently deduplicate repeated feature parts.
});
const kind = (d, type) => d.graph.nodes.find(n => n.type === type);
const edge = (d, from, to) => d.graph.edges.some(e => e.from === from && e.to === to);
const uniqueNames = (d, type) => {
  const names = d.graph.nodes.filter(n => n.type === type).map(n => n.label || n.id);
  assert.equal(new Set(names).size, names.length);
};

test("Ctrl snapping rounds world coordinates; free drag preserves them", () => {
  assert.deepEqual(G.snapPosition(35, -38, true), {x: 24, y: -48});
  assert.deepEqual(G.snapPosition(35.5, -38.5, false), {x: 35.5, y: -38.5});
  const view = {x: 111, y: -37, zoom: .5};
  const world = {x: (view.x + 35 * view.zoom - view.x) / view.zoom,
                 y: (view.y - 38 * view.zoom - view.y) / view.zoom};
  assert.deepEqual(G.snapPosition(world.x, world.y, true), {x: 24, y: -48});
});
test("Adding input constructs an ordered join and preserves the chain", () => {
  const d = seed(), id = G.addInput(d), join = kind(d, "Concat");
  assert.deepEqual(join.sources, ["input", id]);
  assert.ok(edge(d, "input", join.id) && edge(d, id, join.id) && edge(d, join.id, "linear"));
  assert.equal(get(d, "linear").x, 564);
  const second = G.addInput(d);
  assert.deepEqual(join.sources, ["input", id, second]);
  assert.equal(d.graph.nodes.filter(n => n.type === "Concat").length, 1);
});
test("Large named inputs remain drawable; safety is a backend pre-run concern",()=>{
  const d=seed();get(d,"input").features=1024;
  const part=G.addInput(d);
  assert.equal(get(d,"input").features,1024);
  assert.equal(get(d,part).features,1);
  const router=get(d,G.addRouter(d,"Splitter",512));
  assert.deepEqual(router.parts.map(p=>p.features),[256,256]);
});
test("Removing an input collapses a one-part join without losing wires", () => {
  const d = seed(), id = G.addInput(d);
  assert.equal(G.removeBoundary(d, id), "input");
  assert.equal(kind(d, "Concat"), undefined);
  assert.ok(edge(d, "input", "linear"));
  assert.throws(() => G.removeBoundary(d, "input"), /at least one Input/);
});
test("Delete then add uses unique names for input ports", () => {
  const d = seed(), second = G.addInput(d);
  G.addInput(d); G.removeBoundary(d, second); G.addInput(d);
  uniqueNames(d, "Input");
});
test("Split and replica routers own distinct ordered sockets", () => {
  for (const type of ["Splitter", "Replicator"]) {
    const d = seed(), id = G.addRouter(d, type, 4), router = get(d, id);
    assert.ok(edge(d, "linear", id));
    assert.deepEqual(G.slots(router, false), [null]);
    assert.equal(G.slots(router, true).length, 2);
    if (type === "Splitter") assert.deepEqual(router.parts.map(p => p.features), [2, 2]);
    const ports = G.slots(router, true).map(peer => G.portPoint(router, true, peer));
    assert.notEqual(ports[0].y, ports[1].y);
    assert.throws(() => G.addRouter(d, type, 4), /Select the box/);
  }
});
test("Split rejects scalar width and preserves the original graph", () => {
  const d = seed(), before = structuredClone(d);
  assert.throws(() => G.addRouter(d, "Splitter", 1), /at least two/);
  assert.deepEqual(d, before);
});
test("Existing output labels are preserved and new names avoid collisions", () => {
  const d = seed(); get(d, "output").label = "Output 2";
  G.addRouter(d, "Replicator", 4);
  uniqueNames(d, "Output");
  assert.equal(get(d, "output").label, "Output 2");
});
test("Output additions, deletion and collapse preserve the selected survivor", () => {
  for (const type of ["Splitter", "Replicator"]) {
    const d = seed(), router = get(d, G.addRouter(d, type, 4));
    const second = G.slots(router, true)[1];
    G.addRouterPort(d, router); G.removeBoundary(d, second); G.addRouterPort(d, router);
    uniqueNames(d, "Output");
    for (const peer of [...G.slots(router, true)].filter(id => id !== "output")) G.removeBoundary(d, peer);
    assert.equal(kind(d, type), undefined);
    assert.ok(edge(d, "linear", "output"));
  }
});
test("Removing a router restores the single output boundary", () => {
  const d = seed(), id = G.addRouter(d, "Splitter", 4);
  G.removeBoundary(d, id);
  assert.equal(d.graph.nodes.filter(n => n.type === "Output").length, 1);
  assert.ok(edge(d, "linear", "output"));
});
test("Reconnect named sockets preserves other join and router wires", () => {
  const d = seed(), part = G.addInput(d), join = kind(d, "Concat");
  G.connect(d, part, join.id, null, part);
  assert.ok(edge(d, "input", join.id));
  assert.equal(d.graph.edges.filter(e => e.from === part).length, 1);
  const router = get(d, G.addRouter(d, "Replicator", 4)), peers = G.slots(router, true);
  G.connect(d, router.id, peers[0], peers[0]);
  assert.ok(edge(d, router.id, peers[1]));
  G.connect(d, router.id, peers[1], peers[0]);
  assert.equal(d.graph.edges.find(e=>e.to===peers[1]).from_port,peers[0]);
  assert.throws(() => G.connect(d, "output", "input"), /valid source/);
});
test("Input paths are not capped at eight; graph size and router port helpers remain bounded", () => {
  const d = seed();
  for (let i = 1; i < 8; i++) G.addInput(d);
  const join = kind(d, "Concat");
  assert.equal(G.slots(join, false).length, 9);
  assert.equal(G.height(join), G.metrics().socketTop+10+9*G.metrics().rowGap);
  G.addInput(d);assert.equal(join.sources.length,9);
  const router = get(d, G.addRouter(d, "Replicator", 4));
  for (let i = 2; i < 8; i++) G.addRouterPort(d, router);
  assert.equal(G.height(router), G.metrics().socketTop+10+8*G.metrics().rowGap);
  assert.throws(() => G.addRouterPort(d, router), /eight/);
});
