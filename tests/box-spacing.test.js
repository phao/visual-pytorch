'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),G=require('../app/web/graph-tools.js');
test('Box spacing 1–100 derives monotonic geometry and port anchors without editing the graph',()=>{
  const n={id:'fc',type:'Linear',x:100,y:200},router={id:'r',type:'Replicator',outputs:['a','b','c','d'],x:400,y:100};
  const before=JSON.stringify([n,router]);let previous=0,previousHeight=0;
  for(let level=1;level<=100;level++){
    G.setSpacing(level);const m=G.metrics();assert.ok(m.width>previous);previous=m.width;
    assert.ok(G.height(n)>previousHeight);previousHeight=G.height(n);
    assert.ok(m.headerY<m.titleY && m.titleY<m.shapeY && m.shapeY<m.detailY);
    assert.ok(m.titleY-m.headerY>=15 && m.shapeY-m.titleY>=13 && m.detailY-m.shapeY>=11);
    assert.ok(m.detailY<G.height(n)-m.padding && m.padding>0);
    assert.equal(G.portPoint(n,true).x,n.x+m.width);
    for(const id of router.outputs){const p=G.portPoint(router,true,id);assert.ok(p.y>router.y && p.y<router.y+G.height(router));}
  }
  assert.equal(G.metrics().width,166);assert.equal(G.height(n),80);assert.equal(JSON.stringify([n,router]),before);
  G.setSpacing(35);assert.ok(Math.abs(G.metrics().width-145.2)<1e-9);assert.ok(Math.abs(G.height(n)-64.4)<1e-9);
  for(const value of [0,101,1.5,'1'])assert.throws(()=>G.setSpacing(value));
  G.setSpacing(100);
});
test('Explicit sizes change port anchors while all node types keep readable minima',()=>{
  G.setSpacing(35);
  for(const type of ['Input','Linear','Output','Activation','Module','Reshape','Concat','Splitter','Replicator']){
    const n={type,x:13,y:-7,sources:['a','b','c'],outputs:['a','b','c','d'],parts:[{output:'a'},{output:'b'}]};
    const min=G.minimumSize(n);G.resizeBlock(n,320,180);
    assert.equal(G.width(n),320);assert.equal(G.height(n),180);
    for(const p of G.slots(n,true)){assert.equal(G.portPoint(n,true,p).x,333);assert.ok(G.portPoint(n,true,p).y>n.y && G.portPoint(n,true,p).y<n.y+180);}
    G.resizeBlock(n,1,1);assert.equal(G.width(n),min.width);assert.equal(G.height(n),min.height);
    assert.equal(n.x,13);assert.equal(n.y,-7);
  }
  G.setSpacing(100);
});
test('Ctrl block resize snaps bottom-right and preserves top-left above minimum',()=>{
  const n={type:'Linear',x:13,y:-7};G.resizeBlock(n,231.7,122.8,true);
  assert.equal((n.x+n.width)%24,0);assert.equal((n.y+n.height)%24,0);
  G.resizeBlock(n,1,1,true);assert.ok(n.width>=G.minimumSize(n).width && n.height>=G.minimumSize(n).height);
  assert.equal((n.x+n.width)%24,0);assert.equal((n.y+n.height)%24,0);
  assert.throws(()=>G.resizeBlock(n,NaN,100));
  G.resizeBlock(n,1e8,1e8);assert.equal(n.width,100000);assert.equal(n.height,100000);
});
test('Saved overrides survive spacing changes but growing port counts can raise effective minimum',()=>{
  const n={type:'Replicator',x:0,y:0,outputs:['a','b'],width:400,height:100};
  G.setSpacing(1);assert.equal(G.width(n),400);assert.equal(G.height(n),100);
  G.setSpacing(100);n.outputs=['a','b','c','d','e','f','g','h'];
  assert.ok(G.height(n)>100);assert.equal(n.height,100);assert.equal(G.width(n),400);
});
