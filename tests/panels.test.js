"use strict";
const test=require("node:test"),assert=require("node:assert/strict");
const P=require("../app/web/panels.js");
test('Panel shortcuts toggle I/A/B/E but do not steal text or modified keys',()=>{
  for(const [key,id] of Object.entries({i:'inspector-panel',A:'architectures-panel',b:'library-panel',E:'experiment-dock'}))
    assert.equal(P.shortcutPanel({key,target:{closest:()=>null}}),id);
  for(const event of [{key:'i',ctrlKey:true},{key:'a',repeat:true},{key:'e',target:{closest:()=>true}}])
    assert.equal(P.shortcutPanel(event),null);
});

test("Panels may leave bounds while a 24-pixel draggable frame remains visible",()=>{
  assert.deepEqual(P.constrain({x:-200,y:900,width:240,height:600},{width:1000,height:700}),
    {x:-200,y:676,width:240,height:600});
  assert.deepEqual(P.constrain({x:50,y:50,width:1200,height:900},{width:1000,height:700}),
    {x:50,y:50,width:1200,height:900});
  assert.deepEqual(P.constrain({x:-999,y:-999,width:240,height:600},{width:1000,height:700}),
    {x:-216,y:-576,width:240,height:600});
});
test("Overlay scroll geometry has a minimum thumb and no gutter assumptions",()=>{
  const g=P.scrollbar(200,1000,180,400);
  assert.equal(g.max,800);assert.equal(g.length,36);assert.equal(g.top,72);
  assert.equal(P.scrollbar(200,100000,180,0).length,26);
  assert.deepEqual(P.scrollbar(200,100,180,50),{max:0,length:180,travel:0,top:0});
});
test("Fit uses largest free rectangle instead of centering under palettes",()=>{
  const r=P.availableRect({width:1000,height:700},[
    {x:0,y:0,width:200,height:700},{x:800,y:0,width:200,height:700},
    {x:200,y:500,width:600,height:200}]);
  assert.deepEqual(r,{x:200,y:0,width:600,height:500});
  assert.deepEqual(P.availableRect({width:1000,height:700},[]),{x:0,y:0,width:1000,height:700});
  assert.equal(P.availableRect({width:1000,height:700},[{x:0,y:0,width:1000,height:700}]),null);
});
test("Fit still finds free space with a detached rectangular palette",()=>{
  const r=P.availableRect({width:1000,height:700},[{x:300,y:100,width:300,height:200}]);
  assert.ok(r.width*r.height>=400000);
  assert.ok(!(r.x<600 && r.x+r.width>300 && r.y<300 && r.y+r.height>100));
});

// Deliberately tiny DOM harness: behavior tests, not claimed browser layout validation.
class Element {
  constructor(id="",rect={x:0,y:0,width:240,height:400}) {
    this.id=id;this.rect=rect;this.hidden=false;this.dataset={};this.attrs={};this.listeners={};
    this.children=[];this.selectors={};this.focused=false;this.scrollTop=0;
    this.clientHeight=200;this.scrollHeight=800;this.offsetHeight=40;
    const set=new Set();this.classList={add:(...v)=>v.forEach(s=>set.add(s)),
      remove:(...v)=>v.forEach(s=>set.delete(s)),contains:v=>set.has(v),
      toggle:(v,force)=>{const active=force===undefined?!set.has(v):force;active?set.add(v):set.delete(v);return active;}};
    this.style={};
    Object.defineProperty(this.style,"cssText",{get:()=>JSON.stringify(this.style),
      set:text=>{for (const k of Object.keys(this.style)) delete this.style[k];Object.assign(this.style,JSON.parse(text));}});
  }
  setAttribute(key,value){this.attrs[key]=value;}
  removeAttribute(key){if(key==="style") for(const k of Object.keys(this.style))delete this.style[k];else delete this.attrs[key];}
  querySelector(s){return this.selectors[s];}
  querySelectorAll(s){return this.selectors[s] || [];}
  append(...children){this.children.push(...children);}
  addEventListener(name,fn){(this.listeners[name] ||= []).push(fn);}
  dispatchEvent(event){for(const fn of this.listeners[event.type]||[])fn(event);}
  setPointerCapture(){}
  focus(){this.focused=true;}
  closest(s){return s.includes("button") && this.isButton ? this : null;}
  getBoundingClientRect(){
    if(this.hidden)return {left:0,top:0,width:0,height:0};
    return {left:parseFloat(this.style.left)||this.rect.x,top:parseFloat(this.style.top)||this.rect.y,
      width:parseFloat(this.style.width)||this.rect.width,height:parseFloat(this.style.height)||this.rect.height};
  }
}
function fixture(snapDefault=()=>false){
  global.ResizeObserver=class{constructor(fn){this.fn=fn;}observe(){}};
  global.MutationObserver=class{constructor(fn){this.fn=fn;}observe(){}};
  const editor=new Element("editor",{x:0,y:0,width:1000,height:700});
  editor.clientWidth=1000;
  const panels=["library-panel","inspector-panel","experiment-dock","architectures-panel","layout-panel"].map(id=>{
    const p=new Element(id);p.hidden=true;
    for(const selector of ["[data-fold-panel]","[data-expand-panel]","[data-drag-handle]",".scrollport"]){
      const e=new Element();e.isButton=selector.includes("panel]");p.selectors[selector]=e;
    }
    return p;
  });
  const pack=new Element("menu-pack",{x:0,y:0,width:126,height:160}),toggle=new Element("pack-toggle");
  toggle.isButton=true;
  const menus=new Element("pack-menus"),heading=new Element();
  const openers=panels.map(p=>{const b=new Element();b.dataset.openPanel=p.id;b.isButton=true;return b;});
  editor.selectors[".floating-panel"]=panels;pack.selectors["[data-open-panel]"]=openers;
  pack.selectors[".pack-heading"]=heading;
  const all=[editor,pack,toggle,menus,...panels],doc={
    getElementById:id=>all.find(e=>e.id===id),createElement:()=>new Element(),
    documentElement:new Element(),getSelection:()=>({removeAllRanges:()=>doc.selectionCleared=true}),
  };
  return {api:P.initialize(doc,snapDefault),editor,panels,pack,toggle,menus,heading,openers,doc};
}
test("All menus initially closed; pack buttons open/hide independent panels",()=>{
  const f=fixture();assert.ok(f.panels.every(p=>p.hidden));assert.equal(f.menus.hidden,false);
  f.openers[0].onclick();f.openers[1].onclick();
  assert.ok(!f.panels[0].hidden && !f.panels[1].hidden && f.panels[2].hidden);
  assert.equal(f.openers[0].attrs["aria-expanded"],"true");
  f.openers[0].onclick();assert.ok(f.panels[0].hidden);assert.equal(f.openers[0].attrs["aria-expanded"],"false");
});
test("Hide returns focus to the launcher and preserves palette content",()=>{
  const f=fixture(),p=f.panels[0];f.api.open(p.id);
  p.selectors[".scrollport"].draft="unsaved edit";
  p.selectors["[data-fold-panel]"].onclick();
  assert.ok(p.hidden && f.openers[0].focused);
  f.api.open(p.id);assert.equal(p.selectors[".scrollport"].draft,"unsaved edit");
});
test("Shrinking the pack hides every menu; double-click restores buttons only",()=>{
  const f=fixture();f.panels.forEach(p=>f.api.open(p.id));f.toggle.onclick();
  assert.ok(f.pack.classList.contains("collapsed") && f.menus.hidden);
  assert.ok(f.panels.every(p=>p.hidden));f.toggle.onclick();assert.ok(f.menus.hidden);
  f.toggle.ondblclick();assert.equal(f.menus.hidden,false);assert.ok(f.panels.every(p=>p.hidden));
  assert.equal(f.toggle.attrs["aria-expanded"],"true");
});
test("Collapsed icon has keyboard restoration and clears dragged compact dimensions",()=>{
  const f=fixture();f.toggle.onclick();f.pack.style.width="42px";f.pack.style.height="42px";
  let prevented=false;f.toggle.onkeydown({key:"Enter",preventDefault:()=>prevented=true});
  assert.ok(prevented && !f.menus.hidden);assert.equal(f.pack.style.width,"");
});
test("Moving the pack keeps launcher width/height content-driven as fonts change",()=>{
  const f=fixture();
  const target=new Element(),event=(x,y)=>({button:0,shiftKey:false,target,pointerId:1,
    clientX:x,clientY:y,ctrlKey:false,preventDefault:()=>{}});
  f.heading.onpointerdown(event(20,60));f.heading.onpointermove(event(100,140));
  assert.equal(f.pack.style.width,"");assert.equal(f.pack.style.height,"");
  f.pack.rect.width=260;f.pack.rect.height=320;
  assert.equal(f.pack.getBoundingClientRect().width,260);
  assert.equal(f.pack.getBoundingClientRect().height,320);
});
test("Expand/restore and closing expanded menus do not leave stale geometry",()=>{
  const f=fixture(),p=f.panels[0];f.api.open(p.id);
  const button=p.selectors["[data-expand-panel]"];
  p.style.left="150px";p.style.top="80px";p.style.width="280px";p.style.height="300px";
  button.onclick();assert.ok(p.classList.contains("expanded"));assert.equal(p.style.width,"968px");
  button.onclick();assert.equal(p.style.width,"280px");assert.equal(p.style.left,"150px");
  button.onclick();p.selectors["[data-fold-panel]"].onclick();
  assert.ok(p.hidden && !p.classList.contains("expanded"));assert.equal(p.style.width,"280px");
});
test("Reset recovers the pack and closes palettes without discarding their contents",()=>{
  const f=fixture();f.api.open(f.panels[0].id);f.panels[0].style.left="600px";
  f.pack.style.left="900px";f.toggle.onclick();f.api.reset();
  assert.equal(f.menus.hidden,false);assert.equal(f.pack.style.left,undefined);
  assert.ok(f.panels.every(p=>p.hidden));assert.equal(f.panels[0].style.left,undefined);
});
test("Shift-drag pans the menu layer, including closed menus, not the drawing",()=>{
  const f=fixture(),target=new Element();
  let stopped=0,prevented=0;
  const event=(type,x,y,shift=true)=>({type,shiftKey:shift,button:0,target,
    pointerId:1,clientX:x,clientY:y,preventDefault:()=>prevented++,stopPropagation:()=>stopped++});
  f.editor.dispatchEvent(event("pointerdown",100,100,false));
  f.editor.dispatchEvent(event("pointermove",200,200,false));
  assert.equal(f.pack.style.left,undefined);
  f.editor.dispatchEvent(event("pointerdown",100,100));
  f.editor.dispatchEvent(event("pointermove",150,140));
  assert.equal(f.pack.style.left,"50px");assert.equal(f.pack.style.top,"40px");
  assert.ok(f.panels.every(p=>p.hidden && p.style.left==="50px"));
  assert.ok(stopped && prevented);
  f.editor.dispatchEvent(event("pointerup",150,140));
  f.editor.dispatchEvent(event("pointermove",300,300));
  assert.equal(f.pack.style.left,"50px");
});

const pointer=(type,target,x,y,extra={})=>({type,target,button:0,pointerId:1,clientX:x,clientY:y,
  preventDefault(){},stopPropagation(){},...extra});
test('The launcher and palettes take turns at the front of one shared menu layer',()=>{
  const f=fixture(),p=f.panels[0],q=f.panels[1];
  const level=element=>Number(element.style.zIndex);
  f.api.open(p.id);
  f.pack.dispatchEvent(pointer('pointerdown',f.pack,0,0));
  assert.ok(level(f.pack)>level(p));
  p.dispatchEvent(pointer('pointerdown',p.selectors['[data-fold-panel]'],0,0));
  assert.ok(level(p)>level(f.pack));
  f.api.open(q.id);assert.ok(level(q)>level(p));
  f.pack.dispatchEvent(pointer('pointerdown',f.toggle,0,0));
  assert.ok(level(f.pack)>level(q));
  q.dispatchEvent(pointer('pointerdown',q.selectors['.scrollport'],0,0,{button:2}));
  assert.ok(level(q)>level(f.pack));
});
test('Expanded palettes share stacking and restoring geometry preserves the current level',()=>{
  const f=fixture(),p=f.panels[0],q=f.panels[1],expand=p.selectors['[data-expand-panel]'];
  f.api.open(p.id);p.style.left='50px';p.style.width='330px';expand.onclick();
  f.pack.dispatchEvent(pointer('pointerdown',f.pack,0,0));
  assert.ok(Number(f.pack.style.zIndex)>Number(p.style.zIndex));
  f.api.open(q.id);p.dispatchEvent(pointer('pointerdown',p,0,0));
  const current=p.style.zIndex;expand.onclick();
  assert.equal(p.style.zIndex,current);assert.equal(p.style.width,'330px');
  assert.equal(p.style.left,'50px');assert.ok(Number(current)>Number(q.style.zIndex));
});
test('Pack collapse and keyboard restoration raise it without permanently privileging it',()=>{
  const f=fixture(),p=f.panels[0];f.api.open(p.id);f.api.hideAll();
  assert.ok(Number(f.pack.style.zIndex)>Number(p.style.zIndex));
  f.api.open(p.id);
  f.toggle.onkeydown({key:'Enter',preventDefault(){}});
  assert.ok(Number(f.pack.style.zIndex)>Number(p.style.zIndex));
  p.dispatchEvent(pointer('pointerdown',p,0,0));
  assert.ok(Number(p.style.zIndex)>Number(f.pack.style.zIndex));
});
test('Restoring presets keeps palette order but leaves the launcher free to take focus',()=>{
  const f=fixture();f.api.open('library-panel');f.api.open('inspector-panel');
  const preset=f.api.capture();f.api.apply(preset);
  assert.deepEqual(f.api.capture().order,preset.order);
  f.pack.dispatchEvent(pointer('pointerdown',f.pack,0,0));
  assert.ok(f.panels.every(p=>Number(f.pack.style.zIndex)>Number(p.style.zIndex)));
  assert.deepEqual(f.api.capture().order,preset.order);
  f.api.open('library-panel');
  assert.ok(Number(f.panels[0].style.zIndex)>Number(f.pack.style.zIndex));
});
test('Ctrl inverts either grid default; Shift does not change the snap decision',()=>{
  for(const enabled of [false,true])for(const ctrlKey of [false,true])for(const shiftKey of [false,true])
    assert.equal(P.snapEnabled({ctrlKey,shiftKey},enabled),enabled!==ctrlKey);
});
test('Snapped default applies to panel/pack/layer movement and corner resize; Ctrl frees each',()=>{
  let enabled=true;const f=fixture(()=>enabled),p=f.panels[0],heading=p.selectors['[data-drag-handle]'];
  f.api.open(p.id);
  for(const ctrlKey of [false,true]) {
    p.style.left='0px';p.style.top='0px';p.style.width='240px';p.style.height='400px';
    heading.dispatchEvent(pointer('pointerdown',heading,0,0));
    heading.dispatchEvent(pointer('pointermove',heading,50,70,{ctrlKey}));
    heading.dispatchEvent(pointer('pointerup',heading,50,70));
    assert.equal(p.style.left,ctrlKey?'50px':'48px');assert.equal(p.style.top,ctrlKey?'70px':'72px');
    const grip=p.children.find(e=>e.className==='panel-resize');
    grip.dispatchEvent(pointer('pointerdown',grip,0,0));
    grip.dispatchEvent(pointer('pointermove',grip,17,13,{ctrlKey}));
    grip.dispatchEvent(pointer('pointerup',grip,17,13));
    if(ctrlKey){assert.equal(p.style.width,'257px');assert.equal(p.style.height,'413px');}
    else {assert.equal((parseFloat(p.style.left)+parseFloat(p.style.width))%24,0);assert.equal((parseFloat(p.style.top)+parseFloat(p.style.height))%24,0);}
    f.pack.style.left='0px';f.pack.style.top='0px';
    f.heading.onpointerdown(pointer('pointerdown',f.heading,0,0));
    f.heading.onpointermove(pointer('pointermove',f.heading,50,70,{ctrlKey}));
    f.heading.dispatchEvent(pointer('pointerup',f.heading,50,70));
    assert.equal(f.pack.style.left,ctrlKey?'50px':'48px');
    f.pack.style.left='0px';f.pack.style.top='0px';
    f.editor.dispatchEvent(pointer('pointerdown',f.editor,0,0,{shiftKey:true}));
    f.editor.dispatchEvent(pointer('pointermove',f.editor,50,70,{shiftKey:true,ctrlKey}));
    f.editor.dispatchEvent(pointer('pointerup',f.editor,50,70));
    assert.equal(f.pack.style.top,ctrlKey?'70px':'72px');
  }
  enabled=false;
  assert.equal(P.snapEnabled({},enabled),false); // provider follows live Preview/Cancel updates.
});
test("Dragging suppresses selection temporarily and releases the hand cursor on cancel",()=>{
  const f=fixture(),p=f.panels[0],heading=p.selectors['[data-drag-handle]'];
  heading.dispatchEvent(pointer('pointerdown',heading,0,0));
  assert.ok(f.doc.documentElement.classList.contains('loom-dragging'));
  assert.ok(f.doc.selectionCleared);
  heading.dispatchEvent(pointer('pointercancel',heading,0,0));
  assert.ok(!f.doc.documentElement.classList.contains('loom-dragging'));
  const grip=p.children.find(e=>e.className==='panel-resize');
  grip.dispatchEvent(pointer('pointerdown',grip,0,0));
  assert.ok(f.doc.documentElement.classList.contains('loom-resizing'));
  grip.dispatchEvent(pointer('lostpointercapture',grip,0,0));
  assert.ok(!f.doc.documentElement.classList.contains('loom-resizing'));
});
test("Ctrl menu drag snaps its top-left and Ctrl resize snaps bottom-right",()=>{
  const f=fixture(),p=f.panels[0],heading=p.selectors["[data-drag-handle]"];
  f.api.open(p.id);
  heading.dispatchEvent(pointer("pointerdown",heading,0,0));
  heading.dispatchEvent(pointer("pointermove",heading,50,70,{ctrlKey:true}));
  heading.dispatchEvent(pointer("pointerup",heading,50,70));
  assert.equal(p.style.left,"48px");assert.equal(p.style.top,"72px");
  const grip=p.children.find(e=>e.className==="panel-resize");
  grip.dispatchEvent(pointer("pointerdown",grip,0,0));
  grip.dispatchEvent(pointer("pointermove",grip,17,13,{ctrlKey:true}));
  assert.equal((parseFloat(p.style.left)+parseFloat(p.style.width))%24,0);
  assert.equal((parseFloat(p.style.top)+parseFloat(p.style.height))%24,0);
});
test("Empty menu content areas drag but controls and text stay interactive",()=>{
  const f=fixture(),p=f.panels[0],space=p.selectors[".scrollport"];
  f.api.open(p.id);
  p.dispatchEvent(pointer("pointerdown",space,0,0));
  p.dispatchEvent(pointer("pointermove",space,37,53));
  p.dispatchEvent(pointer("pointerup",space,37,53));
  assert.equal(p.style.left,"37px");assert.equal(p.style.top,"53px");
  const button=p.selectors["[data-fold-panel]"];
  p.dispatchEvent(pointer("pointerdown",button,0,0));
  p.dispatchEvent(pointer("pointermove",button,100,100));
  assert.equal(p.style.left,"37px");
  const text=new Element();text.textContent="Selectable explanation";
  p.dispatchEvent(pointer("pointerdown",text,0,0));
  p.dispatchEvent(pointer("pointermove",text,100,100));
  assert.equal(p.style.left,"37px");
});
test("Ctrl+Shift snaps the pack anchor while preserving relative menu positions",()=>{
  const f=fixture(),target=new Element();
  f.editor.dispatchEvent(pointer("pointerdown",target,0,0,{shiftKey:true}));
  f.editor.dispatchEvent(pointer("pointermove",target,17,9,{shiftKey:true,ctrlKey:true}));
  assert.equal(f.pack.style.left,"24px");assert.equal(f.pack.style.top,"0px");
  assert.equal(f.panels[0].style.left,"24px");assert.equal(f.panels[0].style.top,"0px");
});

test('Preset captures hidden panels, CSS rectangles, ordering and pack state with defensive copies',()=>{
  const f=fixture();f.api.open('library-panel');f.api.open('inspector-panel');
  f.panels[0].style.left='-150.25px';f.panels[0].style.width='310.5px';f.pack.style.left='23px';
  const value=f.api.capture();assert.equal(value.panels['library-panel'].x,-150.25);
  assert.equal(value.panels['library-panel'].width,310.5);assert.equal(value.pack.x,23);
  assert.equal(value.panels['architectures-panel'].visible,false);assert.equal(value.order.at(-1),'inspector-panel');
  value.panels['library-panel'].x=999;assert.equal(f.api.capture().panels['library-panel'].x,-150.25);
});
test('Capturing an expanded panel keeps its ordinary rectangle and does not change visible UI',()=>{
  const f=fixture(),p=f.panels[0];f.api.open(p.id);p.style.left='100px';p.style.top='80px';p.style.width='350px';p.style.height='450px';
  p.selectors['[data-expand-panel]'].onclick();const css=p.style.cssText;
  const value=f.api.capture();assert.equal(value.panels[p.id].width,350);assert.equal(value.panels[p.id].x,100);
  assert.ok(value.panels[p.id].expanded&&value.panels[p.id].visible);assert.equal(p.style.cssText,css);
  assert.ok(p.classList.contains('expanded'));
});
test('Applying a preset restores flags and stacking; closing expanded panels restores ordinary geometry',()=>{
  const a=fixture();a.api.open('library-panel');a.panels[0].style.width='390px';
  a.panels[0].selectors['[data-expand-panel]'].onclick();a.api.open('inspector-panel');const value=a.api.capture();
  const b=fixture();b.api.apply(value);assert.ok(b.panels[0].classList.contains('expanded')&&!b.panels[0].hidden);
  assert.equal(b.panels[0].style.width,'968px');assert.ok(Number(b.panels[1].style.zIndex)>Number(b.panels[0].style.zIndex));
  b.panels[0].selectors['[data-fold-panel]'].onclick();assert.ok(b.panels[0].hidden);assert.equal(b.panels[0].style.width,'390px');
  assert.deepEqual(value,a.api.capture());
});
test('Collapsed pack presets restore no visible panels and clear old workspace geometry on null',()=>{
  const a=fixture();a.api.open('library-panel');a.toggle.onclick();const value=a.api.capture();
  const b=fixture();b.api.open('inspector-panel');b.api.apply(value);
  assert.ok(b.pack.classList.contains('collapsed')&&b.panels.every(p=>p.hidden));
  b.api.apply(null);assert.ok(!b.pack.classList.contains('collapsed')&&b.panels.every(p=>p.hidden));
  assert.equal(b.pack.style.left,undefined);assert.equal(b.panels[0].style.left,undefined);
});
test('Preset validation rejects malformed state before touching current controls',()=>{
  const f=fixture();f.api.open('library-panel');const before=f.api.capture();
  for(const mutate of [v=>v.version=2,v=>v.pack.x=NaN,v=>v.pack.x=true,v=>v.panels['library-panel'].width=0,
    v=>v.panels['library-panel'].css='color:red',v=>v.order[0]=v.order[1],v=>v.panels['unknown']=v.panels['library-panel'],
    v=>v.panels['architectures-panel'].expanded=true,v=>v.pack.collapsed=true]) {
    const bad=structuredClone(before);mutate(bad);assert.throws(()=>f.api.apply(bad));assert.deepEqual(f.api.capture(),before);
  }
  const copy=P.validatePreset(before);copy.order.reverse();assert.notDeepEqual(copy.order,before.order);
});
test('Off-stage presets retain a recoverable slice and cannot shrink below readable minima',()=>{
  const f=fixture(),value=f.api.capture();value.pack.x=1000000;value.pack.y=-1000000;
  value.panels['library-panel']={x:-1000000,y:1000000,width:1,height:1,visible:true,expanded:false};
  f.api.apply(value);assert.equal(f.pack.style.left,'976px');assert.equal(f.pack.style.top,'-136px');
  assert.equal(f.panels[0].style.left,'-196px');assert.equal(f.panels[0].style.top,'676px');
  assert.equal(f.panels[0].style.width,'220px');assert.equal(f.panels[0].style.height,'140px');
  assert.equal(value.panels['library-panel'].width,1);
});
test('Applying a workspace preset cancels an active menu gesture without retaining old movement',()=>{
  const f=fixture(),p=f.panels[0],heading=p.selectors['[data-drag-handle]'];
  f.api.open(p.id);const value=f.api.capture();heading.dispatchEvent(pointer('pointerdown',heading,0,0));
  heading.dispatchEvent(pointer('pointermove',heading,72,48));assert.ok(f.doc.documentElement.classList.contains('loom-dragging'));
  f.api.apply(value);heading.dispatchEvent(pointer('pointermove',heading,400,400));
  assert.equal(p.style.left,'0px');assert.ok(!f.doc.documentElement.classList.contains('loom-dragging'));
});
