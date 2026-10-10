"use strict";
((root, factory) => {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LoomGraph = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const GRID = 24;
  let spacing=100;
  function setSpacing(value){
    if(!Number.isInteger(value) || value<1 || value>100)throw new Error('NN box spacing must be 1–100.');
    spacing=value;
  }
  function metrics(){
    const t=spacing/100,padding=3+7*t,width=134+32*t,footerInset=3;
    const headerY=padding+8,titleY=headerY+15+3*t,shapeY=titleY+13+3*t,detailY=shapeY+11+4*t;
    return {width,padding,headerY,titleY,shapeY,detailY,footerInset,baseHeight:detailY+padding+footerInset,socketTop:titleY+5,rowGap:14+5*t};
  }
  const id = () => "n_" + Math.random().toString(36).slice(2, 12);
  const node = (doc, nid) => doc.graph.nodes.find(n => n.id === nid);
  const routers = n => ["Splitter", "Replicator"].includes(n.type);
  const outputs = n => n.type === "Splitter" ? n.parts.map(p => p.output) : n.outputs;
  function unusedLabel(doc, type, prefix, first = 1) {
    const used = new Set(doc.graph.nodes.filter(n => n.type === type).map(n => (n.label || n.id).trim()));
    let number = first;
    while (used.has(prefix + number)) number++;
    return prefix + number;
  }
  function room(doc, extra) {
    if (doc.graph.nodes.length + extra > 64) throw new Error("This prototype supports at most 64 blocks.");
  }
  function snapPosition(x, y, enabled, grid = GRID) {
    return enabled ? {x: Math.round(x / grid) * grid, y: Math.round(y / grid) * grid} : {x, y};
  }
  function minimumSize(n) {
    const count = n.type === "Concat" ? n.sources.length+1 : routers(n) ? outputs(n).length : 1;
    const m=metrics();return {width:m.width,height:Math.max(m.baseHeight,m.socketTop+10+count*m.rowGap)};
  }
  const width = n => n.collapsed ? 72 : Math.max(minimumSize(n).width,n.width || 0);
  const height = n => n.collapsed ? Math.max(n.type==='Linear'?54:44,16*Math.max(slots(n,true).length,slots(n,false).length)+10) : Math.max(minimumSize(n).height,n.height || 0);
  function resizeBlock(n,w,h,snap=false){
    if(!Number.isFinite(w) || !Number.isFinite(h))throw new Error('Use finite block dimensions.');
    const min=minimumSize(n),corner=snapPosition(n.x+Math.max(min.width,w),n.y+Math.max(min.height,h),snap);
    n.width=Math.min(100000,snap?Math.max(Math.ceil((n.x+min.width)/GRID)*GRID,corner.x)-n.x:Math.max(min.width,w));
    n.height=Math.min(100000,snap?Math.max(Math.ceil((n.y+min.height)/GRID)*GRID,corner.y)-n.y:Math.max(min.height,h));
    return {width:n.width,height:n.height};
  }
  function slots(n, output) {
    if ((output && n.type === "Output") || (!output && n.type === "Input")) return [];
    if (!output && n.type === "Concat") return [...n.sources,null];
    if (output && routers(n)) return outputs(n);
    return [null];
  }
  function portPoint(n, output, peer = null) {
    const ports = slots(n, output), index = Math.max(0, ports.indexOf(peer));
    const m=metrics(),top=n.collapsed?5:m.socketTop,y=ports.length>1 ? top+(index+.5)*(height(n)-top-(n.collapsed?5:8))/ports.length : height(n)/2;
    return {x:n.x+(output?width(n):0),y:n.y+y};
  }
  function addInput(doc, independent = false) {
    const inputs = doc.graph.nodes.filter(n => n.type === "Input");
    if (independent) {
      room(doc, 2);
      const part = {id:id(),type:"Input",features:1,label:unusedLabel(doc,"Input","Input part "),
        x:inputs[0]?.x || 90,y:Math.max(...doc.graph.nodes.map(n=>n.y))+150};
      const output = {id:id(),type:"Output",label:unusedLabel(doc,"Output","Output "),x:part.x+448,y:part.y};
      doc.graph.nodes.push(part,output);doc.graph.edges.push({from:part.id,to:output.id});
      return part.id;
    }
    let join = doc.graph.nodes.find(n => n.type === "Concat");
    room(doc, join ? 1 : 2);
    const first = inputs[0];
    if (!join) {
      const edge = doc.graph.edges.find(e => e.from === first.id);
      if (!edge) throw new Error("Connect the current Input before adding another part.");
      join = {id: id(), type: "Concat", sources: [first.id], x: first.x + 224, y: first.y};
      doc.graph.nodes.filter(n => n.type !== "Input").forEach(n => n.x += 224);
      doc.graph.nodes.push(join);
      doc.graph.edges = doc.graph.edges.filter(e => e !== edge);
      doc.graph.edges.push({from: first.id, to: join.id}, {from: join.id, to: edge.to});
    }
    const part = {id: id(), type: "Input", label: unusedLabel(doc, "Input", "Input part ", inputs.length + 1), features: 1,
      x: first.x, y: Math.max(...inputs.map(n => n.y)) + 150};
    doc.graph.nodes.push(part);
    join.sources.push(part.id);
    doc.graph.edges.push({from: part.id, to: join.id});
    return part.id;
  }
  function addOutput(doc, sourceID=null, port=null) {
    room(doc,1);
    let source=node(doc,sourceID);
    if(source?.type==='Output') {
      const edge=doc.graph.edges.find(e=>e.to===source.id);
      source=node(doc,edge?.from);port=edge?.from_port || port;
    }
    source ||= [...doc.graph.nodes].reverse().find(n=>n.type!=='Output' && !doc.graph.edges.some(e=>e.from===n.id));
    source ||= [...doc.graph.nodes].reverse().find(n=>n.type!=='Output');
    if(!source)throw new Error('Add an Input or computational block first.');
    if(routers(source)) {
      port ||= outputs(source)[0];
      if(!outputs(source).includes(port))throw new Error('Select a declared output port.');
    }
    const output={id:id(),type:'Output',label:unusedLabel(doc,'Output','Output '),x:(source.x||0)+224,y:source.y||0};
    doc.graph.nodes.push(output);
    const edge={from:source.id,to:output.id};if(routers(source))edge.from_port=port;
    doc.graph.edges.push(edge);
    return output.id;
  }
  function addRouter(doc, type, width, afterID = null) {
    if (!["Splitter", "Replicator"].includes(type)) throw new Error("Choose a Splitter or Replicator.");
    if (!afterID && doc.graph.nodes.some(routers)) throw new Error("Select the box whose output should be routed.");
    if (type === "Splitter" && width < 2) throw new Error("Set the final Linear to at least two output features before splitting.");
    room(doc, 2);
    let output = afterID ? node(doc,afterID) : doc.graph.nodes.find(n => n.type === "Output");
    if (output.type === "Output") afterID = doc.graph.edges.find(e=>e.to===output.id)?.from;
    const edge = afterID ? doc.graph.edges.find(e=>e.from===afterID) : doc.graph.edges.find(e => e.to === output.id);
    if (!edge) throw new Error("Connect the Output before adding a router.");
    output = node(doc,edge.to);
    const source = node(doc,edge.from);
    const router = {id: id(), type, x: source.x+224, y: source.y};
    if(output.type==="Output")output.label ||= "Output 1";
    const second = {id: id(), type: "Output", label: unusedLabel(doc, "Output", "Output ", 2), x: router.x + 224, y: router.y + 150};
    doc.graph.nodes.filter(n=>n.x>=router.x && n.id!==source.id).forEach(n=>n.x+=224);
    if (type === "Splitter") router.parts = [
      {output: output.id, features: Math.floor(width / 2)},
      {output: second.id, features: width - Math.floor(width / 2)},
    ];
    else router.outputs = [output.id, second.id];
    doc.graph.nodes.push(router, second);
    doc.graph.edges = doc.graph.edges.filter(e => e !== edge);
    doc.graph.edges.push({...edge,to:router.id}, {from: router.id, to: output.id,from_port:output.id}, {from: router.id, to: second.id,from_port:second.id});
    if(output.type==="Concat")output.sources=output.sources.map(src=>src===(edge.from_port?edge.from+"@"+edge.from_port:edge.from)?router.id+"@"+output.id:src);
    return router.id;
  }
  function addRouterPort(doc, router) {
    if (outputs(router).length >= 8) throw new Error("Use up to eight output ports.");
    room(doc, 1);
    const peers = doc.graph.edges.filter(e=>e.from===router.id).map(e=>node(doc,e.to));
    const part = {id: id(), type: "Output", label: unusedLabel(doc, "Output", "Output ", peers.length + 1),
      x: peers[0]?.x ?? router.x+224, y: Math.max(router.y,...peers.map(n => n.y)) + 150};
    doc.graph.nodes.push(part);
    if (router.type === "Splitter") router.parts.push({output: part.id, features: 1});
    else router.outputs.push(part.id);
    doc.graph.edges.push({from: router.id, to: part.id,from_port:part.id});
    return part.id;
  }
  function removeRouter(doc, router, keep = outputs(router)[0]) {
    if(doc.graph.edges.some(e=>e.from===router.id && node(doc,e.to)?.type!=="Output"))
      throw new Error("Remove or rewire the downstream branches before removing this router.");
    const edge = doc.graph.edges.find(e => e.to === router.id);
    const removed = new Set([router.id, ...outputs(router).filter(nid => nid !== keep)]);
    doc.graph.nodes = doc.graph.nodes.filter(n => !removed.has(n.id));
    doc.graph.edges = doc.graph.edges.filter(e => !removed.has(e.from) && !removed.has(e.to));
    if (edge) doc.graph.edges.push({from: edge.from, to: keep});
    return keep;
  }
  function removeBoundary(doc, nid) {
    const n = node(doc, nid);
    if (routers(n)) return removeRouter(doc, n);
    if (n.type === "Input") {
      const join = doc.graph.nodes.find(n => n.type === "Concat");
      if (!join) throw new Error("Keep at least one Input.");
      join.sources = join.sources.filter(value => value !== nid);
      doc.graph.nodes = doc.graph.nodes.filter(n => n.id !== nid);
      doc.graph.edges = doc.graph.edges.filter(e => e.from !== nid);
      if (join.sources.length === 1) {
        const next = doc.graph.edges.find(e => e.from === join.id);
        doc.graph.nodes = doc.graph.nodes.filter(n => n.id !== join.id);
        doc.graph.edges = doc.graph.edges.filter(e => e.to !== join.id && e.from !== join.id);
        if (next) doc.graph.edges.push({from: join.sources[0], to: next.to});
      }
      return join.sources[0];
    }
    const router = doc.graph.nodes.find(n => routers(n) && outputs(n).includes(nid));
    if (!router) throw new Error("Keep at least one Output.");
    const remaining = outputs(router).filter(value => value !== nid);
    if (remaining.length === 1) return removeRouter(doc, router, remaining[0]);
    if (router.type === "Splitter") router.parts = router.parts.filter(p => p.output !== nid);
    else router.outputs = remaining;
    doc.graph.nodes = doc.graph.nodes.filter(n => n.id !== nid);
    doc.graph.edges = doc.graph.edges.filter(e => e.to !== nid);
    return router.id;
  }
  function connect(doc, from, to, fromPeer = null, toPeer = null) {
    const a = node(doc, from), b = node(doc, to);
    if (from === to || a.type === "Output" || b.type === "Input") throw new Error("Choose a valid source and destination.");
    if(routers(a) && (!fromPeer || !outputs(a).includes(fromPeer)))throw new Error("Choose a declared Splitter or Replicator output socket.");
    if(b.type==="Concat") {
      // Normalize legacy router source IDs to explicit source-port keys before edits.
      const oldEdges=doc.graph.edges.filter(e=>e.to===to);
      b.sources=b.sources.map(src=>{
        const old=oldEdges.find(e=>e.from===src && e.from_port);
        return old?old.from+"@"+old.from_port:src;
      });
    }
    doc.graph.edges = doc.graph.edges.filter(e =>
      !(e.to === to && b.type !== "Concat") && !(e.from === from && e.to === to && (e.from_port || null)===(fromPeer || null)));
    const edge={from,to};
    if(routers(a)) {
      edge.from_port=fromPeer;
    }
    doc.graph.edges.push(edge);
    if(b.type==="Concat") {
      const key=fromPeer ? from+"@"+fromPeer : from;
      if(!b.sources.includes(key))b.sources.push(key);
    }
  }
  const sameWire = (a,b) => !!a && !!b && a.from===b.from && a.to===b.to && (a.from_port || null)===(b.from_port || null);
  const sourceKey = e => e.from+(e.from_port?'@'+e.from_port:'');
  function removeWire(doc, edge) {
    const found=doc.graph.edges.find(e=>sameWire(e,edge));
    if(!found)throw new Error('Connection no longer exists.');
    const dest=node(doc,found.to);
    if(dest?.type==='Concat') {
      const incoming=doc.graph.edges.filter(e=>e.to===dest.id);
      dest.sources=dest.sources.map(key=>{
        const legacy=incoming.find(e=>e.from===key && e.from_port);
        return legacy?sourceKey(legacy):key;
      }).filter(key=>key!==sourceKey(found));
    }
    doc.graph.edges=doc.graph.edges.filter(e=>!sameWire(e,found));
  }
  function retargetWire(doc, edge, to, toPeer=null) {
    // Validate on a graph-only copy: a failed drop cannot partially disconnect
    // the original, reorder Concat features, or replace an unrelated wire.
    const draft={graph:JSON.parse(JSON.stringify(doc.graph))};
    const index=draft.graph.edges.findIndex(e=>sameWire(e,edge));
    const found=draft.graph.edges[index],dest=node(draft,to);
    if(!found || !node(draft,found.from) || !dest)throw new Error('Connection or destination no longer exists.');
    if(found.to===to)return false;
    if(dest.type==='Input' || to===found.from)throw new Error('Choose a valid input socket.');
    if(draft.graph.edges.some(e=>sameWire(e,{...found,to})))
      throw new Error('That source is already connected to this destination.');
    const occupied=draft.graph.edges.some(e=>e.to===to);
    if((dest.type!=='Concat' && occupied) || (dest.type==='Concat' && toPeer!==null))
      throw new Error('That input is occupied. Choose an empty socket (the last socket on Concat).');
    const seen=new Set(),todo=[to];
    while(todo.length){
      const current=todo.pop();if(current===found.from)throw new Error('This connection would create a cycle.');
      if(seen.has(current))continue;seen.add(current);
      draft.graph.edges.filter(e=>e.from===current && !sameWire(e,found)).forEach(e=>todo.push(e.to));
    }
    removeWire(draft,found);
    connect(draft,found.from,to,found.from_port || null,toPeer);
    // Retain extension metadata and edge storage order; only `to` changes.
    const replacement=draft.graph.edges.pop();
    Object.assign(replacement,found,{to});
    draft.graph.edges.splice(index,0,replacement);
    doc.graph=draft.graph;
    return true;
  }
  function outputValues(row, group, shape, mode) {
    if(mode!=="split" || !group.shape || !shape || shape.length===1)return row.slice(group.start,group.stop);
    const width=shape[shape.length-1],result=[];
    for(let start=0;start<row.length;start+=width)result.push(...row.slice(start+group.axis_start,start+group.axis_stop));
    return result;
  }
  function removeBlock(doc,nid) {
    const removed=node(doc,nid),incoming=doc.graph.edges.filter(e=>e.to===nid),outgoing=doc.graph.edges.filter(e=>e.from===nid);
    if(!removed)throw new Error('Block does not exist.');
    // Unary operations and tensor aliases can bypass without guessing a merge/slice.
    const distinctDestinations=new Set(outgoing.map(edge=>edge.to)).size===outgoing.length;
    const bypass=incoming.length===1 && !['Input','Output','Concat','Splitter'].includes(removed.type) && (removed.type!=='Replicator' || distinctDestinations);
    const replacements=bypass ? outgoing.map(edge=>({...incoming[0],to:edge.to})) : [];
    const sourceKey=bypass ? incoming[0].from+(incoming[0].from_port?'@'+incoming[0].from_port:'') : null;
    doc.graph.nodes=doc.graph.nodes.filter(n=>n.id!==nid);
    doc.graph.edges=doc.graph.edges.filter(e=>e.from!==nid && e.to!==nid);
    replacements.forEach(e=>{if(!doc.graph.edges.some(old=>old.from===e.from && old.to===e.to && old.from_port===e.from_port))doc.graph.edges.push(e);});
    for(const join of doc.graph.nodes.filter(n=>n.type==='Concat')){
      join.sources=[...new Set(join.sources.flatMap(key=>key===nid || key.startsWith(nid+'@') ? sourceKey?[sourceKey]:[] : [key]))];
    }
    return incoming[0]?.from || doc.graph.nodes[0]?.id || null;
  }
  return {GRID,setSpacing,metrics,snapPosition, minimumSize,width,height,resizeBlock,slots, portPoint, addInput,addOutput, addRouter, addRouterPort, removeBoundary, removeBlock,connect,sameWire,removeWire,retargetWire,outputValues};
});
