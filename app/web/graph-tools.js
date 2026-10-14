"use strict";
((root, factory) => {
  const api = factory(typeof module !== "undefined" && module.exports ? require('./recurrent-editing.js') : root.LoomRecurrent);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LoomGraph = api;
})(typeof globalThis !== "undefined" ? globalThis : this, R => {
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
  function blockToggleShortcut(event){
    if(event.key!==' ' || event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey)return false;
    return !event.target?.isContentEditable && !event.target?.closest?.('input,textarea,select,button,a,summary,[contenteditable],[role="button"],[role="textbox"],[role="combobox"],[role="slider"],[role="checkbox"],[role="switch"]');
  }
  const node = (doc, nid) => doc.graph.nodes.find(n => n.id === nid);
  const routers = n => ["Splitter", "Replicator"].includes(n.type);
  const multiOutput = n => routers(n) || R.multiport(n);
  const outputs = n => n.type === "Splitter" ? n.parts.map(p => p.output) : n.outputs;
  const outputPorts = n => R.multiport(n)?R.ports(n):outputs(n);
  const appendTarget = (n,selected,port) => n.id===selected && port!==null && port!==undefined && multiOutput(n) && slots(n,true).includes(port);
  function tensorShape(n,entry,port=null){
    if(R.multiport(n))return entry?.ports?.[port];
    if(n.type==='Splitter'){
      const base=entry?.input_shape,part=n.parts.find(row=>row.output===port);
      return base&&part?[...base.slice(0,-1),part.features]:undefined;
    }
    return entry?.output_shape;
  }
  function replaceSource(n,oldKey,newKey){
    if(['Concat','Add'].includes(n.type))n.sources=n.sources.map(key=>key===oldKey?newKey:key);
    if(R.multiInput(n))R.replaceSource(n,oldKey,newKey);
  }
  function unusedLabel(doc, type, prefix, first = 1) {
    const used = new Set(doc.graph.nodes.filter(n => n.type === type).map(n => (n.label || n.id).trim()));
    let number = first;
    while (used.has(prefix + number)) number++;
    return prefix + number;
  }
  function room(doc, extra) {
    if (doc.graph.nodes.length + extra > 64) throw new Error("This prototype supports at most 64 blocks.");
  }
  // MLP width syntax: comma-separated terms, each `w` or `count*w` (count copies of w).
  // The first width is the input width. "2, 10*3, 2" -> [2, 3 ×10, 2].
  function parseWidths(text) {
    const terms = String(text).split(',').map(t => t.trim());
    if (!terms.length || terms.some(t => !t)) throw new Error('Write widths separated by commas, e.g. 2, 10*3, 2.');
    const widths = [];
    for (const term of terms) {
      const m = /^(?:(\d+)\s*\*\s*)?(\d+)$/.exec(term);
      if (!m) throw new Error('"' + term + '" is not a width or count*width.');
      const count = m[1] === undefined ? 1 : Number(m[1]), width = Number(m[2]);
      if (!Number.isSafeInteger(count) || count < 1 || !Number.isSafeInteger(width) || width < 1)
        throw new Error('"' + term + '": counts and widths must be positive integers.');
      if (widths.length + count > 64) throw new Error('Too many layers: at most 64 widths.');
      for (let k = 0; k < count; k++) widths.push(width);
    }
    if (widths.length < 2) throw new Error('Give at least two widths: the input width and the output width.');
    return widths;
  }
  // Ordinary Linear/activation blocks: one activation after every hidden Linear, none after the readout.
  // After a selection (inputWidth = its output width) the chain is spliced in like any inserted block;
  // with nothing selected it becomes a new branch Input [n0] -> chain -> Output.
  function addMLP(doc, {after = null, port = null, inputWidth = null, widths, activation = 'Tanh'}) {
    const layers = widths.length - 1, added = 2 * layers - 1 + (after ? 0 : 2);
    if (doc.graph.nodes.length + added > 64)
      throw new Error('This MLP adds ' + added + ' blocks; the limit is 64 and the drawing has ' + doc.graph.nodes.length + '.');
    if (after && inputWidth !== null && widths[0] !== inputWidth)
      throw new Error('The first width is the input width: the selected block outputs ' + inputWidth + ', not ' + widths[0] + '.');
    const anchor = after ? node(doc, after) : null;
    const x0 = anchor ? (anchor.x || 0) + 245 : 90, y0 = anchor ? (anchor.y || 0) : Math.max(40, ...doc.graph.nodes.map(n => (n.y || 0) + 150));
    const chain = [];
    for (let k = 1; k <= layers; k++) {
      chain.push({id: id(), type: 'Linear', features: widths[k], bias: true});
      if (k < layers) {
        const act = {id: id(), type: 'Activation', activation};
        if (['Softmax', 'LogSoftmax'].includes(activation)) act.dim = -1;
        chain.push(act);
      }
    }
    chain.forEach((n, i) => Object.assign(n, {x: x0 + 224 * i, y: y0}));
    const links = chain.slice(1).map((n, i) => ({from: chain[i].id, to: n.id}));
    const first = chain[0], last = chain.at(-1);
    if (anchor) {
      const edge = doc.graph.edges.find(e => e.from === anchor.id && (!port || (e.from_port || e.to) === port));
      const next = edge ? node(doc, edge.to) : null, key = port ? anchor.id + '@' + port : anchor.id;
      if (next) {
        const shift = Math.max(0, x0 + 224 * chain.length - next.x);
        doc.graph.nodes.filter(n => n.x >= next.x && n.id !== anchor.id).forEach(n => n.x += shift);
        doc.graph.edges = doc.graph.edges.filter(e => e !== edge);
        links.push({from: last.id, to: next.id});
        replaceSource(next, key, last.id);
      }
      const incoming = {from: anchor.id, to: first.id}; if (port) incoming.from_port = port;
      links.unshift(incoming);
      doc.graph.nodes.push(...chain);
    } else {
      const input = {id: id(), type: 'Input', features: widths[0], label: unusedLabel(doc, 'Input', 'MLP input '), x: x0 - 245, y: y0};
      const output = {id: id(), type: 'Output', label: unusedLabel(doc, 'Output', 'MLP output '), x: x0 + 224 * chain.length, y: y0};
      if (input.x < 0) { const d = -input.x + 40; [input, output, ...chain].forEach(n => n.x += d); }
      links.unshift({from: input.id, to: first.id}); links.push({from: last.id, to: output.id});
      doc.graph.nodes.push(input, ...chain, output);
    }
    doc.graph.edges.push(...links);
    return {first: first.id, last: last.id, blocks: added};
  }
  // Ready-made blocks expand into simple pieces (vpcc R4). `shape` is the block's contract entry.
  const EXPANDABLE = new Set(['Conv2d', 'MaxPool2d', 'AvgPool2d', 'AdaptiveAvgPool2d']);
  function expansion(n, shape) {
    if (!EXPANDABLE.has(n.type)) throw new Error('Only Conv2d and the pooling blocks expand into simpler blocks.');
    if (!shape?.input_shape || !shape?.output_shape) throw new Error('Connect the block so its shapes are known before expanding it.');
    const [C, H, W] = shape.input_shape, [, Ho, Wo] = shape.output_shape, pair = v => Array.isArray(v) ? v : [v, v];
    const k = pair(n.kernel_size || [1, 1]), window = {kernel_size: k, stride: pair(n.stride || (n.type === 'Conv2d' ? 1 : k)), padding: pair(n.padding || 0), dilation: pair(n.dilation || 1), op_version: 1};
    const back = {type: 'Permute', dims: [2, 0, 1], op_version: 1};
    if (n.type === 'Conv2d') {
      if ((n.groups || 1) !== 1) throw new Error('Grouped convolutions (groups > 1) do not expand into one Linear.');
      return [{type: 'Patches', ...window}, {type: 'Linear', features: n.out_channels, bias: n.bias !== false}, back];
    }
    if (n.type === 'AdaptiveAvgPool2d') {
      if (pair(n.output_size).join() !== '1,1') throw new Error('Only global average pooling (output 1, 1) expands exactly.');
      return [{type: 'Reshape', shape: [C, H * W]}, {type: 'Reduce', mode: 'mean', axis: -1, op_version: 1}, {type: 'Reshape', shape: [C, 1, 1]}];
    }
    if (n.type === 'MaxPool2d' && window.padding.some(Boolean)) throw new Error('Padded max pooling pads with −∞, Patches with zeros: it does not expand exactly.');
    if (n.type === 'AvgPool2d' && n.count_include_pad === false) throw new Error('Average pooling that excludes padded zeros does not expand exactly.');
    return [{type: 'Patches', ...window}, {type: 'Reshape', shape: [Ho, Wo, C, k[0] * k[1]]},
      {type: 'Reduce', mode: n.type === 'MaxPool2d' ? 'max' : 'mean', axis: -1, op_version: 1}, back];
  }
  function expandBlock(doc, nid, shape) {
    const n = node(doc, nid);
    if (!n) throw new Error('The block is missing.');
    if ('share_with' in n || doc.graph.nodes.some(m => m.share_with === nid)) throw new Error('This block shares parameters; make it independent before expanding.');
    const owner = key => { try { return JSON.parse(key)[0]; } catch { return null; } };  // selectors are JSON [block id, ...]
    if (Object.keys(doc.graph.parameter_policy?.parameters || {}).some(key => owner(key) === nid)) throw new Error('This block has a parameter policy; reset it before expanding.');
    const chain = expansion(n, shape).map((piece, i) => ({id: id(), ...piece, x: (n.x || 0) + 224 * i, y: n.y || 0}));
    room(doc, chain.length - 1);
    const shift = 224 * (chain.length - 1);
    doc.graph.nodes.filter(m => m.id !== nid && (m.x || 0) > (n.x || 0)).forEach(m => m.x += shift);
    const first = chain[0], last = chain.at(-1);
    doc.graph.edges = doc.graph.edges.map(e => e.to === nid ? {...e, to: first.id} : e.from === nid ? {...e, from: last.id} : e);
    doc.graph.nodes.forEach(m => replaceSource(m, nid, last.id));
    doc.graph.edges.push(...chain.slice(1).map((m, i) => ({from: chain[i].id, to: m.id})));
    doc.graph.nodes.splice(doc.graph.nodes.indexOf(n), 1, ...chain);
    return {first: first.id, last: last.id, blocks: chain.length};
  }
  function snapPosition(x, y, enabled, grid = GRID) {
    return enabled ? {x: Math.round(x / grid) * grid, y: Math.round(y / grid) * grid} : {x, y};
  }
  function minimumSize(n) {
    const count = Math.max(slots(n,false).length,slots(n,true).length,1);
    const m=metrics();return {width:m.width,height:Math.max(m.baseHeight,m.socketTop+10+count*m.rowGap)};
  }
  const width = n => n.collapsed ? 72 : Math.max(minimumSize(n).width,n.width || 0);
  const height = n => n.collapsed ? Math.max(['Linear','Conv2d','LSTM'].includes(n.type)?54:44,16*Math.max(slots(n,true).length,slots(n,false).length)+10) : Math.max(minimumSize(n).height,n.height || 0);
  function resizeBlock(n,w,h,snap=false){
    if(!Number.isFinite(w) || !Number.isFinite(h))throw new Error('Use finite block dimensions.');
    const min=minimumSize(n),corner=snapPosition(n.x+Math.max(min.width,w),n.y+Math.max(min.height,h),snap);
    n.width=Math.min(100000,snap?Math.max(Math.ceil((n.x+min.width)/GRID)*GRID,corner.x)-n.x:Math.max(min.width,w));
    n.height=Math.min(100000,snap?Math.max(Math.ceil((n.y+min.height)/GRID)*GRID,corner.y)-n.y:Math.max(min.height,h));
    return {width:n.width,height:n.height};
  }
  function slots(n, output) {
    if ((output && n.type === "Output") || (!output && (['Input','Constant','TensorParameter'].includes(n.type) || n.type==='Module' && n.batch_only===true))) return [];
    if (!output && ["Concat","Add"].includes(n.type)) return [...n.sources,null];
    if(!output && R.multiInput(n))return R.sources(n);
    if(output && R.multiport(n))return R.ports(n);
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
    if(multiOutput(source)) {
      port ||= outputPorts(source)[0];
      if(!outputPorts(source).includes(port))throw new Error('Select a declared output port.');
    }
    const output={id:id(),type:'Output',label:unusedLabel(doc,'Output','Output '),x:(source.x||0)+224,y:source.y||0};
    doc.graph.nodes.push(output);
    const edge={from:source.id,to:output.id};if(multiOutput(source))edge.from_port=port;
    doc.graph.edges.push(edge);
    return output.id;
  }
  function addRouter(doc, type, width, afterID = null, port = null) {
    if (!["Splitter", "Replicator"].includes(type)) throw new Error("Choose a Splitter or Replicator.");
    if (!afterID && doc.graph.nodes.some(routers)) throw new Error("Select the box whose output should be routed.");
    if (type === "Splitter" && width < 2) throw new Error("Set the final Linear to at least two output features before splitting.");
    room(doc, 2);
    let output = afterID ? node(doc,afterID) : doc.graph.nodes.find(n => n.type === "Output");
    if (!output)throw new Error('Select an existing tensor-producing box.');
    if (output.type === "Output") {const incoming=doc.graph.edges.find(e=>e.to===output.id);afterID=incoming?.from;port=incoming?.from_port||port;}
    const sourceNode=node(doc,afterID);
    if(sourceNode && multiOutput(sourceNode)) {
      port ||= outputPorts(sourceNode)[0];
      if(!outputPorts(sourceNode).includes(port))throw new Error('Select a declared output port.');
    }
    const edge = afterID ? doc.graph.edges.find(e=>e.from===afterID && (!port || e.from_port===port)) : doc.graph.edges.find(e => e.to === output.id);
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
    replaceSource(output,sourceKey(edge),router.id+"@"+output.id);
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
    if (edge) doc.graph.edges.push({...edge, to: keep});
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
    if (!a || !b || from === to || a.type === "Output" || !slots(b,false).length) throw new Error("Choose a valid source and destination.");
    if(multiOutput(a) && (!fromPeer || !outputPorts(a).includes(fromPeer)))throw new Error("Choose a declared output socket.");
    if(!multiOutput(a) && fromPeer!==null)throw new Error('This box has no named output socket.');
    const key=fromPeer ? from+'@'+fromPeer : from;
    if(R.multiInput(b)) {
      if(!R.sources(b).includes(toPeer))throw new Error('Choose a declared recurrent input socket.');
      // Replace just this physical source binding; other argument wires stay put.
      doc.graph.edges=doc.graph.edges.filter(e=>!(e.to===to && sourceKey(e)===toPeer));
      R.replaceSource(b,toPeer,key);
    }
    if(['Concat','Add'].includes(b.type)) {
      // Normalize legacy router source IDs to explicit source-port keys before edits.
      const oldEdges=doc.graph.edges.filter(e=>e.to===to);
      b.sources=b.sources.map(src=>{
        const old=oldEdges.find(e=>e.from===src && e.from_port);
        return old?old.from+"@"+old.from_port:src;
      });
    }
    doc.graph.edges = doc.graph.edges.filter(e =>
      !(e.to === to && !['Concat','Add','Recurrent','LSTM'].includes(b.type)) && !(e.from === from && e.to === to && (e.from_port || null)===(fromPeer || null)));
    const edge={from,to};
    if(multiOutput(a)) {
      edge.from_port=fromPeer;
    }
    doc.graph.edges.push(edge);
    if(['Concat','Add'].includes(b.type)) {
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
    if(['Concat','Add'].includes(dest?.type)) {
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
    if(found.to===to && (!R.multiInput(dest) || toPeer===sourceKey(found)))return false;
    if(!slots(dest,false).length || to===found.from)throw new Error('Choose a valid input socket.');
    if(draft.graph.edges.some(e=>!sameWire(e,found) && sameWire(e,{...found,to})))
      throw new Error('That source is already connected to this destination.');
    const occupied=draft.graph.edges.some(e=>e.to===to);
    if(R.multiInput(dest) && (!R.sources(dest).includes(toPeer) || draft.graph.edges.some(e=>e.to===to && sourceKey(e)===toPeer)))
      throw new Error('Choose an empty declared recurrent input socket.');
    if((!['Concat','Add','Recurrent','LSTM'].includes(dest.type) && occupied) || (['Concat','Add'].includes(dest.type) && toPeer!==null))
      throw new Error('That input is occupied. Choose an empty socket (the last socket on Concat).');
    const seen=new Set(),todo=[to];
    while(todo.length){
      const current=todo.pop();if(current===found.from)throw new Error('This connection would create a cycle.');
      if(seen.has(current))continue;seen.add(current);
      draft.graph.edges.filter(e=>e.from===current && !sameWire(e,found)).forEach(e=>todo.push(e.to));
    }
    removeWire(draft,found);
    if(found.to===to && R.multiInput(dest)) {
      // The old argument stays visibly disconnected; moving to another socket
      // must not silently bind both arguments to the one remaining wire.
      R.replaceSource(dest,sourceKey(found),'disconnected/'+id());
    }
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
    const bypass=incoming.length===1 && !['Input','Output','Concat','Add','Splitter','Recurrent','LSTM'].includes(removed.type) && (removed.type!=='Replicator' || distinctDestinations);
    const replacements=bypass ? outgoing.map(edge=>({...incoming[0],to:edge.to})) : [];
    const sourceKey=bypass ? incoming[0].from+(incoming[0].from_port?'@'+incoming[0].from_port:'') : null;
    doc.graph.nodes=doc.graph.nodes.filter(n=>n.id!==nid);
    doc.graph.edges=doc.graph.edges.filter(e=>e.from!==nid && e.to!==nid);
    replacements.forEach(e=>{if(!doc.graph.edges.some(old=>old.from===e.from && old.to===e.to && old.from_port===e.from_port))doc.graph.edges.push(e);});
    for(const join of doc.graph.nodes.filter(n=>['Concat','Add'].includes(n.type))){
      join.sources=[...new Set(join.sources.flatMap(key=>key===nid || key.startsWith(nid+'@') ? sourceKey?[sourceKey]:[] : [key]))];
    }
    if(bypass)for(const recurrent of doc.graph.nodes.filter(R.multiInput))
      for(const key of R.sources(recurrent))if(key===nid || key.startsWith(nid+'@'))R.replaceSource(recurrent,key,sourceKey);
    return incoming[0]?.from || doc.graph.nodes[0]?.id || null;
  }
  function fitViewport(nodes,rect){
    if(!Array.isArray(nodes)||!nodes.length||!rect||!['x','y','width','height'].every(k=>Number.isFinite(rect[k]))||rect.width<=0||rect.height<=0)return null;
    const left=Math.min(...nodes.map(n=>n.x)),top=Math.min(...nodes.map(n=>n.y));
    const right=Math.max(...nodes.map(n=>n.x+width(n))),bottom=Math.max(...nodes.map(n=>n.y+height(n)));
    if(![left,top,right,bottom].every(Number.isFinite)||right<=left||bottom<=top)return null;
    // Keep the usual 50-unit margins on roomy views, but don't reject a narrow
    // positive free region or force a scale that overflows it at accessibility zoom.
    const zoom=Math.min(1.3,(rect.width-2*Math.min(50,rect.width*.15))/(right-left),
                               (rect.height-2*Math.min(50,rect.height*.15))/(bottom-top));
    return {zoom,x:rect.x+(rect.width-(right-left)*zoom)/2-left*zoom,
                 y:rect.y+(rect.height-(bottom-top)*zoom)/2-top*zoom,fitted:true};
  }
  return {parseWidths,addMLP,expandBlock,EXPANDABLE,GRID,setSpacing,metrics,snapPosition,blockToggleShortcut, minimumSize,width,height,resizeBlock,slots, portPoint, multiOutput,appendTarget,tensorShape,replaceSource,addInput,addOutput, addRouter, addRouterPort, removeBoundary, removeBlock,connect,sameWire,removeWire,retargetWire,outputValues,fitViewport};
});
