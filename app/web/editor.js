"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const G = window.LoomGraph;
  const floating = window.LoomPanels.initialize(document);
  const activationInfo = {
    ReLU: ["h = max(0, z)", "Keeps positive features; zeros negative ones."],
    Tanh: ["h = tanh(z)", "Smooth, zero-centered output between −1 and 1."],
    Sigmoid: ["h = 1 / (1 + exp(−z))", "Bounds each feature between 0 and 1. Regression still uses MSE, not a classification loss."],
    GELU: ["h = z · Φ(z)", "Smooth gating using the normal CDF. PyTorch uses the exact, non-approximated mode."],
  };
  let host, state = {models: {}, recent: []}, doc = null, description = null;
  let selected = null, selectedPort = null, selectedWire = null, connecting = null, drag = null, panel = "experiment";
  let syncQueue = Promise.resolve(), describeGeneration = 0;
  const notifications=window.LoomNotifications.initialize(document);
  const liveHistories = {};
  const clone = obj => JSON.parse(JSON.stringify(obj));
  const esc = str => String(str).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
  const modelID = () => location.hash.startsWith("#model=") ? location.hash.slice(7) : null;
  const node = id => doc?.graph.nodes.find(n => n.id === id);
  const busy = () => Boolean(state.training_id);
  const shapeText = s => "[B, " + (s?.join(", ") || "?") + "]";
  function prepareModel(value) {
    value.view={x:0,y:0,zoom:1,...(value.view || {})};
    value.training={epochs:400,lr:.03,seed:7,...(value.training || {})};
    value.graph.nodes.forEach(n=>{n.x ??= 0;n.y ??= 0;if(n.type==="Linear")n.bias ??= true;});
    return value;
  }

  function toast(message, error = false) {
    notifications.show(message,error);
  }

  function call(action, args = {}) {
    return new Promise((resolve, reject) => {
      host.request(JSON.stringify({action, ...args}), answer => {
        const result = JSON.parse(answer);
        if (result.ok) resolve(result.value);
        else reject(new Error(result.error));
      });
    });
  }
  async function safe(action, args = {}) {
    try { return await call(action, args); }
    catch (error) { toast(error.message, true); return null; }
  }

  function navigate(id) {
    const hash = id ? "#model=" + id : "#home";
    if (location.hash === hash) route();
    else location.hash = hash;
  }

  function header() {
    document.documentElement.dataset.theme=state.theme || 'dark';
    document.documentElement.dataset.spacing=String(state.display?.spacing ?? 4);
    document.documentElement.style.setProperty('--menu-unit',(state.display?.spacing ?? 4)+'px');
    document.documentElement.style.colorScheme=state.theme_scheme || (state.theme==='light' ? 'light' : 'dark');
    Object.entries(state.theme_tokens || {}).forEach(([key,value])=>document.documentElement.style.setProperty('--'+key,value));
    document.documentElement.style.setProperty("--ui-font-size",(state.display?.font_size || 14)+"px");
    $("file-name").textContent = state.path ? state.path.split(/[\\/]/).pop() : (Object.keys(state.models).length ? "Untitled.vtw" : "Welcome");
    $("save-state").textContent = state.dirty ? "Unsaved changes · Ctrl+S" : state.path ? "All changes saved" : "No workspace open";
    $("save-workspace").disabled = !Object.keys(state.models).length;
    $("constrained-mode").checked = state.constrained_mode !== false;
    $("constrained-mode").parentElement.classList.toggle("unconstrained",state.constrained_mode===false);
    $("constrained-mode").disabled = !Object.keys(state.models).length || busy();
    document.querySelector(".canvas-toolbar").hidden = !doc;
    $("new-tab").disabled = !doc;
    $("resume-area").hidden = !Object.keys(state.models).length;
    $("runtime").textContent = (state.binding || "Qt webview") + " · SQLite workspace" +
      (state.rendering?.mode === "software" ? " · software webview" : "");
  }

  function recentFiles() {
    const list = $("recent-list");
    list.replaceChildren();
    if (!state.recent.length) {
      list.innerHTML = '<div class="empty-recent"><b>No recent workspaces yet.</b>Create your first workspace, then save it as a .vtw file.</div>';
    }
    state.recent.forEach(file => {
      const button = document.createElement("button");
      button.className = "recent-row";
      button.innerHTML = '<span class="file-icon">▣</span><span><b>' + esc(file.name) + '</b><small>' + esc(file.path) + '</small></span><span class="arrow">→</span>';
      button.onclick = () => safe("openWorkspace", {path: file.path});
      list.append(button);
    });
  }

  async function route() {
    const id = modelID();
    endDrag();
    doc = id && state.models[id] ? prepareModel(clone(state.models[id])) : null;
    $("welcome").hidden = Boolean(doc);
    $("editor").hidden = !doc;
    description = null;
    connecting = null;
    selectedWire = null;
    header();
    recentFiles();
    if (!doc) {
      $("status").textContent = "Open a workspace to start building";
      return;
    }
    selected = doc.graph.nodes.find(n => n.type === "Linear" || n.type === "Module")?.id || doc.graph.nodes[0]?.id;
    document.title = doc.name + " · Visual PyTorch";
    render();
    await describe();
    if (!doc.view.fitted) requestAnimationFrame(() => { if (doc?.id === id) fit(); });
  }

  function hydrate(value) {
    state = value;
    if (doc && state.models[doc.id]) {
      doc = prepareModel(clone(state.models[doc.id]));
      render();
      describe();
    } else if (doc && !state.models[doc.id]) {
      navigate(null);
    }
    header();
    recentFiles();
  }

  function renderLibrary() {
    const models=Object.values(state.models).sort((a,b)=>a.name.localeCompare(b.name,undefined,{sensitivity:'base'}) || a.id.localeCompare(b.id));
    renderBookkeeping();
    $("model-list").replaceChildren();
    for (const model of models) {
      const button = document.createElement("button");
      button.className = "model-row" + (doc?.id === model.id ? " selected" : "");
      const refCount = model.graph.nodes.filter(n => n.type === "Module").length;
      button.innerHTML = '<span class="model-glyph">▣</span><span><b>' + esc(model.name) + '</b><small>' + model.graph.nodes.length + ' blocks' + (refCount ? ' · ' + refCount + ' reference' + (refCount > 1 ? 's' : '') : model.graph.nodes.some(n => n.type === "Activation") ? ' · nonlinear graph' : ' · affine graph') + '</small></span><span class="tab-hint">↗</span>';
      button.onclick = event => safe("openModel", {id: model.id, newTab: event.ctrlKey || event.metaKey || Boolean(event.target.closest(".tab-hint"))});
      button.ondblclick = () => safe("openModel", {id: model.id, newTab: true});
      $("model-list").append(button);
    }
    const current = $("reference-select").value;
    $("reference-select").replaceChildren();
    for (const model of models.filter(m => m.id !== doc?.id)) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.name;
      $("reference-select").append(option);
    }
    if (state.models[current]) $("reference-select").value = current;
    $("add-reference").disabled = !$("reference-select").value || busy();
  }

  function renderBookkeeping() {
    const card=$("architecture-bookkeeping");card.replaceChildren();
    if(!doc){card.hidden=true;return;}card.hidden=false;
    const heading=document.createElement('span');heading.className='eyebrow';heading.textContent='CURRENTLY OPEN';
    const name=document.createElement('b');name.textContent=doc.name;
    const stats=document.createElement('p');
    const refs=doc.graph.nodes.filter(n=>n.type==='Module').length;
    const ins=doc.graph.nodes.filter(n=>n.type==='Input').length,outs=doc.graph.nodes.filter(n=>n.type==='Output').length;
    stats.textContent=doc.graph.nodes.length+' blocks · '+doc.graph.edges.length+' wires · '+ins+' inputs · '+outs+' outputs'+(refs?' · '+refs+' references':'');
    const contract=document.createElement('p');
    contract.textContent=description?.valid ? description.contract.parameters.toLocaleString()+' parameters · '+shapeText(description.contract.input_shape)+' → '+shapeText(description.contract.output_shape) : description ? 'Draft: '+description.error : 'Checking graph…';
    contract.className='bookkeeping-contract';
    card.append(heading,name,stats,contract);
  }

  function render() {
    if (!doc) return;
    $("model-title").textContent = doc.name;
    renderLibrary();
    renderGraph();
    renderInspector();
    renderExperiment();
    renderResults();
    const running = busy();
    $("train").disabled = running || !description?.valid || !description?.contract?.parameters || !doc.dataset || description?.run_safety?.allowed === false;
    $("stop").hidden = !running;
    $("add-linear").disabled = running;
    $("add-model").disabled = running;
    $("generate-data").disabled = running || !description?.valid || description?.model_safety?.allowed === false;
    $("import-data").disabled = running || !description?.valid;
    $("rename-model").disabled = running;
    for (const id of ["add-activation", "activation-select", "target-kind", "add-reshape", "add-concat", "add-input", "add-output"]) $(id).disabled = running;
    for (const id of ["add-splitter", "add-replicator"]) $(id).disabled = running || !description?.valid;
  }

  async function describe() {
    if (!doc) return;
    const generation = ++describeGeneration, id = doc.id;
    const value = await safe("describe", {id});
    if (!value || generation !== describeGeneration || doc?.id !== id) return;
    description = value;
    showDescription();
  }

  function showDescription() {
    if (!description || !doc) return;
    $("validation-message").textContent = description.valid ?
      "✓ Valid · " + shapeText(description.contract.input_shape) + " → " + shapeText(description.contract.output_shape) : "⚠ " + description.error;
    $("validation-message").classList.toggle("error", !description.valid);
    const blocked = description.model_safety?.allowed === false;
    $("validation-message").classList.toggle("warning", blocked);
    if (description.valid && blocked) $("validation-message").textContent += " · runs blocked by safety";
    $("code").textContent = description.valid ? description.code : description.error;
    $("parameter-count").textContent = description.valid ? description.contract.parameters.toLocaleString() + " trainable parameters" : "Resolve graph errors before compiling";
    $("status").textContent = description.valid ? "Graph valid · " + description.contract.parameters.toLocaleString() + " parameters · " + (blocked ? "constrained runs blocked" : "PyTorch ready") : description.error;
    render();
  }

  function persist() {
    if (!doc) return Promise.resolve();
    const snapshot = clone(doc), mid = doc.id;
    state.models[mid] = clone(doc);
    state.dirty = true;
    header();
    syncQueue = syncQueue.then(async () => {
      const answer = await safe("updateModel", {document: snapshot});
      if (answer && doc?.id === mid) {
        description = answer;
        showDescription();
      }
    });
    return syncQueue;
  }

  function svgElement(tag, attrs = {}, text = null) {
    const el = document.createElementNS(SVG, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text !== null) el.textContent = text;
    return el;
  }
  function shape(n) { return description?.contract?.shapes[n.id]; }
  function title(n) {
    if (!n) return "Missing port";
    if (n.label) return n.label;
    if (n.type === "Input") return "Input tensor";
    if (n.type === "Output") return "Prediction ŷ";
    if (n.type === "Module") return state.models[n.ref]?.name || "Missing architecture";
    if (n.type === "Activation") return n.activation;
    if (n.type === "Concat") return "Join branches";
    return n.type;
  }
  const portPoint = G.portPoint;
  const linearParameters=(n,s)=>s ? s.input_shape.at(-1)*n.features+(n.bias!==false?n.features:0) : '?';
  function wirePath(a, b) {
    const bend = Math.max(50, Math.abs(b.x - a.x) * .45);
    return "M" + a.x + "," + a.y + " C" + (a.x + bend) + "," + a.y + " " + (b.x - bend) + "," + b.y + " " + b.x + "," + b.y;
  }

  function renderGraph() {
    G.setSpacing(state.display?.box_spacing ?? 35);
    if (!doc) return;
    $("nodes").replaceChildren();
    $("wires").replaceChildren();
    $("wire-handles").replaceChildren();
    if(selectedWire && !doc.graph.edges.some(e=>G.sameWire(e,selectedWire)))selectedWire=null;
    for (const edge of doc.graph.edges) {
      const from = node(edge.from), to = node(edge.to);
      if (!from || !to) continue;
      const path = svgElement("path", {d: wirePath(portPoint(from, true, edge.from_port || to.id), portPoint(to, false, edge.from_port ? from.id+"@"+edge.from_port : from.id)), class: "wire"+(G.sameWire(edge,selectedWire)?" selected-wire":""), "data-from": edge.from, "data-to": edge.to});
      if(edge.from_port)path.dataset.fromPort=edge.from_port;
      const tip = svgElement("title", {}, "Click to select · drag destination endpoint to reconnect · D to delete");
      path.append(tip);
      $("wires").append(path);
    }
    for (const n of doc.graph.nodes) {
      const s = shape(n);
      const group = svgElement("g", {class: "node " + n.type + (n.collapsed ? " collapsed" : "") + (n.id === selected ? " selected" : ""), transform: "translate(" + n.x + "," + n.y + ")", "data-id": n.id});
      const height=G.height(n),m={...G.metrics(),width:G.width(n)};
      const shortened=(text,charWidth,reserve=0)=>{const max=Math.floor((m.width-2*m.padding-reserve)/charWidth);return text.length>max?text.slice(0,max-1)+'…':text;};
      group.append(svgElement("rect", {width:m.width,height,rx:7}));
      group.append(svgElement("text", {x:m.padding,y:m.headerY,class:"node-header"}, n.type === "Module" ? "ARCHITECTURE ↗" : n.type.toUpperCase()));
      const icon=svgElement('g',{class:'node-type-icon',transform:n.collapsed?'translate(11,'+(height/2-10)+') scale(0.833333)':'translate('+(m.width-m.padding-20)+','+Math.max(2,m.padding-7)+') scale(0.833333)',role:'button',tabindex:busy()?-1:0,'aria-expanded':String(!n.collapsed),'aria-disabled':String(busy()),'aria-label':n.collapsed?'Expand '+title(n)+' block (double-click or Enter)':'Collapse '+title(n)+' block'});
      icon.innerHTML=window.LoomTypeIcons.svg(n.type);
      icon.append(svgElement('rect',{class:'icon-hit',x:-3,y:-3,width:30,height:30,rx:3}));
      icon.append(svgElement('title',{},n.type+' block · '+(n.collapsed?'Drag to move · double-click to expand':'Click to collapse')));group.append(icon);
      const label = title(n);
      const labelText=svgElement("text",{x:m.padding,y:m.titleY,class:"node-title"},shortened(label,7.8));
      labelText.append(svgElement('title',{},label));group.append(labelText);
      let line;
      if (n.type === "Input") line = shapeText(n.shape || [n.features]);
      else if (n.type === "Output") line = shapeText(s?.output_shape || n.shape);
      else line = shapeText(s?.input_shape) + " → " + shapeText(s?.output_shape);
      const shapeLine=svgElement("text", {x:m.padding,y:m.shapeY,class:"node-shape"},shortened(line,6.2));
      shapeLine.append(svgElement("title",{},line));group.append(shapeLine);
      const note = n.type === "Linear" ? ('Bias '+(n.bias!==false?'on':'off')+' · '+linearParameters(n,s)+" params · last axis") : n.type === "Reshape" ? "Same elements · batch preserved" : n.type === "Module" ? "Inspect · Ctrl-double-click: open" : n.type === "Concat" ? (n.flatten===false?"Join last axis":"Flatten parts + join") : n.type === "Splitter" ? "Split last axis · branches" : n.type === "Replicator" ? "Shared tensor · branch ports" : n.type === "Activation" ? "Elementwise · 0 params" : "Named tensor · float32";
      const noteText=svgElement("text",{x:m.padding,y:height-m.padding-m.footerInset,class:"node-params"},shortened(note,4.9,14));
      noteText.append(svgElement('title',{},note));group.append(noteText);
      if(n.collapsed){
        group.querySelectorAll('.node-header,.node-title,.node-shape,.node-params').forEach(el=>el.remove());
        group.append(svgElement('title',{},label+' · '+line+' · '+note));
      }
      for (const output of [false, true]) {
        for (const peer of G.slots(n, output)) {
          const p = portPoint(n, output, peer);
          const circle = svgElement("circle", {cx: p.x - n.x, cy: p.y - n.y, r: 5,
            class: "port " + (output ? "port-out" : "port-in") + (connecting?.id === n.id && connecting?.peer === peer ? " armed" : ""), "data-id": n.id});
          if (peer) circle.dataset.peer = peer;
          circle.append(svgElement("title", {}, peer ? (node(peer)?title(node(peer)):peer) : "Packed feature tensor"));
          group.append(circle);
        }
      }
      const grip=svgElement('g',{class:'node-resize',transform:'translate('+(m.width-14)+','+(height-14)+')',tabindex:busy()?-1:0,role:'button','aria-label':'Resize '+label+' block','aria-disabled':String(busy())});
      grip.append(svgElement('path',{d:'M0 0H14V14H0Z',fill:'transparent',stroke:'none'}));
      grip.append(svgElement('path',{d:'M3 11L11 3M7 11L11 7',class:'resize-mark',fill:'none'}));
      grip.append(svgElement('title',{},'Drag to resize · Ctrl snaps bottom-right · arrow keys resize'));if(!n.collapsed)group.append(grip);
      $("nodes").append(group);
    }
    // Above the boxes, but just outside their input sockets. Every destination
    // is directly draggable; selection also reveals a larger endpoint handle.
    for(const edge of doc.graph.edges){
      const from=node(edge.from),to=node(edge.to);if(!from || !to)continue;
      const p=portPoint(to,false,edge.from_port?edge.from+'@'+edge.from_port:edge.from);
      const active=G.sameWire(edge,selectedWire);
      const handle=svgElement('circle',{cx:p.x-11,cy:p.y,r:active?6:4,class:'wire-end'+(active?' selected-wire':''),
        'data-from':edge.from,'data-to':edge.to,role:'button','aria-label':'Move connection destination','aria-disabled':String(busy())});
      if(edge.from_port)handle.dataset.fromPort=edge.from_port;
      handle.append(svgElement('title',{},'Drag to another empty input socket · D deletes selected connection'));
      $('wire-handles').append(handle);
    }
    updateViewport();
  }

  function updateViewport() {
    const view = doc?.view || {x: 0, y: 0, zoom: 1};
    $("viewport").setAttribute("transform", "translate(" + view.x + "," + view.y + ") scale(" + view.zoom + ")");
    $("dots").setAttribute("patternTransform", "translate(" + view.x + "," + view.y + ") scale(" + view.zoom + ")");
    $("zoom-label").textContent = Math.round(view.zoom * 100) + "%";
  }

  function screenPoint(event) {
    const rect = $("canvas").getBoundingClientRect();
    return {x: event.clientX - rect.left, y: event.clientY - rect.top};
  }
  function worldPoint(event) {
    const point = screenPoint(event), view = doc.view;
    return {x: (point.x - view.x) / view.zoom, y: (point.y - view.y) / view.zoom};
  }
  function zoom(factor, point = null) {
    if (!doc) return;
    const rect = $("canvas").getBoundingClientRect();
    const anchor = point || {x: rect.width / 2, y: rect.height / 2}, view = doc.view;
    const x = (anchor.x - view.x) / view.zoom, y = (anchor.y - view.y) / view.zoom;
    view.zoom = Math.min(2.5, Math.max(.05, view.zoom * factor));
    view.x = anchor.x - x * view.zoom;
    view.y = anchor.y - y * view.zoom;
    updateViewport();
    persist();
  }
  function fit() {
    if (!doc) return;
    const rect = floating.available(), nodes = doc.graph.nodes;
    if (!rect || rect.width<80 || rect.height<80) {toast("Hide or move a panel to make room, then Fit the graph.");return;}
    const left = Math.min(...nodes.map(n => n.x)), top = Math.min(...nodes.map(n => n.y));
    const right = Math.max(...nodes.map(n => n.x + G.width(n))), bottom = Math.max(...nodes.map(n => n.y + G.height(n)));
    const scale = Math.min(1.3, (rect.width - 100) / (right - left), (rect.height - 100) / (bottom - top));
    const fittedScale = Math.max(.05, scale);
    doc.view = {zoom: fittedScale, x: rect.x + (rect.width - (right - left) * fittedScale) / 2 - left * fittedScale, y: rect.y + (rect.height - (bottom - top) * fittedScale) / 2 - top * fittedScale, fitted: true};
    updateViewport();
    persist();
  }

  function renderInspector() {
    const n = node(selected), content = $("inspector-content");
    content.replaceChildren();
    if (!n) {
      $('node-kind').textContent=selectedWire?'Connection':'';
      content.innerHTML='<p class="inspect-note">'+(selectedWire?'Connection selected. Drag its arrow end to reconnect, or press D to delete.':'Select a block to inspect it.')+'</p>';return;
    }
    $("node-kind").textContent = n.type;
    const s = shape(n);
    let html = '<div class="eyebrow">' + esc(n.id) + '</div><h3 class="inspect-title">' + esc(title(n)) + '</h3>';
    if (n.type === "Linear") {
      html += '<label class="inspect-label" for="features">Output features</label><input id="features" class="inspect-input" type="number" min="1" value="' + n.features + '"' + (busy() ? ' disabled' : '') + '><label class="inspect-bias"><input id="bias" type="checkbox"' + (n.bias ? ' checked' : '') + (busy() ? ' disabled' : '') + '>Learn a bias b</label>';
      html += '<div class="contract-card">input '+shapeText(s?.input_shape)+'<br>output '+shapeText(s?.output_shape)+'<br>A [' + n.features + ', ' + (s?.input_shape.at(-1) || "?") + ']<br>'+(n.bias!==false?'b ['+n.features+']':'Bias off · no b')+'</div><p class="inspect-note">Linear mixes only the last axis. Earlier dimensions are preserved and share the same weights.</p>';
    } else if (n.type === "Input") {
      html += '<div class="contract-card">x '+shapeText(n.shape || [n.features])+'<br>dtype float32</div><p class="inspect-note">Specify sample dimensions, not B. Example: 3, 8 means [B, 3, 8]. This input can follow its own path before a later join.</p>';
    } else if (n.type === "Reshape") {
      html += '<div class="contract-card">'+shapeText(s?.input_shape)+' → '+shapeText(s?.output_shape)+'</div><p class="inspect-note">Preserve the number of elements in each sample. One dimension may be -1, inferred from the incoming tensor. Batch B never changes.</p>';
    } else if (n.type === "Activation") {
      html += '<label class="inspect-label">Activation</label><select id="node-activation"' + (busy() ? ' disabled' : '') + '>' + Object.keys(activationInfo).sort().map(key => '<option' + (key === n.activation ? ' selected' : '') + '>' + key + '</option>').join('') + '</select><div class="contract-card">'+shapeText(s?.input_shape)+' → '+shapeText(s?.output_shape)+'<br>0 trainable parameters</div><p class="inspect-note">' + esc(activationInfo[n.activation][1]) + '</p><p class="inspect-note">Activations give the network nonlinear expressive power.</p>';
    } else if (n.type === "Concat") {
      html += '<label class="inspect-bias"><input id="flatten-concat" type="checkbox"'+(n.flatten!==false?' checked':'')+(busy()?' disabled':'')+'>Flatten each sample before joining</label><p class="inspect-note">Connect branch outputs to this box. Checked: flatten and concatenate sample values. Unchecked: concatenate the last axis, requiring equal other dimensions. Port order determines concatenation order.</p>';
      n.sources.forEach((nid, i) => html += '<div class="port-row"><span>' + esc(title(node(nid))) + ' · ' + (node(nid)?.features || "?") + '</span><button data-move-input="' + i + '"' + (!i || busy() ? ' disabled' : '') + '>↑</button></div>');
    } else if (["Splitter", "Replicator"].includes(n.type)) {
      const ids = n.type === "Splitter" ? n.parts.map(p => p.output) : n.outputs;
      html += '<p class="inspect-note">' + (n.type === "Splitter" ? 'Partition the incoming tensor on its last axis into ordered slices. Positive sizes must sum to ' + (s?.input || "the incoming width") + '. Connect each socket to any downstream branch.' : 'Each port exposes the same full tensor, without a physical copy. Connect ports to downstream branches.') + '</p>';
      ids.forEach((nid, i) => html += '<div class="port-row"><span>' + esc(title(node(nid))) + '</span>' + (n.type === "Splitter" ? '<input data-split-size="' + i + '" type="number" min="1" value="' + n.parts[i].features + '"' + (busy() ? ' disabled' : '') + '>' : '<small>full tensor</small>') + '</div>');
      html += '<button id="add-output-port"' + (busy() ? ' disabled' : '') + '>+ Output port</button>';
    } else if (n.type === "Module") {
      html += '<div class="contract-card">reference<br>' + esc(state.models[n.ref]?.name || "missing") + '<br>input '+shapeText(s?.input_shape)+'<br>output '+shapeText(s?.output_shape)+'</div><p class="inspect-note">This block refers to a named architecture in this workspace. Editing that architecture updates every referring graph. Each occurrence has independent trainable parameters.</p><div class="inspect-actions"><button id="open-reference">Open details</button><button id="tab-reference">Open in new tab ↗</button></div>';
      const signature = description?.interfaces?.[n.id];
      if (signature) html += '<div class="contract-card">Declared inputs<br>' + signature.input_groups.map(g => esc(g.name) + ' [' + g.features + ']').join('<br>') + '<br>Declared outputs<br>' + signature.output_groups.map(g => esc(g.name) + ' [' + g.features + ']').join('<br>') + '</div><p class="inspect-note">This prototype connects references through packed tensors; the named boundaries remain part of the reusable architecture contract.</p>';
    } else {
      html += '<div class="contract-card">ŷ '+shapeText(s?.output_shape)+'</div><p class="inspect-note">Optional shape declaration must match the incoming wire. Use a Reshape box to change dimensions. Empty means infer from the wire.</p>';
    }
    if (["Input","Output","Reshape"].includes(n.type)) html += '<label class="inspect-label" for="sample-shape">Sample shape (omit B)</label><input id="sample-shape" class="inspect-input" value="'+esc((n.shape || (n.type==="Input"?[n.features]:n.type==="Reshape"?[-1]:[])).join(", "))+'" placeholder="e.g. 3, 8"'+(busy()?' disabled':'')+'>';
    if (["Input", "Output"].includes(n.type)) html += '<label class="inspect-label">Port name</label><input id="port-label" maxlength="60" value="' + esc(n.label || n.id) + '"' + (busy() ? ' disabled' : '') + '>';
    if (["Linear", "Module", "Activation", "Reshape", "Concat", "Splitter", "Replicator"].includes(n.type) ||
        (["Input", "Output"].includes(n.type) && doc.graph.nodes.filter(v => v.type === n.type).length > 1)) html += '<div class="inspect-actions"><button id="remove-node"' + (busy() ? ' disabled' : '') + '>Remove block</button></div>';
    html += '<label class="inspect-label">Diagram size (world units)</label><div class="inline-fields"><label>Width<input id="block-width" type="number" step="any" min="'+G.minimumSize(n).width+'" max="100000" value="'+G.width(n)+'"'+(busy()?' disabled':'')+'></label><label>Height<input id="block-height" type="number" step="any" min="'+G.minimumSize(n).height+'" max="100000" value="'+G.height(n)+'"'+(busy()?' disabled':'')+'></label></div><button id="auto-block-size"'+(busy()?' disabled':'')+'>Automatic size</button>';
    content.innerHTML = html;
    const resize=()=>{if(busy())return;try{G.resizeBlock(n,Number($("block-width").value),Number($("block-height").value));persist();}catch(error){toast(error.message,true);renderInspector();}};
    $("block-width").onchange=$("block-height").onchange=resize;
    $("auto-block-size").onclick=()=>{if(busy())return;delete n.width;delete n.height;persist();};
    if($("sample-shape"))$("sample-shape").onchange=()=>{
      const text=$("sample-shape").value.trim(), dims=text.split(/[,x\s]+/).map(Number);
      if(!text && n.type==="Output")delete n.shape;
      else {
        if(!text || dims.length>8 || dims.some(d=>!Number.isSafeInteger(d) || (d<1 && !(n.type==="Reshape" && d===-1))) || dims.filter(d=>d===-1).length>1 || !Number.isSafeInteger(dims.filter(d=>d!==-1).reduce((a,b)=>a*b,1))) {
          toast("Use 1–8 positive sample dimensions; Reshape may contain one -1.",true);renderInspector();return;
        }
        n.shape=dims;if(n.type==="Input"){delete n.features;doc.dataset=null;}
      }
      invalidateRun();persist();
    };
    if($("flatten-concat"))$("flatten-concat").onchange=()=>{n.flatten=$("flatten-concat").checked;invalidateRun();persist();};
    if ($("features")) $("features").onchange = () => {
      const value = Number($("features").value);
      if (!Number.isSafeInteger(value) || value < 1) { toast("Use a positive integer feature count.", true); renderInspector(); return; }
      if (n.type === "Input" && n.features !== value) doc.dataset = null;
      n.features = value;
      invalidateRun();
      persist();
    };
    if ($("bias")) $("bias").onchange = () => { n.bias = $("bias").checked; invalidateRun(); persist(); };
    if ($("node-activation")) $("node-activation").onchange = () => { n.activation = $("node-activation").value; invalidateRun(); persist(); };
    if ($("port-label")) $("port-label").onchange = () => {
      const label = $("port-label").value.trim();
      if (!label) { toast("Give this port a name.", true); renderInspector(); return; }
      n.label = label; invalidateRun(); persist();
    };
    content.querySelectorAll("[data-split-size]").forEach(input => input.onchange = () => {
      const value = Number(input.value);
      if (!Number.isSafeInteger(value) || value < 1) { toast("Use a positive integer split size.", true); renderInspector(); return; }
      n.parts[Number(input.dataset.splitSize)].features = value; invalidateRun(); persist();
    });
    content.querySelectorAll("[data-move-input]").forEach(button => button.onclick = () => {
      const i = Number(button.dataset.moveInput);
      [n.sources[i-1], n.sources[i]] = [n.sources[i], n.sources[i-1]];
      // Column meaning changed: keep the user from training old data under new labels.
      doc.dataset = null; invalidateRun(); persist();
    });
    if ($("add-output-port")) $("add-output-port").onclick = () => editBoundary(() => G.addRouterPort(doc, n));
    if ($("open-reference")) $("open-reference").onclick = () => safe("openModel", {id: n.ref});
    if ($("tab-reference")) $("tab-reference").onclick = () => safe("openModel", {id: n.ref, newTab: true});
    if ($("remove-node")) $("remove-node").onclick = () => removeNode(n.id);
    $("equation").textContent = n.type === "Activation" ? activationInfo[n.activation][0] : n.type === "Reshape" ? "y = x.reshape(B, …shape)" : n.type === "Concat" ? "x = cat(parts, dim="+(n.flatten===false?"−1":"1")+")" : n.type === "Splitter" ? "parts = split(x, sizes, dim=−1)" : n.type === "Replicator" ? "outᵢ = x" : n.type === "Module" ? "y = module(x)" : n.type === "Linear" ? (n.bias ? "y = x Aᵀ + b" : "y = x Aᵀ") : "ŷ = fθ(x)";
    const activationTex={ReLU:'h=\\max(0,z)',Tanh:'h=\\tanh(z)',Sigmoid:'h=\\frac{1}{1+e^{-z}}',GELU:'h=z\\,\\Phi(z)'};
    const tex=n.type==='Activation'?activationTex[n.activation]:n.type==='Linear'?'y=xA^{\\mathsf T}'+(n.bias?'+b':''):n.type==='Replicator'?'y_i=x':n.type==='Module'?'y=\\operatorname{module}(x)':n.type==='Reshape'?'y=\\operatorname{reshape}(x,B,\\ldots)':n.type==='Concat'?'y=\\operatorname{cat}(x_1,\\ldots)':n.type==='Splitter'?'(y_1,\\ldots)=\\operatorname{split}(x)':'\\hat y=f_\\theta(x)';
    window.LoomMath.set($("equation"),tex).catch(()=>{}); // Keep readable plain formula on failure.
    $("contract-explanation").textContent = n.type === "Linear" ? "Linear mixes the last axis: y = x @ A.T"+(n.bias!==false?" + b.":". Bias is off.")+" All preceding sample dimensions and B are preserved." : "Ports carry float32 tensors [B, …sample dimensions]. Splits use the last axis; reshapes preserve B. Wires name their source ports.";
  }

  function invalidateRun() {
    doc.result = null;
    renderGraph();
    renderResults();
  }
  function insertBlock(type, ref = null) {
    if (!doc || busy()) return;
    let after = node(selected);
    if (!after || after.type === "Output") after = node(doc.graph.edges.find(e => e.to === (after?.id || "output"))?.from) || doc.graph.nodes.find(n => n.type === "Input");
    if (doc.graph.nodes.length >= 64) { toast("Use at most 64 blocks.", true); return; }
    const port=["Splitter","Replicator"].includes(after.type) ? (selectedPort || G.slots(after,true)[0]) : null;
    const edge = doc.graph.edges.find(e => e.from === after.id && (!port || (e.from_port || e.to)===port)), next = edge ? node(edge.to) : null;
    const id = "n_" + Math.random().toString(36).slice(2, 10);
    const block = {id, type, x: after.x + 245, y: after.y};
    if (type === "Linear") Object.assign(block, {features: shape(after)?.output || 1, bias: true});
    else if (type === "Activation") block.activation = $("activation-select").value;
    else if (type === "Reshape") block.shape = shape(after)?.output_shape || [-1];
    else if (type === "Concat") {block.sources=[port?after.id+"@"+port:after.id];block.flatten=true;}
    else block.ref = ref;
    if (next) {
      const move = next.x < block.x + 230 ? block.x + 230 - next.x : 0;
      doc.graph.nodes.filter(n => n.x >= next.x && n.id !== after.id).forEach(n => n.x += move);
      doc.graph.edges = doc.graph.edges.filter(e => e !== edge);
      const incoming={from:after.id,to:id};if(port)incoming.from_port=port;
      doc.graph.edges.push(incoming, {from: id, to: next.id});
      if(next.type==="Concat")next.sources=next.sources.map(src=>src===(port?after.id+"@"+port:after.id)?id:src);
    } else {const incoming={from:after.id,to:id};if(port)incoming.from_port=port;doc.graph.edges.push(incoming);}
    doc.graph.nodes.push(block);
    selected = id;selectedPort=null;
    selectedWire=null;
    invalidateRun();
    persist();
  }
  function removeNode(id) {
    if (!doc || busy()) return;
    if(doc.graph.nodes.length<=2){toast("Keep at least two blocks. Add a replacement before deleting.",true);return;}
    editBoundary(()=>G.removeBlock(doc,id));
  }
  function editBoundary(edit) {
    if (!doc || busy()) return;
    try { selected = edit(); selectedWire=null; doc.dataset = null; invalidateRun(); persist(); }
    catch (error) { toast(error.message, true); }
  }
  const wireFromElement = el => ({from:el.dataset.from,to:el.dataset.to,...(el.dataset.fromPort?{from_port:el.dataset.fromPort}:{})});
  function updateSelection() {
    // Selection must not replace the hit SVG element between consecutive
    // clicks: Chromium otherwise cannot deliver a wire/box double-click.
    document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected',el.dataset.id===selected));
    document.querySelectorAll('.wire[data-from],.wire-end').forEach(el=>{
      const active=G.sameWire(wireFromElement(el),selectedWire);
      el.classList.toggle('selected-wire',active);
      if(el.classList.contains('wire-end'))el.setAttribute('r',active?6:4);
    });
  }
  function selectWire(edge) {
    selectedWire=clone(edge);selected=null;selectedPort=null;connecting=null;
    $('pending-wire').setAttribute('hidden','');updateSelection();renderInspector();$('canvas').focus();
  }
  function deleteSelection() {
    if(!doc || busy() || drag)return;
    if(selectedWire){
      try{G.removeWire(doc,selectedWire);selectedWire=null;invalidateRun();persist();}
      catch(error){toast(error.message,true);}
    } else if(node(selected))removeNode(selected);
  }

  function plot(history) {
    const svg = $("loss-plot");
    svg.replaceChildren();
    svg.append(svgElement("path", {d: "M25 8V101H295", stroke: "var(--line)", fill: "none"}));
    if (!history?.length) {
      svg.append(svgElement("text", {x: 160, y: 60, "text-anchor": "middle", fill: "#68839c", "font-size": 10}, "Loss appears when training begins"));
      return;
    }
    const logs = history.flatMap(p => [Math.log10(Math.max(1e-10,p.train)), Math.log10(Math.max(1e-10,p.test))]);
    const low = Math.min(...logs), high = Math.max(...logs) + .01;
    for (const [key, color] of [["train", "var(--mint)"], ["test", "var(--blue)"]]) {
      const points = history.map(p => {
        const x = 25 + 268 * (p.epoch - 1) / Math.max(1, history[history.length - 1].epoch - 1);
        const y = 98 - 85 * (Math.log10(Math.max(1e-10,p[key])) - low) / (high - low);
        return x + "," + y;
      }).join(" ");
      svg.append(svgElement("polyline", {points, fill: "none", stroke: color, "stroke-width": 1.7}));
    }
    svg.append(svgElement("text", {x: 27, y: 117, fill: "#738da3", "font-size": 8}, "1"));
    svg.append(svgElement("text", {x: 270, y: 117, fill: "#738da3", "font-size": 8}, String(history.at(-1).epoch)));
  }

  function renderExperiment() {
    $("epochs").value = doc.training?.epochs ?? 400;
    $("learning-rate").value = doc.training?.lr ?? .03;
    $("seed").value = doc.training?.seed ?? 7;
    $("data-status").textContent = doc.dataset ? doc.dataset.x.length + " rows" : "No dataset yet";
    const contract = description?.contract;
    if (contract) $("data-help").textContent = "CSV inputs: " + contract.input_groups.map(g => g.name + " [" + g.features + "]").join(" + ") + ". Targets: " + contract.outputs + " packed features. " + (doc.dataset?.truth?.equation || (contract.affine ? "Auto data uses a known affine map." : "Auto data uses a known nonlinear Tanh teacher."));
    $("training-state").textContent = state.training_id === doc.id ? "Training…" : busy() ? "Training another model" : doc.result ? (doc.result.stopped ? "Stopped" : "Complete") : "Ready when data is loaded";
    $("loss-value").textContent = doc.result ? doc.result.mse.toExponential(2) : "—";
    for (const id of ["epochs", "learning-rate", "seed", "sample-count", "noise"]) $(id).disabled = busy();
    const safety = description?.run_safety;
    const blocked = safety?.allowed === false;
    $("safety-summary").textContent = blocked ? "⚠ Run blocked by constrained mode" : state.constrained_mode === false ? "⚠ Unconstrained runs" : "Constrained run checks";
    $("safety-summary").parentElement.classList.toggle("blocked",blocked);
    $("safety-summary").parentElement.open = blocked;
    $("safety-notice").textContent = safety?.violations?.length ?
      safety.violations.map(v=>v.message).join(" ") + (blocked ? " Edit freely; reduce the run or uncheck Constrained." : " Limits are disabled; large runs may exhaust memory or CPU.") :
      (state.constrained_mode === false ? "RAM admission checks are disabled. Stop a run before changing this mode." : "Large specifications remain editable and exportable. Only runs are restricted.");
    const memory=safety?.memory, bytes=n=>n==null ? "unknown" : (n/1024**2).toLocaleString(undefined,{maximumFractionDigits:1})+" MiB";
    $("safety-rules").textContent = memory ? "Estimated run: " + bytes(memory.estimated_bytes) + "; budget: " + bytes(memory.budget_bytes) + " (" + memory.limit_percent + "% of " + bytes(memory.available_bytes) + " available RAM). Whole network, references, activations, optimizer, data and result copies count. Options → Run safety changes the percentage for all workspaces. Revalidate to refresh. Checked again before allocation; this estimate is not a hard memory limit." : "";
    plot(state.training_id === doc.id ? liveHistories[doc.id] : doc.result?.history);
  }
  function showPanel(name) {
    floating.open("experiment-dock");
    panel = name;
    document.querySelectorAll("[data-panel]").forEach(el => el.classList.toggle("active", el.dataset.panel === name));
    for (const p of ["experiment", "code", "results"]) $(p + "-panel").hidden = p !== name;
  }
  function renderResults() {
    const r = doc?.result;
    if (!r) { $("results").textContent = "Train a model to inspect its learned layer weights and predictions."; return; }
    const vector = values => "[ " + values.slice(0,8).map(v=>v.toFixed(5)).join(", ") + (values.length>8 ? ", … ("+values.length+" values)" : "") + " ]";
    const format = values => Array.isArray(values[0]) ?
      values.slice(0,6).map(vector).join("\n") + (values.length>6 ? "\n… ("+values.length+" rows; full weights retained in workspace)" : "") : vector(values);
    const truth = doc.dataset?.truth;
    let html = '<div class="results-summary"><b>Held-out MSE ' + r.mse.toExponential(5) + '</b><span>' + r.train_rows + ' train / ' + r.test_rows + ' held-out · ' + esc(r.engine) + '</span></div>';
    if (r.A && r.b) {
      html += '<div class="results-grid"><section><h3>Effective A · whole affine model</h3><pre>' + esc(format(r.A)) + '</pre>';
      if (r.A_error !== undefined) html += '<p>Max A error: ' + r.A_error.toExponential(3) + '</p>';
      html += '</section><section><h3>Effective b</h3><pre>' + esc(format(r.b)) + '</pre>';
      if (r.b_error !== undefined) html += '<p>Max b error: ' + r.b_error.toExponential(3) + '</p>';
      html += '</section></div>';
    } else html += '<p>'+ (r.kind==="affine"?'This is affine, but its branching or tensor geometry is not represented by the serial-vector A,b report.':'This model is nonlinear: no single matrix A,b describes the full forward map.') +' Inspect the learned layers below.</p>';
    if (truth?.kind === "nonlinear") html += '<p>Known target: ' + esc(truth.equation) + '. Layer weights need not match the teacher; compare predictions and held-out error.</p>';
    html += '<div class="learned-layers">';
    for (const layer of r.layers || []) {
      const weight = r.weights[layer.path + ".weight"], bias = r.weights[layer.path + ".bias"];
      if (!weight) { html += '<div class="learned-activation">' + esc(layer.path) + ' · ' + esc(layer.type) + ' · shape preserved · 0 parameters</div>'; continue; }
      html += '<details><summary>' + esc(layer.architecture) + ' / ' + esc(layer.node_id) + ' · Linear ' + layer.input + ' → ' + layer.output + '</summary><p>' + esc(layer.path) + '.weight · [' + layer.output + ', ' + layer.input + ']</p><pre>' + esc(format(weight)) + '</pre><p>Bias</p><pre>' + (bias ? esc(format(bias)) : 'Disabled') + '</pre></details>';
    }
    html += '</div><h3>Held-out predictions by named output</h3><table class="result-table"><tr><th>Port</th><th>Target</th><th>Prediction</th></tr>';
    const groups = r.output_groups || [{name: "output", start: 0, stop: r.preview[0]?.target.length || 1}];
    for (const row of r.preview) for (const g of groups) html += '<tr><td>' + esc(g.name) + '</td><td>' + esc(format(G.outputValues(row.target,g,r.output_shape,r.output_mode))) + '</td><td>' + esc(format(G.outputValues(row.prediction,g,r.output_shape,r.output_mode))) + '</td></tr>';
    html += '</table>';
    $("results").innerHTML = html;
  }

  async function generateData() {
    if (!description?.valid || busy()) return;
    const count = Number($("sample-count").value), noise = Number($("noise").value), seed = Number($("seed").value);
    if (!Number.isSafeInteger(count) || count < 2 || !Number.isFinite(noise) || noise < 0 || noise > 1 || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
      toast("Use at least 2 samples, noise between 0 and 1, and a nonnegative integer seed.", true); return;
    }
    await syncQueue;
    const generated = await safe("generateData", {id: doc.id, samples: count, noise, seed, kind: $("target-kind").value});
    if (generated) toast("Generated " + generated.rows + " observations from a known " + generated.kind + " target.");
  }

  async function importCSV(file) {
    if (!file || !description?.valid) return;
    if (file.size > 8_000_000) { toast("Use a CSV smaller than 8 MB.", true); return; }
    try {
      const lines = (await file.text()).trim().split(/\r?\n/).filter(line => line.trim());
      let rows = lines.map(line => line.split(",").map(value => value.trim()));
      if (rows[0].some(value => value === "" || !Number.isFinite(Number(value)))) rows.shift();
      const {inputs, outputs} = description.contract;
      if (rows.length < 2 || rows.some(row => row.length !== inputs + outputs || row.some(v => !v || !Number.isFinite(Number(v)) || Math.abs(Number(v)) > 1e6))) throw new Error("CSV needs at least 2 numeric rows: " + inputs + " x columns followed by " + outputs + " y columns. One header row is optional.");
      rows = rows.map(row => row.map(Number));
      doc.dataset = {x: rows.map(row => row.slice(0, inputs)), y: rows.map(row => row.slice(inputs)), source: file.name};
      doc.result = null;
      await persist();
      toast("Imported " + rows.length + " data rows.");
    } catch (error) { toast(error.message, true); }
    $("csv-file").value = "";
  }

  function nameDialog(title, initial = "", buttonLabel = "Create") {
    return new Promise(resolve => {
      $("dialog-title").textContent = title;
      $("name-input").value = initial;
      $("name-dialog").querySelector("button[value=ok]").textContent = buttonLabel;
      $("name-dialog").returnValue = "cancel";
      $("name-dialog").addEventListener("close", () => resolve($("name-dialog").returnValue === "ok" ? $("name-input").value.trim() : null), {once: true});
      $("name-dialog").showModal();
      $("name-input").focus();
    });
  }
  function confirmDialog(title,message,label="Confirm") {
    return new Promise(resolve=>{
      const dialog=$("confirm-dialog");
      $("confirm-title").textContent=title;$("confirm-message").textContent=message;
      $("confirm-ok").textContent=label;dialog.returnValue="cancel";
      dialog.addEventListener("close",()=>resolve(dialog.returnValue==="ok"),{once:true});
      dialog.showModal();dialog.querySelector("button[value=cancel]").focus();
    });
  }

  function selectContextBox(target) {
    const group=target?.closest('.node');
    if(!doc || !group)return false;
    connecting=null;$("pending-wire").setAttribute("hidden", "");
    selected=group.dataset.id;selectedPort=target.classList.contains('port-out') ? target.dataset.peer || null : null;
    selectedWire=null;updateSelection();
    renderInspector();return true;
  }
  function foldBlock(nid,collapsed){
    if(!doc || busy() || !node(nid))return;
    selected=nid;selectedPort=null;selectedWire=null;
    node(nid).collapsed=collapsed;persist();
  }
  $("canvas").addEventListener("click",event=>{
    const icon=event.target.closest('.node-type-icon');
    if(icon && event.button===0 && !event.shiftKey && !node(icon.closest('.node').dataset.id)?.collapsed)
      foldBlock(icon.closest('.node').dataset.id,true);
  });
  $("canvas").addEventListener("pointerdown", event => {
    if(event.button===2){
      const wire=event.target.closest('.wire[data-from],.wire-end');
      if(doc && wire)selectWire(wireFromElement(wire));else selectContextBox(event.target);
      return; // Native menu remains enabled; no drag starts.
    }
    if (!doc || busy() || event.button !== 0 || event.shiftKey) return;
    const target = event.target;
    const blockIcon=target.closest('.node-type-icon');
    // Expanded icons are folding controls; compact icons are also the block's
    // drag surface. Keep the canvas as capture target across SVG redraws.
    if(blockIcon && !node(blockIcon.closest('.node').dataset.id)?.collapsed){selectContextBox(target);event.preventDefault();return;}
    const endpoint=target.closest('.wire-end'),wire=target.closest('.wire');
    if(endpoint){
      const edge=wireFromElement(endpoint);selectWire(edge);
      drag={kind:'rewire',edge,model:doc.id,start:worldPoint(event),moved:false};
      $('canvas').setPointerCapture(event.pointerId);
      document.documentElement.classList.add('loom-dragging');
      document.getSelection()?.removeAllRanges();event.preventDefault();return;
    }
    if(wire){selectWire(wireFromElement(wire));return;}
    if (target.classList.contains("port-out")) {
      selectedWire=null;
      selected=target.dataset.id;selectedPort=target.dataset.peer || null;renderInspector();
      connecting = {id: target.dataset.id, peer: target.dataset.peer || null};
      renderGraph();
      event.stopPropagation();
      return;
    }
    if (target.classList.contains("port-in")) {
      if (connecting && connecting.id !== target.dataset.id) {
        const to = target.dataset.id;
        try { G.connect(doc, connecting.id, to, connecting.peer, target.dataset.peer || null); }
        catch (error) { toast(error.message, true); return; }
        connecting = null;
        $("pending-wire").setAttribute("hidden", "");
        invalidateRun();
        persist();
      }
      return;
    }
    if (connecting) {
      connecting = null;
      $("pending-wire").setAttribute("hidden", "");
      renderGraph();
      return;
    }
    const group = target.closest(".node");
    if (group) {
      selected = group.dataset.id;selectedPort=null;
      selectedWire=null;updateSelection();
      const n=node(selected),resizing=!!target.closest('.node-resize');
      drag = {kind: resizing?'resize':'node', id: selected, start: worldPoint(event), x:n.x, y:n.y,width:G.width(n),height:G.height(n),moved: false,
        deferredCapture:!!n.collapsed,pointerId:event.pointerId};
      // Keep the hit element alive so the browser can deliver a double-click.
      document.querySelectorAll(".node").forEach(el => el.classList.toggle("selected", el.dataset.id === selected));
      $("canvas").focus();
      renderInspector();
    } else {
      drag = {kind: "pan", start: screenPoint(event), x: doc.view.x, y: doc.view.y, moved: false};
    }
    // Capturing a compact icon's stationary click to the canvas would retarget
    // click/dblclick away from the icon. Capture only once movement starts.
    if(!drag.deferredCapture)$("canvas").setPointerCapture(event.pointerId);
    document.documentElement.classList.add(drag.kind==='resize'?'loom-resizing':'loom-dragging');
    document.getSelection()?.removeAllRanges();
    event.preventDefault();
  });
  function moveCanvasPointer(event) {
    if (!doc) return;
    if(drag?.kind==='rewire' && (doc.id!==drag.model || !doc.graph.edges.some(e=>G.sameWire(e,drag.edge))))endDrag();
    if (connecting) {
      $("pending-wire").removeAttribute("hidden");
      $("pending-wire").setAttribute("d", wirePath(portPoint(node(connecting.id), true, connecting.peer), worldPoint(event)));
    }
    if (!drag) return;
    const point = drag.kind !== "pan" ? worldPoint(event) : screenPoint(event);
    const dx = point.x - drag.start.x, dy = point.y - drag.start.y;
    if (Math.abs(dx) + Math.abs(dy) < 2 && !drag.moved) return;
    if(drag.deferredCapture){$("canvas").setPointerCapture(drag.pointerId);drag.deferredCapture=false;}
    drag.moved = true;
    if(drag.kind==='rewire'){
      $('pending-wire').removeAttribute('hidden');
      $('pending-wire').setAttribute('d',wirePath(portPoint(node(drag.edge.from),true,drag.edge.from_port || null),point));
      $('status').textContent='Reconnect destination · drop on an empty input socket · Escape cancels';
    } else if (drag.kind === 'resize') {
      G.resizeBlock(node(drag.id),drag.width+dx,drag.height+dy,event.ctrlKey);
      $("status").textContent=event.ctrlKey?'Resize · bottom-right snaps to grid':'Resize block · hold Ctrl to snap';
      renderGraph();
    } else if (drag.kind === "node") {
      const position = G.snapPosition(drag.x + dx, drag.y + dy, event.ctrlKey);
      Object.assign(node(drag.id), position);
      $("status").textContent = event.ctrlKey ? "Snap to grid · 24 model units" : "Free drag · hold Ctrl to snap";
      renderGraph();
    } else {
      doc.view.x = drag.x + dx;
      doc.view.y = drag.y + dy;
      updateViewport();
    }
  }
  $("canvas").addEventListener("pointermove", moveCanvasPointer);
  window.addEventListener('pointermove',event=>{
    if(drag?.deferredCapture && !$('canvas').contains(event.target))moveCanvasPointer(event);
  });
  function endDrag(event) {
    const ended=drag;
    drag = null;
    document.documentElement.classList.remove('loom-dragging','loom-resizing');
    if(ended?.kind==='rewire'){
      $('pending-wire').setAttribute('hidden','');
      $('status').textContent='Connection unchanged.';
      if(ended.moved && event?.type==='pointerup' && doc?.id===ended.model && !busy()){
        // Pointer capture retargets event.target to the canvas; hit-test the
        // actual screen location instead. No graph edit occurs before release.
        const target=document.elementFromPoint(event.clientX,event.clientY)?.closest('.port-in');
        if(target){
          try{
            if(G.retargetWire(doc,ended.edge,target.dataset.id,target.dataset.peer || null)){
              selectedWire={...ended.edge,to:target.dataset.id};invalidateRun();persist();
            }
          }catch(error){toast(error.message,true);}
        }
      }
      renderGraph();
    } else if (ended?.moved) persist();
  }
  $("canvas").addEventListener("pointerup", endDrag);
  window.addEventListener('pointerup',event=>{if(drag?.deferredCapture)endDrag(event);});
  $("canvas").addEventListener("pointercancel", endDrag);
  $("canvas").addEventListener("lostpointercapture", endDrag);
  window.addEventListener('blur',endDrag);
  $("canvas").addEventListener("dblclick", event => {
    if (!doc) return;
    if(event.target.closest('.node-resize')){event.preventDefault();return;}
    const group = event.target.closest(".node"), path = event.target.closest(".wire");
    if (group) {
      const n = node(group.dataset.id);
      if(event.target.closest('.node-type-icon')){if(n.collapsed)foldBlock(n.id,false);event.preventDefault();return;}
      selected=n.id;renderInspector();floating.open("inspector-panel");
      selectedWire=null;
      if (n.type === "Module" && (event.ctrlKey || event.metaKey)) safe("openModel", {id: n.ref, newTab: true});
    } else if (path?.dataset.from && !busy()) {
      selectWire(wireFromElement(path));deleteSelection();
    }
  });
  $("canvas").addEventListener("wheel", event => {
    if (!doc) return;
    event.preventDefault();
    zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12, screenPoint(event));
  }, {passive: false});
  document.addEventListener("keydown", event => {
    if(document.querySelector('dialog[open]'))return;
    const editing=event.target.isContentEditable || event.target.closest('input,textarea,select,[contenteditable]');
    const blockIcon=event.target.closest('.node-type-icon');
    if(blockIcon && ['Enter',' '].includes(event.key) && !event.repeat && !event.ctrlKey && !event.altKey && !event.metaKey){
      event.preventDefault();const nid=blockIcon.closest('.node').dataset.id;foldBlock(nid,!node(nid).collapsed);return;
    }
    const grip=event.target.closest('.node-resize'),delta={ArrowRight:[8,0],ArrowLeft:[-8,0],ArrowDown:[0,8],ArrowUp:[0,-8]}[event.key];
    if(grip && delta && doc && !busy()){
      event.preventDefault();const n=node(grip.closest('.node').dataset.id),factor=event.shiftKey || event.ctrlKey?3:1;
      G.resizeBlock(n,G.width(n)+delta[0]*factor,G.height(n)+delta[1]*factor,event.ctrlKey);
      persist().then(()=>document.querySelector('.node[data-id="'+CSS.escape(n.id)+'"] .node-resize')?.focus());return;
    }
    if(!editing && !event.altKey && (event.ctrlKey || event.metaKey)) {
      const key=event.key.toLowerCase();
      if(key==='z' || key==='y') {
        event.preventDefault();if(!event.repeat)editHistory(key==='y' || event.shiftKey);return;
      }
    }
    const panel=LoomPanels.shortcutPanel(event);
    if(doc && panel){event.preventDefault();floating.toggle(panel);return;}
    if(!editing && !event.ctrlKey && !event.metaKey && !event.altKey &&
      (event.key.toLowerCase()==='d' || ((event.key==='Delete' || event.key==='Backspace') && event.target===$('canvas')))){
      if(doc){event.preventDefault();if(!event.repeat)deleteSelection();}return;
    }
    if (event.key === "Escape") {
      if(editing || event.ctrlKey || event.altKey || event.metaKey || event.repeat)return;
      event.preventDefault();
      if(drag){endDrag();return;}
      if(connecting){
        connecting=null;$("pending-wire").setAttribute("hidden","");
        if(doc)renderGraph();return;
      }
      if(selected || selectedWire){
        selected=null;selectedPort=null;selectedWire=null;
        updateSelection();renderInspector();
      }else floating.hideAll();
    }
  });
  async function editHistory(redo=false) {
    if(busy())return;
    await syncQueue;
    const changed=await safe(redo?'redo':'undo');
    if(!changed)toast(redo?'Nothing to redo.':'Nothing to undo.');
  }

  $("home").onclick = () => safe("home");
  $("back").onclick = () => history.back();
  $("forward").onclick = () => history.forward();
  $("new-tab").onclick = () => doc && safe("openModel", {id: doc.id, newTab: true});
  $("find").onclick = () => safe("find");
  $("new-workspace").onclick = () => safe("newWorkspace");
  $("open-workspace").onclick = $("welcome-open").onclick = () => safe("openWorkspace");
  $("save-workspace").onclick = async () => { await syncQueue; const ok = await safe("save"); if (ok) toast("Workspace saved."); };
  $("constrained-mode").onchange = async () => {
    const enabled=$("constrained-mode").checked;
    header();
    if (!enabled && !await confirmDialog("Disable constrained mode?",
      "Large runs may exhaust memory or CPU. This setting is saved with the workspace. You can re-enable it later without changing your models.","Disable limits")) return;
    await syncQueue;
    await safe("setConstrainedMode",{enabled});
    header();
  };
  $("reset-panels").onclick = () => {floating.reset();toast("Panels hidden and menu pack restored. Choose a menu to open it.");};
  $("resume-workspace").onclick = () => safe("openModel", {id: Object.keys(state.models)[0]});
  $("add-model").onclick = async () => { const name = await nameDialog("New architecture"); if (name) safe("newModel", {name}); };
  $("rename-model").onclick = async () => { const name = await nameDialog("Rename architecture", doc.name, "Rename"); if (name) { doc.name = name; await persist(); renderLibrary(); } };
  $("add-linear").onclick = () => insertBlock("Linear");
  $("add-reshape").onclick = () => insertBlock("Reshape");
  $("add-concat").onclick = () => insertBlock("Concat");
  $("add-activation").onclick = () => insertBlock("Activation");
  $("add-reference").onclick = () => insertBlock("Module", $("reference-select").value);
  $("add-input").onclick = () => editBoundary(() => G.addInput(doc,true));
  $("add-output").onclick = () => editBoundary(() => G.addOutput(doc,selected,selectedPort));
  function addSelectedRouter(type) {
    let after=node(selected);if(after?.type==="Output")after=node(doc.graph.edges.find(e=>e.to===after.id)?.from);
    if(!after || ["Splitter","Replicator"].includes(after.type)){toast("Select a tensor-producing box, not a router.",true);return;}
    editBoundary(()=>G.addRouter(doc,type,shape(after)?.output || 1,after.id));
  }
  $("add-splitter").onclick = () => addSelectedRouter("Splitter");
  $("add-replicator").onclick = () => addSelectedRouter("Replicator");
  $("fit").onclick = fit;
  $("zoom-in").onclick = () => zoom(1.15);
  $("zoom-out").onclick = () => zoom(1 / 1.15);
  $("validate").onclick = async () => { await syncQueue; await describe(); if (description?.valid) toast("Valid graph. Every block maps to a PyTorch operation."); };
  $("generate-data").onclick = generateData;
  $("import-data").onclick = () => $("csv-file").click();
  $("csv-file").onchange = event => importCSV(event.target.files[0]);
  $("train").onclick = async () => {
    doc.training = {epochs: Number($("epochs").value), lr: Number($("learning-rate").value), seed: Number($("seed").value)};
    doc.result = null;
    liveHistories[doc.id] = [];
    await persist();
    await safe("train", {id: doc.id});
  };
  $("stop").onclick = () => safe("stop");
  for (const [id, key] of [["epochs", "epochs"], ["learning-rate", "lr"], ["seed", "seed"]]) {
    $(id).onchange = () => {
      if (!doc || busy()) return;
      const value = Number($(id).value);
      const valid = key === "lr" ? Number.isFinite(value) && value > 0 && value <= 1 :
        Number.isSafeInteger(value) && value >= (key === "epochs" ? 1 : 0) && (key === "epochs" || value <= 2147483647);
      if (!valid) { toast("The training setting is outside its allowed range.", true); renderExperiment(); return; }
      doc.training[key] = value;
      persist();
    };
  }
  $("export-code").onclick = async () => { await syncQueue; if (await safe("exportCode", {id: doc.id})) toast("PyTorch module exported."); };
  document.querySelectorAll("[data-panel]").forEach(button => button.onclick = () => showPanel(button.dataset.panel));
  window.addEventListener("hashchange", route);

  function onEvent(raw) {
    const event = JSON.parse(raw);
    if (event.kind === "state") hydrate(event.payload);
    else if(event.kind==='history'){state.history=event.payload;if(event.payload.notice)toast(event.payload.notice,true);}
    else if (event.kind === "navigate") navigate(event.payload);
    else if (event.kind === "progress" && doc?.id === event.payload.id) {
      const point = event.payload;
      liveHistories[doc.id] ||= [];
      liveHistories[doc.id].push(point);
      plot(liveHistories[doc.id]);
      $("training-state").textContent = "Epoch " + point.epoch + " / " + doc.training.epochs;
      $("loss-value").textContent = point.test.toExponential(2);
    } else if (event.kind === "complete") {
      if (doc?.id === event.payload.id) {
        doc.result = event.payload.result;
        state.models[doc.id] = clone(doc);
        renderExperiment();
        renderResults();
      }
      toast(event.payload.result.stopped ? "Training stopped; partial result recorded." : "Training complete. Inspect Learned parameters and predictions.");
    } else if (event.kind === "trainingError" || event.kind === "externalConflict") toast(event.payload, true);
    else if (event.kind === "externalReload") toast(event.payload);
  }

  async function initialize() {
    new QWebChannel(qt.webChannelTransport, async channel => {
      host = channel.objects.host;
      host.event.connect(onEvent);
      state = await call("state");
      window.loom = {call, get document() { return clone(doc); }, get state() { return clone(state); }, fit,
        selectAt(x,y){return selectContextBox(document.elementFromPoint(x,y));},
        contextAt(x,y){
          const target=document.elementFromPoint(x,y),wire=target?.closest('.wire[data-from],.wire-end');
          if(doc && wire){selectWire(wireFromElement(wire));return null;}
          if(!selectContextBox(target))return null;
          return {id:selected,modelId:doc.id,canDelete:!busy() && !!node(selected) && doc.graph.nodes.length>2};
        },
        perform(command, blockId=null, modelId=null) {
          if(command==='undo' || command==='redo'){editHistory(command==='redo');return;}
          if(command==="reset-panels"){floating.reset();return;}
          if(!doc)return;
          if(command==="inspect-block"){
            if(doc.id!==modelId || !node(blockId))return;
            selected=blockId;selectedPort=null;selectedWire=null;
            updateSelection();renderInspector();floating.open("inspector-panel");
          }
          else if(command==="delete-block"){if(doc.id===modelId && node(blockId))removeNode(blockId);}
          else if(command==="fit")fit();
          else if(command==="validate")$("validate").click();
          else if(command==="library")floating.open("library-panel");
          else if(command==="architectures")floating.open("architectures-panel");
          else if(command==="inspector")floating.open("inspector-panel");
          else if(command==="experiment")showPanel("experiment");
          else if(command==="code")showPanel("code");
          else if(command==="results")showPanel("results");
        }};
      await route();
    });
  }
  if (window.qt && window.QWebChannel) initialize();
  else {
    $("status").textContent = "Open this editor through the Visual PyTorch desktop launcher";
    $("runtime").textContent = "Desktop webview required";
    for (const id of ["new-workspace", "open-workspace", "welcome-open"]) $(id).disabled = true;
  }
  $("dots").querySelector("circle").setAttribute("cx", 0);
  document.querySelectorAll('.palette').forEach(button=>{
    const name=button.querySelector('b')?.textContent.trim() || button.childNodes[0]?.textContent.trim();
    const description=button.id==='add-reference' ? 'Inspect its interface; open details from Inspector.' : button.querySelector('small')?.textContent.trim();
    if(description){button.title=description;button.setAttribute('aria-label',name+'. '+description);}
  });
  // Sort only browsing groups. Never reorder graph nodes, tensor ports or recents.
  const library=document.querySelector('#library-scroll .sidebar-section'),boundaryHeading=$('add-input').previousElementSibling;
  const buildingGroups=[[$('activation-select').previousElementSibling,$('activation-select'),$('add-activation')],
    [$('reference-select').previousElementSibling,$('reference-select'),$('add-reference')],[$('add-concat')],[$('add-linear')],[$('add-reshape')]];
  buildingGroups.sort((a,b)=>a.at(-1).querySelector('b').textContent.localeCompare(b.at(-1).querySelector('b').textContent));
  buildingGroups.flat().forEach(element=>library.insertBefore(element,boundaryHeading));
  ['add-input','add-output','add-replicator','add-splitter'].map($).sort((a,b)=>a.querySelector('b').textContent.localeCompare(b.querySelector('b').textContent)).forEach(button=>library.append(button));
  $("dots").querySelector("circle").setAttribute("cy", 0);
  document.querySelector(".canvas-note").textContent = "D: delete selection · drag arrow ends: reconnect · Ctrl: snap · wheel: zoom · empty drag: pan · Shift-drag: menus";
})();
