"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const SVG = "http://www.w3.org/2000/svg";
  const G = window.LoomGraph;
  const L = window.LoomLayout;
  const T = window.LoomTensorOperators;
  const R = window.LoomRecurrent;
  const snapEnabled = event => window.LoomPanels.snapEnabled(event,document.documentElement.dataset.gridSnap==='true');
  const floating = window.LoomPanels.initialize(document,()=>document.documentElement.dataset.menuGridSnap==='true');
  const activationInfo = window.LoomActivations.info;
  // Building blocks and the Inspector share one grouped activation list.
  $('activation-select').innerHTML=window.LoomActivations.options('ReLU');
  $('mlp-activation').innerHTML=window.LoomActivations.options('Tanh');
  let host, state = {models: {}, recent: []}, doc = null, description = null;
  let selected = null, selectedPort = null, selectedWire = null, connecting = null, drag = null, panel = "experiment";
  let syncQueue = Promise.resolve(), describeGeneration = 0;
  let panelWorkspace, panelSavePending=false;
  let parameterSizingPending=false;
  let recurrentAuthoringPending=false;
  let lstmUI=null;
  let recurrentEditGeneration=0;
  let policyGeneration=0;
  let pendingPolicyOwner=null;
  let policyMutation=null;
  let policyDraft=null;
  let clearMenusOnRoute=new URLSearchParams(location.search).get('menus')==='hidden';
  const notifications=window.LoomNotifications.initialize(document);
  const liveHistories = {};
  const clone = obj => window.LoomJSON.clone(obj);
  const esc = str => String(str).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
  const modelID = () => location.hash.startsWith("#model=") ? location.hash.slice(7) : null;
  const node = id => doc?.graph.nodes.find(n => n.id === id);
  const busy = () => Boolean(state.training_id || state.experiment_job || state.notebook_job);
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
      host.request(window.LoomJSON.stringify({action, ...args}), answer => {
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
    document.documentElement.dataset.gridSnap=String(state.display?.grid_snap===true);
    document.documentElement.dataset.menuGridSnap=String(state.display?.menu_grid_snap===true);
    document.documentElement.dataset.theme=state.theme || 'dark';
    document.documentElement.dataset.spacing=String(state.display?.spacing ?? 4);
    document.documentElement.style.setProperty('--menu-unit',(state.display?.spacing ?? 4)+'px');
    document.documentElement.style.colorScheme=state.theme_scheme || (state.theme==='light' ? 'light' : 'dark');
    Object.entries(state.theme_tokens || {}).forEach(([key,value])=>document.documentElement.style.setProperty('--'+key,value));
    document.documentElement.style.setProperty("--ui-font-size",(state.display?.font_size || 14)+"px");
    $("file-name").textContent = state.path ? state.path.split(/[\\/]/).pop() : (Object.keys(state.models).length ? "Untitled.vtw" : "Welcome");
    $("file-name").title=$("file-name").textContent;
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
    for(const id of ['minimize-all','expand-all'])$(id).disabled=!doc || busy() || parameterSizingPending;
    $('save-panel-arrangement').disabled=!doc||panelSavePending;
    $('restore-panel-arrangement').disabled=!doc||panelSavePending;
  }
  function restoreWorkspacePanels(force=false) {
    if(!doc||!state.workspace_id)return;
    if(force||panelWorkspace!==state.panel_scope){floating.apply(state.panel_layout??null);panelWorkspace=state.panel_scope;}
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
    if(doc)doc.view={x:0,y:0,zoom:1}; // Saved legacy view is not the initial viewport.
    $("welcome").hidden = Boolean(doc);
    $("editor").hidden = !doc;
    description = null;
    connecting = null;
    selectedWire = null;
    header();
    restoreWorkspacePanels();
    recentFiles();
    if (!doc) {
      $("status").textContent = "Open a workspace to start building";
      return;
    }
    selected = doc.graph.nodes.find(n => n.type === "Linear" || n.type === "Module")?.id || doc.graph.nodes[0]?.id;
    if(pendingPolicyOwner?.model===id){if(node(pendingPolicyOwner.node))selected=pendingPolicyOwner.node;pendingPolicyOwner=null;}
    document.title = doc.name + " · Visual PyTorch";
    render();
    if(clearMenusOnRoute){clearMenusOnRoute=false;selected=null;selectedPort=null;selectedWire=null;connecting=null;updateSelection();renderInspector();floating.hideAll();}
    await describe();
    if (!doc.view.fitted) requestAnimationFrame(() => { if (doc?.id === id) fit(); });
  }

  function hydrate(value) {
    if(state.panel_scope!==value.panel_scope)pendingPolicyOwner=null;
    const viewport=doc&&state.panel_scope===value.panel_scope?clone(doc.view):null;
    const layoutChanged=state.workspace_id!==value.workspace_id ||
      JSON.stringify(state.layout_preferences)!==JSON.stringify(value.layout_preferences);
    state = value;
    window.LoomScience?.stateChanged();
    if(layoutChanged)renderLayoutPreferences();
    if (doc && state.models[doc.id]) {
      doc = prepareModel(clone(state.models[doc.id]));
      if(selectedPort!==null && (!node(selected) || !G.appendTarget(node(selected),selected,selectedPort)))selectedPort=null;
      doc.view=viewport || {x:0,y:0,zoom:1};
      render();
      describe().then(()=>{if(!viewport&&doc)fit();});
    } else if (doc && !state.models[doc.id]) {
      navigate(null);
    }
    header();
    restoreWorkspacePanels();
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
      const open=async newTab=>{if(await safe('openModel',{id:model.id,newTab,clearMenus:true})){selected=null;selectedPort=null;selectedWire=null;connecting=null;updateSelection();renderInspector();floating.hideAll();}};
      button.onclick = event => open(event.ctrlKey || event.metaKey || Boolean(event.target.closest(".tab-hint")));
      button.ondblclick = () => open(true);
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
    $('add-recurrent').disabled = !$('reference-select').value || busy() || recurrentAuthoringPending;
    $('add-lstm').disabled = !doc || busy() || Boolean(lstmUI?.pending());
  }

  function renderBookkeeping() {
    const card=$("architecture-bookkeeping");card.replaceChildren();
    if(!doc){card.hidden=true;return;}card.hidden=false;
    const heading=document.createElement('span');heading.className='eyebrow';heading.textContent='CURRENTLY OPEN';
    const name=document.createElement('b');name.textContent=doc.name;
    const stats=document.createElement('p');
    const refs=doc.graph.nodes.filter(n=>['Module','Recurrent'].includes(n.type)).length;
    const ins=doc.graph.nodes.filter(n=>n.type==='Input').length,outs=doc.graph.nodes.filter(n=>n.type==='Output').length;
    stats.textContent=doc.graph.nodes.length+' blocks · '+doc.graph.edges.length+' wires · '+ins+' inputs · '+outs+' outputs'+(refs?' · '+refs+' references':'');
    const contract=document.createElement('p');
    const policyCounts=description?.contract?.parameter_counts;
    contract.textContent=description?.valid ? (policyCounts?policyCounts.trainable.toLocaleString()+' trainable · '+policyCounts.frozen.toLocaleString()+' frozen · '+policyCounts.stored.toLocaleString()+' stored':description.contract.parameters.toLocaleString()+' parameters')+' · '+shapeText(description.contract.input_shape)+' → '+shapeText(description.contract.output_shape) : description ? 'Draft: '+description.error : 'Checking graph…';
    contract.className='bookkeeping-contract';
    card.append(heading,name,stats,contract);
  }

  function render() {
    if (!doc) return;
    $("model-title").textContent = doc.name;
    $("model-title").title=doc.name;
    renderLibrary();
    renderGraph();
    renderInspector();
    renderExperiment();
    renderResults();
    const running = busy();
    $("train").disabled = running || !description?.valid || description?.contract?.has_gaussian_filter || !description?.contract?.parameters || !doc.dataset || description?.run_safety?.allowed === false || description?.dataset_admission?.allowed === false;
    $("stop").hidden = !running;
    $("add-linear").disabled = running;
    $("add-model").disabled = running;
    $("generate-data").disabled = running || !description?.valid || description?.contract?.has_gaussian_filter || !description?.contract?.inputs || description?.model_safety?.allowed === false;
    $("import-data").disabled = running || !description?.valid;
    $("rename-model").disabled = running;
    for (const id of ["add-activation", "activation-select", "target-kind", "add-reshape", "add-concat", "add-input", "add-output",'add-constant','add-tensor-parameter','add-gaussian']) $(id).disabled = running;
    $("add-splitter").disabled = running || !description?.valid;
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
    const problems=description.valid?[]:description.problems||[],first=problems[0]?problemText(problems[0]):description.error;
    $("validation-message").textContent = description.valid ?
      "✓ Valid · " + shapeText(description.contract.input_shape) + " → " + shapeText(description.contract.output_shape) :
      "⚠ " + first + (problems.length>1?" · +"+(problems.length-1)+" more":"");
    $("validation-message").classList.toggle("locatable", Boolean(problems.length));
    if(problems.length){$("validation-message").setAttribute("role","button");$("validation-message").tabIndex=0;}
    else{$("validation-message").removeAttribute("role");$("validation-message").removeAttribute("tabindex");}
    $("validation-message").title=problems.length?"Click to select "+(problems.length>1?"the first problem block":"the problem block"):"";
    $("validation-message").classList.toggle("error", !description.valid);
    const blocked = description.model_safety?.allowed === false;
    $("validation-message").classList.toggle("warning", blocked);
    if (description.valid && blocked) $("validation-message").textContent += " · runs blocked by safety";
    renderExportSource();
    $("status").textContent = description.valid ? "Graph valid · " + description.contract.parameters.toLocaleString() + " parameters · " + (blocked ? "constrained runs blocked" : "PyTorch ready") : first;
    render();
  }

  function persist() {
    if (!doc) return Promise.resolve();
    let snapshot = clone(doc);const mid = doc.id,policyEdit=policyMutation?.id===mid?policyMutation:null;
    snapshot.view={x:0,y:0,zoom:1}; // Keep viewport state out of persisted edits/history.
    state.models[mid] = clone(snapshot);
    state.dirty = true;
    header();
    syncQueue = syncQueue.then(async () => {
      if(policyEdit?.document){
        if(state.panel_scope!==policyEdit.scope)return;
        snapshot=window.LoomParameterPolicy.reconcile(snapshot,policyEdit.document);
      }
      const answer = await safe("updateModel", {document: snapshot,preserve_parameter_policy:true});
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
  // Invalid drafts still show every determinable shape (display only; never used to run).
  function shape(n) { return description?.valid ? description.contract.shapes[n.id] : description?.partial_shapes?.[n.id]; }
  const problemNodes=()=>new Set((description?.valid?[]:description?.problems||[]).map(p=>p.node));
  function problemText(p){
    const n=node(p.node);if(!n)return p.message;
    const name=['Input','Output'].includes(n.type)?n.type+" '"+title(n)+"'":displayName(n.id);
    const count=doc.graph.edges.filter(e=>e.to===n.id).length;
    return {no_input:name+' has no incoming wire.',extra_inputs:name+' has '+count+' incoming wires; use Concat or Add to join branches.',
      no_output_path:name+' has no outgoing wire; connect it toward an Output or remove it.'}[p.code]||p.message;
  }
  // The selected box is drawn last, above overlapping boxes; stored node order never changes.
  // Never move the hit element while a pointer is down: Chromium would drop the click.
  let pointerDown=false;
  function raiseSelected(){
    if(pointerDown)return;
    const layer=$("nodes"),group=[...layer.children].find(el=>el.dataset.id===selected);
    if(group&&group!==layer.lastElementChild)layer.appendChild(group);
  }
  function sourceShape(key){
    const edge=doc?.graph.edges.find(e=>(e.from_port?e.from+'@'+e.from_port:e.from)===key);
    const from=node(edge?.from);return from?G.tensorShape(from,shape(from),edge.from_port||null):null;
  }
  function title(n) {
    if (!n) return "Missing port";
    if (n.label) return n.label;
    // A port box shows exactly its port name, which falls back to the port ID.
    if (["Input","Output"].includes(n.type)) return n.id;
    if (["Module","Recurrent"].includes(n.type)) return state.models[n.ref]?.name || "Missing architecture";
    if (n.type === "Activation") return n.activation;
    if (n.type === "Concat") return "Join branches";
    if (n.type === "Add") return "Add branches";
    return n.type;
  }
  const portPoint = G.portPoint;
  // Inspector sub-panel fold state: session-only, never saved or undoable.
  const subOpen=new Map();
  // Human-readable block names for the Inspector; internal IDs are never shown.
  function displayName(id){
    const target=node(id);if(!target)return 'Missing block';
    const base=m=>{const s=shape(m);return title(m)+(m.type==='Linear'&&s?.input_shape&&s?.output_shape?' '+s.input_shape.at(-1)+' → '+s.output_shape.at(-1):'');};
    const name=base(target),same=doc.graph.nodes.filter(m=>base(m)===name);
    return same.length>1?name+' ('+(same.indexOf(target)+1)+')':name;
  }
  function blockParameters(n,s){
    if(n.type==='Linear')return linearParameters(n,s);
    if(n.type==='LSTM')return s?.lstm?.parameters;
    if(n.type==='Recurrent')return s?.recurrent?.contract?.parameters;
    if(n.type==='TensorParameter')return (n.shape||[]).reduce((a,b)=>a*b,1);
    return s?.operator?.parameters;
  }
  function summaryLine(n,s){
    const io=n.type==='Input'?shapeText(n.shape||[n.features]):s?.input_shape&&s?.output_shape?shapeText(s.input_shape)+' → '+shapeText(s.output_shape):s?.output_shape?shapeText(s.output_shape):null;
    const count=blockParameters(n,s),params=typeof count==='number'&&count>0?count.toLocaleString()+' parameter'+(count===1?'':'s'):null;
    return [io,params].filter(Boolean).join(' · ');
  }
  function blockHelp(n){
    const id={Linear:'add-linear',Activation:'add-activation',Reshape:'add-reshape',Add:'add-add',Concat:'add-concat',Module:'add-reference',
      Recurrent:'add-recurrent',Input:'add-input',Output:'add-output',Replicator:'add-replicator',Splitter:'add-splitter',LSTM:'add-lstm',
      Constant:'add-constant',TensorParameter:'add-tensor-parameter',GaussianFilter:'add-gaussian',CausalLagWindow:'block-causal-lag'}[n.type];
    return window.LoomBlockHelp.describe(T.types.includes(n.type)?'add-tensor-'+n.type:id,
      {tensorType:T.types.includes(n.type)?n.type:undefined,activation:n.activation,referenceName:state.models[n.ref]?.name});
  }
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
    const problems=problemNodes();
    for (const n of doc.graph.nodes) {
      const s = shape(n);
      const group = svgElement("g", {class: "node " + n.type + (n.collapsed ? " collapsed" : "") + (n.id === selected ? " selected" : "") + (problems.has(n.id) ? " problem" : ""), transform: "translate(" + n.x + "," + n.y + ")", "data-id": n.id});
      const height=G.height(n),m={...G.metrics(),width:G.width(n)};
      const shortened=(text,charWidth,reserve=0)=>{const max=Math.floor((m.width-2*m.padding-reserve)/charWidth);return text.length>max?text.slice(0,max-1)+'…':text;};
      group.append(svgElement("rect", {width:m.width,height,rx:7}));
      group.append(svgElement("text", {x:m.padding,y:m.headerY,class:"node-header"}, n.type === "Module" ? "ARCHITECTURE ↗" : n.type.toUpperCase()));
      const icon=svgElement('g',{class:'node-type-icon',transform:n.collapsed?'translate('+(m.width/2-10)+',4) scale(0.833333)':'translate('+(m.width-m.padding-20)+','+Math.max(2,m.padding-7)+') scale(0.833333)',role:'button',tabindex:busy()?-1:0,'aria-expanded':String(!n.collapsed),'aria-disabled':String(busy()),'aria-label':n.collapsed?'Expand '+title(n)+' block (double-click or Enter)':'Collapse '+title(n)+' block'});
      icon.innerHTML=window.LoomTypeIcons.svg(n.type);
      icon.append(svgElement('rect',{class:'icon-hit',x:-3,y:-3,width:30,height:30,rx:3}));
      icon.append(svgElement('title',{},n.type+' block · '+(n.collapsed?'Drag to move · double-click to expand':'Click to collapse')));group.append(icon);
      const label = title(n);
      const labelText=svgElement("text",{x:m.padding,y:m.titleY,class:"node-title"},shortened(label,7.8));
      labelText.append(svgElement('title',{},label));group.append(labelText);
      let line;
      if (['Input','Constant','TensorParameter'].includes(n.type)) line = shapeText(n.shape || [n.features]);
      else if (n.type === "Output") line = shapeText(s?.output_shape || n.shape);
      else if(R.multiport(n))line=R.sources(n).length+' inputs → '+R.ports(n).length+' ports · T '+(s?.recurrent?.steps??s?.lstm?.steps??s?.gaussian?.steps??'?');
      else line = shapeText(s?.input_shape) + " → " + shapeText(s?.output_shape);
      const shapeLine=svgElement("text", {x:m.padding,y:m.shapeY,class:"node-shape"},shortened(line,6.2));
      shapeLine.append(svgElement("title",{},line));group.append(shapeLine);
      const note = n.type === 'Recurrent' ? 'Shared across T · '+(s?.recurrent?.contract?.parameters??'?')+' params' : n.type === "Conv2d" ? 'Bias '+(n.bias!==false?'on':'off')+' · '+(s?.operator?.parameters ?? '?')+' params · CHW' : T.types.includes(n.type) ? 'Batch preserved · 0 params' : n.type === "Linear" ? ('Bias '+(n.bias!==false?'on':'off')+' · '+linearParameters(n,s)+" params · last axis") : n.type === "Reshape" ? "Same elements · batch preserved" : n.type === "Module" ? "Inspect · Ctrl-double-click: open" : n.type === "Concat" ? (n.flatten===false?"Join last axis":"Flatten parts + join") : n.type === "Splitter" ? "Split last axis · branches" : n.type === "Replicator" ? "Shared tensor · branch ports" : n.type === "Activation" ? (['Softmax','LogSoftmax'].includes(n.activation)?'Normalize axis '+(n.dim??-1)+' · 0 params':"Elementwise · 0 params") : "Named tensor · float32";
      const arithmeticNote=n.type==='GaussianFilter'?'Exact filter · initial before transition · 0 params':n.type==='PositiveVariance'?'Explicit floor '+n.positive_floor+' · 0 params':n.type==='Constant'?'Immutable · float32 · no data input':n.type==='TensorParameter'?'Shared across batch · '+n.shape.reduce((a,b)=>a*b,1)+' params':n.type==='Add'?'Ordered sum · '+n.sources.length+' inputs · 0 params':n.type==='Scale'?'Fixed × '+n.coefficient+' · 0 params':note;
      const lstmNote=n.type==='LSTM'?'Bias '+(n.bias!==false?'on':'off')+' · '+(s?.lstm?.parameters??'?')+' params · '+(n.bidirectional?'bidirectional':'forward'):arithmeticNote;
      const sharingNote=n.share_with?(['Linear','Conv2d','LSTM'].includes(n.type)?'Bias '+(n.bias!==false?'on':'off')+' · ':'')+'Shared: '+(s?.parameter_owner || n.share_with)+' · '+lstmNote:lstmNote;
      const policy=description?.parameter_blocks?.[n.id];
      const policyState=policy?(policy.frozen===0?'Trainable':policy.trainable===0?'Frozen':'Mixed'):'',policyNote=policyState?(sharingNote.startsWith('Bias ')?sharingNote.replace(' · ',' · '+policyState+' · '):policyState+' · '+sharingNote):sharingNote;
      group.dataset.policyStatus=policyState;
      const noteText=svgElement("text",{x:m.padding,y:height-m.padding-m.footerInset,class:"node-params"},shortened(policyNote,4.9,14));
      noteText.append(svgElement('title',{},policyNote));group.append(noteText);
      if(n.collapsed){
        group.querySelectorAll('.node-header,.node-title,.node-shape,.node-params').forEach(el=>el.remove());
        const compact=n.type==='Linear'?(s?.input_shape?.at(-1) || '?')+' → '+n.features:
          n.type==='Conv2d'?(s?.input_shape?.[0] || '?')+' → '+n.out_channels+' · '+(Array.isArray(n.kernel_size)?n.kernel_size.join('×'):n.kernel_size+'×'+n.kernel_size):
          T.types.includes(n.type)?shapeText(s?.output_shape):
          n.type==='Activation'?(['Softmax','LogSoftmax'].includes(n.activation)?'Softmax '+(n.dim??-1):n.activation):['Input','Output','Reshape','Constant','TensorParameter'].includes(n.type)?line:
          R.multiport(n)?'T '+(s?.recurrent?.steps??s?.lstm?.steps??'?')+' · '+R.ports(n).length+' ports':n.type==='Splitter'?'Split · '+n.parts.length:n.type==='Replicator'?'Repeat · '+n.outputs.length:label;
        const small=svgElement('text',{x:m.width/2,y:height-(['Linear','Conv2d','LSTM'].includes(n.type)?17:6),class:'node-compact-label','text-anchor':'middle'},compact.length>13?compact.slice(0,12)+'…':compact);
        small.append(svgElement('title',{},compact));group.append(small);
        if(['Linear','Conv2d','LSTM'].includes(n.type))group.append(svgElement('text',{x:m.width/2,y:height-5,class:'node-compact-bias','text-anchor':'middle'},(n.bias!==false?'Bias on':'Bias off')+(policyState==='Frozen'?' · F':policyState==='Mixed'?' · M':'')));
        group.append(svgElement('title',{},label+' · '+line+' · '+policyNote));
      }
      for (const output of [false, true]) {
        for (const peer of G.slots(n, output)) {
          const p = portPoint(n, output, peer);
          const circle = svgElement("circle", {cx: p.x - n.x, cy: p.y - n.y, r: 5,
            class: "port " + (output ? "port-out" : "port-in") + (connecting?.id === n.id && connecting?.peer === peer ? " armed" : "") + (output && G.appendTarget(n,selected,selectedPort) && selectedPort===peer ? " append-target" : ""), "data-id": n.id});
          if (peer) circle.dataset.peer = peer;
          const portTitle=R.multiport(n)?R.portTitle(n,output,peer)+' · '+shapeText(output?s?.ports?.[peer]:sourceShape(peer)):peer ? (node(peer)?title(node(peer)):peer) : "Packed feature tensor";
          circle.append(svgElement("title", {},portTitle+(output&&G.multiOutput(n)?' · Ctrl-click: choose insertion output':'')));
          if(output)circle.setAttribute('aria-label',portTitle+(circle.classList.contains('append-target')?' · Selected insertion output':''));
          group.append(circle);
        }
      }
      const grip=svgElement('g',{class:'node-resize',transform:'translate('+(m.width-14)+','+(height-14)+')',tabindex:busy()?-1:0,role:'button','aria-label':'Resize '+label+' block','aria-disabled':String(busy())});
      grip.append(svgElement('path',{d:'M0 0H14V14H0Z',fill:'transparent',stroke:'none'}));
      grip.append(svgElement('path',{d:'M3 11L11 3M7 11L11 7',class:'resize-mark',fill:'none'}));
      grip.append(svgElement('title',{},'Drag to resize · Ctrl inverts grid snapping · arrow keys resize'));if(!n.collapsed)group.append(grip);
      $("nodes").append(group);
    }
    raiseSelected();
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
  }
  function fit() {
    if (!doc) return;
    const fitted=G.fitViewport(doc.graph.nodes,floating.available());
    if(!fitted){toast("Hide or move a panel to make room, then Fit the graph.");return;}
    doc.view=fitted;
    updateViewport();
  }

  function renderLayoutPreferences(){
    const options=state.layout_preferences || L.defaults();
    for(const mode of ['collapsed','expanded'])for(const [key,value] of Object.entries({...L.defaults()[mode],...options[mode]})){
      const control=$('layout-'+mode+'-'+key);if(!control)continue;
      if(control.type==='checkbox')control.checked=value;else control.value=value;
    }
  }
  function readLayoutProfile(mode){
    const p={};for(const key of ['style','direction','layer_gap','block_gap','rearrange','fit']){
      const control=$('layout-'+mode+'-'+key);
      p[key]=control.type==='checkbox'?control.checked:control.type==='number'?Number(control.value):control.value;
    }
    if(mode==='expanded')p.parameter_sizing=$('layout-expanded-parameter_sizing').checked;
    return L.validate(p);
  }
  function readLayoutPreferences(){
    return {version:1,collapsed:readLayoutProfile('collapsed'),expanded:readLayoutProfile('expanded')};
  }
  async function bulkBlocks(collapsed){
    if(!doc || busy() || drag || parameterSizingPending)return;
    try{
      const options=readLayoutProfile(collapsed?'collapsed':'expanded');
      let counts=null;
      if(!collapsed && options.parameter_sizing){
        const scope=state.panel_scope,wid=state.workspace_id,mid=doc.id;
        parameterSizingPending=true;header();await syncQueue;
        if(state.panel_scope!==scope || state.workspace_id!==wid || doc?.id!==mid || busy() || drag)throw Error('Workspace changed; retry parameter sizing.');
        const graph=JSON.stringify(doc.graph);
        const result=await call('layoutParameterCounts',{workspace_id:wid,panel_scope:scope,id:mid,graph:clone(doc.graph)});
        if(state.panel_scope!==scope || state.workspace_id!==wid || doc?.id!==mid || JSON.stringify(doc.graph)!==graph || busy() || drag)throw Error('Architecture changed; retry parameter sizing.');
        if(result.workspace_id!==wid || result.panel_scope!==scope || result.id!==mid)throw Error('Parameter sizing response belongs to another architecture.');
        counts=result.counts;
      }
      const updated=L.apply(doc,collapsed,options,G,counts);
      doc=updated;connecting=null;$('pending-wire').setAttribute('hidden','');
      render();if(options.fit)fit(false);persist();
      toast(collapsed?'All blocks minimized. Undo restores the previous layout.':options.parameter_sizing?'Parameter layout applied: zero-parameter blocks collapsed; learnable blocks expanded. Undo restores the previous layout.':'All blocks expanded. Undo restores the previous layout.');
    }catch(error){toast(error.message,true);}
    finally{parameterSizingPending=false;header();}
  }
  $('minimize-all').onclick=()=>bulkBlocks(true);
  $('expand-all').onclick=()=>bulkBlocks(false);
  $('save-layout-preferences').onclick=async()=>{
    try{
      const layout=readLayoutPreferences();
      await call('setLayoutPreferences',{layout});
      state.layout_preferences=layout;
      toast('Layout preferences updated for this workspace. Save workspace to keep them on disk.');
    }catch(error){toast(error.message,true);}
  };

  function renderInspector() {
    const previousPolicy=$('parameter-policy-editor');
    if(previousPolicy?.dataset.architecture&&previousPolicy.querySelector('.policy-owner'))policyDraft={
      scope:previousPolicy.dataset.scope,model:previousPolicy.dataset.model,node:previousPolicy.dataset.node,
      architecture:previousPolicy.dataset.architecture,owners:window.LoomParameterPolicy.drafts(previousPolicy)};
    const policyRequest=++policyGeneration;
    const n = node(selected), content = $("inspector-content");
    content.replaceChildren();
    if (!n) {
      $('node-kind').textContent=selectedWire?'Connection':'';
      $('inspector-header').replaceChildren();$('contract-details').hidden=true;
      content.innerHTML='<p class="inspect-note">'+(selectedWire?'Connection selected. Drag its arrow end to reconnect, or press D to delete.':'Select a block to inspect it.')+'</p>';return;
    }
    $("node-kind").textContent = n.type;
    const s = shape(n);
    const header=$('inspector-header'),help=blockHelp(n);
    header.innerHTML='<h3 class="inspect-title">'+esc(title(n))+'</h3>'+(help?'<button id="inspector-about" type="button" title="About '+esc(help.name)+'">About</button>':'');
    if(help)$('inspector-about').onclick=()=>showAbout(help);
    $('contract-details').hidden=false;$('contract-details').open=Boolean(subOpen.get('contract'));
    // Essentials stay visible; everything else is a folded sub-panel in a fixed order.
    const E=[],shapes=[],axes=[],ports=[],sharing=[],policy=[],advanced=[],dis=busy()?' disabled':'';
    const card=lines=>'<div class="contract-card">'+lines.join('<br>')+'</div>';
    const shapeField=()=>'<label class="inspect-label" for="sample-shape">Sample shape (omit B)</label><input id="sample-shape" class="inspect-input" value="'+esc((n.shape || (n.type==="Input"?[n.features]:n.type==="Reshape"?[-1]:[])).join(", "))+'" placeholder="e.g. 3, 8"'+dis+'>';
    if (["Input", "Output"].includes(n.type)) E.push('<label class="inspect-label" for="port-label">Port name</label><input id="port-label" maxlength="60" value="' + esc(n.label || n.id) + '"' + dis + '>');
    if (["Input","Output","Reshape","Constant","TensorParameter"].includes(n.type)) E.push(shapeField());
    if (n.type === "Linear") {
      E.push('<label class="inspect-label" for="features">Output features</label><input id="features" class="inspect-input" type="number" min="1" value="' + n.features + '"' + dis + '><label class="inspect-bias"><input id="bias" type="checkbox"' + (n.bias ? ' checked' : '') + dis + '>Learn a bias b</label>');
      shapes.push(card(['input '+shapeText(s?.input_shape),'output '+shapeText(s?.output_shape),'A [' + n.features + ', ' + (s?.input_shape.at(-1) || "?") + ']',n.bias!==false?'b ['+n.features+']':'Bias off · no b']));
    } else if(n.type==='PositiveVariance' && n.op_version===2){
      E.push('<label class="inspect-label" for="variance-transform">Variance parameterization</label><select id="variance-transform"'+dis+'>'+['exp','softplus'].map(value=>'<option value="'+value+'"'+(n.transform===value?' selected':'')+'>'+value+'</option>').join('')+'</select><label class="inspect-label" for="variance-floor">Explicit variance floor · zero allowed</label><input id="variance-floor" class="inspect-input" type="number" min="0" step="any" value="'+n.positive_floor+'"'+dis+'>');
      shapes.push(card(['operator version 2',(n.transform==='exp'?'variance = exp(log variance)':'variance = softplus(raw)')+(n.positive_floor?' + '+n.positive_floor:' · no hidden floor')]));
    } else if (T.types.includes(n.type)) {
      for(const [key,kind] of Object.entries(T.definitions[n.type].fields)){
        const value=n[key] ?? T.definitions[n.type].defaults[key],id='op-'+key;
        const label=key.replaceAll('_',' ');
        E.push(kind==='bool'?'<label class="inspect-bias"><input data-op-field="'+key+'" id="'+id+'" type="checkbox"'+(value?' checked':'')+dis+'>'+esc(label)+'</label>':
          kind.startsWith('choice:')?'<label class="inspect-label" for="'+id+'">'+esc(label)+'</label><select data-op-field="'+key+'" id="'+id+'"'+dis+'>'+kind.slice(7).split('|').map(o=>'<option'+(o===value?' selected':'')+'>'+esc(o)+'</option>').join('')+'</select>':
          '<label class="inspect-label" for="'+id+'">'+esc(label)+(kind==='pair' || kind==='padding'?' · H, W':'')+'</label><input data-op-field="'+key+'" id="'+id+'" class="inspect-input" value="'+esc(Array.isArray(value)?value.join(', '):value)+'"'+dis+'>');
      }
      if(G.EXPANDABLE.has(n.type))E.push('<button id="expand-block" type="button"'+dis+' title="Replace this block by the simple blocks it is made of">Expand into simpler blocks</button>');
      shapes.push(card(['input '+shapeText(s?.input_shape),'output '+shapeText(s?.output_shape),...(s?.output_axes?['axes '+esc(s.output_axes.join(', '))]:[]),...(s?.operator?.weight_shape?['W ['+s.operator.weight_shape.join(', ')+']']:[]),'operator version 1']));
    } else if(n.type==='CausalLagWindow'){
      shapes.push(card([n.lags+' preceding observations · oldest to newest','input '+shapeText(s?.input_shape),'output '+shapeText(s?.output_shape)]));
      for(const row of R.bindings(n))ports.push(card([esc(row.role)+' ← '+esc(displayName(row.source))]));
    } else if(n.type==='GaussianFilter'){
      shapes.push(card(['Exact diagonal-Gaussian filter','K '+(s?.gaussian?.states??'?')+' · D '+(s?.gaussian?.dimensions??'?')+' · T '+(s?.gaussian?.steps??'?')]));
      advanced.push('<label class="inspect-label" for="gaussian-config">Role bindings and formats · JSON</label><textarea id="gaussian-config" class="inspect-input" maxlength="2000"'+dis+'>'+esc(window.LoomJSON.stringify(Object.fromEntries(['inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode'].map(k=>[k,n[k]]))))+'</textarea><button id="apply-gaussian-config"'+dis+'>Apply checked bindings</button>');
    } else if (['Constant','TensorParameter'].includes(n.type)) {
      shapes.push(card([esc(n.type)+' '+shapeText(n.shape),'dtype float32 · no incoming data']));
      if(n.type==='Constant'){
        const literal='values' in n;
        E.push('<label class="inspect-label" for="constant-mode">Value declaration</label><select id="constant-mode"'+dis+'><option value="fill"'+(!literal?' selected':'')+'>Uniform fill</option><option value="values"'+(literal?' selected':'')+'>Rectangular JSON literal</option></select>');
        E.push('<label class="inspect-label" for="constant-values">Float32 values · exact declared shape</label><textarea id="constant-values" class="inspect-input" maxlength="2000"'+(!literal?' hidden':'')+dis+'>'+esc(literal?window.LoomJSON.stringify(n.values):'')+'</textarea>');
        E.push('<label class="inspect-label" for="constant-fill">Fill value</label><input id="constant-fill" class="inspect-input" value="'+esc(Object.is(n.fill,-0)?'-0':n.fill??0)+'"'+(literal?' hidden':'')+dis+'>');
      }
    } else if (n.type === "Input") {
      shapes.push(card(['x '+shapeText(n.shape || [n.features]),'dtype float32']));
    } else if (n.type === "Reshape") {
      shapes.push(card([shapeText(s?.input_shape)+' → '+shapeText(s?.output_shape)]));
    } else if (n.type === "Activation") {
      E.push('<label class="inspect-label" for="node-activation">Activation</label><select id="node-activation"' + dis + '>' + window.LoomActivations.options(n.activation, esc) + '</select>');
      if(['Softmax','LogSoftmax'].includes(n.activation)){
        const rank=s?.input_shape?.length,dim=n.dim??-1,axisName=rank?s?.output_axes?.[((dim%rank)+rank)%rank]:null;
        E.push('<label class="inspect-label" for="softmax-dim">Normalization axis · omit B'+(axisName?' ('+esc(axisName)+')':'')+'</label><input id="softmax-dim" class="inspect-input" type="number" min="'+(rank?-rank:-8)+'" max="'+(rank?rank-1:7)+'" step="1" value="'+esc(dim)+'"'+dis+'>');
      }
      shapes.push(card([shapeText(s?.input_shape)+' → '+shapeText(s?.output_shape),'0 trainable parameters']));
    } else if (["Concat","Add"].includes(n.type)) {
      if(n.type==='Concat')E.push('<label class="inspect-bias"><input id="flatten-concat" type="checkbox"'+(n.flatten!==false?' checked':'')+dis+'>Flatten each sample before joining</label>');
      n.sources.forEach((nid, i) => E.push('<div class="port-row"><span>' + esc(displayName(nid)) + ' · ' + (node(nid)?.features || "?") + '</span><button data-move-input="' + i + '"' + (!i || busy() ? ' disabled' : '') + '>↑</button></div>'));
    } else if (["Splitter", "Replicator"].includes(n.type)) {
      const ids = n.type === "Splitter" ? n.parts.map(p => p.output) : n.outputs;
      ids.forEach((nid, i) => E.push('<div class="port-row"><span>' + esc(title(node(nid))) + '</span>' + (n.type === "Splitter" ? '<input data-split-size="' + i + '" type="number" min="1" value="' + n.parts[i].features + '"' + dis + '>' : '<small>full tensor</small>') + '</div>'));
      E.push('<button id="add-output-port"' + dis + '>+ Output port</button>');
    } else if(n.type==='LSTM') {
      E.push('<button id="edit-lstm-interface"'+dis+'>Edit LSTM constructor and bindings…</button>');
      shapes.push(card(['LSTM · '+esc(n.num_layers??1)+' layers · '+(n.bidirectional?'bidirectional':'forward'),'Cell width: '+esc(n.hidden_size),'Hidden width: '+esc(n.proj_size||n.hidden_size),'Bias '+(n.bias!==false?'on':'off')+' · '+esc(s?.lstm?.parameters??'?')+' learned parameters']));
      for(const row of R.bindings(n))ports.push(card([esc(row.role),'source: '+esc(displayName(row.source))+' '+esc(shapeText(sourceShape(row.source))),...(row.adapter?['adapter: '+esc(row.adapter)]:[])]));
      for(const port of R.ports(n))ports.push(card([esc(R.portTitle(n,true,port)),esc(shapeText(s?.ports?.[port]))]));
    } else if(n.type==='Recurrent') {
      E.push('<button id="edit-recurrent-interface"'+dis+'>Edit recurrent interface…</button><div class="inspect-actions"><button id="open-reference">Open transition</button><button id="tab-reference">Open in new tab ↗</button></div>');
      shapes.push(card(['Transition: '+esc(state.models[n.ref]?.name||'missing'),'T: '+esc(s?.recurrent?.steps??'unresolved')+' · one parameter owner',esc(s?.recurrent?.contract?.parameters??'?')+' learned parameters']));
      for(const row of R.bindings(n))ports.push(card([esc(row.role)+' '+esc(row.input||''),'source: '+esc(displayName(row.source))+' '+esc(shapeText(sourceShape(row.source))),...(row.output?['next state: '+esc(row.output)]:[]),...(row.adapter?['adapter: '+esc(row.adapter)]:[])]));
      for(const port of R.ports(n))ports.push(card([esc(R.portTitle(n,true,port)),esc(shapeText(s?.ports?.[port]))]));
    } else if (n.type === "Module") {
      E.push('<div class="inspect-actions"><button id="open-reference">Open details</button><button id="tab-reference">Open in new tab ↗</button></div>');
      shapes.push(card(['reference: ' + esc(state.models[n.ref]?.name || "missing"),'input '+shapeText(s?.input_shape),'output '+shapeText(s?.output_shape)]));
      const signature = description?.interfaces?.[n.id];
      if (signature) ports.push(card(['Declared inputs',...signature.input_groups.map(g => esc(g.name) + ' [' + g.features + ']'),'Declared outputs',...signature.output_groups.map(g => esc(g.name) + ' [' + g.features + ']')]));
    } else {
      shapes.push(card(['ŷ '+shapeText(s?.output_shape)]));
    }
    if(['Input','Constant','TensorParameter'].includes(n.type))axes.push('<label class="inspect-label" for="input-axes">Sample axis names (optional)</label><input id="input-axes" class="inspect-input" value="'+esc(n.axes?.join(', ') || '')+'" placeholder="C, H, W or H, W, C"'+dis+'>');
    if(['Linear','Conv2d','Module','Recurrent','LSTM','TensorParameter'].includes(n.type)){
      const peers=doc.graph.nodes.filter(other=>other.id!==n.id&&other.type===n.type).map(other=>[other.id,displayName(other.id)]).sort((a,b)=>a[1].localeCompare(b[1]));
      if(n.share_with&&!peers.some(([id])=>id===n.share_with))peers.push([n.share_with,displayName(n.share_with)]);
      sharing.push('<label class="inspect-label" for="parameter-owner">Share parameters with</label><select id="parameter-owner"'+dis+'><option value="">Independent parameters</option>'+peers.map(([id,name])=>'<option value="'+esc(id)+'"'+(n.share_with===id?' selected':'')+'>'+esc(name)+'</option>').join('')+'</select>');
      policy.push('<section id="parameter-policy-editor"><p class="inspect-note">Loading parameter owners…</p></section>');
    }
    const summary=summaryLine(n,s);
    if(summary)E.push('<div class="inspect-summary">'+esc(summary)+'</div>');
    const sub=(name,label,items)=>items.length?'<details class="sub" data-sub="'+name+'"'+(subOpen.get(name)?' open':'')+'><summary>'+label+'</summary><div class="sub-body">'+items.join('')+'</div></details>':'';
    content.innerHTML='<div class="essentials">'+E.join('')+'</div>'+sub('shapes','Shapes &amp; parameters',shapes)+sub('axes','Axis names',axes)+
      sub('ports','Ports &amp; bindings',ports)+sub('sharing','Parameter sharing',sharing)+sub('policy','Parameter policy',policy)+sub('advanced','Advanced configuration',advanced);
    content.querySelectorAll('details.sub').forEach(d=>d.addEventListener('toggle',()=>subOpen.set(d.dataset.sub,d.open)));
    if($('variance-transform'))$('variance-transform').onchange=()=>{n.transform=$('variance-transform').value;invalidateRun();persist();};
    if($('variance-floor'))$('variance-floor').onchange=()=>{
      const text=$('variance-floor').value.trim(),value=Number(text),rounded=Math.fround(value);
      if(!text||!Number.isFinite(rounded)||value<0||value>0&&rounded===0){toast('Choose an explicit nonnegative representable float32 floor.',true);return;}
      n.positive_floor=rounded;invalidateRun();persist();
    };
    if($('apply-gaussian-config'))$('apply-gaussian-config').onclick=async()=>{
      if(busy())return;
      try{await syncQueue;const fields=JSON.parse($('gaussian-config').value);
        const allowed=['inputs','initial_representation','transition_representation','transition_mode','mean_mode','variance_mode'];
        if(Object.keys(fields).some(k=>!allowed.includes(k))||allowed.some(k=>!(k in fields)))throw Error('Use all six checked configuration fields; no extra fields.');
        await call('editGaussian',{id:doc.id,node:n.id,architecture:description.architecture,fields,workspace_id:state.workspace_id,panel_scope:state.panel_scope});hydrate(await call('state'));}
      catch(error){toast(error.message,true);}
    };
    if($('parameter-policy-editor')){
      const mid=doc.id,nid=n.id,context={workspace_id:state.workspace_id,panel_scope:state.panel_scope};
      syncQueue.then(()=>call('parameterPolicy',{...context,id:mid,node:nid})).then(info=>{
        if(policyRequest!==policyGeneration||doc?.id!==mid||selected!==nid)return;
        const policySection=$('parameter-policy-editor');Object.assign(policySection.dataset,{architecture:info.architecture,scope:context.panel_scope,model:mid,node:nid});
        function policyEdit(action,args){
            if(busy())throw new Error('Stop the current run before editing.');
            if(policyMutation)throw new Error('Wait for the current parameter policy edit.');
            if(policyRequest!==policyGeneration||doc?.id!==mid||selected!==nid)throw new Error('Selection changed; reopen parameter policy.');
            const transaction={id:mid,scope:context.panel_scope,document:null};policyMutation=transaction;
            const operation=syncQueue.then(async()=>{
              try{
                const updated=await call(action,{...context,id:mid,node:nid,architecture:info.architecture,...args});
                transaction.document=updated.document;
                if(updated.binding)toast('Bound initial tensor · attempt '+updated.binding.source.attempt+' · '+updated.binding.source.checkpoint_id.slice(0,12)+' · trainability preserved.');
                const authoritative=await call('state');
                if(authoritative.panel_scope===context.panel_scope&&doc?.id===mid){
                  authoritative.models[mid]=window.LoomParameterPolicy.reconcile(doc,updated.document);
                  hydrate(authoritative);
                }
              }catch(error){toast(error.message,true);throw error;}
              finally{if(policyMutation===transaction)policyMutation=null;if(state.panel_scope===context.panel_scope)renderInspector();}
            });
            syncQueue=operation.catch(()=>{});
            return operation;
        }
        window.LoomParameterPolicy.mount($('parameter-policy-editor'),info,{busy:()=>busy()||Boolean(policyMutation),
          commit:(selector,changes)=>policyEdit('editParameterPolicy',{selector,changes_json:window.LoomJSON.stringify(changes)}),
          resetBlock:()=>policyEdit('resetBlockParameterPolicy',{}),
          binding:{
            call,workspace:()=>context.workspace_id,busy:()=>busy()||Boolean(policyMutation),toast,
            valid:()=>policyRequest===policyGeneration&&doc?.id===mid&&selected===nid&&state.panel_scope===context.panel_scope,
            inventory:(run,attempt,offset)=>call('parameterSourceInventory',{...context,id:mid,node:nid,architecture:info.architecture,run,attempt,offset}),
            bind:(selector,source)=>policyEdit('bindParameter',{selector,source})
          },
          ownerName:name=>name.split(' / ').map(part=>node(part)?displayName(part):part).join(' · '),
          openDefinition:async definition=>{
            if(definition.model===doc?.id){selected=definition.node;updateSelection();renderInspector();}
            else{pendingPolicyOwner=definition;if(!await safe('openModel',{id:definition.model,newTab:false}))pendingPolicyOwner=null;}
          }});
        if(policyDraft?.scope===context.panel_scope&&policyDraft.model===mid&&policyDraft.node===nid&&policyDraft.architecture===info.architecture)
          window.LoomParameterPolicy.restore(policySection,policyDraft.owners);
      }).catch(error=>{if(policyRequest===policyGeneration&&$('parameter-policy-editor'))$('parameter-policy-editor').textContent=error.message;});
    }
    if($('expand-block'))$('expand-block').onclick=()=>expandInPlace(n.id);
    if($('edit-recurrent-interface'))$('edit-recurrent-interface').onclick=()=>recurrentEditDialog(n.id);
    if($('edit-lstm-interface'))$('edit-lstm-interface').onclick=()=>lstmUI.open(n.id);
    if($('parameter-owner'))$('parameter-owner').onchange=()=>{if(busy())return;const id=$('parameter-owner').value;if(id)n.share_with=id;else delete n.share_with;invalidateRun();persist();};
    content.querySelectorAll('[data-op-field]').forEach(input=>input.onchange=()=>{
      if(busy())return;
      try{n[input.dataset.opField]=T.parse(n.type,input.dataset.opField,input.type==='checkbox'?input.checked:input.value);invalidateRun();persist();}
      catch(error){toast(error.message,true);renderInspector();}
    });
    if($('input-axes'))$('input-axes').onchange=()=>{
      if(busy())return;
      const text=$('input-axes').value.trim(),axes=text.split(/[,\s]+/),rank=(n.shape || [n.features]).length;
      if(text && (axes.length!==rank || new Set(axes).size!==rank || axes.some(a=>a.length>32))){toast('Give one unique axis name per sample dimension, or leave empty.',true);renderInspector();return;}
      if(text)n.axes=axes;else delete n.axes;
      invalidateRun();persist();
    };
    if($('constant-mode'))$('constant-mode').onchange=()=>{
      if(busy())return;
      const literal=$('constant-mode').value==='values';
      $('constant-values').hidden=!literal;$('constant-fill').hidden=literal;
      // Changing editor mode is not a graph edit. Commit only a checked value.
    };
    if($('constant-fill'))$('constant-fill').onchange=()=>{
      if(busy())return;const text=$('constant-fill').value.trim(),value=Number(text);
      if(!text || !Number.isFinite(value) || Math.abs(value)>3.4028234663852886e38){toast('Use a finite float32 value.',true);renderInspector();return;}
      n.fill=value;delete n.values;invalidateRun();persist();
    };
    if($('constant-values'))$('constant-values').onchange=()=>{
      if(busy())return;
      try{const values=JSON.parse($('constant-values').value);
        const valid=(value,shape)=>shape.length?Array.isArray(value)&&value.length===shape[0]&&value.every(v=>valid(v,shape.slice(1))):typeof value==='number'&&Number.isFinite(value)&&Math.abs(value)<=3.4028234663852886e38;
        if(!valid(values,n.shape))throw Error('Use finite float32 values in a rectangular array matching the declared shape.');
        n.values=values;delete n.fill;invalidateRun();persist();}
      catch(error){toast(error.message,true);renderInspector();}
    };
    if($("sample-shape"))$("sample-shape").onchange=()=>{
      const text=$("sample-shape").value.trim(), dims=text.split(/[,x\s]+/).map(Number);
      if(!text && n.type==="Output")delete n.shape;
      else {
        if(!text || dims.length>8 || dims.some(d=>!Number.isSafeInteger(d) || (d<1 && !(n.type==="Reshape" && d===-1))) || dims.filter(d=>d===-1).length>1 || !Number.isSafeInteger(dims.filter(d=>d!==-1).reduce((a,b)=>a*b,1))) {
          toast("Use 1–8 positive sample dimensions; Reshape may contain one -1.",true);renderInspector();return;
        }
        if(n.type==='Constant' && 'values' in n && dims.join(',')!==n.shape.join(',')){toast('Switch to a fill value before changing literal shape, or use an atomic CLI edit of shape and values.',true);renderInspector();return;}
        n.shape=dims;if(n.type==="Input"){delete n.features;doc.dataset=null;}if(n.axes && n.axes.length!==dims.length)delete n.axes;
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
    if ($("node-activation")) $("node-activation").onchange = () => {
      n.activation = $("node-activation").value;
      if(['Softmax','LogSoftmax'].includes(n.activation))n.dim=-1;else delete n.dim;
      invalidateRun(); persist();
    };
    if($('softmax-dim'))$('softmax-dim').onchange=()=>{
      if(busy())return;
      try{n.dim=window.LoomActivations.axis($('softmax-dim').value,s?.input_shape?.length??null);}
      catch(error){toast(error.message,true);renderInspector();return;}
      invalidateRun();persist();
    };
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
    $("equation").textContent = n.type === "Activation" ? activationInfo[n.activation][0] : n.type === "Reshape" ? "y = x.reshape(B, …shape)" : n.type === "Concat" ? "x = cat(parts, dim="+(n.flatten===false?"−1":"1")+")" : n.type === "Splitter" ? "parts = split(x, sizes, dim=−1)" : n.type === "Replicator" ? "outᵢ = x" : n.type === "Module" ? "y = module(x)" : n.type === "Linear" ? (n.bias ? "y = x Aᵀ + b" : "y = x Aᵀ") : "ŷ = fθ(x)";
    const activationTex=window.LoomActivations.tex;
    const tex=n.type==='Activation'?activationTex[n.activation]:n.type==='Linear'?'y=xA^{\\mathsf T}'+(n.bias?'+b':''):n.type==='Replicator'?'y_i=x':n.type==='Module'?'y=\\operatorname{module}(x)':n.type==='Reshape'?'y=\\operatorname{reshape}(x,B,\\ldots)':n.type==='Concat'?'y=\\operatorname{cat}(x_1,\\ldots)':n.type==='Splitter'?'(y_1,\\ldots)=\\operatorname{split}(x)':'\\hat y=f_\\theta(x)';
    window.LoomMath.set($("equation"),tex).catch(()=>{}); // Keep readable plain formula on failure.
    if(T.types.includes(n.type)){
      $('equation').textContent=T.definitions[n.type].equation;
      window.LoomMath.set($('equation'),T.definitions[n.type].tex).catch(()=>{});
    }
    if(n.type==='Add'){
      $('equation').textContent='y = x₁ + … + xₙ';
      window.LoomMath.set($('equation'),'y=\\sum_i x_i').catch(()=>{});
    }
    if(n.type==='Recurrent'){
      $('equation').textContent='(yₜ, hₜ₊₁) = transition(xₜ, hₜ)';
      window.LoomMath.set($('equation'),'(y_t,h_{t+1})=\\operatorname{transition}(x_t,h_t)').catch(()=>{});
    }
  }

  function invalidateRun() {
    doc.result = null;
    renderGraph();
    renderResults();
  }
  // Nothing selected: add the block unconnected, at the centre of the visible canvas (vpid R3).
  function insertLoose(type, ref) {
    if (doc.graph.nodes.length >= 64) { toast("Use at most 64 blocks.", true); return; }
    const rect=$("canvas").getBoundingClientRect(),view=doc.view,id="n_" + Math.random().toString(36).slice(2, 10);
    const block={id,type,x:(rect.width/2-view.x)/view.zoom-90,y:(rect.height/2-view.y)/view.zoom-40};
    if (type === "Linear") Object.assign(block, {features: 1, bias: true});
    else if (type === "Activation") {block.activation = $("activation-select").value;if(['Softmax','LogSoftmax'].includes(block.activation))block.dim=-1;}
    else if (type === "Reshape") block.shape = [-1];
    else if(T.types.includes(type))Object.assign(block,T.defaults(type,null));
    else if (["Concat","Add"].includes(type)) {block.sources=[];if(type==='Concat')block.flatten=true;else block.op_version=1;}
    else block.ref = ref;
    // Nudge along the grid until the new box overlaps no existing box.
    const overlaps=()=>doc.graph.nodes.some(n=>block.x<n.x+G.width(n)&&n.x<block.x+G.width(block)&&block.y<n.y+G.height(n)&&n.y<block.y+G.height(block));
    for(let k=0;k<40&&overlaps();k++)block.y+=G.GRID*2;
    doc.graph.nodes.push(block);
    selected=id;selectedPort=null;selectedWire=null;
    invalidateRun();persist();
  }
  // Replace a ready-made block by its simple pieces: one persist, so one undo step.
  function expandInPlace(nid) {
    if (!doc || busy() || !node(nid)) return;
    try {
      const result = G.expandBlock(doc, nid, shape(node(nid)));
      selected = result.first; selectedPort = null; selectedWire = null;
      invalidateRun(); persist(); toast('Expanded into ' + result.blocks + ' simpler blocks. Undo restores the original.');
    } catch (error) { toast(error.message, true); }
  }
  function insertBlock(type, ref = null) {
    if (!doc || busy()) return;
    if (!node(selected) && !selectedPort && !selectedWire) { insertLoose(type, ref); return; }
    let after = node(selected),chosenPort=selectedPort;
    if (!after || after.type === "Output") {
      const incoming=doc.graph.edges.find(e=>e.to===(after?.id||'output'));
      after=node(incoming?.from)||doc.graph.nodes.find(n=>['Input','Constant','TensorParameter'].includes(n.type) || n.type==='Module' && n.batch_only);chosenPort=incoming?.from_port||chosenPort;
    }
    if(!after){toast('Select a tensor-producing box first.',true);return;}
    if (doc.graph.nodes.length >= 64) { toast("Use at most 64 blocks.", true); return; }
    const port=G.multiOutput(after) ? (chosenPort || G.slots(after,true)[0]) : null;
    const edge = doc.graph.edges.find(e => e.from === after.id && (!port || (e.from_port || e.to)===port)), next = edge ? node(edge.to) : null;
    const id = "n_" + Math.random().toString(36).slice(2, 10);
    const block = {id, type, x: after.x + 245, y: after.y};
    const sourceShape=G.tensorShape(after,shape(after),port);
    if (type === "Linear") Object.assign(block, {features: sourceShape?.at(-1) || 1, bias: true});
    else if (type === "Activation") {block.activation = $("activation-select").value;if(['Softmax','LogSoftmax'].includes(block.activation))block.dim=-1;}
    else if (type === "Reshape") block.shape = sourceShape || [-1];
    else if(T.types.includes(type))Object.assign(block,T.defaults(type,sourceShape));
    else if (["Concat","Add"].includes(type)) {block.sources=[port?after.id+"@"+port:after.id];if(type==='Concat')block.flatten=true;else block.op_version=1;}
    else block.ref = ref;
    if (next) {
      const move = next.x < block.x + 230 ? block.x + 230 - next.x : 0;
      doc.graph.nodes.filter(n => n.x >= next.x && n.id !== after.id).forEach(n => n.x += move);
      doc.graph.edges = doc.graph.edges.filter(e => e !== edge);
      const incoming={from:after.id,to:id};if(port)incoming.from_port=port;
      doc.graph.edges.push(incoming, {from: id, to: next.id});
      G.replaceSource(next,port?after.id+'@'+port:after.id,id);
    } else {const incoming={from:after.id,to:id};if(port)incoming.from_port=port;doc.graph.edges.push(incoming);}
    doc.graph.nodes.push(block);
    selected = id;selectedPort=null;
    selectedWire=null;
    invalidateRun();
    persist();
  }
  function insertSource(type,ref=null){
    if(!doc||busy())return;
    if(doc.graph.nodes.length>62){toast('Leave room for the source and its Output.',true);return;}
    const selectedNode=node(selected),dims=selectedNode?G.tensorShape(selectedNode,shape(selectedNode),selectedPort):null;
    const id='n_'+Math.random().toString(36).slice(2,10),x=40,y=Math.max(40,...doc.graph.nodes.map(n=>(n.y||0)+G.height(n)))+50;
    const source={id,type,x,y,op_version:1,shape:dims||[1]};
    if(type==='Constant')source.fill=0;
    if(type==='Module'){delete source.shape;delete source.op_version;source.ref=ref;source.batch_only=true;}
    doc.graph.nodes.push(source);G.addOutput(doc,id);selected=id;selectedPort=null;selectedWire=null;
    invalidateRun();persist();
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
  async function recurrentEditDialog(nid){
    if(!doc||busy()||$('recurrent-edit-dialog').open)return;
    const id=doc.id,workspace_id=state.workspace_id,panel_scope=state.panel_scope;
    const generation=++recurrentEditGeneration;
    const current=()=>generation===recurrentEditGeneration&&doc?.id===id&&state.workspace_id===workspace_id&&state.panel_scope===panel_scope&&!busy();
    const dialog=$('recurrent-edit-dialog'),form=$('recurrent-edit-form');
    let token=null,loading=false;
    $('recurrent-edit-ref').innerHTML=Object.values(state.models).filter(m=>m.id!==id).sort((a,b)=>a.name.localeCompare(b.name)).map(m=>'<option value="'+esc(m.id)+'">'+esc(m.name)+' · '+esc(m.id)+'</option>').join('');
    $('recurrent-edit-ref').value=node(nid)?.ref||'';
    $('recurrent-edit-error').textContent='';
    async function read(first=false){
      if(loading)return;loading=true;token=null;$('recurrent-edit-apply').disabled=true;
      $('recurrent-edit-ref').disabled=true;
      $('recurrent-edit-error').textContent='';
      try{
        await syncQueue;if(!current()||!dialog.open)throw Error('Workspace changed; reopen recurrent editing.');
        const ref=$('recurrent-edit-ref').value;
        const context={id,ref,node:nid,workspace_id,panel_scope};
        const info=await call('recurrentEditing',context);
        if(!current()||!dialog.open)throw Error('Workspace changed; reopen recurrent editing.');
        token=R.editResponse(info,context);
        $('recurrent-edit-signature').textContent='Transition inputs: '+info.inputs.map(g=>g.id+' '+shapeText(g.shape)).join('; ')+'\nTransition outputs: '+info.outputs.map(g=>g.id+' '+shapeText(g.shape)).join('; ');
        $('recurrent-edit-sources').textContent='Existing source sockets: '+doc.graph.nodes.filter(n=>n.id!==nid&&n.type!=='Output').flatMap(n=>G.slots(n,true).map(port=>port?n.id+'@'+port:n.id)).join(', ');
        if(first)$('recurrent-edit-json').value=JSON.stringify(info.options,null,2);
        $('recurrent-edit-apply').disabled=false;
      }catch(error){if(current()&&dialog.open)$('recurrent-edit-error').textContent=error.message;else if(generation===recurrentEditGeneration)dialog.close();}
      finally{loading=false;if(generation===recurrentEditGeneration)$('recurrent-edit-ref').disabled=false;}
    }
    $('recurrent-edit-ref').onchange=()=>read();
    $('recurrent-edit-cancel').onclick=()=>dialog.close();
    form.onsubmit=async event=>{
      event.preventDefault();if(!token||loading||$('recurrent-edit-apply').disabled)return;
      $('recurrent-edit-apply').disabled=true;$('recurrent-edit-error').textContent='';
      $('recurrent-edit-ref').disabled=true;
      try{
        await syncQueue;if(!current())throw Error('Workspace changed; reopen recurrent editing.');
        const result=await call('editRecurrent',{...token,workspace_id,panel_scope,options_json:$('recurrent-edit-json').value});
        if(result.id!==id||result.node!==nid||result.workspace_id!==workspace_id||result.panel_scope!==panel_scope)throw Error('Recurrent acknowledgement belongs to another block or workspace.');
        if(current()){
          const latest=await call('state');
          if(current()&&latest.workspace_id===workspace_id&&latest.panel_scope===panel_scope)hydrate(latest);
          if(current()&&node(nid)){selected=nid;selectedPort=null;selectedWire=null;await describe();floating.open('inspector-panel');}
        }
        if(generation===recurrentEditGeneration){dialog.close();toast('Recurrent interface accepted. Existing block identities and downstream wires were kept.');}
      }catch(error){if(generation===recurrentEditGeneration){if(dialog.open)$('recurrent-edit-error').textContent=error.message;else toast(error.message,true);}}
      finally{if(generation===recurrentEditGeneration){$('recurrent-edit-apply').disabled=false;$('recurrent-edit-ref').disabled=false;}}
    };
    dialog.showModal();await read(true);
  }
  async function recurrentDialog(){
    if(!doc||busy()||recurrentAuthoringPending)return;
    const id=doc.id,ref=$('reference-select').value,workspace_id=state.workspace_id,panel_scope=state.panel_scope;
    if(!ref)return;
    const current=()=>doc?.id===id&&state.workspace_id===workspace_id&&state.panel_scope===panel_scope&&!busy();
    recurrentAuthoringPending=true;renderLibrary();
    try{
      await syncQueue;if(!current())return;
      const info=await call('recurrentAuthoring',{id,ref,workspace_id,panel_scope});
      if(!current())return;
      if(info.id!==id||info.ref!==ref||info.workspace_id!==workspace_id||info.panel_scope!==panel_scope)throw Error('Recurrent authoring response belongs to another workspace.');
      const dialog=$('recurrent-dialog'),form=$('recurrent-form');
      $('recurrent-title').textContent='Recurrent transition · '+info.name;
      $('recurrent-label').value=('Recurrent '+info.name).slice(0,60);$('recurrent-steps').value='4';$('recurrent-error').textContent='';
      $('recurrent-inputs').innerHTML=info.inputs.map((g,i)=>{
        const compatible=info.outputs.filter(out=>JSON.stringify(out.shape)===JSON.stringify(g.shape)&&JSON.stringify(out.axes??null)===JSON.stringify(g.axes??null));
        return '<fieldset class="recurrent-input-row" data-input="'+esc(g.id)+'"><legend>'+esc(g.name)+' · '+esc(g.id)+' '+esc(shapeText(g.shape))+'</legend><label for="recurrent-role-'+i+'">Input role</label><select id="recurrent-role-'+i+'" data-role required><option value="">Choose role…</option><option value="sequence">Sequence over T</option><option value="state">Explicit initial state</option></select><label for="recurrent-next-'+i+'">Next-state Output (shape/axes must match)</label><select id="recurrent-next-'+i+'" data-next disabled><option value="">Choose Output…</option>'+compatible.map(out=>'<option value="'+esc(out.id)+'">'+esc(out.name)+' · '+esc(out.id)+'</option>').join('')+'</select></fieldset>';
      }).join('');
      $('recurrent-outputs').innerHTML=info.outputs.map(g=>'<label class="checkbox"><input type="checkbox" data-sequence-output="'+esc(g.id)+'" checked> '+esc(g.name)+' · '+esc(g.id)+' '+esc(shapeText(g.shape))+'</label>').join('');
      form.querySelectorAll('[data-recurrent-control]').forEach(input=>input.checked=false);
      form.querySelectorAll('.recurrent-input-row').forEach(row=>row.querySelector('[data-role]').onchange=()=>{
        const state=row.querySelector('[data-role]').value==='state',next=row.querySelector('[data-next]');
        next.disabled=!state;next.required=state;
      });
      $('recurrent-cancel').onclick=()=>dialog.close();
      form.onsubmit=async event=>{
        event.preventDefault();if($('recurrent-create').disabled)return;
        $('recurrent-create').disabled=true;$('recurrent-error').textContent='';
        try{
          await syncQueue;if(!current())throw Error('Workspace changed; reopen recurrent authoring.');
          const options={version:1,steps:Number($('recurrent-steps').value),label:$('recurrent-label').value,
            inputs:[...form.querySelectorAll('.recurrent-input-row')].map(row=>({input:row.dataset.input,role:row.querySelector('[data-role]').value,output:row.querySelector('[data-role]').value==='state'?row.querySelector('[data-next]').value:null})),
            sequence_outputs:[...form.querySelectorAll('[data-sequence-output]:checked')].map(input=>input.dataset.sequenceOutput),
            controls:[...form.querySelectorAll('[data-recurrent-control]:checked')].map(input=>input.dataset.recurrentControl)};
          const result=await call('addRecurrent',{id,ref,workspace_id,panel_scope,root_graph:info.root_graph,transition:info.transition,options});
          if(result.id!==id||result.workspace_id!==workspace_id||result.panel_scope!==panel_scope)throw Error('Recurrent acknowledgement belongs to another workspace.');
          if(current()){
            const latest=await call('state');
            if(current()&&latest.workspace_id===workspace_id&&latest.panel_scope===panel_scope)hydrate(latest);
            if(current()&&node(result.node)){selected=result.node;selectedPort=null;selectedWire=null;await describe();floating.open('inspector-panel');fit();}
          }
          dialog.close();toast('Recurrent branch added. Supply initial states and explicit task/output bindings before training.');
        }catch(error){if(dialog.open)$('recurrent-error').textContent=error.message;else toast(error.message,true);}
        finally{$('recurrent-create').disabled=false;}
      };
      $('recurrent-create').disabled=false;dialog.showModal();
    }catch(error){if(current())toast(error.message,true);}
    finally{recurrentAuthoringPending=false;renderLibrary();}
  }
  const wireFromElement = el => ({from:el.dataset.from,to:el.dataset.to,...(el.dataset.fromPort?{from_port:el.dataset.fromPort}:{})});
  function updateSelection() {
    // Selection must not replace the hit SVG element between consecutive
    // clicks: Chromium otherwise cannot deliver a wire/box double-click.
    document.querySelectorAll('.node').forEach(el=>el.classList.toggle('selected',el.dataset.id===selected));
    raiseSelected();
    document.querySelectorAll('.port-out').forEach(el=>el.classList.toggle('append-target',!!node(el.dataset.id)&&G.appendTarget(node(el.dataset.id),selected,selectedPort)&&el.dataset.peer===selectedPort));
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
    $("data-status").textContent = Array.isArray(doc.dataset?.x) ? doc.dataset.x.length + " rows" : "No dataset yet";
    const contract = description?.contract;
    if (contract) $("data-help").textContent = "CSV inputs: " + contract.input_groups.map(g => g.name + " [" + g.features + "]").join(" + ") + ". Targets: " + contract.outputs + " packed features. " + (doc.dataset?.truth?.equation || (contract.affine ? "Auto data uses a known affine map." : "Auto data uses a known nonlinear Tanh teacher."))+" Target family does not guarantee this architecture can represent the teacher.";
    const admission=description?.dataset_admission, evidence=description?.checkpoint_status;
    const verified=evidence?.status==="current", legacy=evidence?.status==="unverified";
    $("binding-status").textContent = ({compatible:"Bound",incompatible:"Needs rebind",unbound:"Unbound",invalid:"Invalid data",none:"No dataset"})[admission?.status] || "Checking…";
    $("binding-notice").textContent = admission?.reason || "Checking declared columns…";
    const groupList=value=>Array.isArray(value)?value.filter(g=>g && typeof g==="object"):[];
    const columns=(binding)=>[...groupList(binding?.inputs).map(g=>"x["+g.start+":"+g.stop+"] → "+g.id+" "+shapeText(Array.isArray(g.shape)?g.shape:[])),
      ...groupList(binding?.outputs).map(g=>"y["+g.start+":"+g.stop+"] → "+g.id+" "+shapeText(Array.isArray(g.shape)?g.shape:[]))].join("\n");
    $("binding-columns").textContent = admission ? "Recorded columns\n"+(columns(admission.binding)||"Not recorded")+"\n\nCurrent declared columns\n"+columns(admission.declared_binding) : "";
    $("rebind-data").hidden = !admission?.can_rebind;
    $("rebind-data").disabled = busy();
    if(doc.dataset && admission?.allowed===false)$("dataset-bindings").open=true;
    $("training-state").textContent = state.training_id === doc.id ? "Training…" : state.notebook_job ? (state.notebook_job.operation==='inspect'?'Inspecting notebook':'Executing notebook') : busy() ? "Training another model" : verified ? (doc.result.stopped ? "Stopped · current checkpoint" : "Complete · current checkpoint") : legacy ? "Unverified legacy result" : doc.result ? "Checkpoint not current" : admission?.allowed===false && doc.dataset ? "Dataset blocked" : "Ready when data is loaded";
    $("loss-value").textContent = (verified || legacy) && Number.isFinite(doc.result?.mse) ? (legacy?"Unverified · ":"")+doc.result.mse.toExponential(2) : "—";
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
    plot(state.training_id === doc.id ? liveHistories[doc.id] : verified || legacy ? doc.result?.history : null);
  }
  function showPanel(name) {
    floating.open("experiment-dock");
    panel = name;
    $("panel-context").textContent = name === "science" ? "Exact revisions · explicit runs" : "CPU · full-batch Adam · MSE";
    document.querySelectorAll("[data-panel]").forEach(el => el.classList.toggle("active", el.dataset.panel === name));
    for (const p of ["experiment", "results", "science"]) $(p + "-panel").hidden = p !== name;
  }
  function renderResults() {
    const r = doc?.result;
    if (!r) { $("results").textContent = "Train a model to inspect its learned layer weights and predictions."; return; }
    const evidence=description?.checkpoint_status;
    if(evidence && !["current","unverified"].includes(evidence.status)){
      $("results").textContent="Checkpoint not current. "+evidence.reason;return;
    }
    const vector = values => "[ " + values.slice(0,8).map(v=>Number.isFinite(v)?v.toFixed(5):String(v)).join(", ") + (values.length>8 ? ", … ("+values.length+" values)" : "") + " ]";
    const format = values => !Array.isArray(values) ? "Not recorded" : Array.isArray(values[0]) ?
      values.slice(0,6).map(vector).join("\n") + (values.length>6 ? "\n… ("+values.length+" rows; full weights retained in workspace)" : "") : vector(values);
    const truth = doc.dataset?.truth;
    let html = '<p class="tiny">'+esc(evidence?.reason || "Checkpoint status is being checked.")+'</p>';
    html += '<div class="results-summary"><b>'+(evidence?.status==="unverified"?'Unverified legacy · ':'')+'Held-out MSE ' + (Number.isFinite(r.mse)?r.mse.toExponential(5):"not recorded") + '</b><span>' + (r.train_rows ?? '?') + ' train / ' + (r.test_rows ?? '?') + ' held-out · ' + esc(r.engine || 'unknown engine') + '</span></div>';
    if(r.completed_updates!==undefined)html+='<p class="tiny">'+esc(r.completed_updates)+' / '+esc(r.requested_epochs)+' updates · '+esc(r.stop_reason)+' · '+esc(r.metric_protocol?.timing || 'unknown timing')+' / '+esc(r.metric_protocol?.mode || 'unknown mode')+'</p>';
    if(r.affine_analysis?.status && r.affine_analysis.status!=="available")html+='<p class="tiny">Optional affine analysis: '+esc(r.affine_analysis.status)+' · '+esc(r.affine_analysis.reason || '')+'</p>';
    if (r.A && r.b) {
      html += '<div class="results-grid"><section><h3>Effective A · whole affine model</h3><pre>' + esc(format(r.A)) + '</pre>';
      if (Number.isFinite(r.A_error)) html += '<p>Max A error: ' + r.A_error.toExponential(3) + '</p>';
      html += '</section><section><h3>Effective b</h3><pre>' + esc(format(r.b)) + '</pre>';
      if (Number.isFinite(r.b_error)) html += '<p>Max b error: ' + r.b_error.toExponential(3) + '</p>';
      html += '</section></div>';
    } else html += '<p>'+ (r.kind==="affine"?'This is affine, but its branching or tensor geometry is not represented by the serial-vector A,b report.':r.kind==="nonlinear"?'This model is nonlinear: no single matrix A,b describes the full forward map.':'Optional model analysis was not recorded.') +' Inspect the learned layers below.</p>';
    if (["nonlinear","spatial"].includes(truth?.kind)) html += '<p>Known target: ' + esc(truth.equation) + '. Layer weights need not match the teacher; compare predictions and held-out error.</p>';
    html += '<div class="learned-layers">';
    for (const layer of Array.isArray(r.layers)?r.layers:[]) {
      const weight = r.weights?.[layer.path + ".weight"], bias = r.weights?.[layer.path + ".bias"];
      if (!weight) { html += '<div class="learned-activation">' + esc(layer.path) + ' · ' + esc(layer.type) + ' · '+esc(shapeText(layer.input_shape))+' → '+esc(shapeText(layer.output_shape))+' · 0 parameters</div>'; continue; }
      const weightShape=layer.operator?.weight_shape || [layer.output,layer.input];
      html += '<details><summary>' + esc(layer.architecture) + ' / ' + esc(layer.node_id) + ' · '+esc(layer.type)+' '+esc(shapeText(layer.input_shape))+' → '+esc(shapeText(layer.output_shape))+'</summary><p>' + esc(layer.path) + '.weight · [' + weightShape.join(', ') + ']</p><pre>' + esc(format(weight)) + '</pre><p>Bias</p><pre>' + (bias ? esc(format(bias)) : 'Disabled') + '</pre></details>';
    }
    html += '</div><h3>Held-out predictions by named output</h3><table class="result-table"><tr><th>Port</th><th>Target</th><th>Prediction</th></tr>';
    const groups = r.output_groups || [{name: "output", start: 0, stop: r.preview?.[0]?.target.length || 1}];
    for (const row of r.preview || []) for (const g of groups) html += '<tr><td>' + esc(g.name) + '</td><td>' + esc(format(G.outputValues(row.target,g,r.output_shape,r.output_mode))) + '</td><td>' + esc(format(G.outputValues(row.prediction,g,r.output_shape,r.output_mode))) + '</td></tr>';
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
  function aboutPalette(id){
    const button=document.getElementById(id);
    if(!button?.matches('#library-panel .palette'))return;
    const help=window.LoomBlockHelp.describe(id,{tensorType:button.dataset.tensorType,
      activation:$('activation-select').value,referenceName:state.models[$('reference-select').value]?.name});
    if(help)showAbout(help);
  }
  function showAbout(help){
    const dialog=$('block-about-dialog');if(dialog.open)return;
    $('block-about-title').textContent='About '+help.name;
    const content=$('block-about-content');content.replaceChildren();
    help.paragraphs.forEach(value=>{const p=document.createElement('p');p.textContent=value;content.append(p);});
    if(help.equation){const formula=document.createElement('div');formula.className='contract-card';formula.textContent=help.equation;content.append(formula);if(help.tex)window.LoomMath.set(formula,help.tex).catch(()=>{});}
    if(help.docs?.length){
      const heading=document.createElement('h3');heading.textContent='PyTorch documentation';content.append(heading);
      const list=document.createElement('ul');list.className='about-docs';
      for(const name of help.docs){
        const item=document.createElement('li'),link=document.createElement('a');
        link.href='#';link.textContent=name;link.dataset.helpResource=name;link.title='Opens in your web browser';
        link.onclick=event=>{event.preventDefault();safe('openHelpResource',{id:name});};
        item.append(link);list.append(item);
      }
      content.append(list);
    }
    const note=document.createElement('p');note.textContent='B means batch size. Reading this explanation does not add a block or start training.';content.append(note);
    dialog.showModal();dialog.querySelector('button').focus();
  }
  function foldBlock(nid,collapsed){
    if(!doc || busy() || !node(nid))return;
    selected=nid;selectedPort=null;selectedWire=null;
    node(nid).collapsed=collapsed;persist();
  }
  $("canvas").addEventListener("pointerdown",()=>{pointerDown=true;},true);
  for(const name of ["pointerup","pointercancel"])window.addEventListener(name,()=>{if(pointerDown){pointerDown=false;setTimeout(raiseSelected,0);}},true);
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
      connecting = event.ctrlKey ? null : {id: target.dataset.id, peer: target.dataset.peer || null};
      if(event.ctrlKey)$("pending-wire").setAttribute("hidden", "");
      renderGraph();
      event.preventDefault();
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
      G.resizeBlock(node(drag.id),drag.width+dx,drag.height+dy,snapEnabled(event));
      $("status").textContent=snapEnabled(event)?'Resize · bottom-right snaps to grid · Ctrl for free resizing':'Free resize · Ctrl to snap';
      renderGraph();
    } else if (drag.kind === "node") {
      const position = G.snapPosition(drag.x + dx, drag.y + dy, snapEnabled(event));
      Object.assign(node(drag.id), position);
      $("status").textContent = snapEnabled(event) ? "Snap to grid · 24 model units · Ctrl for free drag" : "Free drag · Ctrl to snap";
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
    } else if (ended?.moved && ended.kind!=='pan') persist();
  }
  $("canvas").addEventListener("pointerup", endDrag);
  window.addEventListener('pointerup',event=>{if(drag?.deferredCapture)endDrag(event);});
  $("canvas").addEventListener("pointercancel", endDrag);
  $("canvas").addEventListener("lostpointercapture", endDrag);
  window.addEventListener('blur',endDrag);
  $("canvas").addEventListener("dblclick", event => {
    if (!doc) return;
    // Pointer capture from a block drag retargets dblclick to the canvas;
    // hit-test the actual screen location, as rewiring does.
    const hit=event.target===$("canvas") ? document.elementFromPoint(event.clientX,event.clientY) || event.target : event.target;
    if(hit.closest('.node-resize')){event.preventDefault();return;}
    const group = hit.closest(".node"), path = hit.closest(".wire");
    if (group) {
      const n = node(group.dataset.id);
      if(hit.closest('.node-type-icon')){if(n.collapsed)foldBlock(n.id,false);event.preventDefault();return;}
      selected=n.id;renderInspector();floating.open("inspector-panel");
      selectedWire=null;
      if (["Module","Recurrent"].includes(n.type) && (event.ctrlKey || event.metaKey)) safe("openModel", {id: n.ref, newTab: true});
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
    if(doc && node(selected) && !busy() && !drag && !connecting && G.blockToggleShortcut(event)){
      event.preventDefault();if(!event.repeat)foldBlock(selected,!node(selected).collapsed);return;
    }
    const grip=event.target.closest('.node-resize'),delta={ArrowRight:[8,0],ArrowLeft:[-8,0],ArrowDown:[0,8],ArrowUp:[0,-8]}[event.key];
    if(grip && delta && doc && !busy()){
      event.preventDefault();const n=node(grip.closest('.node').dataset.id),factor=event.shiftKey || event.ctrlKey?3:1;
      G.resizeBlock(n,G.width(n)+delta[0]*factor,G.height(n)+delta[1]*factor,snapEnabled(event));
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
  $('restore-panel-arrangement').onclick=()=>{restoreWorkspacePanels(true);toast('Saved workspace panel arrangement restored.');};
  $('save-panel-arrangement').onclick=async()=>{
    if(!doc||panelSavePending)return;
    const workspace_id=state.workspace_id,panel_scope=state.panel_scope,expected=clone(state.panel_layout??null);
    panelSavePending=true;header();
    try {
      const result=await call('setPanelLayout',{workspace_id,panel_scope,expected,layout:floating.capture()});
      if(state.workspace_id!==workspace_id||state.panel_scope!==panel_scope)return;
      state.panel_layout=result.layout;state.dirty=result.dirty;
      toast('Panel arrangement saved in this workspace draft. Save workspace to keep it on disk.');
    } catch(error){if(state.workspace_id===workspace_id&&state.panel_scope===panel_scope)toast(error.message,true);}
    finally{panelSavePending=false;header();}
  };
  $("resume-workspace").onclick = () => safe("openModel", {id: Object.keys(state.models)[0]});
  $("add-model").onclick = async () => { const name = await nameDialog("New architecture"); if (name) safe("newModel", {name}); };
  $("rename-model").onclick = async () => { const name = await nameDialog("Rename architecture", doc.name, "Rename"); if (name) { doc.name = name; await persist(); renderLibrary(); } };
  $("add-linear").onclick = () => insertBlock("Linear");
  // MLP: a compact form; builds ordinary Linear/activation blocks (vpbm).
  $("add-mlp").onclick = () => {
    const open=$("mlp-form").hidden;$("mlp-form").hidden=!open;$("add-mlp").setAttribute("aria-expanded",String(open));
    if(open){$("mlp-error").textContent="";$("mlp-widths").focus();}
  };
  $("mlp-cancel").onclick = () => {$("mlp-form").hidden=true;$("add-mlp").setAttribute("aria-expanded","false");};
  $("mlp-form").onsubmit = event => {
    event.preventDefault();
    if (!doc || busy()) return;
    try {
      const widths=G.parseWidths($("mlp-widths").value);
      let after=node(selected),port=selectedPort;
      if(after?.type==="Output"){const incoming=doc.graph.edges.find(e=>e.to===after.id);after=node(incoming?.from);port=incoming?.from_port||null;}
      const inputWidth=after?G.tensorShape(after,shape(after),G.multiOutput(after)?(port||G.slots(after,true)[0]):null)?.at(-1)??null:null;
      if(after&&G.multiOutput(after))port=port||G.slots(after,true)[0];
      const result=G.addMLP(doc,{after:after?.id||null,port:after&&G.multiOutput(after)?port:null,inputWidth,widths,activation:$("mlp-activation").value});
      selected=result.last;selectedPort=null;selectedWire=null;
      $("mlp-form").hidden=true;$("add-mlp").setAttribute("aria-expanded","false");
      invalidateRun();persist();toast("Added an MLP: "+(widths.length-1)+" Linear layers, "+result.blocks+" blocks.");
    } catch(error){$("mlp-error").textContent=error.message;}
  };
  $("add-reshape").onclick = () => insertBlock("Reshape");
  $("add-concat").onclick = () => insertBlock("Concat");
  $("add-add").onclick = () => insertBlock("Add");
  $("add-activation").onclick = () => insertBlock("Activation");
  $('add-constant').onclick=()=>insertSource('Constant');
  $('add-tensor-parameter').onclick=()=>insertSource('TensorParameter');
  $('add-gaussian').onclick=async()=>{
    if(!doc||busy())return;
    try{await syncQueue;const id=doc.id,info=await call('describe',{id});
      if(!info.valid)throw Error('Repair the current architecture before adding the checked component.');
      const result=await call('addGaussian',{id,architecture:info.architecture,workspace_id:state.workspace_id,panel_scope:state.panel_scope});
      hydrate(await call('state'));selected=result.node;selectedPort=null;selectedWire=null;await describe();floating.open('inspector-panel');fit();}
    catch(error){toast(error.message,true);}
  };
  $("add-reference").onclick = () => {const ref=$("reference-select").value,child=state.models[ref];
    if(child && !child.graph.nodes.some(n=>n.type==='Input'))insertSource('Module',ref);else insertBlock('Module',ref);};
  $('add-recurrent').onclick = recurrentDialog;
  lstmUI=window.LoomLSTM.mount(document,{
    call,context:()=>({id:doc?.id,workspace_id:state.workspace_id,panel_scope:state.panel_scope}),
    busy,sync:()=>syncQueue,refresh:renderLibrary,toast,
    owners:nid=>doc.graph.nodes.filter(n=>n.type==='LSTM'&&n.id!==nid)
      .map(n=>({key:n.id,name:(n.label||n.id)+' · '+n.id})).sort((a,b)=>a.name.localeCompare(b.name)||a.key.localeCompare(b.key)),
    accept:async(result,current)=>{const latest=await call('state');
      if(!current())return;
      if(latest.workspace_id!==state.workspace_id||latest.panel_scope!==state.panel_scope)
        throw Error('Workspace changed; reopen LSTM authoring.');
      hydrate(latest);if(current()&&node(result.node)){selected=result.node;selectedPort=null;selectedWire=null;
        await describe();if(current()){floating.open('inspector-panel');fit();}}
    }
  });
  $('add-lstm').onclick=()=>lstmUI.open();
  $("add-input").onclick = () => editBoundary(() => G.addInput(doc,true));
  $("add-output").onclick = () => editBoundary(() => G.addOutput(doc,selected,selectedPort));
  function addSelectedRouter(type) {
    let after=node(selected),chosenPort=selectedPort;if(after?.type==="Output"){const incoming=doc.graph.edges.find(e=>e.to===after.id);after=node(incoming?.from);chosenPort=incoming?.from_port||chosenPort;}
    if(!after || ["Splitter","Replicator"].includes(after.type)){toast("Select a tensor-producing box, not a router.",true);return;}
    const port=G.multiOutput(after)?chosenPort||G.slots(after,true)[0]:null;
    const dims=G.tensorShape(after,shape(after),port);
    editBoundary(()=>G.addRouter(doc,type,dims?.at(-1)||1,after.id,port));
  }
  $("add-splitter").onclick = () => addSelectedRouter("Splitter");
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
  $('contract-details').addEventListener('toggle',()=>subOpen.set('contract',$('contract-details').open));
  $("export-code").onclick = async () => { if(!doc)return; await syncQueue; if (await safe("exportCode", {id: doc.id})) toast("PyTorch module exported."); };
  // Export panel: the open architecture's PyTorch source, read-only and highlighted.
  function renderExportSource(){
    const view=$("export-source"),valid=Boolean(description?.valid);
    view.classList.toggle("invalid",!valid);
    if(valid)view.innerHTML=window.LoomPythonHighlight.highlight(description.code);
    else view.textContent=description?.error||"Open a valid architecture to see its PyTorch source.";
    $("export-copy").disabled=$("export-code").disabled=!valid;
  }
  $("validation-message").onkeydown=event=>{if(["Enter"," "].includes(event.key)){event.preventDefault();$("validation-message").click();}};
  $("validation-message").onclick=()=>{
    const problem=(description?.valid?[]:description?.problems||[]).find(p=>node(p.node));
    if(!problem)return;
    selected=problem.node;selectedPort=null;selectedWire=null;updateSelection();renderInspector();floating.open("inspector-panel");
    const box=document.querySelector('.node[data-id="'+CSS.escape(problem.node)+'"]')?.getBoundingClientRect(),view=$("canvas").getBoundingClientRect();
    if(!box||box.right<view.left||box.left>view.right||box.bottom<view.top||box.top>view.bottom)fit();
  };
  $("export-copy").onclick=async()=>{
    if(!description?.valid)return;
    if(await safe("copyText",{text:description.code}))toast("PyTorch source copied to the clipboard.");
  };
  $("export-source").addEventListener("keydown",event=>{
    // Ctrl/Cmd+A selects only the code, never the whole page.
    if((event.ctrlKey||event.metaKey)&&!event.altKey&&event.key.toLowerCase()==="a"){
      event.preventDefault();event.stopPropagation();
      const range=document.createRange();range.selectNodeContents($("export-source"));
      const selection=document.getSelection();selection.removeAllRanges();selection.addRange(range);
    }
  });
  document.querySelectorAll("[data-export-tab]").forEach(tab=>tab.onclick=()=>{
    document.querySelectorAll("[data-export-tab]").forEach(other=>{
      const on=other===tab;other.classList.toggle("active",on);other.setAttribute("aria-selected",String(on));
      $("export-"+other.dataset.exportTab).hidden=!on;
    });
  });
  $("rebind-data").onclick = async () => { await syncQueue; if (busy()) return; if(await safe("rebindDataset",{id:doc.id}))toast("Dataset columns rebound to the reviewed port order."); };
  document.querySelectorAll("[data-panel]").forEach(button => button.onclick = () => showPanel(button.dataset.panel));
  window.addEventListener("hashchange", route);

  function onEvent(raw) {
    const event = JSON.parse(raw);
    window.LoomScience?.event(event);
    if (event.kind === "state") hydrate(event.payload);
    else if(event.kind==='panelLayout'&&event.payload.workspace_id===state.workspace_id&&event.payload.panel_scope===state.panel_scope){state.panel_layout=event.payload.layout;state.dirty=event.payload.dirty;header();}
    else if(event.kind==='history'){state.history=event.payload;if(event.payload.notice)toast(event.payload.notice,true);}
    else if (event.kind === "navigate") navigate(event.payload);
    else if(event.kind==='navigateClean'){clearMenusOnRoute=true;navigate(event.payload);}
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
    else if (event.kind === "externalReload") {restoreWorkspacePanels(true);toast(event.payload);}
  }

  async function initialize() {
    new QWebChannel(qt.webChannelTransport, async channel => {
      host = channel.objects.host;
      host.event.connect(onEvent);
      state = await call("state");
      window.LoomScience.mount(document, {call, toast, busy, model:()=>doc?.id, workspace:()=>state.workspace_id,
        python:()=>state.torch_python,notebookJob:()=>state.notebook_job,recoveryHeader:()=>state.experiment_recoveries});
      window.loom = {call, get document() { return clone(doc); }, get state() { return clone(state); }, fit,
        selectAt(x,y){return selectContextBox(document.elementFromPoint(x,y));},
        contextAt(x,y){
          const target=document.elementFromPoint(x,y),palette=target?.closest('#library-panel .palette');
          if(palette){const name=palette.querySelector('b')?.textContent.trim();return {palette:{id:palette.id,name}};}
          const wire=target?.closest('.wire[data-from],.wire-end');
          if(doc && wire){selectWire(wireFromElement(wire));return null;}
          if(!selectContextBox(target))return null;
          return {id:selected,modelId:doc.id,canDelete:!busy() && !!node(selected) && doc.graph.nodes.length>2,canResize:!busy(),canExpand:!busy() && G.EXPANDABLE.has(node(selected)?.type)};
        },
        perform(command, blockId=null, modelId=null) {
          if(command==='about-building-block'){aboutPalette(blockId);return;}
          if(command==='undo' || command==='redo'){editHistory(command==='redo');return;}
          if(command==="reset-panels"){floating.reset();return;}
          if(!doc)return;
          if(command==="inspect-block"){
            if(doc.id!==modelId || !node(blockId))return;
            selected=blockId;selectedPort=null;selectedWire=null;
            updateSelection();renderInspector();floating.open("inspector-panel");
          }
          else if(command==="delete-block"){if(doc.id===modelId && node(blockId))removeNode(blockId);}
          else if(command==="expand-block"){if(doc.id===modelId)expandInPlace(blockId);}
          else if(command==="automatic-block-size"){
            if(busy() || doc.id!==modelId)return;
            const target=node(blockId);if(!target)return;
            if(!('width' in target) && !('height' in target))return;
            delete target.width;delete target.height;persist();
          }
          else if(command==="fit")fit();
          else if(command==="validate")$("validate").click();
          else if(command==="library")floating.open("library-panel");
          else if(command==="architectures")floating.open("architectures-panel");
          else if(command==='layout')floating.open('layout-panel');
          else if(command==='compact-all')bulkBlocks(true);
          else if(command==='expand-all')bulkBlocks(false);
          else if(command==="inspector")floating.open("inspector-panel");
          else if(command==="experiment")showPanel("experiment");
          else if(command==="code"||command==="export")floating.open("export-panel");
          else if(command==="results")showPanel("results");
        }};
      renderLayoutPreferences();await route();
    });
  }
  if (window.qt && window.QWebChannel) initialize();
  else {
    $("status").textContent = "Open this editor through the Visual PyTorch desktop launcher";
    $("runtime").textContent = "Desktop webview required";
    for (const id of ["new-workspace", "open-workspace", "welcome-open"]) $(id).disabled = true;
  }
  $("dots").querySelector("circle").setAttribute("cx", 0);
  T.types.forEach(type=>{
    const button=document.createElement('button');button.className='palette';button.id='add-tensor-'+type;button.dataset.tensorType=type;
    button.innerHTML='<svg class="type-symbol" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">'+window.LoomTypeIcons.svg(type)+'</svg><span><b>'+type+'</b><small>'+T.definitions[type].explanation+'</small></span><span>+</span>';
    button.onclick=()=>insertBlock(type);$('tensor-palette').append(button);
  });
  document.querySelectorAll('.palette').forEach(button=>{
    const name=button.querySelector('b')?.textContent.trim() || button.childNodes[0]?.textContent.trim();
    const description=button.id==='add-reference' ? 'Inspect its interface; open details from Inspector.' : button.querySelector('small')?.textContent.trim();
    if(description){button.title=description;button.setAttribute('aria-label',name+'. '+description);}
  });
  // Sort only browsing groups. Never reorder graph nodes, tensor ports or recents.
  // Building blocks end where the Tensor sources heading starts (not at Named boundaries, which would put them under other headings).
  const library=document.querySelector('#library-scroll .sidebar-section'),boundaryHeading=$('add-constant').previousElementSibling;
  const buildingGroups=[[$('activation-palette-group')],[$('mlp-palette-group')],[$('reference-palette-group')],[$('add-add')],[$('add-concat')],[$('add-linear')],[$('add-lstm')],[$('add-reshape')],
    ...[...$('tensor-palette').children].map(button=>[button])];
  // Simple pieces first; blocks that expand into pieces form the Ready-made group (vpcc decision 5).
  const readyMade=new Set(['mlp-palette-group','add-tensor-Conv2d','add-tensor-MaxPool2d','add-tensor-AvgPool2d','add-tensor-AdaptiveAvgPool2d']);
  const byName=(a,b)=>a.at(-1).querySelector('b').textContent.localeCompare(b.at(-1).querySelector('b').textContent);
  buildingGroups.filter(g=>!readyMade.has(g.at(-1).id)).sort(byName).flat().forEach(element=>library.insertBefore(element,boundaryHeading));
  const readyHeading=document.createElement('div');readyHeading.className='eyebrow';readyHeading.id='ready-made-heading';
  readyHeading.textContent='READY-MADE · EXPAND INTO PIECES';library.insertBefore(readyHeading,boundaryHeading);
  buildingGroups.filter(g=>readyMade.has(g.at(-1).id)).sort(byName).flat().forEach(element=>library.insertBefore(element,boundaryHeading));
  ['add-input','add-output','add-splitter'].map($).sort((a,b)=>a.querySelector('b').textContent.localeCompare(b.querySelector('b').textContent)).forEach(button=>library.append(button));
  $("dots").querySelector("circle").setAttribute("cy", 0);
  document.querySelectorAll('.help-sections a').forEach(link=>link.onclick=event=>{
    event.preventDefault();$(link.getAttribute('href').slice(1)).scrollIntoView({block:'start'});
  });
  document.querySelectorAll('[data-help-resource]').forEach(link=>link.onclick=event=>{
    event.preventDefault();safe('openHelpResource',{id:link.dataset.helpResource});
  });
})();
