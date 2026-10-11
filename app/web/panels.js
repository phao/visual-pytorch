"use strict";
((root, factory) => {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LoomPanels = api;
})(typeof globalThis !== "undefined" ? globalThis : this, () => {
  const clamp = (v, low, high) => Math.max(low, Math.min(high, v));
  const snapEnabled = (event, defaultEnabled=false) => Boolean(defaultEnabled) !== Boolean(event.ctrlKey);
  const panelIDs=['architectures-panel','experiment-dock','inspector-panel','layout-panel','library-panel'];
  function validatePreset(value) {
    if(value===null)return null;
    const keys=(o,names)=>{if(!o||typeof o!=='object'||Array.isArray(o)||Object.keys(o).length!==names.length||names.some(k=>!Object.hasOwn(o,k)))throw Error('Panel arrangement has missing or unsupported fields.');};
    const number=(v,lo,hi)=>{if(typeof v!=='number'||!Number.isFinite(v)||v<lo||v>hi)throw Error('Panel arrangement coordinates/sizes must be finite numbers in range.');};
    keys(value,['version','pack','panels','order']);if(value.version!==1)throw Error('Panel arrangement requires version 1.');
    keys(value.pack,['x','y','collapsed']);number(value.pack.x,-1000000,1000000);number(value.pack.y,-1000000,1000000);
    if(typeof value.pack.collapsed!=='boolean')throw Error('Panel pack collapsed must be a boolean.');
    keys(value.panels,panelIDs);
    for(const p of Object.values(value.panels)) {
      keys(p,['x','y','width','height','visible','expanded']);number(p.x,-1000000,1000000);number(p.y,-1000000,1000000);
      number(p.width,1,100000);number(p.height,1,100000);
      if(typeof p.visible!=='boolean'||typeof p.expanded!=='boolean')throw Error('Panel visibility/expansion must be booleans.');
      if(p.expanded&&!p.visible)throw Error('A hidden panel cannot be expanded.');
      if(value.pack.collapsed&&p.visible)throw Error('A collapsed menu pack cannot contain visible panels.');
    }
    if(!Array.isArray(value.order)||value.order.length!==panelIDs.length||new Set(value.order).size!==panelIDs.length||value.order.some(id=>!panelIDs.includes(id)))throw Error('Panel order must contain each supported panel exactly once.');
    return JSON.parse(JSON.stringify(value));
  }
  function shortcutPanel(event) {
    if(event.ctrlKey || event.metaKey || event.altKey || event.repeat || event.target?.isContentEditable ||
       event.target?.closest('input,textarea,select,[contenteditable]'))return null;
    return {i:'inspector-panel',a:'architectures-panel',b:'library-panel',e:'experiment-dock'}[event.key.toLowerCase()] || null;
  }
  function constrain(rect, bounds) {
    const {width,height} = rect, visible = 24;
    // Off-stage is fine; keep a recoverable slice of the draggable frame visible.
    return {x: clamp(rect.x, visible-width, bounds.width-visible),
      y: clamp(rect.y, visible-height, bounds.height-visible), width, height};
  }
  function scrollbar(view, content, track, offset) {
    const max = Math.max(0, content - view);
    const length = max ? Math.min(track, Math.max(26, track * view / content)) : track;
    const travel = track - length;
    return {max, length, travel, top: max ? clamp(offset, 0, max) / max * travel : 0};
  }
  function availableRect(bounds, panels) {
    // Largest unobscured axis-aligned rectangle; Fit never puts nodes behind a panel.
    const obstacles = panels.map(p => ({
      x: clamp(p.x, 0, bounds.width), y: clamp(p.y, 0, bounds.height),
      right: clamp(p.x + p.width, 0, bounds.width),
      bottom: clamp(p.y + p.height, 0, bounds.height),
    }));
    const xs = [...new Set([0, bounds.width, ...obstacles.flatMap(p => [p.x, p.right])])].sort((a,b) => a-b);
    const ys = [...new Set([0, bounds.height, ...obstacles.flatMap(p => [p.y, p.bottom])])].sort((a,b) => a-b);
    let best = null, area = -1;
    for (let i=0; i<xs.length-1; i++) for (let j=i+1; j<xs.length; j++)
      for (let k=0; k<ys.length-1; k++) for (let l=k+1; l<ys.length; l++) {
        const r = {x: xs[i], y: ys[k], width: xs[j]-xs[i], height: ys[l]-ys[k]};
        if (obstacles.some(p => r.x < p.right && r.x+r.width > p.x && r.y < p.bottom && r.y+r.height > p.y)) continue;
        if (r.width*r.height > Math.max(0,area)) { best = r; area = r.width*r.height; }
      }
    return best;
  }
  function initialize(document, snapDefault=()=>false) {
    function dragVisual(active,resizing=false) {
      document.documentElement?.classList.toggle('loom-dragging',active);
      document.documentElement?.classList.toggle('loom-resizing',active && resizing);
      if(active)document.getSelection?.()?.removeAllRanges();
    }
    const snap = (x,y,enabled) => enabled ? {x:Math.round(x/24)*24,y:Math.round(y/24)*24} : {x,y};
    const editor = document.getElementById("editor");
    const panels = [...editor.querySelectorAll(".floating-panel")];
    const pack = document.getElementById("menu-pack");
    const packToggle = document.getElementById("pack-toggle");
    const openers = [...pack.querySelectorAll("[data-open-panel]")];
    let z = 3;
    const cancelGestures=[];
    const bounds = () => editor.getBoundingClientRect();
    const relative = panel => {
      const hidden=panel.hidden;
      if(hidden)panel.hidden=false; // Measure closed palettes without painting/opening them.
      const p = panel.getBoundingClientRect(), b = bounds();
      panel.hidden=hidden;
      return {x: p.left-b.left, y: p.top-b.top, width: p.width, height: p.height};
    };
    function place(panel, r) {
      r = constrain(r, bounds());
      Object.assign(panel.style, {left:r.x+"px", top:r.y+"px", right:"auto", bottom:"auto"});
      if(panel===pack) {
        // Launcher follows its labels/font. Dragging must not freeze its size.
        panel.style.width="";panel.style.height="";
      } else Object.assign(panel.style,{width:r.width+"px",height:r.height+"px"});
    }
    function minimum(panel) {
      const css=document.defaultView?.getComputedStyle(panel);
      return {width:Math.max(220,parseFloat(css?.minWidth)||0),height:Math.max(140,parseFloat(css?.minHeight)||0)};
    }
    function controls(panel) {
      const fold = panel.querySelector("[data-fold-panel]");
      const expanded = panel.classList.contains("expanded");
      fold.textContent = "×";
      fold.title = "Hide panel";
      fold.setAttribute("aria-expanded", String(!panel.hidden));
      fold.setAttribute("aria-label", "Hide " + panel.dataset.label);
      openers.find(b => b.dataset.openPanel === panel.id).setAttribute("aria-expanded",String(!panel.hidden));
      const expand = panel.querySelector("[data-expand-panel]");
      expand.textContent = expanded ? "↙" : "⤢";
      expand.setAttribute("aria-pressed", String(expanded));
      expand.setAttribute("aria-label", (expanded ? "Restore " : "Expand ") + panel.dataset.label);
      panel.dispatchEvent(new Event("panelresize"));
    }
    function restore(panel) {
      panel.classList.remove("expanded");
      if (panel.restoreStyle !== undefined) {
        panel.style.cssText = panel.restoreStyle;
        delete panel.restoreStyle;
      }
    }
    function close(panel) {
      restore(panel);
      panel.hidden = true;
      controls(panel);
    }
    function open(id) {
      const panel = panels.find(p => p.id === id);
      if (!panel) return;
      panel.hidden = false;
      panel.style.zIndex = ++z;
      if (panel.style.left) place(panel,relative(panel));
      controls(panel);
    }
    panels.forEach(panel => {
      panel.dataset.label = panel.id === "library-panel" ? "building blocks" : panel.id === "architectures-panel" ? "architectures" : panel.id === "inspector-panel" ? "inspector" : "experiment panel";
      const heading = panel.querySelector("[data-drag-handle]");
      const scroll = panel.querySelector(".scrollport");
      panel.addEventListener("pointerdown", () => panel.style.zIndex = ++z);
      panel.querySelector("[data-fold-panel]").onclick = () => {
        close(panel);
        openers.find(b => b.dataset.openPanel === panel.id).focus();
      };
      panel.querySelector("[data-expand-panel]").onclick = () => {
        if (panel.classList.contains("expanded")) restore(panel);
        else {
          panel.restoreStyle = panel.style.cssText;
          panel.classList.add("expanded");
          place(panel, {x:16,y:16,width:bounds().width-32,height:bounds().height-32});
          panel.style.zIndex = ++z;
        }
        controls(panel);
      };
      const resize = document.createElement("button");
      resize.className = "panel-resize";
      resize.setAttribute("aria-label", "Resize " + panel.dataset.label + " (arrow keys)");
      resize.title = "Drag to resize · arrow keys when focused";
      panel.append(resize);
      resize.onkeydown = event => {
        const changes = {ArrowRight:[20,0],ArrowLeft:[-20,0],ArrowDown:[0,20],ArrowUp:[0,-20]}[event.key];
        if (!changes) return;
        event.preventDefault();
        const r = relative(panel);
        const min=minimum(panel);
        place(panel, {...r, width:Math.max(min.width,r.width+changes[0]),height:Math.max(min.height,r.height+changes[1])});
        panel.dispatchEvent(new Event("panelresize"));
      };
      function emptyArea(event) {
        const target=event.target;
        if(target===panel)return true;
        if(target.closest("[data-drag-handle],button,input,select,textarea,a,label,summary,pre,code,[contenteditable],[data-no-drag],.overlay-scrollbar,.panel-resize"))return false;
        // Text stays selectable; only the actual empty area around it is draggable.
        if(document.createTreeWalker && document.createRange) {
          const walker=document.createTreeWalker(target,4),range=document.createRange();
          let text;
          while((text=walker.nextNode())) {
            if(!text.textContent.trim())continue;
            range.selectNodeContents(text);
            if([...range.getClientRects()].some(r=>event.clientX>=r.left && event.clientX<=r.right && event.clientY>=r.top && event.clientY<=r.bottom))return false;
          }
        } else if(target.textContent?.trim())return false;
        return true;
      }
      function gesture(handle, resizing, emptyOnly=false) {
        let start = null;
        handle.addEventListener("pointerdown", event => {
          if (event.button !== 0 || event.shiftKey || (emptyOnly && !emptyArea(event)) ||
              (!resizing && event.target.closest("button,input,select,textarea")) ||
              panel.classList.contains("expanded")) return;
          start = {...relative(panel), px:event.clientX, py:event.clientY};
          dragVisual(true,resizing);
          handle.setPointerCapture(event.pointerId);
          event.preventDefault();
        });
        handle.addEventListener("pointermove", event => {
          if (!start) return;
          const dx=event.clientX-start.px,dy=event.clientY-start.py;
          const min=minimum(panel);
          let width=Math.max(min.width,start.width+dx),height=Math.max(min.height,start.height+dy);
          if(resizing && snapEnabled(event,snapDefault())) {
            const corner=snap(start.x+width,start.y+height,true);
            width=Math.max(Math.ceil((start.x+min.width)/24)*24,corner.x)-start.x;
            height=Math.max(Math.ceil((start.y+min.height)/24)*24,corner.y)-start.y;
          }
          place(panel, resizing ?
            {...start,width,height} :
            {...start,...snap(start.x+dx,start.y+dy,snapEnabled(event,snapDefault()))});
          panel.dispatchEvent(new Event("panelresize"));
        });
        const finish = () => { if(start){start=null;dragVisual(false);} };
        cancelGestures.push(finish);
        handle.addEventListener("pointerup", finish);
        handle.addEventListener("pointercancel", finish);
        handle.addEventListener("lostpointercapture", finish);
        document.defaultView?.addEventListener('blur',finish);
      }
      gesture(heading, false);
      gesture(panel, false, true); // Exposed frame and empty content areas, never text/controls.
      gesture(resize, true);
      // Own overlay rail: no layout gutter, regardless of the desktop scrollbar theme.
      const rail = document.createElement("div"), thumb = document.createElement("div");
      rail.className = "overlay-scrollbar"; thumb.className = "scroll-thumb";
      thumb.tabIndex = 0; thumb.setAttribute("role","scrollbar");
      thumb.setAttribute("aria-controls",scroll.id);
      thumb.setAttribute("aria-orientation","vertical");
      thumb.setAttribute("aria-label","Scroll " + panel.dataset.label);
      rail.append(thumb); panel.append(rail);
      let timer, dragging = null;
      function geometry() {
        return scrollbar(scroll.clientHeight,scroll.scrollHeight,rail.clientHeight,scroll.scrollTop);
      }
      function update() {
        rail.style.top = heading.offsetHeight+6+"px";
        // Reveal before measuring: a hidden rail has zero clientHeight.
        rail.hidden = panel.hidden || scroll.scrollHeight <= scroll.clientHeight;
        const g = geometry();
        thumb.style.height = g.length+"px"; thumb.style.transform = "translateY("+g.top+"px)";
        thumb.setAttribute("aria-valuemin","0"); thumb.setAttribute("aria-valuemax",String(g.max));
        thumb.setAttribute("aria-valuenow",String(Math.round(scroll.scrollTop)));
      }
      scroll.addEventListener("scroll", () => {
        update(); panel.classList.add("scrolling"); clearTimeout(timer);
        timer=setTimeout(()=>panel.classList.remove("scrolling"),850);
      }, {passive:true});
      rail.addEventListener("pointerdown", event => {
        if (event.button !== 0) return;
        const g = geometry();
        if (event.target !== thumb) scroll.scrollTop = clamp(event.clientY-rail.getBoundingClientRect().top-g.length/2,0,g.travel)/Math.max(1,g.travel)*g.max;
        dragging={y:event.clientY,offset:scroll.scrollTop};
        dragVisual(true);
        rail.setPointerCapture(event.pointerId); event.preventDefault();
      });
      rail.addEventListener("pointermove", event => {
        if (!dragging) return;
        const g=geometry();
        scroll.scrollTop=dragging.offset+(event.clientY-dragging.y)/Math.max(1,g.travel)*g.max;
      });
      const finishScroll=()=>{if(dragging){dragging=null;dragVisual(false);}};
      cancelGestures.push(finishScroll);
      for (const name of ["pointerup","pointercancel","lostpointercapture"]) rail.addEventListener(name,finishScroll);
      document.defaultView?.addEventListener('blur',finishScroll);
      thumb.onkeydown = event => {
        const step = {ArrowDown:40,ArrowUp:-40,PageDown:scroll.clientHeight*.8,PageUp:-scroll.clientHeight*.8}[event.key];
        if (step !== undefined) scroll.scrollTop+=step;
        else if (event.key==="Home") scroll.scrollTop=0;
        else if (event.key==="End") scroll.scrollTop=scroll.scrollHeight;
        else return;
        event.preventDefault();
      };
      new ResizeObserver(update).observe(scroll);
      new MutationObserver(update).observe(scroll,{childList:true,subtree:true,attributes:true,characterData:true});
      panel.addEventListener("panelresize",update);
      controls(panel);
    });
    function reset() {
      cancelGestures.forEach(cancel=>cancel());
      panels.forEach(p => {
        delete p.restoreStyle; p.classList.remove("expanded"); p.removeAttribute("style"); p.hidden=true; controls(p);
      });
      pack.removeAttribute("style");
      pack.classList.remove("collapsed");
      packControls();
    }
    function onResize() {
      if (!editor.clientWidth) return;
      panels.forEach(p => {
        if (p.hidden) return;
        if (p.classList.contains("expanded")) place(p,{x:16,y:16,width:bounds().width-32,height:bounds().height-32});
        else if (p.style.left) place(p,relative(p));
      });
      if (pack.style.left) place(pack,relative(pack));
    }
    function packControls() {
      const collapsed=pack.classList.contains("collapsed");
      packToggle.setAttribute("aria-expanded",String(!collapsed));
      packToggle.setAttribute("aria-label",collapsed ? "Expand menu pack (double-click or Enter)" : "Shrink menu pack");
      document.getElementById("pack-menus").hidden=collapsed;
    }
    function shrink() {
      panels.forEach(close);
      pack.classList.add("collapsed");pack.style.width="";pack.style.height="";
      packControls();
    }
    function unshrink() {
      pack.classList.remove("collapsed");pack.style.width="";pack.style.height="";packControls();
      if (pack.style.left) place(pack,relative(pack));
    }
    packToggle.onclick=()=> {if (!pack.classList.contains("collapsed")) shrink();};
    packToggle.ondblclick=()=> {if (pack.classList.contains("collapsed")) unshrink();};
    packToggle.onkeydown=event => {
      if (["Enter"," "].includes(event.key) && pack.classList.contains("collapsed")) {
        event.preventDefault();unshrink();
      }
    };
    openers.forEach(button=> button.onclick=()=> {
      const p=panels.find(p=>p.id===button.dataset.openPanel);
      if (p.hidden) open(p.id);else close(p);
    });
    let packDrag=null;
    const packHeading=pack.querySelector(".pack-heading");
    packHeading.onpointerdown=event=>{
      if (event.button!==0 || event.shiftKey || (event.target.closest("button") && !pack.classList.contains("collapsed"))) return;
      packDrag={...relative(pack),px:event.clientX,py:event.clientY};
      dragVisual(true);
      // Keep the collapsed icon as the click target so double-click still fires.
      (event.target.closest("button") || packHeading).setPointerCapture(event.pointerId);
    };
    packHeading.onpointermove=event=>{
      if (!packDrag) return;
      place(pack,{...packDrag,...snap(packDrag.x+event.clientX-packDrag.px,packDrag.y+event.clientY-packDrag.py,snapEnabled(event,snapDefault()))});
    };
    const finishPack=()=>{if(packDrag){packDrag=null;dragVisual(false);}};
    cancelGestures.push(finishPack);
    for (const name of ["pointerup","pointercancel","lostpointercapture"]) packHeading.addEventListener(name,finishPack);
    document.defaultView?.addEventListener('blur',finishPack);
    let layerDrag=null;
    editor.addEventListener("pointerdown",event=>{
      if (!event.shiftKey || event.button!==0 || event.target.closest("input,select,textarea")) return;
      layerDrag={px:event.clientX,py:event.clientY,
        items:[pack,...panels.filter(p=>!p.classList.contains("expanded"))].map(p=>({p,r:relative(p)}))};
      dragVisual(true);
      editor.setPointerCapture(event.pointerId);
      event.preventDefault();event.stopPropagation();
    },true);
    editor.addEventListener("pointermove",event=>{
      if (!layerDrag) return;
      let dx=event.clientX-layerDrag.px,dy=event.clientY-layerDrag.py;
      if(snapEnabled(event,snapDefault())){const r=layerDrag.items[0].r,anchor=snap(r.x+dx,r.y+dy,true);dx=anchor.x-r.x;dy=anchor.y-r.y;}
      layerDrag.items.forEach(({p,r})=>place(p,{...r,x:r.x+dx,y:r.y+dy}));
      event.preventDefault();event.stopPropagation();
    },true);
    for(const name of ["pointerup","pointercancel","lostpointercapture"]) editor.addEventListener(name,event=>{
      if (layerDrag) {layerDrag=null;dragVisual(false);event.stopPropagation();}
    },true);
    document.defaultView?.addEventListener('blur',()=>{if(layerDrag){layerDrag=null;dragVisual(false);}});
    cancelGestures.push(()=>{layerDrag=null;dragVisual(false);});
    function capture() {
      if(bounds().width<=0||bounds().height<=0)throw Error('Open an architecture before saving its panel arrangement.');
      const round=v=>Math.round(v*1000)/1000,items={};
      for(const p of panels) {
        // Read ordinary geometry, never save full-stage expanded bounds as it.
        const expanded=p.classList.contains('expanded'),style=p.style.cssText;
        let r;
        try {
          if(expanded){p.classList.remove('expanded');p.style.cssText=p.restoreStyle;}
          r=relative(p);
        } finally {
          if(expanded){p.style.cssText=style;p.classList.add('expanded');}
        }
        items[p.id]={...Object.fromEntries(Object.entries(r).map(([k,v])=>[k,round(v)])),visible:!p.hidden,expanded};
      }
      const r=relative(pack);
      return validatePreset({version:1,pack:{x:round(r.x),y:round(r.y),collapsed:pack.classList.contains('collapsed')},panels:items,
        order:[...panels].sort((a,b)=>(parseFloat(a.style.zIndex)||0)-(parseFloat(b.style.zIndex)||0)).map(p=>p.id)});
    }
    function apply(value) {
      value=validatePreset(value); // Reject first: malformed presets cannot partly change the UI.
      reset();
      if(value===null)return;
      for(const id of value.order) {
        const p=panels.find(p=>p.id===id),r=value.panels[id],min=minimum(p);
        place(p,{...r,width:Math.max(min.width,r.width),height:Math.max(min.height,r.height)});
        p.hidden=!r.visible;p.style.zIndex=++z;
        if(r.expanded){p.restoreStyle=p.style.cssText;p.classList.add('expanded');place(p,{x:16,y:16,width:bounds().width-32,height:bounds().height-32});}
        controls(p);
      }
      pack.classList.toggle('collapsed',value.pack.collapsed);packControls();
      place(pack,{...relative(pack),x:value.pack.x,y:value.pack.y});
    }
    packControls();
    new ResizeObserver(onResize).observe(editor);
    new ResizeObserver(onResize).observe(pack);
    return {reset,open,capture,apply,hideAll:shrink,toggle(id){const p=panels.find(p=>p.id===id);if(p){if(p.hidden)open(id);else close(p);}}, available: () => availableRect(bounds(),
      [...panels.filter(p=>!p.hidden),pack].map(relative))};
  }
  return {constrain,scrollbar,availableRect,initialize,shortcutPanel,validatePreset,snapEnabled};
});
