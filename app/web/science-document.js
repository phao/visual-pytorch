"use strict";
(() => {
  let root,api,selected=null,page=null,generation=0,reading=null;
  let location={path:[],part:'value',offset:0,backs:[]};
  const $=id=>root.getElementById(id);
  const el=(tag,value)=>{const node=root.createElement(tag);if(value!==undefined)node.textContent=value;return node;};
  const act=fn=>Promise.resolve().then(fn).catch(error=>api.toast(error.message,true));
  function controls(){
    if(!api)return;
    $('science-tree-root').disabled=!selected||reading!==null;
    $('science-tree-parent').disabled=!selected||reading!==null||(!location.path.length&&location.part==='value');
    $('science-tree-prev').disabled=reading!==null||!location.backs.length;
    $('science-tree-next').disabled=reading!==null||!page||page.next_offset===null;
    $('science-tree-retry').disabled=!selected||reading!==null;
    $('science-tree-rows').querySelectorAll('button').forEach(b=>b.disabled=reading!==null);
    const edit=$('science-tree-edit');if(edit)edit.disabled=!selected||reading!==null||!page||location.part!=='value'||
      !location.path.length||selected.recovery||!['dataset','task','experiment'].includes(selected.reference.kind)||api.busy?.()||api.editing?.();
  }
  function preview(value){
    if(value.type==='object'||value.type==='array')return value.type+' · '+value.size+' members';
    return value.type+' · '+value.value.text+(value.value.next_offset!==null?' … [partial preview]':'');
  }
  function render(value){
    const end=value.offset+(value.scalar?Array.from(value.scalar.text).length:value.entries.length);
    $('science-tree-status').textContent=(value.part==='key'?'Member key':'Stored value')+' · '+value.type+
      ' · '+value.offset+'–'+end+' of '+value.total+(value.scalar?' code points':' members')+
      (selected.recovery?' · session recovery evidence':' · exact revision')+' · preview, not a complete JSON document';
    $('science-tree-path').textContent='Stored ordinal path: '+JSON.stringify(value.path)+' · '+value.part;
    const rows=$('science-tree-rows');rows.replaceChildren();
    if(value.scalar){const text=el('pre',value.scalar.text);text.className='science-scalar';rows.append(text);}
    else{
      const table=el('table'),head=el('thead'),heading=el('tr');
      for(const label of ['Member','Preview','Inspect']){const th=el('th',label);th.scope='col';heading.append(th);}
      head.append(heading);table.append(head);const body=el('tbody');table.append(body);
      value.entries.forEach(entry=>{
        const row=el('tr');row.dataset.specOrdinal=entry.ordinal;
        row.append(el('td',entry.key?entry.key.text+(entry.key.next_offset!==null?' … [partial key]':''):'['+entry.ordinal+']'),el('td',preview(entry.preview)));
        const actions=el('td'),open=el('button','Open value');open.dataset.specValue='';
        open.onclick=()=>act(()=>navigate([...location.path,entry.ordinal],'value'));actions.append(open);
        if(entry.key&&entry.key.next_offset!==null){const key=el('button','Read full key');key.onclick=()=>act(()=>navigate([...location.path,entry.ordinal],'key'));actions.append(key);}
        row.append(actions);body.append(row);
      });rows.append(table);
    }
    controls();
  }
  async function read(){
    const target=selected,ticket=++generation,identity=api.workspace(),cursor={...location,path:[...location.path]};
    if(!target||target.workspace!==identity)return;
    reading=ticket;page=null;$('science-tree-rows').replaceChildren();
    $('science-tree-path').textContent='Stored ordinal path: '+JSON.stringify(cursor.path)+' · '+cursor.part;
    $('science-tree-status').textContent='Reading an exact bounded page…';controls();
    try{
      const arguments_={path:cursor.path,part:cursor.part,offset:cursor.offset,limit:25};
      const result=await api.call(target.recovery?'scienceRecoveryInspect':'scienceInspect',
        target.recovery?{...target.recovery,...arguments_}:{workspace_id:identity,reference:target.reference,...arguments_});
      if(ticket!==generation||selected!==target||identity!==api.workspace())return;
      const matches=target.recovery?
        result.version===1&&result.scope==='experiment-recovery-inspection'&&result.scope_id===target.recovery.scope_id&&
          result.sequence===target.recovery.sequence&&result.id===target.recovery.id&&result.limit===25:
        result.scope==='exact-spec-inspection-page'&&result.reference.kind===target.reference.kind&&
          result.reference.id===target.reference.id&&result.reference.revision===target.reference.revision;
      if(!matches||
        JSON.stringify(result.path)!==JSON.stringify(cursor.path)||result.part!==cursor.part||result.offset!==cursor.offset)
        throw new Error('Inspection response does not match the requested exact revision and cursor.');
      page=result;render(result);return true;
    }catch(error){if(ticket===generation&&selected===target&&identity===api.workspace()){
      page=null;$('science-tree-rows').replaceChildren();$('science-tree-status').textContent=error.message;api.toast(error.message,true);
    }return false;}finally{if(reading===ticket)reading=null;controls();}
  }
  async function navigate(path,part){location={path,part,offset:0,backs:[]};page=null;return read();}
  window.LoomScienceDocument={
    mount(document,services){root=document;api=services;
      $('science-details').addEventListener('toggle',()=>{if($('science-details').open&&selected&&!page&&reading===null)act(read);});
      $('science-tree-root').onclick=()=>act(()=>navigate([],'value'));
      $('science-tree-parent').onclick=()=>act(()=>navigate(location.part==='key'?location.path:location.path.slice(0,-1),'value'));
      $('science-tree-next').onclick=()=>act(async()=>{if(reading!==null||!page||page.next_offset===null)return;
        const cursor=location,target=selected,old=cursor.offset;cursor.offset=page.next_offset;const ok=await read();
        if(location!==cursor||selected!==target)return;if(ok)cursor.backs.push(old);else cursor.offset=old;controls();});
      $('science-tree-prev').onclick=()=>act(async()=>{if(reading!==null||!location.backs.length)return;
        const cursor=location,target=selected,old=cursor.offset;cursor.offset=cursor.backs.at(-1);const ok=await read();
        if(location!==cursor||selected!==target)return;if(ok)cursor.backs.pop();else cursor.offset=old;controls();});
      $('science-tree-retry').onclick=()=>act(()=>reading===null?read():undefined);controls();
      const edit=$('science-tree-edit');if(edit)edit.onclick=()=>api.editValue?.({reference:{...selected.reference},workspace:selected.workspace,path:[...location.path]});
    },
    select(reference){this.clear();selected={reference:{...reference},workspace:api.workspace()};controls();if($('science-details').open)act(read);},
    selectRecovery(identity){this.clear();selected={recovery:{...identity},workspace:api.workspace()};controls();if($('science-details').open)act(read);},
    clearRecovery(){if(selected?.recovery){this.clear();$('science-document').textContent='Recovery session changed. Select evidence again.';delete $('science-document').dataset.projection;}},
    stateChanged(){controls();},
    clear(){if(!api)return;generation++;reading=null;selected=null;page=null;location={path:[],part:'value',offset:0,backs:[]};
      $('science-tree-rows').replaceChildren();$('science-tree-path').textContent='';$('science-tree-status').textContent='Open a retained specification to inspect its bounded pages. No training.';controls();}
  };
})();
