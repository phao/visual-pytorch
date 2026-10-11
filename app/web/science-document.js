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
    $('science-tree-rows').querySelectorAll('button').forEach(b=>b.disabled=reading!==null);
    const edit=$('science-tree-edit');if(edit)edit.disabled=!selected||reading!==null||!page||location.part!=='value'||
      !location.path.length||!['dataset','task','experiment'].includes(selected.reference.kind)||api.busy?.()||api.editing?.();
  }
  function preview(value){
    if(value.type==='object'||value.type==='array')return value.type+' · '+value.size+' members';
    return value.type+' · '+value.value.text+(value.value.next_offset!==null?' … [partial preview]':'');
  }
  function render(value){
    const end=value.offset+(value.scalar?Array.from(value.scalar.text).length:value.entries.length);
    $('science-tree-status').textContent=(value.part==='key'?'Member key':'Stored value')+' · '+value.type+
      ' · '+value.offset+'–'+end+' of '+value.total+(value.scalar?' code points':' members')+
      ' · exact revision · preview, not a complete JSON document';
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
      const result=await api.call('scienceInspect',{workspace_id:identity,reference:target.reference,
        path:cursor.path,part:cursor.part,offset:cursor.offset,limit:25});
      if(ticket!==generation||selected!==target||identity!==api.workspace())return;
      if(result.scope!=='exact-spec-inspection-page'||result.reference.kind!==target.reference.kind||
        result.reference.id!==target.reference.id||result.reference.revision!==target.reference.revision||
        JSON.stringify(result.path)!==JSON.stringify(cursor.path)||result.part!==cursor.part||result.offset!==cursor.offset)
        throw new Error('Inspection response does not match the requested exact revision and cursor.');
      page=result;render(result);
    }catch(error){if(ticket===generation&&selected===target&&identity===api.workspace()){
      page=null;$('science-tree-rows').replaceChildren();$('science-tree-status').textContent=error.message;api.toast(error.message,true);
    }}finally{if(reading===ticket)reading=null;controls();}
  }
  async function navigate(path,part){location={path,part,offset:0,backs:[]};page=null;return read();}
  window.LoomScienceDocument={
    mount(document,services){root=document;api=services;
      $('science-details').addEventListener('toggle',()=>{if($('science-details').open&&selected&&!page&&reading===null)act(read);});
      $('science-tree-root').onclick=()=>act(()=>navigate([],'value'));
      $('science-tree-parent').onclick=()=>act(()=>navigate(location.part==='key'?location.path:location.path.slice(0,-1),'value'));
      $('science-tree-next').onclick=()=>act(()=>{if(reading!==null||!page||page.next_offset===null)return;location.backs.push(location.offset);location.offset=page.next_offset;page=null;return read();});
      $('science-tree-prev').onclick=()=>act(()=>{if(reading!==null||!location.backs.length)return;location.offset=location.backs.pop();page=null;return read();});controls();
      const edit=$('science-tree-edit');if(edit)edit.onclick=()=>api.editValue?.({reference:{...selected.reference},workspace:selected.workspace,path:[...location.path]});
    },
    select(reference){this.clear();selected={reference:{...reference},workspace:api.workspace()};controls();if($('science-details').open)act(read);},
    stateChanged(){controls();},
    clear(){if(!api)return;generation++;reading=null;selected=null;page=null;location={path:[],part:'value',offset:0,backs:[]};
      $('science-tree-rows').replaceChildren();$('science-tree-path').textContent='';$('science-tree-status').textContent='Open a retained specification to inspect its bounded pages. No training.';controls();}
  };
})();
