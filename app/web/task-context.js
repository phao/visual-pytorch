"use strict";
(() => {
  let root,api,state=null,generation=0,groups=[];
  const $=id=>root.getElementById(id),el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const sameRef=(a,b)=>a&&b&&a.kind===b.kind&&a.id===b.id&&a.revision===b.revision;
  const live=(owner,ticket)=>state===owner&&ticket===generation&&owner.workspace===api.workspace()&&owner.valid();
  const shape=row=>'['+row.shape.join(', ')+(row.shape_complete?'':', …')+']'+(row.shape_complete?'':' · preview, rank '+row.shape_rank);
  function controls(){if(!api)return;for(const group of groups){
    const blocked=!state||!state.valid()||state.workspace!==api.workspace()||api.busy()||api.blocked()||group.pending;
    for(const n of [group.query,group.pin,group.search])n.disabled=blocked;
    group.prev.disabled=blocked||!group.backs.length;group.next.disabled=blocked||!group.page||group.page.next_offset===null;
  }}
  function paint(group,page){
    const rows=[...page.rows];if(page.pinned&&!rows.some(row=>row.id===page.pinned.id))rows.push(page.pinned);
    $(group.list).replaceChildren();for(const row of rows){const option=el('option',(row.name||row.id)+(row.name_complete===false?'… (name preview)':'')+' · '+row.dtype+' '+shape(row));option.value=row.id;
      option.title='Stable ID '+row.id+' · original position '+(row.ordinal+1)+' · '+row.dtype+' '+shape(row)+(row.vocabulary_count===null?'':' · '+row.vocabulary_count+' fixed categories');
      $(group.list).append(option);}
    group.range.textContent=(page.rows.length?page.offset+1:0)+'–'+(page.offset+page.rows.length)+' / '+page.total+' matching · '+page.source_total+' total'+
      (page.missing_pin?' · ID unavailable: '+page.missing_pin:page.pinned?' · exact ID: '+page.pinned.id+' '+shape(page.pinned):'');
    $('science-task-context-status').textContent='Exact context · '+page.record_count+' observations · independently paged suggestions; bindings keep their stable IDs. No fitting or execution.';
  }
  async function read(group){
    const owner=state,ticket=generation,cursor={query:group.queryText,offset:group.offset,pin:group.pinText||null};
    const result=await api.call('scienceTaskContextPage',{workspace_id:owner.workspace,...owner.refs,collection:group.collection,...cursor,limit:25});
    if(!live(owner,ticket))return false;
    if(result.scope!=='task-context-page'||result.version!==1||result.workspace_id!==owner.workspace||!sameRef(result.model,owner.refs.model)||
      !sameRef(result.dataset,owner.refs.dataset)||result.collection!==group.collection||result.query!==cursor.query||result.offset!==cursor.offset||result.requested_pin!==cursor.pin)
      throw Error('Task suggestions do not match the exact context, collection and cursor.');
    group.page=result;paint(group,result);return true;
  }
  async function run(group,fn,initial=false){
    if(!state||group.pending||!state.valid()||api.busy()||(!initial&&api.blocked()))return;const owner=state,ticket=generation;group.pending=true;controls();
    try{await fn();}catch(error){if(live(owner,ticket)){group.range.textContent=error.message+' · retry Search without changing bindings';api.toast(error.message,true);}}
    finally{if(state===owner){group.pending=false;api.updateControls();controls();}}
  }
  function clear(){state=null;generation++;for(const group of groups){group.pending=false;group.page=null;group.backs=[];group.offset=0;$(group.list).replaceChildren();group.range.textContent='No exact context selected';}controls();}
  window.LoomTaskContext={
    mount(document,services){root=document;api=services;const container=$('science-task-context-pages');
      for(const [collection,name,list] of [['fields','Dataset fields','science-task-fields'],['input_ports','Input ports','science-task-input-ports'],['output_ports','Output ports','science-task-output-ports']]){
        const box=el('details');box.dataset.contextCollection=collection;box.append(el('summary',name+' · suggestions'));
        const grid=el('div');grid.className='science-author-grid';
        const group={collection,list,query:el('input'),pin:el('input'),search:el('button','Search / inspect ID'),prev:el('button','‹'),next:el('button','›'),range:el('p'),queryText:'',pinText:'',offset:0,backs:[],page:null,pending:false};
        // HTML maxlength counts UTF-16 units; 400 permits all valid native
        // 200-code-point IDs/searches, including supplementary characters.
        for(const [control,label] of [[group.query,'Search '+name],[group.pin,'Exact '+name+' ID (optional)']]){const wrapper=el('label',label);control.maxLength=400;control.dataset.authorControl='';wrapper.append(control);grid.append(wrapper);}
        const actions=el('div');actions.className='inline-actions';for(const [control,label] of [[group.search,'Search '+name+' and inspect exact ID'],[group.prev,'Previous '+name+' suggestions'],[group.next,'Next '+name+' suggestions']]){
          control.dataset.authorControl='';control.setAttribute('aria-label',label);actions.append(control);}
        group.range.className='tiny';group.range.setAttribute('role','status');box.append(grid,actions,group.range);container.append(box);groups.push(group);
        group.search.onclick=()=>run(group,async()=>{
          const before={queryText:group.queryText,pinText:group.pinText,offset:group.offset,backs:group.backs};
          group.queryText=group.query.value;group.pinText=group.pin.value;group.offset=0;
          try{if(await read(group))group.backs=[];else Object.assign(group,before);}catch(error){Object.assign(group,before);throw error;}
        });
        for(const [control,forward] of [[group.prev,false],[group.next,true]])control.onclick=()=>run(group,async()=>{
          if(!group.page||(forward?group.page.next_offset===null:!group.backs.length))return;
          const prior=group.offset;group.offset=forward?group.page.next_offset:group.backs.at(-1);
          try{if(await read(group)){if(forward)group.backs.push(prior);else group.backs.pop();}else group.offset=prior;}
          catch(error){group.offset=prior;throw error;}
        });
      }clear();
    },
    async show(refs,valid){
      if(!api)return;const same=state&&sameRef(state.refs.model,refs.model)&&sameRef(state.refs.dataset,refs.dataset);
      generation++;for(const group of groups){group.pending=false;group.page=null;if(!same){group.offset=0;group.backs=[];group.queryText='';group.pinText='';group.query.value='';group.pin.value='';}$(group.list).replaceChildren();}
      state={refs:JSON.parse(JSON.stringify(refs)),valid,workspace:api.workspace()};
      $('science-task-context-status').textContent='Reading bounded exact field/port suggestions; no fitting or execution…';
      await Promise.all(groups.map(group=>run(group,()=>read(group),true)));
    },
    clear,controls
  };
})();
