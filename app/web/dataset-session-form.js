"use strict";
(() => {
  let root,api,saved,state=null,pending=false,flushing=false,epoch=0,workspaceSeen=null;
  const $=id=>root.getElementById(id),el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const live=(owner,ticket)=>state===owner&&!owner.stale&&ticket===epoch&&owner.header.workspace_id===api.workspace();
  function issue(error){$('science-author-status').textContent=error.message;api.toast(error.message,true);}
  function controls(){
    if(!api)return;
    if(state){const blocked=pending||state.stale||state.header.workspace_id!==api.workspace()||api.busy()||window.LoomDraftValueEditor?.active?.();
      for(const id of ['science-author-save','science-author-check','science-author-add-field','science-author-add-record'])$(id).disabled=blocked||!state.page;
      $('science-author').querySelectorAll('[data-dataset-control]').forEach(n=>n.disabled=blocked||!state.page||n.dataset.large==='true');
      $('science-author-retry-page').hidden=!state.pageFailed;
      $('science-author-retry-page').disabled=blocked;
      $('science-author-fields-prev').disabled=blocked||!state.fieldBacks.length;
      $('science-author-fields-next').disabled=blocked||!state.page||state.page.next_field_offset===null;
      $('science-author-rows-prev').disabled=blocked||!state.rowBacks.length;
      $('science-author-rows-next').disabled=blocked||!state.page||state.page.next_row_offset===null;
      $('science-author-cancel').disabled=pending||window.LoomDraftValueEditor?.active?.();
    }
    window.LoomScienceAuthor?.stateChanged?.();window.LoomScienceDocument?.stateChanged?.();
  }
  async function command(fn){
    if(pending)return;captureInput();const owner=state,identity=api.workspace(),ticket=epoch;pending=true;controls();
    try{await fn();}catch(error){if(ticket===epoch&&identity===api.workspace()&&(!owner||owner===state))issue(error);}
    finally{pending=false;controls();}
  }
  const act=fn=>Promise.resolve().then(()=>command(fn));
  const args=()=>({session:state.header.session,sequence:state.header.sequence});
  function enqueue(key,operation,node){
    if(!state||state.stale||(pending&&!flushing)||api.busy()||window.LoomDraftValueEditor?.active?.())return;
    if(typeof operation.text==='string'&&Array.from(operation.text).length>2000){node.dataset.invalid='';issue(Error('Use staged value editing for more than 2000 code points.'));return;}
    state.edits=state.edits.filter(entry=>entry.key!==key);state.edits.push({key,operation,node});
    delete node.dataset.invalid;node.removeAttribute('aria-invalid');delete node.dataset.dirty;
    $('science-author-status').textContent='Unsaved dataset draft · '+state.edits.length+' pending typed edits · no training';
  }
  function asciiSize(value){return JSON.stringify(value).replace(/[\u0080-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')).length;}
  async function mutate(operations){
    const owner=state,ticket=epoch,result=await api.call('scienceDraftDatasetEdit',{...args(),operations});
    if(!live(owner,ticket))return false;
    if(result.scope!=='owned-authoring-session'||result.version!==1||result.kind!=='dataset'||result.workspace_id!==owner.header.workspace_id||
      result.session!==owner.header.session||result.sequence!==owner.header.sequence+1||result.id!==owner.header.id)
      throw Error('Typed edit acknowledgement does not match the owned dataset sequence.');
    owner.header=result;return true;
  }
  async function flush(){
    captureInput();
    if([...$('science-author').querySelectorAll('[data-dataset-control]')].some(n=>'invalid' in n.dataset))throw Error('Correct invalid draft fields before continuing. Your text has not been discarded.');
    while(state.edits.length){
      const owner=state,batch=[];
      for(const entry of owner.edits.slice(0,25)){
        const candidate=[...batch,entry];if(asciiSize({...args(),action:'scienceDraftDatasetEdit',operations:candidate.map(e=>e.operation)})>65536)break;
        batch.push(entry);
      }
      if(!batch.length)throw Error('Typed edit exceeds the byte budget. Use staged value editing.');
      try{if(!await mutate(batch.map(e=>e.operation)))return false;}
      catch(error){const match=/Dataset edit #(\d+):/.exec(error.message),entry=match?batch[Number(match[1])]:null;
        if(entry){entry.node.dataset.invalid='';entry.node.setAttribute('aria-invalid','true');}throw error;}
      owner.edits=owner.edits.filter(entry=>!batch.includes(entry));
    }
    return true;
  }
  function captureInput(){
    const active=root.activeElement;
    if(state&&active?.matches('input,textarea')&&'dirty' in active.dataset&&$('science-author').contains(active)){
      flushing=true;try{active.dispatchEvent(new Event('change'));}finally{flushing=false;}
    }
  }
  async function read(){
    const owner=state,ticket=epoch,cursor={field_offset:owner.fieldOffset,row_offset:owner.rowOffset};
    try{
      const result=await api.call('scienceDraftDatasetPage',{...args(),...cursor});if(!live(owner,ticket))return false;
      if(result.scope!=='owned-dataset-page'||result.session!==owner.header.session||result.sequence!==owner.header.sequence||
        result.id!==owner.header.id||result.workspace_id!==owner.header.workspace_id||result.field_offset!==cursor.field_offset||result.row_offset!==cursor.row_offset)
        throw Error('Dataset page does not match its exact owner, sequence and cursors.');
      owner.page=result;owner.pageFailed=false;render();return true;
    }catch(error){if(live(owner,ticket))owner.pageFailed=true;throw error;}
  }
  function valueOf(descriptor,fallback){return descriptor.present&&descriptor.text!==null?JSON.parse(descriptor.text):fallback;}
  function rawButton(target){const b=el('button','Edit value…');b.title='Stage raw JSON fragments without changing other pending dataset edits';b.dataset.datasetControl='';b.onclick=()=>act(async()=>{
    if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftDatasetValue',{...args(),target});
    if(!live(owner,ticket))return;
    const matchesTarget=result.target&&Object.keys(result.target).length===Object.keys(target).length&&Object.keys(target).every(key=>result.target[key]===target[key]);
    if(result.scope!=='owned-dataset-value'||result.session!==owner.header.session||result.sequence!==owner.header.sequence||!matchesTarget)
      throw Error('Staged target does not match its stable dataset identities.');
    await window.LoomDraftValueEditor.openOwned({header:{...owner.header},path:[...result.path],label:JSON.stringify(target),onClosed:async header=>{
      if(state!==owner||owner.stale||header.session!==owner.header.session)return;
      owner.header=header;await read();
    }});
  });return b;}
  function control(label,descriptor,target,operation,convert=x=>x,display=x=>x,{area=false}={}){
    const wrapper=el('label',label),n=el(area?'textarea':'input');n.dataset.datasetControl='';n.dataset.authorControl='';
    const large=descriptor.present&&descriptor.text===null;n.dataset.large=String(large);
    n.value=large?'':display(descriptor.present?descriptor.text:'');
    if(large)n.placeholder='Large '+descriptor.summary.type+' · '+descriptor.summary.size+' · use staged editor';
    n.oninput=()=>{n.dataset.dirty='';};n.onchange=()=>{try{enqueue(JSON.stringify(target),{...operation,text:convert(n.value)},n);}catch(error){n.dataset.invalid='';issue(error);}};
    wrapper.append(n);if(descriptor.present)wrapper.append(rawButton(target));return wrapper;
  }
  function fields(){
    const container=$('science-author-fields');container.replaceChildren();
    for(const field of state.page.fields){
      const card=el('fieldset');card.dataset.fieldId=field.id;card.append(el('legend',field.id));
      const props=Object.fromEntries(field.properties.map(p=>[p.key,p])),target=key=>({kind:'field',field:field.id,key}),op=key=>({op:'field',field:field.id,key});
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      grid.append(control('Field name',props.name,target('name'),op('name'),v=>v===''?'':JSON.stringify(v),v=>v===''?'':JSON.parse(v)));
      const label=el('label','Dtype'),dtype=el('select');dtype.dataset.datasetControl='';dtype.dataset.authorControl='';
      for(const value of ['bool','category','float32','int64','string']){const option=el('option',value);option.value=value;dtype.append(option);}
      dtype.value=valueOf(props.dtype,'');dtype.onchange=()=>{enqueue(JSON.stringify(target('dtype')),{...op('dtype'),text:JSON.stringify(dtype.value)},dtype);act(async()=>{if(await flush())await read();});};label.append(dtype);grid.append(label);
      const listDisplay=v=>{if(v==='')return '';const parsed=JSON.parse(v);return Array.isArray(parsed)?parsed.join(', '):v;};
      grid.append(control('Sample shape',props.shape,target('shape'),op('shape'),v=>JSON.stringify(window.LoomDatasetAuthoring.dimensions(v)),listDisplay));
      grid.append(control('Axis names',props.axes,target('axes'),op('axes'),v=>v.trim()===''?'':JSON.stringify(v.split(',').map(s=>s.trim())),listDisplay));
      const nullableLabel=el('label','Nullable '),nullable=el('input');nullable.type='checkbox';nullable.dataset.datasetControl='';nullable.dataset.authorControl='';nullable.checked=valueOf(props.nullable,false)===true;
      nullable.onchange=()=>enqueue(JSON.stringify(target('nullable')),{...op('nullable'),text:String(nullable.checked)},nullable);nullableLabel.append(nullable);grid.append(nullableLabel);
      if(valueOf(props.dtype,'')==='category')grid.append(control('Ordered vocabulary',props.vocabulary,target('vocabulary'),op('vocabulary'),v=>v,v=>v,{area:true}));
      const extra=el('details');extra.append(el('summary','Field metadata'));
      for(const key of ['coordinates','modality','unit'])extra.append(control(key,props[key],target(key),op(key),v=>v,v=>v,{area:true}));card.append(extra);
      const remove=el('button','Remove field');remove.dataset.datasetControl='';remove.onclick=()=>act(async()=>{
        if(!await flush()||!await mutate([{op:'field-remove',field:field.id}]))return;
        state.fieldOffset=Math.min(state.fieldOffset,Math.max(0,state.page.field_total-2));state.fieldBacks=[];await read();
      });card.append(remove);container.append(card);
    }
  }
  function records(){
    const page=state.page,table=el('table'),head=el('thead'),heading=el('tr'),body=el('tbody');table.className='science-dataset-table';heading.append(el('th','Observation ID'));
    for(const field of page.fields){const dtype=field.properties.find(p=>p.key==='dtype');heading.append(el('th',field.id+' · '+valueOf(dtype,'invalid')));}
    heading.append(el('th',''));head.append(heading);table.append(head,body);
    for(const row of page.rows){const tr=el('tr');tr.append(el('th',row.id));
      row.cells.forEach((cell,index)=>{const field=page.fields[index],target={kind:'cell',record:row.id,field:field.id},td=el('td');
        const wrapper=control('',cell,target,{op:'cell',record:row.id,field:field.id},v=>v,v=>v,{area:true});wrapper.querySelector('textarea').setAttribute('aria-label',row.id+' / '+field.id);td.append(wrapper);tr.append(td);});
      const remove=el('button','Remove');remove.dataset.datasetControl='';remove.onclick=()=>act(async()=>{
        if(!await flush()||!await mutate([{op:'record-remove',record:row.id}]))return;
        state.rowOffset=Math.min(state.rowOffset,Math.max(0,page.row_total-2));state.rowBacks=[];await read();
      });const td=el('td');td.append(remove);tr.append(td);body.append(tr);
    }$('science-author-records').replaceChildren(table);
  }
  function render(){
    const h=state.header,page=state.page;$('science-author').hidden=false;$('science-author-title').textContent='Dataset draft';
    $('science-author-id').textContent='Stable dataset ID: '+h.id+' · '+(h.base?'editing exact '+h.base.revision.slice(0,12):'new object')+(h.base&&h.base.revision!==h.expected_head?' · historical base, not current head':'');
    $('science-author-dataset').hidden=false;$('science-author-task').hidden=true;$('science-author-experiment').hidden=true;
    const name=$('science-author-name');name.dataset.datasetControl='';name.value=h.name;
    name.oninput=()=>{name.dataset.dirty='';};name.onchange=()=>enqueue('name',{op:'name',text:JSON.stringify(name.value)},name);
    fields();records();const metadata=$('science-author-metadata');metadata.replaceChildren();
    for(const value of page.metadata)metadata.append(control(value.key,value,{kind:'metadata',key:value.key},{op:'metadata',key:value.key},v=>v,v=>v,{area:true}));
    $('science-author-field-page').textContent='Fields '+(page.fields.length?page.field_offset+1:0)+'–'+(page.field_offset+page.fields.length)+' / '+page.field_total;
    $('science-author-row-page').textContent='Observations '+(page.rows.length?page.row_offset+1:0)+'–'+(page.row_offset+page.rows.length)+' / '+page.row_total;
    $('science-author-status').textContent='Unsaved dataset draft · '+page.field_total+' fields · '+page.row_total+' observations · bounded page, no training';controls();
  }
  function closeUI(){
    state=null;$('science-author').hidden=true;$('science-author-retry-page').hidden=true;
    for(const id of ['science-author-fields','science-author-records','science-author-metadata'])$(id).replaceChildren();
    const name=$('science-author-name');for(const key of ['datasetControl','dirty','invalid'])delete name.dataset[key];name.removeAttribute('aria-invalid');
  }
  function route(id,fn){const previous=$(id).onclick;$(id).onclick=()=>state||pending?act(fn):previous?.();}
  window.LoomDatasetSessionForm={
    active(){return Boolean(state)||pending;},
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;workspaceSeen=api.workspace();
      $('science-author-new').onclick=()=>this.open();const oldEdit=$('science-author-edit').onclick;
      $('science-author-edit').onclick=()=>{const ref=$('science-object').value?JSON.parse($('science-object').value):null;return ref?.kind==='dataset'?this.open(ref):oldEdit?.();};
      route('science-author-cancel',async()=>{const owner=state;await api.call('scienceDraftClose',{session:owner.header.session});if(state===owner)closeUI();});
      route('science-author-retry-page',async()=>{if(await flush())await read();});
      route('science-author-check',async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCheck',args());
        if(live(owner,ticket))$('science-author-status').textContent='Valid DatasetSpec · prospective revision '+result.revision+' · not saved, no training';});
      route('science-author-save',async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCommit',args());
        if(!live(owner,ticket))return;if(result.kind!=='dataset'||result.id!==owner.header.id)throw Error('Retained dataset has the wrong identity.');closeUI();await saved(result);api.toast('Dataset revision retained. Save workspace to persist it on disk.');});
      route('science-author-add-field',async()=>{if(!await flush())return;const id=$('science-author-field-id').value,count=state.page.field_total;
        if(!await mutate([{op:'field-add',id}]))return;$('science-author-field-id').value='';state.fieldOffset=Math.floor(count/6)*6;state.fieldBacks=state.fieldOffset?[0]:[];await read();});
      route('science-author-add-record',async()=>{if(!await flush())return;const id=$('science-author-record-id').value||crypto.randomUUID(),count=state.page.row_total;
        if(!await mutate([{op:'record-add',id}]))return;$('science-author-record-id').value='';state.rowOffset=Math.floor(count/20)*20;state.rowBacks=state.rowOffset?[0]:[];await read();});
      for(const [id,axis,forward] of [['science-author-fields-next','field',true],['science-author-fields-prev','field',false],['science-author-rows-next','row',true],['science-author-rows-prev','row',false]])route(id,async()=>{
        if(!await flush())return;const owner=state,offset=axis+'Offset',backs=axis+'Backs',prior=owner[offset],next=forward?owner.page['next_'+axis+'_offset']:owner[backs].at(-1);
        owner[offset]=next;try{if(await read()){if(forward)owner[backs].push(prior);else owner[backs].pop();}else owner[offset]=prior;}
        catch(error){owner[offset]=prior;throw error;}
      });
    },
    open(reference=null){return act(async()=>{
      if(state||window.LoomTaskSessionForm?.active?.()||window.LoomScienceAuthor?.ownActive?.()||window.LoomDraftValueEditor?.active?.())throw Error('Finish or discard the current draft before opening another.');
      if(api.busy())throw Error('Stop the run before opening a draft.');const identity=api.workspace(),ticket=epoch;
      const header=await api.call('scienceDraftOpen',{kind:'dataset',...(reference?{reference}:{})});
      if(header.scope!=='owned-authoring-session'||header.version!==1||header.kind!=='dataset'||header.workspace_id!==identity||header.sequence!==0||
        (reference?(header.id!==reference.id||header.base?.kind!=='dataset'||header.base?.id!==reference.id||header.base?.revision!==reference.revision):header.base!==null))
        throw Error('Dataset draft does not match its exact base and workspace.');
      state={header,stale:identity!==api.workspace()||ticket!==epoch,page:null,pageFailed:false,fieldOffset:0,rowOffset:0,fieldBacks:[],rowBacks:[],edits:[]};
      $('science-author').hidden=false;$('science-author-title').textContent='Dataset draft';
      $('science-author-dataset').hidden=false;$('science-author-task').hidden=true;$('science-author-experiment').hidden=true;
      for(const id of ['science-author-fields','science-author-records','science-author-metadata'])$(id).replaceChildren();
      $('science-author-name').dataset.datasetControl='';$('science-author-name').value=header.name;
      $('science-author-status').textContent='Reading bounded dataset page…';
      if(state.stale){$('science-author').hidden=false;$('science-author-status').textContent='Workspace changed. Unsaved dataset draft retained; cancel and reopen.';return;}
      await read();
    });},
    stateChanged(){if(!api)return;if(workspaceSeen!==api.workspace()){workspaceSeen=api.workspace();epoch++;if(state){state.stale=true;$('science-author-status').textContent='Workspace changed. Unsaved dataset draft retained; cancel and reopen.';}}controls();},
    event(event){if(event.kind==='externalReload'){epoch++;if(state){state.stale=true;$('science-author-status').textContent='Workspace reloaded. Unsaved dataset draft retained; cancel and reopen.';}controls();}}
  };
})();
