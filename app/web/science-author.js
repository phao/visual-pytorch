"use strict";
(() => {
  let root, api, saved, draft=null, rowOffset=0, fieldOffset=0, generation=0, pending=false, contextGeneration=0, editGeneration=0;
  let draftPage=null;
  const $=id=>root.getElementById(id), helpers=()=>window.LoomDatasetAuthoring;
  const el=(tag,value)=>{const result=root.createElement(tag);if(value!==undefined)result.textContent=value;return result;};
  const uid=()=>crypto.randomUUID();
  const issue=error=>{$('science-author-status').textContent=error.message;api.toast(error.message,true);};
  function act(fn){Promise.resolve().then(fn).catch(issue);}
  async function command(fn){if(pending)return;pending=true;controls();try{return await fn();}finally{pending=false;controls();}}
  function input(label,value,onchange,options={}){
    const wrapper=el('label',label),control=el(options.area?'textarea':'input');
    control.value=value;control.dataset.authorControl='';if(options.readOnly)control.readOnly=true;
    if(options.title)control.title=options.title;
    if(options.list)control.setAttribute('list',options.list);
    control.addEventListener('input',changed);
    control.addEventListener('change',()=>{changed();try{onchange(control.value);control.removeAttribute('aria-invalid');delete control.dataset.invalid;status();}catch(error){control.setAttribute('aria-invalid','true');control.dataset.invalid='';issue(error);}});
    wrapper.append(control);return wrapper;
  }
  function optional(container,object,key,label){
    container.append(input(label,Object.hasOwn(object,key)?JSON.stringify(object[key]):'',value=>{
      if(value.trim()==='')delete object[key];else object[key]=helpers().json(value);
    },{area:true,title:'Optional finite JSON metadata. Blank removes this property.'}));
  }
  function status(){
    if(!draft)return;
    const stale=draft.workspace_id!==api.workspace();
    if(stale&&draft.kind==='task'){contextGeneration++;changed();window.LoomTaskForm.context(root,null);}
    if(stale&&draft.kind==='experiment'){contextGeneration++;draftPage=null;window.LoomExperimentForm.clearPreview(root);}
    const counts=draft.kind==='dataset'?draft.document.fields.length+' fields · '+draft.document.records.length+' observations':
      draft.kind==='task'?draft.document.inputs.length+' input bindings · '+draft.document.heads.length+' objectives':
      draft.document.models.length+' architectures · '+draft.document.data_tasks.length+' dataset/task pairings · '+draft.document.initialization_seeds.length+' initialization seeds';
    $('science-author-status').textContent=stale?'Workspace changed. This unsaved draft is retained for inspection; cancel and reopen it in the intended workspace.':
      'Unsaved '+draft.kind+' draft · '+counts+' · validation never trains';
    controls();
  }
  function changed(){
    editGeneration++;
    $('science-task-compatibility-status').textContent='';
    if(draft?.kind==='experiment'){draftPage=null;window.LoomExperimentForm.clearPreview(root);}
    if(draft&&draft.workspace_id===api.workspace())status();
  }
  function controls(){
    if(!api)return;
    const blocked=pending||!draft||api.busy()||draft.workspace_id!==api.workspace();
    for(const id of ['science-author-save','science-author-check','science-author-add-field','science-author-add-record'])$(id).disabled=blocked;
    $('science-author-new').disabled=pending||api.busy()||Boolean(draft)||!api.workspace();
    $('science-author-new-task').disabled=$('science-author-new').disabled;
    $('science-author-new-experiment').disabled=$('science-author-new').disabled;
    $('science-author-edit').disabled=pending||api.busy()||Boolean(draft)||!api.workspace()||!['dataset','task','experiment'].includes($('science-kind').value)||!$('science-object').value;
    $('science-author-cancel').disabled=pending;
    if(draft)$('science-author').querySelectorAll('[data-author-control]').forEach(control=>control.disabled=blocked||control.dataset.orderDisabled==='true');
    $('science-experiment-prev').disabled=blocked||!draftPage||draftPage.preview.offset===0;
    $('science-experiment-next').disabled=blocked||!draftPage||draftPage.preview.next_offset===null||!Number.isSafeInteger(draftPage.preview.next_offset);
  }
  function fields(){
    const container=$('science-author-fields');container.replaceChildren();
    const data=helpers().page(draft.document,rowOffset,fieldOffset);
    data.fields.forEach((_,index)=>{
      const field=draft.document.fields[fieldOffset+index];
      const card=el('fieldset'),legend=el('legend',field.id);card.dataset.fieldId=field.id;card.append(legend);
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      grid.append(input('Field name',field.name||'',value=>{if(value==='')delete field.name;else field.name=value;}));
      const dtypeLabel=el('label','Dtype'),dtype=el('select');dtype.dataset.authorControl='';
      ['bool','category','float32','int64','string'].forEach(value=>{const option=el('option',value);option.value=value;dtype.append(option);});dtype.value=field.dtype;
      dtype.onchange=()=>{if($('science-author').querySelector('[data-invalid]')){dtype.value=field.dtype;issue(Error('Correct invalid fields before changing dtype.'));return;}
        field.dtype=dtype.value;if(field.dtype!=='category')delete field.vocabulary;else if(!field.vocabulary)field.vocabulary={version:1,values:[]};render();};
      dtypeLabel.append(dtype);grid.append(dtypeLabel);
      grid.append(input('Sample shape',field.shape.join(', '),value=>{field.shape=helpers().dimensions(value);},{title:'No batch axis. Blank = scalar; e.g. 3, 32, 32; -1, 3 = ragged first axis.'}));
      grid.append(input('Axis names',field.axes?.join(', ')||'',value=>{if(value.trim()==='')delete field.axes;else field.axes=value.split(',').map(v=>v.trim());},{title:'Optional comma-separated sample-axis names, in tensor order.'}));
      const nullableLabel=el('label','Nullable '),nullable=el('input');nullable.type='checkbox';nullable.dataset.authorControl='';nullable.checked=field.nullable===true;
      nullable.onchange=()=>{field.nullable=nullable.checked;status();};nullableLabel.append(nullable);grid.append(nullableLabel);
      if(field.dtype==='category')grid.append(input('Ordered vocabulary',JSON.stringify(field.vocabulary?.values||[]),value=>{const values=helpers().cell(value,'category');if(!Array.isArray(values))throw Error('Vocabulary is an ordered JSON array.');field.vocabulary={version:1,values};},{area:true,title:'Fixed ordered labels. For example ["cat","dog"]. Reordering changes class indices.'}));
      const extra=el('details');extra.append(el('summary','Field metadata'));
      for(const key of ['coordinates','modality','unit'])optional(extra,field,key,key);
      card.append(extra);
      const remove=el('button','Remove field');remove.type='button';remove.dataset.authorControl='';remove.title='Remove this field and its values from the draft only. Cancel discards the entire draft.';
      remove.onclick=()=>{if(!flush())return;draft.document=helpers().removeField(draft.document,field.id);fieldOffset=Math.min(fieldOffset,Math.max(0,draft.document.fields.length-6));render();};card.append(remove);
      container.append(card);
    });
    $('science-author-field-page').textContent='Fields '+(data.fields.length?fieldOffset+1:0)+'–'+(fieldOffset+data.fields.length)+' / '+data.fieldTotal;
    $('science-author-fields-prev').disabled=fieldOffset===0;$('science-author-fields-next').disabled=fieldOffset+6>=data.fieldTotal;
  }
  function records(){
    const data=helpers().page(draft.document,rowOffset,fieldOffset),table=el('table'),thead=el('thead'),heading=el('tr');
    heading.append(el('th','Observation ID'));data.fields.forEach(field=>heading.append(el('th',field.id+' · '+field.dtype)));heading.append(el('th',''));thead.append(heading);table.append(thead);
    const tbody=el('tbody');table.append(tbody);
    data.rows.forEach((_,i)=>{
      const record=draft.document.records[rowOffset+i],row=el('tr');row.append(el('th',record.id));
      data.fields.forEach(field=>{
        const td=el('td'),cell=el('textarea');cell.dataset.authorControl='';cell.setAttribute('aria-label',record.id+' / '+field.id);cell.value=JSON.stringify(record.values[field.id]);
        cell.onchange=()=>{try{record.values[field.id]=helpers().cell(cell.value,field.dtype);cell.removeAttribute('aria-invalid');delete cell.dataset.invalid;status();}catch(error){cell.setAttribute('aria-invalid','true');cell.dataset.invalid='';issue(error);}};td.append(cell);row.append(td);
      });
      const remove=el('button','Remove');remove.dataset.authorControl='';remove.title='Remove this observation from the draft only.';
      remove.onclick=()=>{if(!flush())return;draft.document.records.splice(rowOffset+i,1);rowOffset=Math.min(rowOffset,Math.max(0,draft.document.records.length-20));render();};
      const td=el('td');td.append(remove);row.append(td);tbody.append(row);
    });
    $('science-author-records').replaceChildren(table);
    $('science-author-row-page').textContent='Observations '+(data.rows.length?rowOffset+1:0)+'–'+(rowOffset+data.rows.length)+' / '+data.rowTotal;
    $('science-author-rows-prev').disabled=rowOffset===0;$('science-author-rows-next').disabled=rowOffset+20>=data.rowTotal;
  }
  function flush(){
    // Select handlers themselves call flush. Re-dispatching their change would
    // recurse; only text/checkbox controls have an uncommitted value to flush.
    const active=root.activeElement;if(active?.matches('input,textarea')&&$('science-author').contains(active))active.dispatchEvent(new Event('change'));
    if($('science-author').querySelector('[data-invalid]')){issue(Error('Correct invalid draft fields before continuing. Your text has not been discarded.'));return false;}return true;
  }
  function render(){
    $('science-author').hidden=!draft;if(!draft){controls();return;}
    $('science-author-title').textContent=draft.kind[0].toUpperCase()+draft.kind.slice(1)+' draft';
    $('science-author-id').textContent='Stable '+draft.kind+' ID: '+draft.id+' · '+(draft.base?'editing exact '+draft.base.revision.slice(0,12):'new object')+(draft.base&&draft.base.revision!==draft.expected_head?' · historical base, not current head':'');
    const name=$('science-author-name');name.value=draft.document.name;name.oninput=changed;name.onchange=()=>{changed();draft.document.name=name.value;status();};
    $('science-author-dataset').hidden=draft.kind!=='dataset';$('science-author-task').hidden=draft.kind!=='task';
    $('science-author-experiment').hidden=draft.kind!=='experiment';
    if(draft.kind==='dataset'){
      fields();records();const metadata=$('science-author-metadata');metadata.replaceChildren();
      for(const key of ['provenance','usage'])optional(metadata,draft.document,key,key);
    }else if(draft.kind==='task')window.LoomTaskForm.render(root,draft.document,{el,input,flush,rerender:render,changed});
    else window.LoomExperimentForm.render(root,draft.document,{el,input,flush,rerender:render,changed});
    status();
  }
  async function context(){
    const selected=draft,ticket=++contextGeneration,identity=api.workspace();
    const m=$('science-task-model').value,d=$('science-task-dataset').value;
    $('science-task-compatibility-status').textContent='';
    window.LoomTaskForm.context(root,null);
    if(!m||!d)return;
    $('science-task-context-status').textContent='Reading exact interfaces; no fitting or execution…';
    const value=await api.call('scienceTaskContext',{model:JSON.parse(m),dataset:JSON.parse(d)});
    if(ticket!==contextGeneration||draft!==selected||identity!==api.workspace())return;
    window.LoomTaskForm.context(root,value);
  }
  async function choices(){
    const selected=draft,ticket=++contextGeneration,identity=api.workspace();
    if(!selected||selected.kind!=='task')return;
    const rows=await api.call('scienceIndex');
    if(ticket!==contextGeneration||draft!==selected||identity!==api.workspace())return;
    for(const [id,kind] of [['science-task-model','model'],['science-task-dataset','dataset']]){
      const select=$(id),old=select.value;select.replaceChildren();
      rows.filter(row=>row.kind===kind).forEach(row=>{const option=el('option',row.name+' · '+row.revision.slice(0,12)+(row.selected?' · current head':' · historical'));option.value=JSON.stringify({kind:row.kind,id:row.id,revision:row.revision});select.append(option);});
      if([...select.options].some(option=>option.value===old))select.value=old;
      else if(old){const missing=el('option','Unavailable exact '+kind+' revision · choose another explicitly');missing.value=old;select.append(missing);select.value=old;}
      else{const head=rows.find(row=>row.kind===kind&&row.selected);if(head)select.value=JSON.stringify({kind:head.kind,id:head.id,revision:head.revision});}
    }
    await context();
  }
  async function begin(kind='dataset',reference=null){
    const ticket=++generation,identity=api.workspace();
    const result=await api.call('scienceAuthorBegin',{kind,...(reference?{reference}:{})});
    if(ticket!==generation||identity!==api.workspace())return;
    draft=result;draftPage=null;rowOffset=0;fieldOffset=0;if(kind==='experiment')window.LoomExperimentForm.reset();render();
    if(kind==='task'){window.LoomTaskForm.context(root,null);await choices();}
    else if(kind==='experiment')await experimentChoices();
  }
  async function experimentChoices(){
    const selected=draft,ticket=++contextGeneration,identity=api.workspace();
    const rows=await api.call('scienceIndex');
    if(ticket!==contextGeneration||draft!==selected||identity!==api.workspace())return;
    window.LoomExperimentForm.refresh(root,draft.document,rows);
  }
  async function previewDraft(offset=0){
    if(!flush())return;
    const selected=draft,ticket=generation,edit=editGeneration,identity=api.workspace();
    const result=await api.call('scienceAuthorPreview',{draft,offset,limit:25});
    if(ticket!==generation||draft!==selected||edit!==editGeneration||identity!==api.workspace())return;
    draftPage=result;const page=result.preview,container=$('science-experiment-preview-rows');container.replaceChildren();
    page.candidates.forEach(row=>{const card=el('div');card.className='science-candidate';card.dataset.draftCandidate=row.status;
      card.append(el('span',row.status+' · model '+row.coordinates.model.id+' @ '+row.coordinates.model.revision.slice(0,12)+' · '+row.coordinates.data_task+' · seed '+row.coordinates.initialization+' · '+row.coordinates.split+' · '+row.coordinates.protocol));
      if(row.reason)card.append(el('small',row.reason));container.append(card);});
    $('science-experiment-preview-status').textContent=page.offset+'–'+(page.offset+page.candidates.length)+' of '+result.total_combinations_exact+' · prospective '+page.experiment.revision.slice(0,12)+' · interface/split preview only, no fitting or RAM admission. Retain revision before explicit runs.';
    controls();
  }
  window.LoomScienceAuthor={
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;
      $('science-author-new').onclick=()=>act(()=>command(()=>begin()));
      $('science-author-new-task').onclick=()=>act(()=>command(()=>begin('task')));
      $('science-author-new-experiment').onclick=()=>act(()=>command(()=>begin('experiment')));
      $('science-author-edit').onclick=()=>act(()=>command(()=>{const reference=JSON.parse($('science-object').value);return begin(reference.kind,reference);}));
      $('science-author-cancel').onclick=()=>{draft=null;generation++;contextGeneration++;render();};
      $('science-experiment-refresh').onclick=()=>act(()=>command(experimentChoices));
      $('science-experiment-preview').onclick=()=>act(()=>command(()=>previewDraft()));
      $('science-experiment-prev').onclick=()=>act(()=>command(()=>previewDraft(Math.max(0,draftPage.preview.offset-25))));
      $('science-experiment-next').onclick=()=>act(()=>command(()=>previewDraft(draftPage.preview.next_offset)));
      $('science-task-context-refresh').onclick=()=>act(()=>command(choices));
      for(const id of ['science-task-model','science-task-dataset'])$(id).onchange=()=>act(()=>command(context));
      $('science-task-compatibility').onclick=()=>act(()=>command(async()=>{
        if(!flush())return;const selected=draft,ticket=generation,edit=editGeneration,identity=api.workspace(),model=$('science-task-model').value,dataset=$('science-task-dataset').value;
        if(!model||!dataset)throw Error('Choose exact architecture and dataset revisions first.');
        const result=await api.call('scienceTaskCompatibility',{draft,model:JSON.parse(model),dataset:JSON.parse(dataset)});
        if(ticket!==generation||edit!==editGeneration||draft!==selected||identity!==api.workspace()||model!==$('science-task-model').value||dataset!==$('science-task-dataset').value)return;
        $('science-task-compatibility-status').textContent=result.status+(result.reason?' · '+result.reason:' · interface only; no fitting, execution or RAM admission')+' · model '+result.model.revision.slice(0,12)+' · dataset '+result.dataset.revision.slice(0,12);
      }));
      $('science-author-add-field').onclick=()=>{try{if(!flush())return;const fid=$('science-author-field-id').value;
        draft.document=helpers().addField(draft.document,{id:fid,dtype:'float32',shape:[]});$('science-author-field-id').value='';fieldOffset=Math.floor((draft.document.fields.length-1)/6)*6;render();}catch(error){issue(error);}};
      $('science-author-add-record').onclick=()=>{try{if(!flush())return;const control=$('science-author-record-id');draft.document=helpers().addRecord(draft.document,control.value||uid());control.value='';rowOffset=Math.floor((draft.document.records.length-1)/20)*20;render();}catch(error){issue(error);}};
      for(const [id,axis,delta] of [['science-author-fields-prev','field',-6],['science-author-fields-next','field',6],['science-author-rows-prev','row',-20],['science-author-rows-next','row',20]])$(id).onclick=()=>{if(!flush())return;if(axis==='field')fieldOffset=Math.max(0,fieldOffset+delta);else rowOffset=Math.max(0,rowOffset+delta);render();};
      $('science-author-check').onclick=()=>act(()=>command(async()=>{if(!flush())return;const selected=draft,ticket=generation,edit=editGeneration,identity=api.workspace();
        const result=await api.call('scienceAuthorCheck',{draft});if(ticket!==generation||edit!==editGeneration||draft!==selected||identity!==api.workspace())return;
        $('science-author-status').textContent='Valid '+draft.kind[0].toUpperCase()+draft.kind.slice(1)+'Spec · prospective revision '+result.revision+' · not saved, no training';}));
      $('science-author-save').onclick=()=>act(()=>command(async()=>{if(!flush())return;const selected=draft,ticket=generation,identity=api.workspace();
        const result=await api.call('scienceAuthorSave',{draft});if(ticket!==generation||draft!==selected||identity!==api.workspace())return;
        draft=null;contextGeneration++;render();await saved(result);api.toast(result.kind[0].toUpperCase()+result.kind.slice(1)+' revision retained. Save workspace to persist it on disk.');}));
      controls();
    },
    stateChanged(){if(!api)return;if(draft)status();else controls();},
    selectionChanged(){controls();}
  };
})();
