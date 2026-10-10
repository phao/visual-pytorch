"use strict";
(() => {
  let root, api, saved, draft=null, rowOffset=0, fieldOffset=0, generation=0, pending=false;
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
    control.addEventListener('change',()=>{try{onchange(control.value);control.removeAttribute('aria-invalid');delete control.dataset.invalid;status();}catch(error){control.setAttribute('aria-invalid','true');control.dataset.invalid='';issue(error);}});
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
    $('science-author-status').textContent=stale?'Workspace changed. This unsaved draft is retained for inspection; cancel and reopen it in the intended workspace.':
      'Unsaved dataset draft · '+draft.document.fields.length+' fields · '+draft.document.records.length+' observations · validation never trains';
    controls();
  }
  function controls(){
    if(!api)return;
    const blocked=pending||!draft||api.busy()||draft.workspace_id!==api.workspace();
    for(const id of ['science-author-save','science-author-check','science-author-add-field','science-author-add-record'])$(id).disabled=blocked;
    $('science-author-new').disabled=pending||api.busy()||Boolean(draft)||!api.workspace();
    $('science-author-edit').disabled=pending||api.busy()||Boolean(draft)||!api.workspace()||$('science-kind').value!=='dataset'||!$('science-object').value;
    $('science-author-cancel').disabled=pending;
    if(draft)$('science-author').querySelectorAll('[data-author-control]').forEach(control=>control.disabled=blocked);
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
    const active=root.activeElement;if(active&&$('science-author').contains(active))active.dispatchEvent(new Event('change'));
    if($('science-author').querySelector('[data-invalid]')){issue(Error('Correct invalid draft fields before continuing. Your text has not been discarded.'));return false;}return true;
  }
  function render(){
    $('science-author').hidden=!draft;if(!draft){controls();return;}
    $('science-author-id').textContent='Stable dataset ID: '+draft.id+' · '+(draft.base?'editing exact '+draft.base.revision.slice(0,12):'new object')+(draft.base&&draft.base.revision!==draft.expected_head?' · historical base, not current head':'');
    const name=$('science-author-name');name.value=draft.document.name;name.onchange=()=>{draft.document.name=name.value;status();};
    fields();records();const metadata=$('science-author-metadata');metadata.replaceChildren();
    for(const key of ['provenance','usage'])optional(metadata,draft.document,key,key);
    status();
  }
  async function begin(reference=null){
    const ticket=++generation,identity=api.workspace();
    const result=await api.call('scienceAuthorBegin',{kind:'dataset',...(reference?{reference}:{})});
    if(ticket!==generation||identity!==api.workspace())return;
    draft=result;rowOffset=0;fieldOffset=0;render();
  }
  window.LoomScienceAuthor={
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;
      $('science-author-new').onclick=()=>act(()=>command(()=>begin()));
      $('science-author-edit').onclick=()=>act(()=>command(()=>begin(JSON.parse($('science-object').value))));
      $('science-author-cancel').onclick=()=>{draft=null;generation++;render();};
      $('science-author-add-field').onclick=()=>{try{if(!flush())return;const fid=$('science-author-field-id').value;
        draft.document=helpers().addField(draft.document,{id:fid,dtype:'float32',shape:[]});$('science-author-field-id').value='';fieldOffset=Math.floor((draft.document.fields.length-1)/6)*6;render();}catch(error){issue(error);}};
      $('science-author-add-record').onclick=()=>{try{if(!flush())return;const control=$('science-author-record-id');draft.document=helpers().addRecord(draft.document,control.value||uid());control.value='';rowOffset=Math.floor((draft.document.records.length-1)/20)*20;render();}catch(error){issue(error);}};
      for(const [id,axis,delta] of [['science-author-fields-prev','field',-6],['science-author-fields-next','field',6],['science-author-rows-prev','row',-20],['science-author-rows-next','row',20]])$(id).onclick=()=>{if(!flush())return;if(axis==='field')fieldOffset=Math.max(0,fieldOffset+delta);else rowOffset=Math.max(0,rowOffset+delta);render();};
      $('science-author-check').onclick=()=>act(()=>command(async()=>{if(!flush())return;const selected=draft,ticket=generation;
        const result=await api.call('scienceAuthorCheck',{draft});if(ticket!==generation||draft!==selected)return;
        $('science-author-status').textContent='Valid DatasetSpec · prospective revision '+result.revision+' · not saved, no training';}));
      $('science-author-save').onclick=()=>act(()=>command(async()=>{if(!flush())return;const selected=draft,ticket=generation;
        const result=await api.call('scienceAuthorSave',{draft});if(ticket!==generation||draft!==selected)return;
        draft=null;render();await saved(result);api.toast('Dataset revision retained. Save workspace to persist it on disk.');}));
      controls();
    },
    stateChanged(){if(!api)return;if(draft)status();else controls();},
    selectionChanged(){controls();}
  };
})();
