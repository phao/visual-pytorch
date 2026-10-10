"use strict";
(() => {
  let root, api, rows=[], current=null, page=null, offset=0, generation=0, workspace=null;
  const $=id=>root.getElementById(id);
  const ref=row=>({kind:row.kind,id:row.id,revision:row.revision});
  const key=row=>JSON.stringify(ref(row));
  const text=(tag,value)=>{const el=root.createElement(tag);el.textContent=value;return el;};
  async function request(action,args={}){try{return await api.call(action,args);}catch(error){api.toast(error.message,true);error.vpwNotified=true;throw error;}}
  async function refresh(preferred=null){
    if(!api)return;
    const ticket=++generation, identity=api.workspace();
    if(!identity){rows=[];current=null;return;}
    const result=await request('scienceIndex');
    if(ticket!==generation || identity!==api.workspace())return;
    rows=result;workspace=identity;
    const select=$('science-object'),previous=preferred?key(preferred):select.value,kind=$('science-kind').value;
    select.replaceChildren();
    rows.filter(r=>r.kind===kind).forEach(row=>{const option=text('option',row.name+' · '+row.revision.slice(0,12)+(row.selected?' · current head':' · historical'));option.value=key(row);select.append(option);});
    if([...select.options].some(o=>o.value===previous))select.value=previous;
    else {const head=rows.find(r=>r.kind===kind && r.selected);if(head)select.value=key(head);}
    await inspect();
  }
  async function inspect(){
    const ticket=++generation,identity=api.workspace(),value=$('science-object').value;
    current=null;page=null;
    for(const id of ['science-selection','science-run','science-report'])$(id).hidden=true;
    $('science-candidates').replaceChildren();$('science-document').textContent='';
    if(!value){$('science-status').textContent='No '+$('science-kind').selectedOptions[0].textContent.toLowerCase()+' stored yet. Use New dataset, import a specification, or create it using the CLI.';window.LoomScienceAuthor?.selectionChanged();return;}
    const reference=JSON.parse(value),document=await request('scienceShow',{reference});
    if(ticket!==generation || identity!==api.workspace())return;
    current={reference,document};$('science-document').textContent=JSON.stringify(document,null,2);
    $('science-status').textContent=reference.kind+' '+reference.id+' · '+reference.revision+' · browsing does not execute';
    if(reference.kind==='experiment'){
      $('science-selection').hidden=false;$('science-question').textContent=document.question;
      await preview();
    }else if(reference.kind==='run'){
      $('science-run').hidden=false;$('science-details').open=true;
      const last=document.attempts.at(-1);
      $('science-status').textContent='Attempt '+last.number+' · '+last.status+' · '+last.stage+(last.error?' · '+last.error.message:'')+' · exact '+reference.revision;
    }else if(reference.kind==='report'){
      const evidence=await request('scienceReport',{reference});
      if(ticket!==generation || identity!==api.workspace())return;
      $('science-report').hidden=false;
      $('science-report-mode').textContent=document.mode==='result-report'?'Saved-result report: reads these exact runs; no training.':'Explicit rerun recipe: generation does not train; running its notebook will train.';
      $('science-document').textContent=JSON.stringify(evidence,null,2);$('science-details').open=true;
    }
    controls();
    window.LoomScienceAuthor?.selectionChanged();
  }
  async function preview(){
    const selected=current,ticket=++generation,identity=api.workspace();
    const result=await request('sciencePreview',{reference:selected.reference,offset,limit:25});
    if(ticket!==generation || current!==selected || identity!==api.workspace())return;
    page=result;$('science-page').textContent=result.offset+'–'+(result.offset+result.candidates.length)+' of '+result.total_combinations+' · no training';
    const container=$('science-candidates');container.replaceChildren();
    result.candidates.forEach(row=>{
      const card=root.createElement('div');card.className='science-candidate';
      card.append(text('span',row.status+' · '+row.coordinates.model.id+' · '+row.coordinates.data_task+' · seed '+row.coordinates.initialization+' · '+row.coordinates.split+' · '+row.coordinates.protocol));
      if(row.reason)card.append(text('small',row.reason));
      const run=text('button','Run this candidate');run.dataset.scienceRun='';run.disabled=api.busy()||row.status!=='compatible';
      run.addEventListener('click',()=>act(async()=>{await request('scienceTrain',{reference:selected.reference,coordinates:row.coordinates});api.toast('Explicit experiment run started. Results will be retained as independent evidence.');controls();}));
      card.append(run);container.append(card);
    });controls();
  }
  function controls(){
    if(!api)return;
    const active=api.busy();
    for(const id of ['science-import','science-freeze','science-retry','science-make-report','science-generate'])$(id).disabled=active;
    $('science-freeze').disabled=active||!api.model();$('science-stop').hidden=!active;
    $('science-previous').disabled=offset===0;$('science-next').disabled=!page||page.next_offset===null;
    $('science-candidates').querySelectorAll('[data-science-run]').forEach((button,i)=>button.disabled=active||page.candidates[i].status!=='compatible');
    $('science-recovery-export').disabled=active;
  }
  async function recoveries(){
    const result=await request('scienceRecoveries'),select=$('science-recovery-object'),previous=select.value;
    select.replaceChildren();
    result.forEach(row=>{const option=text('option',row.run_id+' · '+row.error+(row.exported_to?' · exported':''));option.value=row.id;select.append(option);});
    if([...select.options].some(option=>option.value===previous))select.value=previous;
    $('science-recovery').hidden=result.length===0;
    controls();
  }
  function act(fn){Promise.resolve().then(fn).catch(error=>{if(!error.vpwNotified)api.toast(error.message,true);});}
  window.LoomScience={
    mount(document,services){root=document;api=services;
      window.LoomScienceAuthor.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      root.querySelector('[data-panel=science]').addEventListener('click',()=>act(refresh));
      $('science-refresh').onclick=()=>act(refresh);
      $('science-kind').onchange=()=>{offset=0;act(refresh);};
      $('science-object').onchange=()=>{offset=0;act(inspect);};
      $('science-next').onclick=()=>{offset=page.next_offset;act(preview);};
      $('science-previous').onclick=()=>{offset=Math.max(0,offset-25);act(preview);};
      $('science-stop').onclick=()=>act(()=>request('stop'));
      $('science-freeze').onclick=()=>act(async()=>{const result=await request('scienceFreeze',{id:api.model()});$('science-kind').value='model';await refresh(result);});
      $('science-import').onclick=()=>act(async()=>{const result=await request('scienceImport',{kind:$('science-kind').value});await refresh(result);});
      $('science-retry').onclick=()=>act(async()=>{const run=current.document;await request('scienceTrain',{reference:run.selection.experiment,coordinates:run.selection.coordinates,run_id:run.id,name:run.name,retry:true});controls();});
      $('science-make-report').onclick=()=>act(async()=>{const result=await request('scienceResultReport',{runs:[current.reference],question:$('science-report-question').value});$('science-kind').value='report';await refresh(result);});
      $('science-generate').onclick=()=>act(async()=>{const result=await request('scienceGenerateNotebook',{reference:current.reference});if(result)api.toast('Notebook generated, not executed: '+result.path);});
      $('science-recovery-show').onclick=()=>act(async()=>{const result=await request('scienceRecoveryShow',{id:$('science-recovery-object').value});$('science-document').textContent=JSON.stringify(result,null,2);$('science-details').open=true;});
      $('science-recovery-export').onclick=()=>act(async()=>{const result=await request('scienceExportRecovery',{id:$('science-recovery-object').value});if(result)api.toast('Recovery evidence exported: '+result.path);await recoveries();});
      root.querySelector('[data-panel=science]').addEventListener('click',()=>act(recoveries));
    },
    stateChanged(){if(!api)return;if(workspace!==api.workspace()){workspace=api.workspace();rows=[];current=null;page=null;offset=0;generation++;$('science-object').replaceChildren();$('science-document').textContent='';$('science-candidates').replaceChildren();for(const id of ['science-selection','science-run','science-report'])$(id).hidden=true;if(!$('science-panel').hidden)act(refresh);}controls();window.LoomScienceAuthor?.stateChanged();},
    event(event){if(!api)return;
      if(event.kind==='scienceChanged' && !$('science-panel').hidden)act(refresh);
      if(event.kind==='scienceProgress')$('science-status').textContent='Active run '+event.payload.run_id+' · '+JSON.stringify(event.payload.point);
      if(event.kind==='scienceComplete'){const attempt=event.payload.record.attempts.at(-1);api.toast('Experiment attempt '+attempt.number+': '+attempt.status+'. Save workspace to retain it on disk.');}
      if(event.kind==='scienceRecovery'){$('science-document').textContent=JSON.stringify(event.payload,null,2);$('science-details').open=true;act(recoveries);api.toast('Result admission failed. Export recovery evidence before closing the app.',true);}
    }
  };
})();
