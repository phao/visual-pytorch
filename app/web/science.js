"use strict";
(() => {
  let root, api, rows=[], current=null, generation=0, workspace=null;
  let indexPage=null,indexOffset=0,indexBacks=[],indexReading=null;
  const $=id=>root.getElementById(id);
  const ref=row=>({kind:row.kind,id:row.id,revision:row.revision});
  const key=row=>JSON.stringify(ref(row));
  const text=(tag,value)=>{const el=root.createElement(tag);el.textContent=value;return el;};
  async function request(action,args={}){try{return await api.call(action,args);}catch(error){api.toast(error.message,true);error.vpwNotified=true;throw error;}}
  async function refresh(preferred=null){
    if(!api)return;
    const ticket=++generation, identity=api.workspace();
    if(!identity){rows=[];current=null;return;}
    const select=$('science-object'),kind=$('science-kind').value,previous=preferred?key(preferred):select.value;
    const pinned=previous?JSON.parse(previous):null;
    indexReading=ticket;indexPage=null;controls();
    try{
    const result=await api.call('scienceBrowse',{workspace_id:identity,kind,offset:indexOffset,limit:25,
      query:$('science-object-query').value,preferred:pinned?.kind===kind?pinned:null});
    if(ticket!==generation || identity!==api.workspace())return;
    indexPage=result;rows=[...result.rows];workspace=identity;
    if(result.pinned&&!rows.some(r=>key(r)===key(result.pinned)))rows.push({...result.pinned,pinned:true});
    $('science-object-page').textContent=result.offset+'–'+(result.offset+result.rows.length)+' of '+result.total+
      ' exact revisions'+(rows.some(r=>r.pinned)?' · selected identity pinned outside this page':'');
    select.replaceChildren();
    rows.forEach(row=>{const option=text('option',row.name+' · '+row.revision.slice(0,12)+(row.selected?' · current head':' · historical')+(row.pinned?' · pinned selection':''));option.value=key(row);select.append(option);});
    if([...select.options].some(o=>o.value===previous))select.value=previous;
    else {const head=rows.find(r=>r.kind===kind && r.selected);if(head)select.value=key(head);}
    await inspect();
    }catch(error){if(ticket===generation&&identity===api.workspace())api.toast(error.message,true);}
    finally{if(indexReading===ticket)indexReading=null;controls();}
  }
  async function inspect(){
    const ticket=++generation,identity=api.workspace(),value=$('science-object').value;
    current=null;window.LoomScienceCandidates?.clear();
    window.LoomScienceResults.clear();
    window.LoomScienceDocument.clear();
    for(const id of ['science-selection','science-run','science-report'])$(id).hidden=true;
    $('science-candidates').replaceChildren();$('science-document').textContent='';
    if(!value){const kind=$('science-kind').selectedOptions[0].textContent.toLowerCase();
      $('science-status').textContent=$('science-object-query').value||indexOffset>0?
        'No matching '+kind+' revisions on this page. Clear the search or return to an earlier page.':
        'No '+kind+' stored yet. Use the dataset/task/experiment authoring controls, freeze an architecture, import a specification, or use the CLI.';
      window.LoomScienceAuthor?.selectionChanged();return;}
    const reference=JSON.parse(value);let document;
    try{document=await api.call('scienceSummary',{workspace_id:identity,reference});}
    catch(error){if(ticket===generation&&identity===api.workspace())api.toast(error.message,true);return;}
    if(ticket!==generation || identity!==api.workspace())return;
    current={reference,document};$('science-document').dataset.projection='summary';$('science-document').textContent=JSON.stringify(document,null,2);
    window.LoomScienceDocument.select(reference);
    $('science-status').textContent=reference.kind+' '+reference.id+' · '+reference.revision+' · browsing does not execute';
    if(reference.kind==='experiment'){
      $('science-selection').hidden=false;$('science-question').textContent=document.question.text+(document.question.next_offset!==null?' … [partial question; inspect full text below]':'');
      await window.LoomScienceCandidates.show(reference);
    }else if(reference.kind==='run'){
      $('science-run').hidden=false;
      const last=document.latest;
      $('science-status').textContent='Attempt '+last.number+' · '+last.status+' · '+last.stage+(last.error?' · '+last.error.message.text+(last.error.message.next_offset!==null?' … [partial diagnostic]':''):'')+' · exact '+reference.revision;
      await window.LoomScienceResults.show(reference,document);
    }else if(reference.kind==='report'){
      $('science-report').hidden=false;
      $('science-report-mode').textContent=document.mode==='result-report'?'Saved-result report: reads these exact runs; no training.':'Explicit rerun recipe: generation does not train; running its notebook will train.';
      await window.LoomScienceResults.show(reference,document);
    }
    controls();
    window.LoomScienceAuthor?.selectionChanged();
  }
  function controls(){
    if(!api)return;
    const active=api.busy();
    for(const id of ['science-import','science-freeze','science-retry','science-make-report','science-generate'])$(id).disabled=active;
    $('science-freeze').disabled=active||!api.model();$('science-stop').hidden=!active;
    $('science-stop').textContent=api.notebookJob()?'Stop notebook work':'Stop run';
    window.LoomScienceCandidates?.stateChanged();
    window.LoomScienceRecoveries?.stateChanged();
    $('science-object-prev').disabled=indexReading!==null||!indexBacks.length;
    $('science-object-next').disabled=indexReading!==null||!indexPage||indexPage.next_offset===null;
  }
  function act(fn){Promise.resolve().then(fn).catch(error=>{if(!error.vpwNotified)api.toast(error.message,true);});}
  window.LoomScience={
    mount(document,services){root=document;api=services;
      window.LoomScienceCandidates?.mount(document,services,()=>indexReading!==null);
      window.LoomScienceResults.mount(document,services,async reference=>{$('science-kind').value=reference.kind;await refresh(reference);});
      window.LoomScienceAuthor.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      window.LoomScienceNotebooks.mount(document,services);
      window.LoomDraftValueEditor?.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      window.LoomDatasetSessionForm?.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      window.LoomTaskSessionForm?.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      window.LoomExperimentSessionForm?.mount(document,services,async result=>{$('science-kind').value=result.kind;await refresh(result);});
      window.LoomScienceDocument.mount(document,{...services,editing:()=>window.LoomDraftValueEditor?.active?.()||window.LoomScienceAuthor?.active?.(),editValue:target=>window.LoomDraftValueEditor?.open(target)});
      window.LoomScienceRecoveries?.mount(document,services);
      root.querySelector('[data-panel=science]').addEventListener('click',()=>act(refresh));
      $('science-refresh').onclick=()=>act(refresh);
      $('science-kind').onchange=()=>{indexOffset=0;indexBacks=[];act(refresh);};
      $('science-object-search').onclick=()=>{indexOffset=0;indexBacks=[];act(refresh);};
      $('science-object-query').onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();indexOffset=0;indexBacks=[];act(refresh);}};
      $('science-object-next').onclick=()=>act(()=>{if(indexReading!==null||!indexPage||indexPage.next_offset===null)return;indexBacks.push(indexOffset);indexOffset=indexPage.next_offset;return refresh();});
      $('science-object-prev').onclick=()=>act(()=>{if(indexReading!==null||!indexBacks.length)return;indexOffset=indexBacks.pop();return refresh();});
      $('science-object').onchange=()=>act(inspect);
      $('science-stop').onclick=()=>act(()=>request('stop'));
      $('science-freeze').onclick=()=>act(async()=>{const result=await request('scienceFreeze',{id:api.model()});$('science-kind').value='model';await refresh(result);});
      $('science-import').onclick=()=>act(async()=>{const result=await request('scienceImport',{kind:$('science-kind').value});await refresh(result);});
      $('science-retry').onclick=()=>act(async()=>{const run=current.document;await request('scienceTrain',{workspace_id:api.workspace(),reference:run.selection.experiment,coordinates:run.selection.coordinates,run_id:run.id,name:run.name,retry:true});controls();});
      $('science-make-report').onclick=()=>act(async()=>{const result=await request('scienceResultReport',{runs:[current.reference],question:$('science-report-question').value});$('science-kind').value='report';await refresh(result);});
      $('science-generate').onclick=()=>act(async()=>{const identity=api.workspace(),reference=current.reference;const result=await request('scienceGenerateNotebook',{reference});if(result&&identity===api.workspace()){api.toast('Notebook generated, not executed: '+result.path);await window.LoomScienceNotebooks.generated(result,identity);}});
      root.querySelector('[data-panel=science]').addEventListener('click',()=>window.LoomScienceRecoveries?.refresh());
    },
    stateChanged(){if(!api)return;window.LoomScienceResults.stateChanged();window.LoomScienceNotebooks.stateChanged();if(workspace!==api.workspace()){workspace=api.workspace();rows=[];current=null;indexPage=null;indexOffset=0;indexBacks=[];generation++;window.LoomScienceCandidates?.clear();window.LoomScienceDocument.clear();$('science-object').replaceChildren();$('science-document').textContent='';$('science-candidates').replaceChildren();for(const id of ['science-selection','science-run','science-report'])$(id).hidden=true;if(!$('science-panel').hidden)act(refresh);}controls();window.LoomScienceAuthor?.stateChanged();window.LoomDraftValueEditor?.stateChanged?.();window.LoomDatasetSessionForm?.stateChanged?.();window.LoomTaskSessionForm?.stateChanged?.();window.LoomExperimentSessionForm?.stateChanged?.();window.LoomScienceDocument.stateChanged?.();},
    event(event){if(!api)return;
      window.LoomDraftValueEditor?.event?.(event);
      window.LoomDatasetSessionForm?.event?.(event);
      window.LoomTaskSessionForm?.event?.(event);
      window.LoomExperimentSessionForm?.event?.(event);
      window.LoomScienceAuthor?.event?.(event);
      window.LoomScienceNotebooks.event(event);
      if(event.kind==='externalReload'){
        generation++;indexReading=null;current=null;indexPage=null;window.LoomScienceCandidates?.clear();
        window.LoomScienceResults.clear();window.LoomScienceDocument.clear();
        $('science-document').textContent='';$('science-candidates').replaceChildren();
        for(const id of ['science-selection','science-run','science-report'])$(id).hidden=true;
        controls();if(!$('science-panel').hidden)act(refresh);
      }
      if(event.kind==='scienceChanged' && !$('science-panel').hidden)act(refresh);
      if(event.kind==='scienceProgress')$('science-status').textContent='Active run '+event.payload.run_id+' · '+JSON.stringify(event.payload.point);
      if(event.kind==='scienceComplete'&&event.payload.scope==='experiment-completion-notification'){
        const attempt=event.payload.attempt;api.toast('Experiment attempt '+attempt.number+': '+attempt.status+'. Save workspace to retain it on disk.');}
      if(event.kind==='scienceRecovery'&&event.payload.scope==='experiment-recovery-notification'){
        api.toast('Result admission failed. Inspect or export recovery evidence before closing the app.',true);}
    }
  };
})();
