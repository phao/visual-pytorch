"use strict";
(() => {
  let root,api,workspace=null,selected=null,generation=0,reading=false,starting=false,rows=[],historyGeneration=0;
  const $=id=>root.getElementById(id),H=window.LoomNotebookSelection;
  const text=(tag,value)=>{const el=root.createElement(tag);el.textContent=value;return el;};
  function fail(error){if(!error.vpwNotified)api.toast(error.message,true);}
  function act(fn){Promise.resolve().then(fn).catch(fail);}
  function controls(){
    if(!api)return;
    const blocked=api.busy()||reading||starting;
    for(const id of ['science-notebook-inspect','science-notebook-python','science-notebook-timeout'])$(id).disabled=blocked;
    $('science-notebook-execute').disabled=blocked||!selected;
    $('science-notebook-execute').textContent=selected?.mode==='rerun'?'Run rerun notebook (trains)…':'Run saved-result notebook…';
    $('science-notebook-stop').hidden=!api.notebookJob();
    $('science-notebook-stop').textContent=api.notebookJob()?.operation==='inspect'?'Stop inspection':'Stop notebook';
  }
  function clear(){
    selected=null;generation++;reading=false;starting=false;
    $('science-notebook-selection').textContent='No bundle inspected. Generation does not run cells.';
    $('science-notebook-status').textContent='';
    $('science-notebook-evidence').replaceChildren();
    rows=[];historyGeneration++;$('science-notebook-history').replaceChildren();controls();
  }
  async function inspect(directory=null){
    const ticket=++generation,identity=api.workspace(),python=$('science-notebook-python').value.trim()||api.python();
    reading=true;controls();let admitted=false;
    try{
      const result=await api.call('scienceNotebookInspect',{workspace_id:identity,python,request_id:String(ticket),...(directory?{directory}:{})});
      if(ticket!==generation||identity!==api.workspace())return;
      if(!result)return;
      admitted=true;
      if(reading)$('science-notebook-status').textContent='Inspecting input closure asynchronously · '+result.job_id+' · no kernel execution.';
      $('science-notebook').open=true;
    }catch(error){if(ticket===generation&&identity===api.workspace())throw error;}
    finally{if(ticket===generation&&!admitted){reading=false;controls();}}
  }
  async function execute(){
    const ticket=++generation,identity=api.workspace(),captured=selected;
    const req=H.request(captured,identity,$('science-notebook-python').value,Number($('science-notebook-timeout').value));
    starting=true;controls();
    try{
      const result=await api.call('scienceExecuteNotebook',req);
      if(ticket!==generation||identity!==api.workspace()||selected!==captured)return;
      if(result)$('science-notebook-status').textContent='Started, not completed · '+result.job_id+' · '+H.behavior(captured.mode);
    }catch(error){if(ticket===generation&&identity===api.workspace())throw error;}
    finally{if(ticket===generation){starting=false;controls();}}
  }
  function evidence(row){
    const area=$('science-notebook-evidence');area.replaceChildren();
    const card=root.createElement('section');card.className='science-evidence';
    card.append(text('h4','Notebook execution · '+row.status),text('p',H.behavior(row.selection.mode)),
      text('p','Origin workspace: '+row.workspace_id+' · '+(row.workspace_id===api.workspace()?'current workspace':'another workspace')),
      text('p','Exact recipe: '+row.selection.recipe.id+' · '+row.selection.recipe.revision));
    const receipt=row.result?.execution;
    if(receipt){
      card.append(text('p','Cells with execution counts: '+receipt.executed_code_cells+' · elapsed '+receipt.elapsed_seconds.toFixed(3)+' s'),
        text('p','Artifacts: '+row.result.path),text('p','Python: '+receipt.interpreter+' · per-cell timeout '+receipt.cell_timeout_seconds+' s'),
        text('p','Notebook completion is not a claim that every training candidate succeeded. Runs stay in the bundle; no workspace import.'));
      if(receipt.error)card.append(text('p',receipt.error.type+': '+receipt.error.message));
      const list=root.createElement('dl');list.className='science-evidence-facts';
      for(const [name,info] of Object.entries(receipt.artifacts)){list.append(text('dt',name),text('dd',info.bytes+' bytes · '+info.sha256));}
      const disclosure=root.createElement('details');disclosure.append(text('summary','Artifacts · '+Object.keys(receipt.artifacts).length+' retained files · exact byte/hash index'),list);
      card.append(disclosure);
    }else card.append(text('p','No verified terminal receipt: '+row.error),text('p','Inspect retained files in '+row.selection.path+'; no completed/cancelled notebook is fabricated.'));
    area.append(card);$('science-notebook').open=true;
  }
  async function history(preferred=null){
    const ticket=++historyGeneration,identity=api.workspace(),result=await api.call('scienceNotebookHistory');
    if(ticket!==historyGeneration||identity!==api.workspace())return;
    rows=result;const select=$('science-notebook-history'),previous=preferred||select.value;select.replaceChildren();
    rows.forEach(row=>{const option=text('option',row.status+' · '+row.selection.recipe.id+' · '+row.id.slice(0,12)+(row.workspace_id===identity?'':' · other workspace'));option.value=row.id;select.append(option);});
    if([...select.options].some(o=>o.value===previous))select.value=previous;
    if(select.value)await readEvidence();
  }
  async function readEvidence(){
    const id=$('science-notebook-history').value,identity=api.workspace(),ticket=++historyGeneration;
    if(!id)return;
    $('science-notebook-evidence').replaceChildren();
    const row=await api.call('scienceNotebookEvidence',{id});
    if(ticket!==historyGeneration||identity!==api.workspace()||$('science-notebook-history').value!==id)return;
    evidence(row);
  }
  window.LoomScienceNotebooks={
    mount(document,services){root=document;api=services;workspace=api.workspace();$('science-notebook-python').value=api.python()||'';
      $('science-notebook-inspect').onclick=()=>act(()=>inspect());
      $('science-notebook-execute').onclick=()=>act(execute);
      $('science-notebook-stop').onclick=()=>act(()=>api.call('stop'));
      $('science-notebook-refresh').onclick=()=>act(()=>history());
      $('science-notebook-history').onchange=()=>act(readEvidence);controls();
    },
    generated(value,origin){if(origin===api.workspace())return inspect(value.path);return Promise.resolve();},
    stateChanged(){if(!api)return;if(workspace!==api.workspace()){workspace=api.workspace();clear();}controls();},
    event(event){if(!api||event.payload?.workspace_id!==api.workspace())return;
      if(event.kind==='scienceNotebookInspected'&&event.payload.request_id===String(generation)){
        reading=false;
        const result=event.payload.inspection;
        if(result){
          selected=H.selection(result.selection);$('science-notebook-python').value=result.python;
          $('science-notebook-selection').textContent=H.behavior(selected.mode)+'\nBundle: '+selected.path+'\nRecipe: '+selected.recipe.id+' · '+selected.recipe.revision+
            '\nRequested notebook: '+selected.requested_sha256+'\nInput index: '+selected.input_index_hash;
          $('science-notebook-status').textContent='Verified input closure. No kernel launched; hashes are not authenticity or a sandbox.';
        }else{
          $('science-notebook-status').textContent=event.payload.status+' · '+event.payload.error;
          api.toast(event.payload.error,true);
        }
        controls();
      }
      if(event.kind==='scienceNotebookProgress'){
        const p=event.payload.progress;
        $('science-notebook-status').textContent='Active notebook '+event.payload.job_id+' · '+p.stage+(p.cell_index===null?'':' · cell '+(p.cell_index+1)+'/'+p.cell_count+' · '+(p.section||'code'));
      }
      if(event.kind==='scienceNotebookComplete'){
        $('science-notebook-status').textContent='Notebook '+event.payload.status+' · '+(event.payload.result?.path||event.payload.selection.path);
        api.toast('Notebook '+event.payload.status+'. '+(event.payload.result?'Receipt and artifacts retained in '+event.payload.result.path:'No verified receipt; inspect the bundle directory.'),event.payload.status!=='completed');
        act(()=>history(event.payload.id));
      }
    }
  };
})();
