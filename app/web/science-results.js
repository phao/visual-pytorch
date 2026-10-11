"use strict";
(() => {
  let root,api,selectExact,current=null,workspace=null,selection=[],runOffset=0,reportOffset=0,attempt=null;
  let page=null,generation=0,reading=null,writing=false;
  let sourceGeneration=0,sourceReading=null,exportGeneration=0,exporting=null;
  const $=id=>root.getElementById(id),H=()=>window.LoomRunEvidence;
  const el=(tag,value)=>{const node=root.createElement(tag);if(value!==undefined)node.textContent=value;return node;};
  const act=fn=>Promise.resolve().then(fn).catch(error=>api.toast(error.message,true));
  function controls(){
    if(!api)return;
    const busy=api.busy()||writing;
    $('science-report-add').disabled=busy||current?.reference.kind!=='run';
    $('science-comparison-create').disabled=busy||selection.length===0||!api.workspace();
    $('science-report-selection').querySelectorAll('button').forEach(b=>b.disabled=busy);
    for(const id of ['science-comparison-name','science-comparison-question'])$(id).disabled=busy;
    const wait=reading!==null;
    $('science-evidence-read').disabled=wait||current?.reference.kind!=='run';
    $('science-evidence-attempt').disabled=wait;
    $('science-metrics-prev').disabled=wait||current?.reference.kind!=='run'||!page||runOffset===0;
    $('science-metrics-next').disabled=wait||current?.reference.kind!=='run'||!page||page.metrics.next_offset===null;
    $('science-report-prev').disabled=wait||current?.reference.kind!=='report'||!page||reportOffset===0;
    $('science-report-next').disabled=wait||current?.reference.kind!=='report'||!page||page.runs.next_offset===null;
    const checkpoint=current?.reference.kind==='run'&&page?.execution;
    $('science-predictor-view').disabled=wait||sourceReading!==null||!checkpoint;
    $('science-predictor-export').disabled=wait||exporting!==null||api.busy()||!checkpoint;
  }
  function clearSource(){
    sourceGeneration++;exportGeneration++;sourceReading=null;exporting=null;
    $('science-predictor').hidden=true;$('science-predictor-code').textContent='';$('science-predictor-status').textContent='';
  }
  async function predictor(save){
    const selected=current,evidence=page,identity=api.workspace(),request=H().predictorRequest(evidence,identity);
    const ticket=save?++exportGeneration:++sourceGeneration;
    if(save)exporting=ticket;else sourceReading=ticket;controls();
    try{
      const result=await api.call(save?'scienceExportRunPredictor':'scienceRunPredictor',request);
      if(!result||identity!==api.workspace())return;
      if(save){if(ticket===exportGeneration)api.toast('Checkpoint predictor source exported (not executed): '+result.path);}
      else if(ticket===sourceGeneration&&current===selected&&page===evidence){
        $('science-predictor').hidden=false;$('science-predictor').open=true;
        $('science-predictor-status').textContent='Retained attempt '+result.provenance.attempt+' · '+result.provenance.status+
          ' · '+result.provenance.completed_updates+' successful updates · source SHA-256 '+result.code_sha256+
          ' · stable output IDs; classification logits; declared transformed target space. No optimizer resume.';
        $('science-predictor-code').textContent=result.code;
      }
    }finally{if(save&&exporting===ticket)exporting=null;if(!save&&sourceReading===ticket)sourceReading=null;controls();}
  }
  function selectionView(){
    const list=$('science-report-selection');list.replaceChildren();
    selection.forEach(ref=>{const card=el('div');card.className='science-report-choice';card.dataset.reportRun=ref.id;card.dataset.reportRevision=ref.revision;
      card.append(el('span',ref.id+' @ '+ref.revision.slice(0,12)));
      const remove=el('button','Remove selection');remove.onclick=()=>{selection=selection.filter(r=>r.id!==ref.id);selectionView();};card.append(remove);list.append(card);});
    $('science-report-selection-label').textContent='Compare exact saved runs · '+selection.length+' selected';controls();
  }
  async function read(){
    if(!current)return;
    const selected=current,ticket=++generation,identity=api.workspace();
    reading=ticket;controls();
    try{
      const result=await api.call(selected.reference.kind==='run'?'scienceRunEvidence':'scienceReportEvidence',
        {reference:selected.reference,...(selected.reference.kind==='run'?{attempt,offset:runOffset,limit:25}:{offset:reportOffset,limit:5})});
      if(ticket!==generation||current!==selected||identity!==api.workspace())return;
      page=result;
      clearSource();
      if(selected.reference.kind==='run'){
        $('science-evidence-attempt').value=result.attempt.number;
        $('science-evidence-attempt').max=result.attempt_count;
        window.LoomEvidenceView.render(root,$('science-run-evidence'),result);
      }else{
        const list=$('science-report-evidence');list.replaceChildren();
        const description=el('p',result.question+' · '+result.execution_behavior+' · descriptive selected runs, not independent replicates by assertion. No metrics are pooled across runs.');list.append(description);
        result.runs.rows.forEach(run=>{const card=el('section');card.className='science-evidence';list.append(card);window.LoomEvidenceView.render(root,card,run,ref=>act(()=>selectExact(ref)));});
        $('science-report-page').textContent='Exact saved runs '+result.runs.offset+'–'+(result.runs.offset+result.runs.rows.length)+' of '+result.runs.total+
          (result.mode==='rerun'?' · no saved runs: execution requires an explicit notebook action.':' · no training');
      }
    }finally{if(reading===ticket)reading=null;controls();}
  }
  async function show(reference,document){
    const changed=!current||H().key(current.reference)!==H().key(reference);
    current={reference:{...reference},document};workspace=api.workspace();
    if(changed){attempt=null;runOffset=0;reportOffset=0;page=null;
      $('science-run-evidence').replaceChildren();delete $('science-run-evidence').dataset.runEvidence;$('science-report-evidence').replaceChildren();}
    await read();
  }
  async function create(){
    if(writing)return;
    const identity=api.workspace(),runs=selection.map(r=>({...r}));
    if(!runs.length)throw Error('Select at least one exact saved run.');
    writing=true;controls();
    try{
      const result=await api.call('scienceResultReport',{workspace_id:identity,runs,
        name:$('science-comparison-name').value,question:$('science-comparison-question').value});
      if(identity!==api.workspace())return;
      await selectExact(result);api.toast('Saved comparison recipe retained; no training. Save workspace to persist it.');
    }finally{writing=false;controls();}
  }
  window.LoomScienceResults={
    mount(document,services,onSelect){
      root=document;api=services;selectExact=onSelect;workspace=api.workspace();
      $('science-report-add').onclick=()=>act(()=>{if(current?.reference.kind!=='run')throw Error('Select an exact run first.');
        selection=H().add(selection,current.reference);$('science-report-composer').open=true;selectionView();});
      $('science-comparison-create').onclick=()=>act(create);
      $('science-evidence-read').onclick=()=>act(()=>{attempt=window.LoomExperimentAuthoring.number($('science-evidence-attempt').value,
        {integer:true,minimum:1,maximum:current.document.attempt_count});runOffset=0;return read();});
      $('science-metrics-prev').onclick=()=>act(()=>{runOffset=Math.max(0,runOffset-25);return read();});
      $('science-metrics-next').onclick=()=>act(()=>{runOffset=page.metrics.next_offset;return read();});
      $('science-report-prev').onclick=()=>act(()=>{reportOffset=Math.max(0,reportOffset-5);return read();});
      $('science-report-next').onclick=()=>act(()=>{reportOffset=page.runs.next_offset;return read();});
      $('science-predictor-view').onclick=()=>act(()=>predictor(false));
      $('science-predictor-export').onclick=()=>act(()=>predictor(true));
      selectionView();
    },
    show,
    clear(){current=null;page=null;generation++;reading=null;clearSource();$('science-run-evidence').replaceChildren();delete $('science-run-evidence').dataset.runEvidence;$('science-report-evidence').replaceChildren();$('science-report-page').textContent='';controls();},
    stateChanged(){
      if(workspace!==api.workspace()){workspace=api.workspace();selection=[];writing=false;this.clear();selectionView();}
      controls();
    }
  };
})();
