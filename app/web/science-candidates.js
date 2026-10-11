"use strict";
(() => {
  let root,api,blocked,selection=null,page=null,offset=0,backs=[],generation=0,reading=null,launching=null;
  const $=id=>root.getElementById(id);
  const text=(tag,value)=>{const el=root.createElement(tag);el.textContent=value;return el;};
  const same=(a,b)=>a?.kind===b?.kind&&a?.id===b?.id&&a?.revision===b?.revision;
  const live=(s,ticket)=>selection===s&&ticket===generation&&s.workspace===api.workspace();
  function controls(){
    if(!api)return;
    const pending=reading!==null;
    $('science-previous').disabled=pending||!selection||!backs.length;
    $('science-next').disabled=pending||!selection||!Number.isSafeInteger(page?.next_offset);
    $('science-candidate-refresh').disabled=pending||!selection;
    $('science-candidates').querySelectorAll('[data-science-run]').forEach((button,i)=>{
      button.disabled=pending||launching!==null||api.busy()||blocked()||!selection||
        selection.workspace!==api.workspace()||page?.candidates[i]?.status!=='compatible';
    });
  }
  function clear(){generation++;selection=null;page=null;offset=0;backs=[];reading=null;
    if(root){$('science-candidates').replaceChildren();$('science-page').textContent='';controls();}}
  async function run(s,p,row){
    if(selection!==s||page!==p||s.workspace!==api.workspace()||reading!==null||launching!==null||api.busy()||blocked()||row.status!=='compatible')return;
    const ticket=generation,token={};launching=token;controls();
    try{await api.call('scienceTrain',{workspace_id:s.workspace,reference:s.reference,coordinates:row.coordinates});
      if(live(s,ticket))api.toast('Explicit experiment run started. Results will be retained as independent evidence.');
    }catch(error){if(live(s,ticket))api.toast(error.message,true);}
    finally{if(launching===token)launching=null;controls();}
  }
  async function read(target=offset,move=null){
    if(!selection||reading!==null)return;
    const s=selection,ticket=++generation,previous=offset,history=backs;
    reading=ticket;page=null;$('science-candidates').replaceChildren();$('science-page').textContent='Reading exact candidates…';controls();
    try{
      const result=await api.call('sciencePreviewPage',{workspace_id:s.workspace,reference:s.reference,offset:target,limit:25});
      if(!live(s,ticket))return;
      if(result.version!==1||result.scope!=='stored-experiment-preview'||result.workspace_id!==s.workspace||
         !same(result.experiment,s.reference)||result.offset!==target||result.limit!==25||
         result.execution!=='no-fitting-or-training'||!/^(0|[1-9][0-9]*)$/.test(result.total_combinations_exact)||
         typeof result.total_combinations_exact!=='string'||!Array.isArray(result.candidates)||result.candidates.length>25||
         (result.next_offset!==null&&(!Number.isInteger(result.next_offset)||result.next_offset!==target+result.candidates.length)))
        throw Error('Candidate page does not match the exact selection and cursor. Refresh candidates.');
      if(move==='next')history.push(previous);else if(move==='back')history.pop();
      offset=target;page=result;
      $('science-page').textContent=result.offset+'–'+(result.offset+result.candidates.length)+' of '+result.total_combinations_exact+
        ' · no training'+(result.next_offset!==null&&!Number.isSafeInteger(result.next_offset)?' · native page offset limit reached':'');
      result.candidates.forEach(row=>{
        const card=text('div','');card.className='science-candidate';card.dataset.candidateStatus=row.status;
        const c=row.coordinates;card.append(text('span',row.status+' · '+c.model.id+' · '+c.data_task+' · seed '+c.initialization+' · '+c.split+' · '+c.protocol));
        if(row.counts)card.append(text('small',Object.entries(row.counts).map(([role,count])=>role+': '+count).join(' · ')));
        if(row.reason)card.append(text('small',row.reason.text+(row.reason.next_offset!==null?' … [partial reason; inspect exact experiment and dependencies]':'')));
        const button=text('button','Run this candidate');button.dataset.scienceRun='';button.onclick=()=>run(s,result,row);card.append(button);$('science-candidates').append(card);
      });
    }catch(error){if(live(s,ticket)){$('science-page').textContent=error.message;api.toast(error.message,true);}}
    finally{if(reading===ticket)reading=null;controls();}
  }
  window.LoomScienceCandidates={
    mount(document,services,isBlocked=()=>false){root=document;api=services;blocked=isBlocked;
      $('science-next').onclick=()=>{if(reading===null&&Number.isSafeInteger(page?.next_offset))return read(page.next_offset,'next');};
      $('science-previous').onclick=()=>{if(reading===null&&backs.length)return read(backs.at(-1),'back');};
      $('science-candidate-refresh').onclick=()=>read();clear();},
    show(reference){clear();selection={reference:{...reference},workspace:api.workspace()};return read();},
    clear,stateChanged:controls
  };
})();
