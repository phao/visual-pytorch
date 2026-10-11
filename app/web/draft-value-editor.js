"use strict";
(() => {
  let root,api,saved,state=null,pending=false,generation=0,epoch=0,workspaceSeen=null,page=null,backs=[];
  const $=id=>root.getElementById(id),bytes=text=>new TextEncoder().encode(text).length;
  const dirty=()=>page&&$('science-value-text').value!==page.text;
  function controls(){
    if(!api)return;const stale=state&&(state.stale||state.header.workspace_id!==api.workspace());
    const blocked=!state||pending||stale||api.busy();
    for(const [id,extra] of [['start',Boolean(state?.value)],['apply',!state?.value||!dirty()],
      ['finish',!state?.value||dirty()],['check',Boolean(state?.value)],['save',Boolean(state?.value)]])
      $('science-value-'+id).disabled=blocked||extra||Boolean(state?.onClosed&&['check','save'].includes(id));
    $('science-value-prev').disabled=!state?.value||pending||dirty()||!backs.length;
    $('science-value-next').disabled=!state?.value||pending||dirty()||!page||page.next_offset===null;
    $('science-value-text').disabled=!state?.value||pending||stale||api.busy();
    $('science-value-discard').disabled=!state||pending;
    $('science-value-discard').textContent=state?.onClosed?'Discard staged value':'Discard draft';
    window.LoomScienceAuthor?.stateChanged?.();
    window.LoomDatasetSessionForm?.stateChanged?.();
    window.LoomScienceDocument?.stateChanged?.();
  }
  const valid=owner=>state===owner;
  const current=(owner,capturedEpoch)=>valid(owner)&&!owner.stale&&capturedEpoch===epoch&&owner.header.workspace_id===api.workspace();
  async function command(fn){if(pending)return;const owner=state,workspace=api.workspace(),capturedEpoch=epoch;pending=true;controls();try{await fn();}catch(error){
    if(workspace===api.workspace()&&capturedEpoch===epoch&&(!owner||state===owner)){
      $('science-value-status').textContent=error.message;api.toast(error.message,true);
    }
  }finally{pending=false;controls();}}
  const act=fn=>Promise.resolve().then(()=>command(fn));
  function argumentsFor(owner){return {session:owner.header.session,sequence:owner.header.sequence,
    value:owner.value.value,value_revision:owner.value.value_revision};}
  function matchesHeader(result,owner,sequence){return result.scope==='owned-authoring-session'&&result.version===1&&
    result.session===owner.header.session&&result.sequence===sequence&&result.id===owner.header.id&&
    result.kind===owner.header.kind&&result.workspace_id===owner.header.workspace_id;}
  async function read(offset=0){
    const owner=state,ticket=++generation,args=argumentsFor(owner);
    const result=await api.call('scienceDraftValueRead',{...args,offset});
    if(!valid(owner)||ticket!==generation)return;
    if(result.scope!=='staged-draft-value'||result.session!==args.session||result.sequence!==args.sequence||
      result.value!==args.value||result.value_revision!==args.value_revision||result.offset!==offset||
      JSON.stringify(result.path)!==JSON.stringify(owner.value.path)||result.offset_units!=='utf8-bytes')
      throw Error('Value fragment does not match its session, sequence and cursor.');
    page=result;$('science-value-text').value=result.text;
    $('science-value-status').textContent='JSON fragment · UTF-8 bytes '+offset+'–'+(offset+bytes(result.text))+' of '+result.total_bytes+
      ' · whole value is checked only on Use staged value · no training';
    return true;
  }
  async function start(fromBase=state.fromBase){
    const owner=state,ticket=++generation;
    const sourcePath=[...owner.path],value=await api.call('scienceDraftValueStart',{session:owner.header.session,sequence:owner.header.sequence,path:sourcePath,from_base:fromBase});
    if(!valid(owner)||ticket!==generation)return;
    if(value.scope!=='staged-draft-value'||value.session!==owner.header.session||value.sequence!==owner.header.sequence||
      JSON.stringify(value.source_path)!==JSON.stringify(sourcePath)||value.from_base!==fromBase)
      throw Error('Staged value does not match the owned draft.');
    owner.value=value;owner.path=[...value.path];owner.fromBase=false;page=null;backs=[];await read();
  }
  window.LoomDraftValueEditor={
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;workspaceSeen=api.workspace();
      $('science-value-start').onclick=()=>act(()=>start());
      $('science-value-apply').onclick=()=>act(async()=>{const owner=state,capturedEpoch=epoch,text=$('science-value-text').value;
        if(!page||!owner.value)return;
        if(Array.from(text).length>2000)throw Error('A fragment edit may contain at most 2000 Unicode code points.');
        const offset=page.offset,value=await api.call('scienceDraftValueSplice',{...argumentsFor(owner),offset,delete_bytes:bytes(page.text),text});
        if(!current(owner,capturedEpoch))return;
        if(value.session!==owner.header.session||value.sequence!==owner.header.sequence||value.value!==owner.value.value||value.value_revision!==owner.value.value_revision+1)
          throw Error('Staged edit response has the wrong identity or sequence.');
        owner.value=value;await read(offset);
      });
      $('science-value-prev').onclick=()=>act(async()=>{if(dirty())throw Error('Apply the unfinished fragment before navigating.');
        if(await read(backs.at(-1)))backs.pop();});
      $('science-value-next').onclick=()=>act(async()=>{if(dirty())throw Error('Apply the unfinished fragment before navigating.');
        const prior=page.offset;if(await read(page.next_offset))backs.push(prior);});
      $('science-value-finish').onclick=()=>act(async()=>{if(dirty())throw Error('Apply the unfinished fragment first.');
        const owner=state,capturedEpoch=epoch,result=await api.call('scienceDraftValueFinish',argumentsFor(owner));if(!current(owner,capturedEpoch))return;
        if(!matchesHeader(result,owner,owner.header.sequence+1))throw Error('Accepted value has the wrong draft identity or sequence.');
        owner.header=result;owner.value=null;page=null;backs=[];
        if(owner.onClosed){state=null;generation++;$('science-value-editor').hidden=true;await owner.onClosed(result);return;}
        $('science-value-status').textContent='Value accepted into the unsaved draft. Validate, then explicitly retain its revision; no training.';
      });
      $('science-value-check').onclick=()=>act(async()=>{const owner=state,workspace=api.workspace();
        if(owner.onClosed)throw Error('Validate through the parent dataset form, not its child value editor.');
        const result=await api.call('scienceDraftCheck',{session:owner.header.session,sequence:owner.header.sequence});
        if(!valid(owner)||workspace!==api.workspace()||owner.stale)return;
        $('science-value-status').textContent='Valid draft · prospective '+result.revision+' · not retained, no training';
      });
      $('science-value-save').onclick=()=>act(async()=>{const owner=state,workspace=api.workspace();
        if(owner.onClosed)throw Error('Retain through the parent dataset form, not its child value editor.');
        const result=await api.call('scienceDraftCommit',{session:owner.header.session,sequence:owner.header.sequence});
        if(!valid(owner)||workspace!==api.workspace()||owner.stale)return;
        state=null;generation++;$('science-value-editor').hidden=true;await saved(result);
        api.toast('Revision retained. Save workspace to persist it on disk.');
      });
      $('science-value-discard').onclick=()=>act(async()=>{const owner=state;
        let header=owner.header;
        if(owner.onClosed){if(owner.value){header=await api.call('scienceDraftValueDiscard',argumentsFor(owner));
          if(!matchesHeader(header,owner,owner.header.sequence))throw Error('Discarded value has the wrong parent identity or sequence.');}}
        else await api.call('scienceDraftClose',{session:owner.header.session});if(!valid(owner))return;
        state=null;page=null;generation++;$('science-value-editor').hidden=true;
        if(owner.onClosed)await owner.onClosed(header);
      });
      $('science-value-text').oninput=controls;controls();
    },
    open(target){return act(async()=>{
      if(state||window.LoomScienceAuthor?.active?.())throw Error('Finish or discard the current draft before opening another.');
      if(target.workspace!==api.workspace()||api.busy())throw Error('Workspace or active run changed. Select the exact value again.');
      const capturedEpoch=epoch,header=await api.call('scienceDraftOpen',{kind:target.reference.kind,reference:target.reference});
      if(header.scope!=='owned-authoring-session'||header.kind!==target.reference.kind||header.id!==target.reference.id||
        header.workspace_id!==target.workspace||header.base?.kind!==target.reference.kind||
        header.base?.id!==target.reference.id||header.base?.revision!==target.reference.revision)
        throw Error('Draft header does not match the selected exact base and workspace.');
      state={header,path:[...target.path],fromBase:true,value:null,stale:target.workspace!==api.workspace()||capturedEpoch!==epoch};
      $('science-value-editor').hidden=false;
      $('science-value-identity').textContent='Exact base '+target.reference.kind+' '+target.reference.id+' @ '+target.reference.revision+' · stored path '+JSON.stringify(target.path);
      if(state.stale){$('science-value-status').textContent='Workspace changed. Draft is retained for inspection/discard; do not admit into the replacement.';return;}
      await start(true);
    });},
    openOwned(target){return act(async()=>{
      if(state||api.busy()||target.header.workspace_id!==api.workspace())throw Error('Finish the current value or reopen the intended workspace first.');
      state={header:{...target.header},path:[...target.path],fromBase:false,value:null,stale:false,onClosed:target.onClosed};
      $('science-value-editor').hidden=false;
      $('science-value-identity').textContent='Owned dataset draft '+target.header.id+' · '+target.label+' · sequence '+target.header.sequence;
      await start(false);
    });},
    active(){return Boolean(state)||pending;},
    stateChanged(){if(!api)return;if(workspaceSeen!==api.workspace()){workspaceSeen=api.workspace();epoch++;generation++;}
      if(state&&state.header.workspace_id!==api.workspace()){state.stale=true;$('science-value-status').textContent='Workspace changed. Unsaved draft retained; only inspection/discard is available.';}controls();},
    event(event){if(event.kind==='externalReload'){epoch++;generation++;if(state){state.stale=true;$('science-value-status').textContent='Workspace reloaded. Unsaved draft retained; discard and reopen before further admission.';}controls();}}
  };
})();
