"use strict";
(() => {
  let root,api,saved,state=null,pending=false,epoch=0,controlling=false,flushing=false;
  const $=id=>root.getElementById(id),el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const collections={templates:'Concrete templates',symbols:'Scoped integer symbols',assignments:'Derived field assignments',repeats:'Finite repeats · independent or tied'};
  const args=()=>({session:state.header.session,sequence:state.header.sequence});
  const live=(owner,ticket)=>state===owner&&!owner.stale&&ticket===epoch&&owner.header.workspace_id===api.workspace();
  const issue=e=>{$(state?'science-family-status':'science-family-instance-status').textContent=e.message;api.toast(e.message,true);};
  const active=()=>Boolean(state)||pending;
  const blocked=()=>!state||state.stale||state.header.workspace_id!==api.workspace()||pending||api.busy()||window.LoomDraftValueEditor?.active?.();
  function invalidate(){if(state){state.edit++;state.preview=null;$('science-family-preview-result').textContent='';
    $('science-family-status').textContent='Unsaved family draft · '+state.edits.length+' pending edits · no training';}}
  function controls(){if(!api||controlling)return;controlling=true;
    try{
      const other=window.LoomScienceAuthor?.active?.()||active()||window.LoomDraftValueEditor?.active?.();
      $('science-author-new-family').disabled=Boolean(other)||pending||api.busy()||!api.workspace()||!api.model();
      const selected=$('science-kind').value==='family'&&Boolean($('science-object').value);
      $('science-family-instance').hidden=!selected||active();
      $('science-family-create').disabled=pending||api.busy()||Boolean(other)||!selected;
      for(const id of ['science-family-instance-bindings','science-family-materialize'])$(id).disabled=$('science-family-create').disabled;
      if(state){
        for(const id of ['preview','check','save','raw','retry'])$('science-family-'+id).disabled=blocked()||(['preview','check','save'].includes(id)&&state.failed);
        $('science-family-cancel').disabled=pending||Boolean(window.LoomDraftValueEditor?.active?.());
        $('science-family-editor').querySelectorAll('[data-family-control]').forEach(n=>n.disabled=blocked()||state.failed||n.dataset.large==='true'||n.dataset.readOnly==='true'||n.dataset.orderDisabled==='true');
        $('science-family-bindings').disabled=blocked();
      }
      window.LoomScienceAuthor?.stateChanged?.();
    }finally{controlling=false;}
  }
  async function run(fn){if(pending)return;capture();pending=true;controls();try{await fn();}catch(e){issue(e);}finally{pending=false;controls();}}
  const act=fn=>Promise.resolve().then(()=>run(fn));
  function capture(){const n=root.activeElement;if(state&&n&&'familyControl' in n.dataset&&'dirty' in n.dataset){
    flushing=true;try{n.dispatchEvent(new Event('change'));}finally{flushing=false;}}}
  function enqueue(target,key,text,n){if(!state||state.stale||(pending&&!flushing))return;
    if(text!==null&&text.length>2000)throw Error('Use staged JSON for values over 2000 characters.');
    const identity=JSON.stringify([target,key]);state.edits=state.edits.filter(e=>e.identity!==identity);
    state.edits.push({identity,operation:{op:'property',target,key,text}});delete n.dataset.invalid;delete n.dataset.dirty;n.removeAttribute('aria-invalid');invalidate();
  }
  async function mutate(operations){const owner=state,ticket=epoch,sequence=owner.header.sequence;
    const h=await api.call('scienceDraftFamilyEdit',{...args(),operations});
    if(!live(owner,ticket))return false;
    if(h.version!==1||h.scope!=='owned-authoring-session'||h.kind!=='family'||h.session!==owner.header.session||h.sequence!==sequence+1||h.id!==owner.header.id||h.workspace_id!==owner.header.workspace_id)
      throw Error('Family edit acknowledgement has a different owner or sequence.');
    owner.header=h;return true;
  }
  async function flush(){capture();if($('science-family-editor').querySelector('[data-invalid]'))throw Error('Correct invalid family fields; your text is retained.');
    while(state.edits.length){const batch=[];
      for(const entry of state.edits.slice(0,25)){const request={action:'scienceDraftFamilyEdit',...args(),operations:[...batch,entry].map(e=>e.operation)};
        if(JSON.stringify(request).replace(/[\u0080-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')).length>65536)break;batch.push(entry);}
      if(!batch.length)throw Error('Family edit exceeds the wire budget; use staged JSON.');
      if(!await mutate(batch.map(e=>e.operation)))return false;state.edits=state.edits.filter(e=>!batch.includes(e));}return true;
  }
  function button(text,fn){const b=el('button',text);b.type='button';b.dataset.familyControl='';b.onclick=()=>act(fn);return b;}
  async function raw(path,label){if(!await flush())return;const owner=state,ticket=epoch,sequence=owner.header.sequence;
    await window.LoomDraftValueEditor.openOwned({header:{...owner.header},path:[...path],label,onClosed:async h=>{
      if(!live(owner,ticket))return;
      if(h.kind!=='family'||h.session!==owner.header.session||h.id!==owner.header.id||h.workspace_id!==owner.header.workspace_id||![sequence,sequence+1].includes(h.sequence))
        throw Error('Staged family value belongs to another draft.');
      owner.header=h;invalidate();await act(readAll);
    }});
  }
  function control(d,target,readOnly=false){
    const key=d.key,wrapper=el('label',key),json=['value','count','nodes','graph'].includes(key),
      numeric=['default','minimum','maximum','version'].includes(key),n=el(key==='mode'?'select':json?'textarea':'input');
    if(key==='mode'){
      const current=d.present?JSON.parse(d.text):'independent';
      for(const value of [...new Set(['independent','tied',current])].sort()){
        const option=el('option',value);option.value=value;n.append(option);
      }
      n.title='Independent: fresh parameters per use. Tied: one shared parameter set across finite uses; family version 2.';
    }
    n.dataset.familyControl='';n.dataset.key=key;const large=d.present&&d.text===null;
    n.dataset.large=String(large);n.dataset.readOnly=String(readOnly);
    if(large)n.placeholder='Large '+d.summary.type+' · '+d.summary.size+' · staged editing';
    else n.value=d.present?(json||numeric?d.text:JSON.parse(d.text)):'';
    n.oninput=()=>{n.dataset.dirty='';invalidate();};
    n.onchange=()=>{try{
      let text;if((numeric||['default','minimum','maximum'].includes(key))&&n.value.trim()==='')text=null;
      else if(json||numeric){text=n.value;JSON.parse(text);
        if(numeric&&!/^-?(0|[1-9][0-9]*)$/.test(text))throw Error('Enter an exact decimal integer, or blank for an absent default/bound.');
      }else text=JSON.stringify(n.value);
      if(key==='mode'&&n.value==='tied')enqueue({kind:'document'},'version','2',n);
      enqueue(target,key,text,n);
    }catch(e){n.dataset.invalid='';n.setAttribute('aria-invalid','true');issue(e);}};
    wrapper.append(n);if(!readOnly&&d.path&&(large||json))wrapper.append(button('Edit staged JSON…',()=>raw(d.path,key)));
    if(readOnly)n.readOnly=true;return wrapper;
  }
  async function readGroup(collection){const owner=state,ticket=epoch,g=owner.groups[collection],request={...args(),collection,offset:g.offset,limit:6};
    const page=await api.call('scienceDraftFamilyPage',request);if(!live(owner,ticket))return;
    if(page.version!==1||page.scope!=='owned-family-page'||page.session!==request.session||page.sequence!==request.sequence||page.workspace_id!==owner.header.workspace_id||page.id!==owner.header.id||page.collection!==collection||page.offset!==g.offset||page.limit!==6)
      throw Error('Family page does not match its owner, sequence and cursor.');
    g.page=page;g.body.replaceChildren();g.range.textContent=page.offset+'–'+(page.offset+page.rows.length)+' of '+page.total;
    g.prev.dataset.orderDisabled=String(!g.backs.length);g.next.dataset.orderDisabled=String(page.next_offset===null);
    for(const row of page.rows){const card=el('fieldset'),target={kind:'entry',collection,ordinal:row.ordinal};
      card.dataset.familyCard=collection;card.dataset.ordinal=String(row.ordinal);card.append(el('legend',row.template||collection+' '+(row.ordinal+1)));
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      for(const d of row.properties)grid.append(control(d,target,collection==='templates'&&d.key==='id'));
      if(collection!=='templates'){
        card.append(button('Remove',()=>structure(collection,{op:'remove',collection,index:row.ordinal})));
        if(row.ordinal>0)card.append(button('Move up',()=>structure(collection,{op:'move',collection,from:row.ordinal,to:row.ordinal-1})));
        if(row.ordinal+1<page.total)card.append(button('Move down',()=>structure(collection,{op:'move',collection,from:row.ordinal,to:row.ordinal+1})));
      }g.body.append(card);
    }
    if(collection==='templates'){$('science-family-roots').replaceChildren(...page.roots.map(d=>control(d,{kind:'document'})));}
  }
  async function readAll(){const owner=state;owner.failed=true;const results=await Promise.allSettled(Object.keys(collections).map(readGroup));
    const bad=results.find(r=>r.status==='rejected');if(bad)throw bad.reason;if(state===owner&&!owner.stale)owner.failed=false;controls();
  }
  async function structure(collection,op){if(!await flush())return;if(await mutate([op])){invalidate();await readAll();}}
  async function navigate(collection,forward){if(!await flush())return;const g=state.groups[collection],old=g.offset,next=forward?g.page?.next_offset:g.backs.at(-1);
    if(next===null||next===undefined)return;g.offset=next;
    try{await readGroup(collection);if(forward)g.backs.push(old);else g.backs.pop();}catch(e){g.offset=old;throw e;}
  }
  function layout(){$('science-family-editor').hidden=false;$('science-family-identity').textContent=state.header.id+' · '+(state.header.base?'exact base '+state.header.base.revision:'new drawing seed');
    const container=$('science-family-cards');container.replaceChildren();
    for(const [collection,label] of Object.entries(collections)){const section=el('details');section.open=true;section.append(el('summary',label));
      const g=state.groups[collection]={offset:0,backs:[],page:null,body:el('div'),range:el('span')},nav=el('div');nav.className='inline-actions';
      g.prev=button('‹ Cards',()=>navigate(collection,false));g.next=button('Cards ›',()=>navigate(collection,true));nav.append(g.prev,g.range,g.next);
      if(collection!=='templates')nav.append(button('Add '+collection.slice(0,-1),async()=>{
        if(!await flush())return;const page=await api.call('scienceDraftFamilyPage',{...args(),collection,offset:0,limit:6});
        const scope=state.groups.templates.page?.rows[0]?.template||'',id=crypto.randomUUID();
        const value=collection==='symbols'?{id,scope,name:'width',default:1,minimum:1}:
          collection==='assignments'?{model:scope,node:'',field:'features',value:1}:
          {id,model:scope,nodes:[],count:1,mode:'independent'};
        await structure(collection,{op:'insert',collection,index:page.total,value});
      }));section.append(nav,g.body);container.append(section);
    }
  }
  function closeUI(){state=null;epoch++;$('science-family-editor').hidden=true;$('science-family-cards').replaceChildren();$('science-family-roots').replaceChildren();}
  async function open(reference=null){if(state||window.LoomScienceAuthor?.legacyActive?.()||window.LoomDatasetSessionForm?.active?.()||window.LoomTaskSessionForm?.active?.()||window.LoomExperimentSessionForm?.active?.()||window.LoomDraftValueEditor?.active?.())throw Error('Finish or discard the current draft first.');
    const identity=api.workspace(),ticket=epoch,h=await api.call('scienceDraftOpen',{kind:'family',...(reference?{reference}:{model:api.model()})});
    if(h.version!==1||h.scope!=='owned-authoring-session'||h.kind!=='family'||h.workspace_id!==identity||h.sequence!==0||
       (reference?(h.base?.kind!=='family'||h.base?.id!==reference.id||h.base?.revision!==reference.revision||h.id!==reference.id):h.base!==null))throw Error('Family draft does not match its exact base/workspace.');
    state={header:h,stale:identity!==api.workspace()||ticket!==epoch,groups:{},edits:[],edit:0,failed:true,preview:null};layout();await readAll();invalidate();
  }
  window.LoomFamilySessionForm={
    active,open(reference=null){return act(()=>open(reference));},
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;
      $('science-family-bindings').oninput=invalidate;
      $('science-author-new-family').onclick=()=>act(()=>open());
      const prior=$('science-author-edit').onclick;$('science-author-edit').onclick=()=>{$('science-kind').value==='family'?act(()=>open(JSON.parse($('science-object').value))):prior?.();};
      for(const id of ['science-kind','science-object'])$(id).addEventListener('change',controls);
      $('science-family-preview').onclick=()=>act(async()=>{if(!await flush())return;const owner=state,ticket=epoch,edit=owner.edit;
        const result=await api.call('scienceDraftFamilyPreview',{...args(),bindings:JSON.parse($('science-family-bindings').value)});
        if(!live(owner,ticket)||edit!==owner.edit)return;
        if(result.scope!=='owned-family-preview'||result.session!==owner.header.session||result.sequence!==owner.header.sequence||result.workspace_id!==owner.header.workspace_id||result.id!==owner.header.id)throw Error('Family preview returned a different owner/sequence.');
        owner.preview=result;$('science-family-preview-result').textContent=JSON.stringify(result,(key,value)=>key==='parameters'&&result.parameters_exact!==undefined&&!Number.isSafeInteger(value)?result.parameters_exact:value,2);$('science-family-status').textContent=result.status+' · no model allocation or training';});
      $('science-family-check').onclick=()=>act(async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCheck',args());if(!live(owner,ticket))return;
        if(result.kind!=='family'||result.id!==owner.header.id||!/^[a-f0-9]{64}$/.test(result.revision))throw Error('Family validation returned another identity.');
        $('science-family-status').textContent='Valid family · prospective '+result.revision+' · not retained';});
      $('science-family-save').onclick=()=>act(async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCommit',args());if(!live(owner,ticket))return;
        if(result.kind!=='family'||result.id!==owner.header.id||!/^[a-f0-9]{64}$/.test(result.revision))throw Error('Retained family identity does not match the draft.');
        closeUI();await saved(result);api.toast('Family revision retained. Save workspace to persist it.');});
      $('science-family-raw').onclick=()=>act(()=>raw([],'Complete family definition'));
      $('science-family-retry').onclick=()=>act(async()=>{if(await flush())await readAll();});
      $('science-family-cancel').onclick=()=>act(async()=>{await api.call('scienceDraftClose',{session:state.header.session});closeUI();});
      $('science-family-create').onclick=()=>act(async()=>{const identity=api.workspace(),family=JSON.parse($('science-object').value);
        if(family.kind!=='family')throw Error('Select an exact retained family revision.');
        const result=await api.call('scienceFamilyResolve',{workspace_id:identity,family,bindings:JSON.parse($('science-family-instance-bindings').value),materialize:$('science-family-materialize').checked});
        if(identity!==api.workspace())return;$('science-family-instance-status').textContent='Retained specialization '+result.specialization.revision+' · '+(result.parameters_exact??result.parameters)+' parameters · '+(result.materialized?'editable drawing added':'registry only');
        api.toast('Concrete instance retained. Save workspace to persist it.');});
      controls();
    },
    stateChanged(){if(!api)return;if(state&&state.header.workspace_id!==api.workspace()&&!state.stale){state.stale=true;epoch++;$('science-family-status').textContent='Workspace changed. Draft text is retained; discard/reopen to edit the intended working copy.';}controls();},
    event(event){if(event.kind==='externalReload'&&state){state.stale=true;epoch++;$('science-family-status').textContent='Working copy reloaded. Draft retained for inspection; discard/reopen before editing.';}controls();}
  };
})();
