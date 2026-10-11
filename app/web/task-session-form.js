"use strict";
(() => {
  let root,api,saved,state=null,pending=false,flushing=false,epoch=0,contextEpoch=0,workspaceSeen=null,controlling=false;
  const $=id=>root.getElementById(id),el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const live=(owner,ticket)=>state===owner&&!owner.stale&&ticket===epoch&&owner.header.workspace_id===api.workspace();
  const sameRef=(a,b)=>a?.kind===b.kind&&a?.id===b.id&&a?.revision===b.revision;
  const taskRef=(value,owner)=>value?.kind==='task'&&value.id===owner.header.id&&/^[a-f0-9]{64}$/.test(value.revision||'');
  const args=()=>({session:state.header.session,sequence:state.header.sequence});
  const size=value=>JSON.stringify(value).replace(/[\u0080-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')).length;
  function issue(error){$('science-author-status').textContent=error.message;api.toast(error.message,true);}
  function changed(){if(!state)return;state.edit++;$('science-task-compatibility-status').textContent='';$('science-author-status').textContent='Unsaved task draft · '+state.edits.length+' pending edits · no training';}
  function blocked(){return !state||state.stale||pending||api.busy()||state.header.workspace_id!==api.workspace()||window.LoomDraftValueEditor?.active?.();}
  function controls(){
    if(!api||controlling)return;controlling=true;
    try{
      if(state){const stop=blocked(),ready=Object.values(state.groups).every(g=>g.page)&&!state.needsRead;
        state.choices.forEach(c=>c.controls());
        for(const id of ['science-author-save','science-author-check'])$(id).disabled=stop||!ready;
        $('science-author-cancel').disabled=pending||window.LoomDraftValueEditor?.active?.();
        $('science-author-retry-page').hidden=!Object.values(state.groups).some(g=>g.failed)&&!state.needsRead;
        $('science-author-retry-page').disabled=stop;
        $('science-author').querySelectorAll('[data-task-control]').forEach(n=>n.disabled=stop||state.needsRead||n.dataset.large==='true'||n.dataset.orderDisabled==='true');
        for(const g of Object.values(state.groups)){g.add.disabled=stop||!g.page||state.needsRead;g.prev.disabled=stop||!g.backs.length||state.needsRead;g.next.disabled=stop||!g.page||g.page.next_offset===null||state.needsRead;}
        for(const p of state.pipelines.values()){
          p.prev.disabled=stop||!p.page||!p.backs.length||state.needsRead;
          p.next.disabled=stop||!p.page||p.page.next_offset===null||state.needsRead;
        }
      }
      window.LoomScienceAuthor?.stateChanged?.();window.LoomScienceDocument?.stateChanged?.();window.LoomTaskContext?.controls();
    }finally{controlling=false;}
  }
  function capture(){const n=root.activeElement;if(state&&n?.matches('input,textarea')&&'dirty' in n.dataset&&'taskControl' in n.dataset){
    flushing=true;try{n.dispatchEvent(new Event('change'));}finally{flushing=false;}
  }}
  async function command(fn){if(pending)return;capture();const owner=state,ticket=epoch,identity=api.workspace();pending=true;controls();
    try{await fn();}catch(error){if(ticket===epoch&&identity===api.workspace()&&(!owner||owner===state))issue(error);}
    finally{pending=false;controls();}
  }
  const act=fn=>Promise.resolve().then(()=>command(fn));
  async function settled(reads){
    const results=await Promise.allSettled(reads),failed=results.find(r=>r.status==='rejected');
    if(failed)throw failed.reason;
  }
  function enqueue(target,key,text,node){
    if(!state||state.stale||(pending&&!flushing)||api.busy()||window.LoomDraftValueEditor?.active?.())return;
    if(text!==null&&Array.from(text).length>2000)throw Error('Use staged value editing for more than 2000 code points.');
    const identity=JSON.stringify([target,key]);state.edits=state.edits.filter(e=>e.identity!==identity);
    state.edits.push({identity,node,operation:{op:'property',target,key,text}});
    delete node.dataset.invalid;delete node.dataset.dirty;node.removeAttribute('aria-invalid');changed();
  }
  function matchesHeader(h,owner,sequence){return h.scope==='owned-authoring-session'&&h.version===1&&h.kind==='task'&&h.id===owner.header.id&&
    h.session===owner.header.session&&h.workspace_id===owner.header.workspace_id&&h.sequence===sequence;}
  async function mutate(operations){
    const owner=state,ticket=epoch,sequence=owner.header.sequence,result=await api.call('scienceDraftTaskEdit',{...args(),operations});
    if(!live(owner,ticket))return false;
    if(!matchesHeader(result,owner,sequence+1))throw Error('Task edit acknowledgement does not match the owned sequence.');
    owner.header=result;return true;
  }
  async function flush(){
    capture();if([...$('science-author').querySelectorAll('[data-task-control]')].some(n=>'invalid' in n.dataset))
      throw Error('Correct invalid task fields before continuing. Your text has not been discarded.');
    while(state.edits.length){
      const owner=state,batch=[];
      for(const entry of owner.edits.slice(0,25)){const candidate=[...batch,entry];if(size({action:'scienceDraftTaskEdit',...args(),operations:candidate.map(e=>e.operation)})>65536)break;batch.push(entry);}
      if(!batch.length)throw Error('Task edit exceeds the byte budget. Use staged value editing.');
      try{if(!await mutate(batch.map(e=>e.operation)))return false;}
      catch(error){const entry=batch[Number(/Task edit #(\d+):/.exec(error.message)?.[1])];if(entry){entry.node.dataset.invalid='';entry.node.setAttribute('aria-invalid','true');}throw error;}
      owner.edits=owner.edits.filter(e=>!batch.includes(e));
    }return true;
  }
  async function page(collection,offset,limit=6,binding=null){
    const owner=state,ticket=epoch,request={...args(),collection,offset,limit,...(binding===null?{}:{binding})};
    const result=await api.call(binding===null?'scienceDraftTaskPage':'scienceDraftTaskTransforms',request);
    if(!live(owner,ticket))return null;
    if(result.scope!==(binding===null?'owned-task-page':'owned-task-transform-page')||result.version!==1||
      result.session!==request.session||result.sequence!==request.sequence||result.workspace_id!==owner.header.workspace_id||result.id!==owner.header.id||
      result.collection!==collection||result.offset!==offset||result.limit!==limit||(binding!==null&&result.binding!==binding))
      throw Error('Task page does not match its exact owner, sequence and cursor.');
    return result;
  }
  async function readGroup(collection){
    const owner=state,ticket=epoch,g=owner.groups[collection];
    try{const result=await page(collection,g.offset);if(!result)return false;g.page=result;g.failed=false;renderGroup(collection);
      await settled([...owner.pipelines.values()].filter(p=>p.collection===collection&&p.box.open).map(p=>readPipeline(p)));return live(owner,ticket);
    }catch(error){if(live(owner,ticket)){g.failed=true;g.range.textContent=error.message+' · retry page';}throw error;}
  }
  async function readAll(){const owner=state;await settled(['inputs','heads'].map(readGroup));if(state!==owner||owner.stale)return false;
    owner.needsRead=false;renderDescription();controls();return true;}
  async function readPipeline(p){
    const owner=state,ticket=epoch;try{const result=await page(p.collection,p.offset,6,p.binding);if(!result||owner.pipelines.get(p.key)!==p)return false;
      p.page=result;p.failed=false;renderPipeline(p);return true;
    }catch(error){if(live(owner,ticket)&&owner.pipelines.get(p.key)===p){p.failed=true;p.retry.hidden=false;p.range.textContent=error.message+' · Retry transforms';}throw error;}
  }
  function button(name,fn,title){const b=el('button',name);b.dataset.taskControl='';if(title)b.title=title;b.onclick=()=>act(fn);return b;}
  async function structure(collection,build){
    if(!await flush())return;const owner=state,fresh=await page(collection,owner.groups[collection].offset);if(!fresh)return;
    const operations=await build(fresh);if(!operations||!await mutate(operations))return;
    owner.needsRead=true;owner.pipelines.clear();await readAll();
  }
  function navigate(g,read,forward){return async()=>{if(!await flush())return;
    const prior=g.offset,next=forward?g.page?.next_offset:g.backs.at(-1);if(next===null||next===undefined)return;
    g.offset=next;try{if(await read()){if(forward)g.backs.push(prior);else g.backs.pop();}else g.offset=prior;}catch(error){g.offset=prior;throw error;}
  };}
  async function resolve(target,key){
    if(target.kind==='document'){const result=await page('inputs',0,1);return result?.description;}
    const result=await page(target.collection,target.kind==='binding'?target.binding:target.transform,1,
      target.kind==='binding'?null:target.binding);
    return result?.rows[0]?.properties.find(p=>p.key===key);
  }
  function rawButton(target,key){return button('Edit value…',async()=>{
    if(!await flush())return;const owner=state,ticket=epoch,descriptor=await resolve(target,key);
    if(!descriptor?.present||!live(owner,ticket))throw Error('Reload the current task property before staging it.');
    await window.LoomDraftValueEditor.openOwned({header:{...owner.header},path:[...descriptor.path],label:JSON.stringify([target,key]),onClosed:async header=>{
      if(!live(owner,ticket))return;
      if(!matchesHeader(header,owner,owner.header.sequence)&&!matchesHeader(header,owner,owner.header.sequence+1))throw Error('Staged task value returned a different parent.');
      owner.header=header;owner.needsRead=true;await command(readAll);
    }});
  },'Stage exact JSON without replacing other task properties');}
  function control(label,descriptor,target,key,{mode='string',optional=false,list,area=false}={}){
    const wrapper=el('label',label),n=el(area?'textarea':'input');n.dataset.taskControl='';n.dataset.authorControl='';
    const large=descriptor.present&&descriptor.text===null;n.dataset.large=String(large);if(list)n.setAttribute('list',list);
    if(large){n.value='';n.placeholder='Large '+descriptor.summary.type+' · '+descriptor.summary.size+' · use staged editor';}
    else if(!descriptor.present)n.value='';
    else if(mode==='number')n.value=descriptor.text;
    else{const value=JSON.parse(descriptor.text);n.value=typeof value==='string'?value:descriptor.text;}
    n.oninput=()=>{n.dataset.dirty='';changed();};
    n.onchange=()=>{try{
      let text=optional&&n.value===''?null:mode==='number'?n.value.trim():JSON.stringify(n.value);
      if(mode==='number'&&text!==null){const value=JSON.parse(text);if(typeof value!=='number'||!Number.isFinite(value)||(key==='weight'&&value<=0)||
        (key==='length'&&(!Number.isSafeInteger(value)||value<=0)))throw Error('Enter a finite '+(key==='length'?'positive exact integer.':key==='weight'?'positive number.':'number.'));}
      enqueue(target,key,text,n);
    }catch(error){n.dataset.invalid='';n.setAttribute('aria-invalid','true');issue(error);}};
    wrapper.append(n);if(large)wrapper.append(rawButton(target,key));return wrapper;
  }
  function select(label,options,value,change){
    const wrapper=el('label',label),n=el('select');n.dataset.taskControl='';n.dataset.authorControl='';
    for(const [id,name] of options){const option=el('option',name);option.value=id;n.append(option);}n.value=value;
    n.onchange=()=>{const chosen=n.value;act(async()=>{try{await change(chosen);}catch(error){n.value=value;throw error;}});};wrapper.append(n);return wrapper;
  }
  function properties(row){return Object.fromEntries(row.properties.map(p=>[p.key,p]));}
  const scalar=(d,fallback='')=>d.present&&d.text!==null?JSON.parse(d.text):fallback;
  function renderGroup(collection){
    const g=state.groups[collection],owner=state;g.cards.replaceChildren();
    for(const [key,p] of owner.pipelines)if(p.collection===collection)owner.pipelines.delete(key);
    for(const row of g.page.rows){
      const target={kind:'binding',collection,binding:row.ordinal},props=properties(row),card=el('fieldset');
      card.dataset[collection==='inputs'?'taskInput':'taskHead']=row.ordinal;card.append(el('legend',(collection==='inputs'?'Input binding ':'Loss head ')+(row.ordinal+1)));
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      if(collection==='heads')grid.append(control('Head ID',props.id,target,'id'));
      grid.append(control(collection==='inputs'?'Input port ID':'Output port ID',props.port,target,'port',{list:collection==='inputs'?'science-task-input-ports':'science-task-output-ports'}),
        control(collection==='inputs'?'Source field ID':'Target field ID',props.field,target,'field',{list:'science-task-fields'}));
      if(collection==='inputs'){
        grid.append(select('Derived input',[['length','Sequence length'],['mask','Padding mask'],['none','None · transformed field']],scalar(props.derived,'none'),async mode=>{
          if(!await flush())return;const ops=mode==='none'?['derived','length'].map(key=>({op:'property',target,key,text:null})):
            [{op:'property',target,key:'derived',text:JSON.stringify(mode)},...(!props.length.present?[{op:'property',target,key:'length',text:'1'}]:[])];
          if(await mutate(ops)){owner.needsRead=true;await readAll();}
        }));
        if(props.derived.present)grid.append(control('Padded length',props.length,target,'length',{mode:'number'}));
      }else{
        grid.append(select('Loss',[['cross-entropy','Cross-entropy'],['mae','MAE'],['mse','MSE']],scalar(props.loss),async loss=>{
          if(!await flush())return;if(await mutate([{op:'property',target,key:'loss',text:JSON.stringify(loss)}]))await readAll();
        }));
        grid.append(control('Objective weight',props.weight,target,'weight',{mode:'number'}));
        for(const [key,label] of [['mask','Mask field ID (optional)'],['sample_weight','Sample-weight field ID (optional)']])
          grid.append(control(label,props[key],target,key,{optional:true,list:'science-task-fields'}));
      }
      const box=el('details');box.open=row.transforms.total>0;box.append(el('summary','Ordered transforms · '+row.transforms.total));
      const actions=el('div');actions.className='inline-actions';const range=el('span');range.className='tiny';range.setAttribute('role','status');
      const key=collection+'/'+row.ordinal,p={key,collection,binding:row.ordinal,box,range,offset:0,backs:[],page:null,failed:false,rows:el('div')};
      owner.pipelines.set(key,p);p.prev=button('‹ Transforms',navigate(p,()=>readPipeline(p),false));p.next=button('Transforms ›',navigate(p,()=>readPipeline(p),true));
      p.retry=button('Retry transforms',async()=>{if(await flush())await readPipeline(p);});p.retry.hidden=true;
      actions.append(p.prev,range,p.next,p.retry);box.append(actions,p.rows);
      const add=el('div');add.className='inline-actions';const choice=el('select');choice.dataset.taskControl='';choice.setAttribute('aria-label','Transform to add');
      for(const kind of window.LoomTaskAuthoring.kinds){const option=el('option',kind);option.value=kind;choice.append(option);}choice.value='standardize';
      add.append(choice,button('Add transform',()=>structure(collection,async()=>{const fresh=await page(collection,row.ordinal,1);if(!fresh)return;
        return [{op:'insert-json',path:fresh.rows[0].transforms.path,index:fresh.rows[0].transforms.total,text:JSON.stringify(window.LoomTaskAuthoring.transform(choice.value))}];
      })));box.append(add);card.append(box);
      box.ontoggle=()=>{if(box.open&&!p.page&&!pending)act(async()=>{if(await flush())await readPipeline(p);});};
      const order=el('div');order.className='inline-actions';order.dataset.bindingOrder='';
      for(const [name,delta] of [['↑',-1],['↓',1]]){
        const b=button(name,()=>structure(collection,fresh=>[{op:'move',path:fresh.path,from:row.ordinal,to:row.ordinal+delta}]),'Move binding '+(delta<0?'earlier':'later'));
        b.dataset.orderDisabled=String(delta<0?row.ordinal===0:row.ordinal===g.page.total-1);order.append(b);
      }
      order.append(button('Remove binding',()=>structure(collection,fresh=>[{op:'remove',path:[...fresh.path,row.ordinal]}])));card.append(order);g.cards.append(card);
    }
    g.range.textContent=(g.page.rows.length?g.offset+1:0)+'–'+(g.offset+g.page.rows.length)+' / '+g.page.total+' · scientific order';controls();
  }
  function renderPipeline(p){
    p.rows.replaceChildren();for(const row of p.page.rows){
      const props=properties(row),target={kind:'transform',collection:p.collection,binding:p.binding,transform:row.ordinal},kind=scalar(props.kind);
      const grid=el('div');grid.className='science-author-grid science-task-transform';grid.dataset.transformIndex=row.ordinal;grid.append(el('span',(row.ordinal+1)+'. '+kind));
      for(const key of kind==='affine'?['scale','offset']:kind==='pad'?['length','value']:[])grid.append(control(key,props[key],target,key,{mode:'number'}));
      for(const [name,delta] of [['↑',-1],['↓',1]]){
        const b=button(name,()=>structure(p.collection,async()=>{const fresh=await page(p.collection,0,1,p.binding);if(!fresh)return;
          return [{op:'move',path:fresh.path,from:row.ordinal,to:row.ordinal+delta}];}),'Move transform '+(delta<0?'earlier':'later'));
        b.dataset.orderDisabled=String(delta<0?row.ordinal===0:row.ordinal===p.page.total-1);grid.append(b);
      }
      grid.append(button('Remove transform',()=>structure(p.collection,async()=>{const fresh=await page(p.collection,0,1,p.binding);if(!fresh)return;
        return [{op:'remove',path:[...fresh.path,row.ordinal]}];})));p.rows.append(grid);
    }
    p.range.textContent=(p.page.rows.length?p.offset+1:0)+'–'+(p.offset+p.page.rows.length)+' / '+p.page.total;
    p.retry.hidden=!p.failed;controls();
  }
  function renderDescription(){const g=state.groups.inputs;if(!g.page)return;state.description.replaceChildren(
    control('Task description',g.page.description,{kind:'document'},'description',{optional:true,area:true}));}
  function layout(){
    const owner=state,h=owner.header;$('science-author').hidden=false;$('science-author-title').textContent='Task draft';
    $('science-author-id').textContent='Stable task ID: '+h.id+' · '+(h.base?'editing exact '+h.base.revision.slice(0,12):'new object')+(h.base&&h.base.revision!==h.expected_head?' · historical base, not current head':'');
    $('science-author-dataset').hidden=true;$('science-author-task').hidden=false;$('science-author-experiment').hidden=true;
    $('science-task-bindings').replaceChildren();const name=$('science-author-name');name.dataset.taskControl='';name.value=h.name;
    name.oninput=()=>{name.dataset.dirty='';changed();};name.onchange=()=>{try{enqueue({kind:'document'},'name',JSON.stringify(name.value),name);}catch(error){name.dataset.invalid='';issue(error);}};
    for(const collection of ['inputs','heads']){
      const container=$('science-task-bindings');container.append(el('h4',collection==='inputs'?'Named inputs':'Named output objectives'));
      const g={offset:0,backs:[],page:null,failed:false,cards:el('div'),range:el('span')};g.range.className='tiny';g.range.setAttribute('role','status');
      g.prev=button('‹ Bindings',navigate(g,()=>readGroup(collection),false));g.next=button('Bindings ›',navigate(g,()=>readGroup(collection),true));
      const actions=el('div');actions.className='inline-actions';actions.append(g.prev,g.range,g.next);container.append(actions,g.cards);
      g.add=button(collection==='inputs'?'Add input binding':'Add loss head',()=>structure(collection,fresh=>[{
        op:'insert-json',path:fresh.path,index:fresh.total,text:JSON.stringify(collection==='inputs'?window.LoomTaskAuthoring.input():window.LoomTaskAuthoring.head(crypto.randomUUID()))
      }]));container.append(g.add);owner.groups[collection]=g;
    }
    owner.description=el('div');$('science-task-bindings').append(owner.description);$('science-task-compatibility-status').textContent='';
    for(const id of ['science-task-model','science-task-dataset','science-task-context-refresh','science-task-compatibility'])$(id).dataset.taskControl='';
    owner.choices=[['science-task-model','model'],['science-task-dataset','dataset']].map(([id,kind])=>window.LoomRevisionChoice.attach(root,$(id),
      {...api,busy:()=>api.busy()||pending,updateControls:controls},{kind,defaultHead:true,
        getReference:()=>$(id).value?JSON.parse($(id).value):null,valid:()=>state===owner&&!owner.stale}));
  }
  async function context(){
    const owner=state,ticket=++contextEpoch,m=$('science-task-model').value,d=$('science-task-dataset').value;
    $('science-task-compatibility-status').textContent='';
    if(!m||!d){window.LoomTaskContext.clear();return;}
    await window.LoomTaskContext.show({model:JSON.parse(m),dataset:JSON.parse(d)},()=>state===owner&&!owner.stale&&ticket===contextEpoch);
  }
  async function choices(){const owner=state;await Promise.all(owner.choices.map(c=>c.refresh()));if(state===owner&&!owner.stale)await context();}
  function closeUI(){
    state.choices.forEach(c=>c.dispose());state=null;contextEpoch++;window.LoomTaskContext.clear();$('science-task-bindings').replaceChildren();
    $('science-author').hidden=true;$('science-author-retry-page').hidden=true;
    const name=$('science-author-name');for(const key of ['taskControl','dirty','invalid'])delete name.dataset[key];name.removeAttribute('aria-invalid');
  }
  function route(id,fn){const prior=$(id).onclick;$(id).onclick=()=>state||pending?act(fn):prior?.();}
  window.LoomTaskSessionForm={
    active:()=>Boolean(state)||pending,contextBlocked:blocked,
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;workspaceSeen=api.workspace();
      $('science-author-new-task').onclick=()=>this.open();const prior=$('science-author-edit').onclick;
      $('science-author-edit').onclick=()=>{const ref=$('science-object').value?JSON.parse($('science-object').value):null;return ref?.kind==='task'?this.open(ref):prior?.();};
      route('science-author-cancel',async()=>{const owner=state;await api.call('scienceDraftClose',{session:owner.header.session});if(state===owner)closeUI();});
      route('science-author-retry-page',async()=>{if(await flush())await readAll();});
      route('science-author-check',async()=>{if(!await flush())return;const owner=state,ticket=epoch,edit=owner.edit,result=await api.call('scienceDraftCheck',args());
        if(!live(owner,ticket)||edit!==owner.edit)return;if(!taskRef(result,owner))throw Error('Task validation returned a different identity.');
        $('science-author-status').textContent='Valid TaskSpec · prospective revision '+result.revision+' · not saved, no training';});
      route('science-author-save',async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCommit',args());
        if(!live(owner,ticket))return;if(!taskRef(result,owner))throw Error('Retained task has the wrong identity.');
        closeUI();await saved(result);api.toast('Task revision retained. Save workspace to persist it on disk.');});
      route('science-task-context-refresh',choices);
      for(const id of ['science-task-model','science-task-dataset'])$(id).onchange=()=>state?act(context):null;
      route('science-task-compatibility',async()=>{if(!await flush())return;const owner=state,ticket=epoch,edit=owner.edit,m=$('science-task-model').value,d=$('science-task-dataset').value;
        if(!m||!d)throw Error('Choose exact architecture and dataset revisions first.');
        const result=await api.call('scienceDraftTaskCompatibility',{...args(),model:JSON.parse(m),dataset:JSON.parse(d)});
        if(!live(owner,ticket)||edit!==owner.edit||m!==$('science-task-model').value||d!==$('science-task-dataset').value)return;
        if(result.scope!=='interface-only-no-fitting-or-execution'||result.version!==1||!taskRef(result.task,owner)||
          !sameRef(result.model,JSON.parse(m))||!sameRef(result.dataset,JSON.parse(d))||!['compatible','incompatible'].includes(result.status))
          throw Error('Task compatibility returned a different exact context.');
        $('science-task-compatibility-status').textContent=result.status+(result.reason?' · '+result.reason:' · interface only; no fitting, execution or RAM admission')+' · model '+result.model.revision.slice(0,12)+' · dataset '+result.dataset.revision.slice(0,12);
      });
    },
    open(reference=null){return act(async()=>{
      if(state||window.LoomDatasetSessionForm?.active?.()||window.LoomExperimentSessionForm?.active?.()||window.LoomScienceAuthor?.ownActive?.()||window.LoomDraftValueEditor?.active?.())throw Error('Finish or discard the current draft before opening another.');
      if(api.busy())throw Error('Stop the run before opening a draft.');const identity=api.workspace(),ticket=epoch;
      const header=await api.call('scienceDraftOpen',{kind:'task',...(reference?{reference}:{})});
      if(header.scope!=='owned-authoring-session'||header.version!==1||header.kind!=='task'||header.workspace_id!==identity||header.sequence!==0||
        (reference?(header.id!==reference.id||header.base?.kind!=='task'||header.base?.id!==reference.id||header.base?.revision!==reference.revision):header.base!==null))
        throw Error('Task draft does not match its exact base and workspace.');
      state={header,stale:identity!==api.workspace()||ticket!==epoch,groups:{},pipelines:new Map(),edits:[],edit:0,choices:[],needsRead:false};layout();
      if(state.stale){$('science-author-status').textContent='Workspace changed. Unsaved task retained; cancel and reopen.';return;}
      const owner=state;await settled([readAll(),choices()]);
      if(live(owner,ticket))$('science-author-status').textContent='Unsaved task draft · bounded scientific-order cards · no training';
    });},
    stateChanged(){if(!api)return;if(workspaceSeen!==api.workspace()){workspaceSeen=api.workspace();epoch++;if(state){
      state.stale=true;contextEpoch++;window.LoomTaskContext?.clear();$('science-author-status').textContent='Workspace changed. Unsaved task retained; cancel and reopen.';
      if(!state.reloaded&&workspaceSeen===state.header.workspace_id){
        const owner=state,ticket=epoch,sequence=owner.header.sequence;
        api.call('scienceDraftOrigin',{session:owner.header.session,sequence}).then(header=>{
          if(state!==owner||ticket!==epoch||owner.reloaded||api.workspace()!==owner.header.workspace_id)return;
          if(!matchesHeader(header,owner,sequence))throw Error('Draft origin proof has a different identity or sequence.');
          owner.stale=false;changed();controls();
        }).catch(error=>{if(state===owner&&ticket===epoch){issue(error);controls();}});
      }
    }}controls();},
    event(event){if(event.kind==='externalReload'){epoch++;contextEpoch++;if(state){state.stale=true;state.reloaded=true;window.LoomTaskContext.clear();$('science-author-status').textContent='Workspace reloaded. Unsaved task retained; cancel and reopen.';}controls();}}
  };
})();
