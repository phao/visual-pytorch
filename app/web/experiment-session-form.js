"use strict";
(() => {
  let root,api,saved,state=null,pending=false,flushing=false,epoch=0,workspaceSeen=null,controlling=false;
  const $=id=>root.getElementById(id),el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const axes={models:['Frozen architecture selections','model','Architecture selection','Add architecture selection'],
    data_tasks:['Dataset/task pairings','data-task','Dataset/task pairing','Add dataset/task pairing'],
    splits:['Split selections','split','Split','Add split'],initialization_seeds:['Independent initialization seeds','seed','Initialization seed','Add initialization seed'],
    protocols:['Training configurations','protocol','Training configuration','Add training configuration'],
    pairs:['Pair whitelist','pairs','Whitelisted comparison','Add pair whitelist entry'],exclusions:['Explicit exclusions','exclusions','Excluded comparison','Add exclusion']};
  const args=()=>({session:state.header.session,sequence:state.header.sequence});
  const live=(owner,ticket)=>state===owner&&!owner.stale&&ticket===epoch&&owner.header.workspace_id===api.workspace();
  const size=value=>JSON.stringify(value).replace(/[\u0080-\uffff]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0')).length;
  const exact=(r,owner)=>r?.kind==='experiment'&&r.id===owner.header.id&&/^[a-f0-9]{64}$/.test(r.revision||'');
  const scalar=(d,fallback='')=>d?.present&&d.text!==null?JSON.parse(d.text):fallback;
  const properties=row=>Object.fromEntries(row.properties.map(p=>[p.key,p]));
  function issue(error){$('science-author-status').textContent=error.message;api.toast(error.message,true);}
  function clearPreview(){if(!state)return;state.preview=null;state.previewBacks=[];
    $('science-experiment-preview-rows').replaceChildren();$('science-experiment-preview-status').textContent='Draft preview is not current. Check it explicitly; no training.';
  }
  function changed(){if(!state)return;state.edit++;clearPreview();$('science-author-status').textContent='Unsaved experiment draft · '+state.edits.length+' pending edits · no training';}
  const blocked=()=>!state||state.stale||pending||api.busy()||state.header.workspace_id!==api.workspace()||window.LoomDraftValueEditor?.active?.();
  function controls(){if(!api||controlling)return;controlling=true;
    try{if(state){const stop=blocked(),ready=!state.needsRead&&Object.values(state.groups).every(g=>g.page&&!g.failed);
      for(const g of Object.values(state.groups))g.choices.forEach(c=>c.controls());
      for(const id of ['science-author-save','science-author-check','science-experiment-preview'])$(id).disabled=stop||!ready;
      $('science-author-cancel').disabled=pending||window.LoomDraftValueEditor?.active?.();
      $('science-author-retry-page').hidden=ready;$('science-author-retry-page').disabled=stop;
      $('science-experiment-refresh').disabled=stop;
      $('science-experiment-prev').disabled=stop||!state.preview||!state.previewBacks.length;
      $('science-experiment-next').disabled=stop||!state.preview||state.preview.next_offset===null||!Number.isSafeInteger(state.preview.next_offset);
      $('science-author').querySelectorAll('[data-experiment-control]').forEach(n=>n.disabled=stop||
        (state.needsRead&&n.dataset.repair!=='true')||n.dataset.large==='true'||n.dataset.orderDisabled==='true');
      for(const g of Object.values(state.groups)){g.add.disabled=stop||!g.page||state.needsRead;g.prev.disabled=stop||!g.backs.length||state.needsRead;
        g.next.disabled=stop||!g.page||g.page.next_offset===null||state.needsRead;}
    }window.LoomScienceAuthor?.stateChanged?.();window.LoomScienceDocument?.stateChanged?.();
    }finally{controlling=false;}
  }
  function capture(){const n=root.activeElement;if(state&&n?.matches('input,textarea')&&'dirty' in n.dataset&&'experimentControl' in n.dataset){
    flushing=true;try{n.dispatchEvent(new Event('change'));}finally{flushing=false;}
  }}
  async function command(fn){if(pending)return;capture();const owner=state,ticket=epoch,identity=api.workspace();pending=true;controls();
    try{await fn();}catch(error){if(ticket===epoch&&identity===api.workspace()&&(!owner||owner===state))issue(error);}
    finally{pending=false;controls();}
  }
  const act=fn=>Promise.resolve().then(()=>command(fn));
  async function settled(reads){const results=await Promise.allSettled(reads),failed=results.find(r=>r.status==='rejected');if(failed)throw failed.reason;}
  function enqueue(target,key,text,node){if(!state||state.stale||(pending&&!flushing)||api.busy()||window.LoomDraftValueEditor?.active?.())return;
    if(text!==null&&Array.from(text).length>2000)throw Error('Use staged value editing for more than 2000 code points.');
    const identity=JSON.stringify([target,key]);state.edits=state.edits.filter(e=>e.identity!==identity);
    state.edits.push({identity,node,operation:{op:'property',target,key,text}});delete node.dataset.invalid;delete node.dataset.dirty;node.removeAttribute('aria-invalid');changed();
  }
  function matchesHeader(h,owner,sequence){return h.scope==='owned-authoring-session'&&h.version===1&&h.kind==='experiment'&&h.id===owner.header.id&&
    h.session===owner.header.session&&h.workspace_id===owner.header.workspace_id&&h.sequence===sequence;}
  async function mutate(operations){const owner=state,ticket=epoch,sequence=owner.header.sequence;
    const result=await api.call('scienceDraftExperimentEdit',{...args(),operations});if(!live(owner,ticket))return false;
    if(!matchesHeader(result,owner,sequence+1))throw Error('Experiment edit acknowledgement does not match the owned sequence.');owner.header=result;return true;
  }
  async function flush(){capture();if([...$('science-author').querySelectorAll('[data-experiment-control]')].some(n=>'invalid' in n.dataset))
    throw Error('Correct invalid experiment fields before continuing. Your text has not been discarded.');
    while(state.edits.length){const owner=state,batch=[];
      for(const entry of owner.edits.slice(0,25)){const next=[...batch,entry];if(size({action:'scienceDraftExperimentEdit',...args(),operations:next.map(e=>e.operation)})>65536)break;batch.push(entry);}
      if(!batch.length)throw Error('Experiment edit exceeds the byte budget. Use staged value editing.');
      try{if(!await mutate(batch.map(e=>e.operation)))return false;}
      catch(error){const entry=batch[Number(/Experiment edit #(\d+):/.exec(error.message)?.[1])];if(entry){entry.node.dataset.invalid='';entry.node.setAttribute('aria-invalid','true');}throw error;}
      owner.edits=owner.edits.filter(e=>!batch.includes(e));
    }return true;
  }
  function matching(result,request,scope,owner){return result.scope===scope&&result.version===1&&result.session===request.session&&
    result.sequence===request.sequence&&result.workspace_id===owner.header.workspace_id&&result.id===owner.header.id;}
  async function page(collection,offset,limit=6){const owner=state,ticket=epoch,request={...args(),collection,offset,limit};
    const result=await api.call('scienceDraftExperimentPage',request);if(!live(owner,ticket))return null;
    if(!matching(result,request,'owned-experiment-page',owner)||result.collection!==collection||result.offset!==offset||result.limit!==limit)
      throw Error('Experiment page does not match its exact owner, sequence and cursor.');return result;
  }
  async function readGroup(collection){const owner=state,ticket=epoch,g=owner.groups[collection];
    try{const result=await page(collection,g.offset);if(!result)return false;g.choices.forEach(c=>c.dispose());g.choices=[];g.page=result;g.failed=false;
      renderGroup(collection);await settled(g.choices.map(c=>c.refresh()));return live(owner,ticket);
    }catch(error){if(live(owner,ticket)){g.failed=true;g.range.textContent=error.message+' · retry page';}throw error;}
  }
  async function readAll(){const owner=state;await settled(Object.keys(axes).map(readGroup));if(state!==owner||owner.stale)return false;
    owner.needsRead=false;renderRoots();controls();return true;
  }
  function button(name,fn,title){const b=el('button',name);b.type='button';b.dataset.experimentControl='';b.dataset.authorControl='';if(title)b.title=title;b.onclick=()=>act(fn);return b;}
  async function structure(collection,build){if(!await flush())return;const owner=state,fresh=await page(collection,owner.groups[collection].offset);if(!fresh)return;
    const operations=await build(fresh);if(!operations||!await mutate(operations))return;owner.needsRead=true;changed();await readAll();
  }
  function navigate(g,forward){return async()=>{if(!await flush())return;const old=g.offset,next=forward?g.page?.next_offset:g.backs.at(-1);if(next===null||next===undefined)return;
    g.offset=next;try{if(await readGroup(g.collection)){if(forward)g.backs.push(old);else g.backs.pop();}else g.offset=old;}catch(error){g.offset=old;throw error;}
  };}
  async function raw(target,key,path=null){if(!await flush())return;const owner=state,ticket=epoch;
    if(path===null){const request={...args(),target,key},result=await api.call('scienceDraftExperimentValue',request);if(!live(owner,ticket))return;
      if(!matching(result,request,'owned-experiment-value',owner)||JSON.stringify(result.target)!==JSON.stringify(target)||result.key!==key||!result.property.present)
        throw Error('Reload the current experiment property before staging it.');path=result.property.path;
    }
    await window.LoomDraftValueEditor.openOwned({header:{...owner.header},path:[...path],label:key||'Complete experiment JSON',onClosed:async header=>{
      if(!live(owner,ticket))return;if(!matchesHeader(header,owner,owner.header.sequence)&&!matchesHeader(header,owner,owner.header.sequence+1))
        throw Error('Staged experiment value returned a different parent.');owner.header=header;owner.needsRead=true;changed();await command(readAll);
    }});
  }
  function rawButton(target,key){return button('Edit value…',()=>raw(target,key),'Stage exact JSON in the same unsaved experiment');}
  function control(label,d,target,key,{mode='string',area=false,optional=false,integer=false,min=-Infinity,max=Infinity,exclusiveMax=false}={}){
    const wrapper=el('label',label),n=el(area?'textarea':'input');n.dataset.experimentControl='';n.dataset.authorControl='';
    const large=d?.present&&d.text===null;n.dataset.large=String(large);
    if(large){n.placeholder='Large '+d.summary.type+' · '+d.summary.size+' · use staged editor';}
    else if(!d?.present)n.value='';else if(mode==='number')n.value=d.text==='null'?'':d.text;
    else if(mode==='json')n.value=d.text;
    else if(mode==='ids')n.value=scalar(d,[]).join('\n');else n.value=scalar(d);
    n.oninput=()=>{n.dataset.dirty='';changed();};n.onchange=()=>{try{let text;
      if(mode==='number'){text=n.value.trim();if(optional&&text==='')text='null';else{
        if(integer){if(!/^-?(0|[1-9]\d*)$/.test(text))throw Error('Enter an exact decimal integer.');const v=BigInt(text);
          if((Number.isFinite(min)&&v<BigInt(min))||(Number.isFinite(max)&&v>BigInt(max)))throw Error('Integer is outside its declared range.');
        }else{const v=JSON.parse(text);if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max||(exclusiveMax&&v===max))throw Error('Enter a finite number in its declared range.');}
      }}else if(mode==='json'){text=n.value.trim();JSON.parse(text);}
      else text=JSON.stringify(mode==='ids'?window.LoomExperimentAuthoring.ids(n.value):n.value);
      enqueue(target,key,text,n);
    }catch(error){n.dataset.invalid='';n.setAttribute('aria-invalid','true');issue(error);}};
    wrapper.append(n);if(large){if(d.summary.preview!==null)wrapper.append(el('small','Partial preview · '+d.summary.preview));wrapper.append(rawButton(target,key));}return wrapper;
  }
  function select(label,options,value,target,key){const wrapper=el('label',label),n=el('select');n.dataset.experimentControl='';n.dataset.authorControl='';
    for(const [id,name] of options){const o=el('option',name);o.value=id;n.append(o);}
    if(!options.some(([id])=>id===value)){const o=el('option','Unavailable value · '+value);o.value=value;n.append(o);}n.value=value;
    n.onchange=()=>{try{enqueue(target,key,JSON.stringify(n.value),n);}catch(error){issue(error);}};wrapper.append(n);return wrapper;
  }
  function choice(label,d,target,key,g,kind,local=false){let value=scalar(d,null);
    if(d?.present&&d.text===null)return control(label,d,target,key);
    const wrapper=el('label',label),n=el('select');n.dataset.authorControl='';n.dataset.experimentControl='';wrapper.append(n);
    n.onchange=()=>{try{value=kind==='data_tasks'?n.value:n.value?JSON.parse(n.value):kind==='models'?window.LoomExperimentAuthoring.reference('model'):kind==='model'||kind==='dataset'||kind==='task'?window.LoomExperimentAuthoring.reference(kind):'';
      enqueue(target,key,JSON.stringify(value),n);
    }catch(error){n.dataset.invalid='';issue(error);}};
    // Attachment needs a connected label to keep tools outside its label scope.
    g.cards.append(wrapper);const services={...api,busy:()=>api.busy()||pending||state?.needsRead||window.LoomDraftValueEditor?.active?.(),updateControls:controls};
    const c=local?window.LoomExperimentLocalChoice.attach(root,n,{...services,arguments:args,id:()=>state.header.id,
      run:fn=>act(async()=>{if(await flush())await fn();})},{collection:kind,getValue:()=>value,valid:()=>state===g.owner&&!g.owner.stale}):
      window.LoomRevisionChoice.attach(root,n,services,{kind,getReference:()=>value,valid:()=>state===g.owner&&!g.owner.stale});
    g.choices.push(c);return n.closest('.revision-choice-field')||wrapper;
  }
  function renderGroup(collection){const g=state.groups[collection];g.cards.replaceChildren();
    for(const row of g.page.rows){const target={kind:'entry',collection,ordinal:row.ordinal},p=properties(row),card=el('fieldset');
      card.dataset.experimentAxis=axes[collection][1];card.dataset.axisIndex=row.ordinal;card.append(el('legend',axes[collection][2]+' '+(row.ordinal+1)));
      const grid=el('div');grid.className='science-author-grid';card.append(grid);g.cards.append(card);
      const text=(label,key,opts={})=>control(label,p[key],target,key,opts),num=(label,key,opts={})=>text(label,key,{mode:'number',...opts});
      const ref=(label,key,kind,local=false)=>choice(label,p[key],target,key,g,kind,local);
      if(collection==='models'){
        if(p.family){grid.append(text('Family choice ID','id'),ref('Exact family','family','family'),
          text('Fixed bindings JSON','bindings',{mode:'json',area:true}),text('Ordered binding axes JSON','axes',{mode:'json',area:true}));
          card.append(el('p','Each axis: {"symbol":"stable-id","values":[2,4]}. Last binding axis changes fastest. Missing defaults stay conditional; incompatible combinations remain visible. No automatic training.'));
        }else grid.append(ref('Exact architecture','reference','model'));
      }
      else if(collection==='data_tasks')grid.append(text('Pairing ID','id'),ref('Exact dataset','dataset','dataset'),ref('Exact task','task','task'));
      else if(collection==='initialization_seeds')grid.append(num('Initialization seed','value',{integer:true,min:0,max:2**31-1}));
      else if(collection==='splits'){const method=scalar(p.method);grid.append(text('Split ID','id'));
        if(method==='row'||method==='group')grid.append(num('Split seed','seed',{integer:true,min:0,max:2**31-1}));
        if(method==='group'||method==='temporal')grid.append(text(method==='group'?'Grouping field ID':'Time field ID','field'));
        if(method==='external'){for(const role of ['train','validation','test'])grid.append(text(role+' observation IDs','roles.'+role,{mode:'ids',area:true}));
          grid.append(text('Excluded observation IDs','excluded',{mode:'ids',area:true}));
        }else if(['row','group','temporal'].includes(method))for(const [i,label] of ['Train fraction','Validation fraction','Test fraction'].entries())grid.append(num(label,'fractions.'+i,{min:0,max:1}));
        else card.append(el('p','Incomplete split method · repair the exact entry explicitly.'));
        card.append(el('p','Method: '+method+' · remove/add to change it. IDs and scientific order are not rewritten.'));
      }else if(collection==='protocols'){
        grid.append(text('Configuration ID','id'),num('Batch size','batch_size',{integer:true,min:1}),num('Evaluate every updates','evaluate_every',{integer:true,min:1}),num('Retained curve points','history.max_points',{integer:true,min:1}));
        card.append(el('p','CPU float32 · Adam · no augmentation · last successful checkpoint · completed-test evaluation.'));
        for(const key of ['epochs','updates','examples','seconds'])grid.append(num('Maximum '+key,'budget.'+key,{optional:true,integer:key!=='seconds',min:key==='seconds'?Number.MIN_VALUE:1}));
        card.append(el('p','Blank disables a bound; at least one is required. Stops at the first observed active bound.'));
        for(const [label,key,opts] of [['Learning rate','optimizer.lr',{min:Number.MIN_VALUE,max:1}],['Adam epsilon','optimizer.eps',{min:Number.MIN_VALUE}],['Weight decay','optimizer.weight_decay',{min:0}]])grid.append(num(label,key,opts));
        for(let i=0;i<2;i++)grid.append(num('Adam beta '+(i+1),'optimizer.betas.'+i,{min:0,max:1,exclusiveMax:true}));
        grid.append(num('Protocol initialization seed (axis overrides)','seeds.initialization',{integer:true,min:0,max:2**31-1}),num('Independent shuffle seed','seeds.shuffle',{integer:true,min:0,max:2**31-1}));
        const label=el('label','Shuffle training observations '),n=el('input');n.type='checkbox';n.dataset.experimentControl='';n.dataset.authorControl='';n.checked=scalar(p.shuffle,false);
        n.onchange=()=>enqueue(target,'shuffle',JSON.stringify(n.checked),n);label.append(n);grid.append(label);
      }else{grid.append(scalar(p.model,null)?.bindings?
        text('Exact family coordinate JSON','model',{mode:'json',area:true}):ref('Selected exact architecture','model','models',true),ref('Dataset/task pairing ID','data_task','data_tasks',true));
        if(collection==='exclusions')grid.append(text('Exclusion reason','reason',{area:true}));
      }
      const order=el('div');order.className='inline-actions';order.dataset.axisOrder='';
      for(const [name,delta] of [['↑',-1],['↓',1]]){const b=button(name,()=>structure(collection,f=>[{op:'move',path:f.path,from:row.ordinal,to:row.ordinal+delta}]));
        b.dataset.orderDisabled=String(delta<0?row.ordinal===0:row.ordinal===g.page.total-1);order.append(b);
      }order.append(button('Remove entry',()=>structure(collection,f=>[{op:'remove',path:[...f.path,row.ordinal]}])),button('Edit entry JSON…',async()=>{
        if(!await flush())return;const fresh=await page(collection,row.ordinal,1);if(fresh?.rows[0])await raw(null,null,fresh.rows[0].path);
      }));card.append(order);
    }g.range.textContent=(g.page.rows.length?g.offset+1:0)+'–'+(g.offset+g.page.rows.length)+' / '+g.page.total+' · scientific order';controls();
  }
  function renderRoots(){const roots=Object.fromEntries(state.groups.models.page.roots.map(p=>[p.key,p]));state.rootControls.replaceChildren(
    control('Research question',roots.question,{kind:'document'},'question',{area:true}),
    select('Comparison mode',[['crossed','Crossed · every architecture × dataset/task'],['paired','Paired · explicit whitelist']],scalar(roots['selection.kind']),{kind:'document'},'selection.kind'),
    el('p','Crossed requires an empty whitelist; paired needs explicit entries. Changing mode never removes entries; exclusions keep reasons.'));
  }
  function layout(){const owner=state,h=owner.header;$('science-author').hidden=false;$('science-author-title').textContent='Experiment draft';
    $('science-author-id').textContent='Stable experiment ID: '+h.id+' · '+(h.base?'editing exact '+h.base.revision.slice(0,12):'new object')+(h.base&&h.base.revision!==h.expected_head?' · historical base, not current head':'');
    $('science-author-dataset').hidden=true;$('science-author-task').hidden=true;$('science-author-experiment').hidden=false;
    const container=$('science-experiment-bindings');container.replaceChildren();owner.rootControls=el('div');container.append(owner.rootControls);
    const repair=button('Edit complete experiment JSON…',()=>raw(null,null,[]));repair.dataset.repair='true';container.append(repair);
    const name=$('science-author-name');name.dataset.experimentControl='';name.value=h.name;
    name.oninput=()=>{name.dataset.dirty='';changed();};name.onchange=()=>{try{enqueue({kind:'document'},'name',JSON.stringify(name.value),name);}catch(error){name.dataset.invalid='';issue(error);}};
    for(const [collection,[label,,,addLabel]] of Object.entries(axes)){const section=el('details');section.open=true;section.dataset.experimentCollection=collection;section.append(el('summary',label));container.append(section);
      const g={owner,collection,offset:0,backs:[],page:null,failed:false,choices:[],cards:el('div'),range:el('span')};g.range.className='tiny';g.range.setAttribute('role','status');owner.groups[collection]=g;
      g.prev=button('‹ Entries',navigate(g,false));g.next=button('Entries ›',navigate(g,true));const actions=el('div');actions.className='inline-actions';actions.append(g.prev,g.range,g.next);section.append(actions,g.cards);
      if(collection==='splits'){g.method=el('select');g.method.dataset.experimentControl='';g.method.dataset.authorControl='';g.method.setAttribute('aria-label','Split method to add');
        for(const method of window.LoomExperimentAuthoring.methods){const o=el('option',method);o.value=method;g.method.append(o);}g.method.value='row';section.append(g.method);
      }
      g.add=button(addLabel,()=>structure(collection,f=>[{op:'insert-json',path:f.path,index:f.total,text:JSON.stringify(
        collection==='models'?window.LoomExperimentAuthoring.reference('model'):collection==='data_tasks'?window.LoomExperimentAuthoring.pairing(crypto.randomUUID()):
        collection==='splits'?window.LoomExperimentAuthoring.split(crypto.randomUUID(),g.method.value):collection==='initialization_seeds'?7:
        collection==='protocols'?window.LoomExperimentAuthoring.protocol(crypto.randomUUID()):window.LoomExperimentAuthoring.comparison(collection==='exclusions'))}]));section.append(g.add);
      if(collection==='models')section.append(button('Add family selection',()=>structure(collection,f=>[
        {op:'property',target:{kind:'document'},key:'version',text:'2'},
        {op:'insert-json',path:f.path,index:f.total,text:JSON.stringify(window.LoomExperimentAuthoring.family(crypto.randomUUID()))}])));
    }$('science-experiment-context-status').textContent='Independent exact stored/local choices · bounded scientific-order axes · no training';clearPreview();controls();
  }
  async function refreshChoices(){const owner=state;await settled(Object.values(owner.groups).flatMap(g=>g.choices.map(c=>c.refresh())));}
  async function preview(offset=0){if(!await flush())return false;const owner=state,ticket=epoch,edit=owner.edit,request={...args(),offset,limit:25};
    const result=await api.call('scienceDraftExperimentPreview',request);if(!live(owner,ticket)||edit!==owner.edit)return false;
    if(!matching(result,request,'owned-experiment-preview',owner)||result.offset!==offset||result.limit!==25||!exact(result.experiment,owner)||
      result.execution!=='no-fitting-or-training'||!/^\d+$/.test(result.total_combinations_exact))throw Error('Preview does not match the exact experiment, owner and cursor.');
    owner.preview=result;const box=$('science-experiment-preview-rows');box.replaceChildren();for(const row of result.candidates){const card=el('div');card.className='science-candidate';card.dataset.draftCandidate=row.status;
      card.append(el('span',row.status+' · '+window.LoomExperimentAuthoring.modelLabel(row.coordinates.model)+' · '+row.coordinates.data_task+' · seed '+row.coordinates.initialization+' · '+row.coordinates.split+' · '+row.coordinates.protocol));
      if(row.parameters_exact!==undefined&&row.parameters_exact!==null)card.append(el('small',row.parameters_exact+' trainable parameters'));
      if(row.reason)card.append(el('small',row.reason.text+(row.reason.next_offset===null?'':' · partial reason; inspect/export full experiment')));
      if(row.counts)card.append(el('small','Observations · train '+row.counts.train+' · validation '+row.counts.validation+' · test '+row.counts.test+' · excluded '+row.counts.excluded));box.append(card);
    }$('science-experiment-preview-status').textContent=offset+'–'+(offset+result.candidates.length)+' of '+result.total_combinations_exact+' · prospective '+result.experiment.revision.slice(0,12)+' · interface/split preview only, no fitting or execution RAM admission. Retain before runs.';controls();return true;
  }
  function closeUI(){Object.values(state.groups).forEach(g=>g.choices.forEach(c=>c.dispose()));clearPreview();state=null;$('science-experiment-bindings').replaceChildren();$('science-author').hidden=true;$('science-author-retry-page').hidden=true;
    const name=$('science-author-name');for(const key of ['experimentControl','dirty','invalid'])delete name.dataset[key];name.removeAttribute('aria-invalid');
  }
  function route(id,fn){const prior=$(id).onclick;$(id).onclick=()=>state||pending?act(fn):prior?.();}
  window.LoomExperimentSessionForm={active:()=>Boolean(state)||pending,
    mount(document,services,onSaved){root=document;api=services;saved=onSaved;workspaceSeen=api.workspace();$('science-author-new-experiment').onclick=()=>this.open();const prior=$('science-author-edit').onclick;
      $('science-author-edit').onclick=()=>{const ref=$('science-object').value?JSON.parse($('science-object').value):null;return ref?.kind==='experiment'?this.open(ref):prior?.();};
      route('science-author-cancel',async()=>{const owner=state;await api.call('scienceDraftClose',{session:owner.header.session});if(state===owner)closeUI();});
      route('science-author-retry-page',async()=>{if(await flush())await readAll();});route('science-experiment-refresh',refreshChoices);
      route('science-author-check',async()=>{if(!await flush())return;const owner=state,ticket=epoch,edit=owner.edit,result=await api.call('scienceDraftCheck',args());
        if(!live(owner,ticket)||edit!==owner.edit)return;if(!exact(result,owner))throw Error('Experiment validation returned a different identity.');$('science-author-status').textContent='Valid ExperimentSpec · prospective revision '+result.revision+' · not saved, no training';});
      route('science-author-save',async()=>{if(!await flush())return;const owner=state,ticket=epoch,result=await api.call('scienceDraftCommit',args());if(!live(owner,ticket))return;
        if(!exact(result,owner))throw Error('Retained experiment has the wrong identity.');closeUI();await saved(result);api.toast('Experiment revision retained. Save workspace to persist it on disk.');});
      route('science-experiment-preview',async()=>{state.previewBacks=[];await preview();});
      route('science-experiment-prev',async()=>{if(!state.previewBacks.length)return;if(await preview(state.previewBacks.at(-1)))state.previewBacks.pop();});
      route('science-experiment-next',async()=>{if(!state.preview)return;const old=state.preview.offset;if(await preview(state.preview.next_offset))state.previewBacks.push(old);});
    },
    open(reference=null){return act(async()=>{if(state||window.LoomDatasetSessionForm?.active?.()||window.LoomTaskSessionForm?.active?.()||window.LoomScienceAuthor?.ownActive?.()||window.LoomDraftValueEditor?.active?.())throw Error('Finish or discard the current draft before opening another.');
      if(api.busy())throw Error('Stop the run before opening a draft.');const identity=api.workspace(),ticket=epoch,header=await api.call('scienceDraftOpen',{kind:'experiment',...(reference?{reference}:{})});
      if(header.scope!=='owned-authoring-session'||header.version!==1||header.kind!=='experiment'||header.workspace_id!==identity||header.sequence!==0||
        (reference?(header.id!==reference.id||header.base?.kind!=='experiment'||header.base?.id!==reference.id||header.base?.revision!==reference.revision):header.base!==null))throw Error('Experiment draft does not match its exact base and workspace.');
      state={header,stale:identity!==api.workspace()||ticket!==epoch,groups:{},edits:[],edit:0,needsRead:true,preview:null,previewBacks:[]};layout();
      if(state.stale){$('science-author-status').textContent='Workspace changed. Unsaved experiment retained; cancel and reopen.';return;}
      const owner=state;await readAll();if(live(owner,ticket))$('science-author-status').textContent='Unsaved experiment draft · bounded scientific-order cards · no training';
    });},
    stateChanged(){if(!api)return;if(workspaceSeen!==api.workspace()){workspaceSeen=api.workspace();epoch++;if(state){state.stale=true;clearPreview();$('science-author-status').textContent='Workspace changed. Unsaved experiment retained; cancel and reopen.';
      if(!state.reloaded&&workspaceSeen===state.header.workspace_id){const owner=state,ticket=epoch,sequence=owner.header.sequence;api.call('scienceDraftOrigin',{session:owner.header.session,sequence}).then(header=>{
        if(state!==owner||ticket!==epoch||owner.reloaded||api.workspace()!==owner.header.workspace_id)return;if(!matchesHeader(header,owner,sequence))throw Error('Draft origin proof has a different identity or sequence.');owner.stale=false;changed();controls();
      }).catch(error=>{if(state===owner&&ticket===epoch){issue(error);controls();}});}
    }}controls();},
    event(event){if(event.kind==='externalReload'){epoch++;if(state){state.stale=true;state.reloaded=true;clearPreview();$('science-author-status').textContent='Workspace reloaded. Unsaved experiment retained; cancel and reopen.';}controls();}}
  };
})();
