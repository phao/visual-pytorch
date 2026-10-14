(function(root,factory){const api=factory(typeof module==='object'&&module.exports?require('./json-codec.js'):root.LoomJSON);if(typeof module==='object'&&module.exports)module.exports=api;else root.LoomParameterPolicy=api;})(typeof globalThis!=='undefined'?globalThis:this,function(J){
  'use strict';
  const kinds=['constructor','normal','uniform','values','zeros'];
  function defaultInitializer(kind){
    if(kind==='normal')return {kind,seed:7,mean:0,std:.05};
    if(kind==='uniform')return {kind,seed:7,low:-.1,high:.1};
    if(kind==='values')return {kind,dtype:'float32',values:[]};
    if(['constructor','zeros'].includes(kind))return {kind};
    throw new Error('Unsupported initializer.');
  }
  function parse(kind,fields){
    const value=defaultInitializer(kind);
    if(kind==='values'){value.values=JSON.parse(fields.values);if(!Array.isArray(value.values))throw new Error('Values must be a JSON array with the exact displayed shape.');}
    else if(['normal','uniform'].includes(kind)){
      for(const key of Object.keys(value).filter(k=>k!=='kind')){
        if(String(fields[key]).trim()==='')throw new Error(key+' cannot be empty.');
        value[key]=Number(fields[key]);if(!Number.isFinite(value[key]))throw new Error(key+' must be finite.');
      }
      if(!Number.isInteger(value.seed)||value.seed<0||value.seed>2147483647)throw new Error('Seed must be an integer from 0 to 2147483647.');
      if(kind==='normal'&&!(value.std>0))throw new Error('Standard deviation must be positive.');
      if(kind==='uniform'&&!(value.low<value.high))throw new Error('Uniform bounds must be ordered.');
    }
    return value;
  }
  function reconcile(snapshot,authoritative){
    if(snapshot.id!==authoritative.id)throw new Error('Cannot reconcile different architectures.');
    const result=J.clone(snapshot);
    if(Object.hasOwn(authoritative.graph,'parameter_policy'))result.graph.parameter_policy=J.clone(authoritative.graph.parameter_policy);
    else delete result.graph.parameter_policy;
    if(Object.hasOwn(authoritative,'result'))result.result=J.clone(authoritative.result);else delete result.result;
    return result;
  }
  function drafts(container){
    if(!container)return [];
    return [...container.querySelectorAll('.policy-owner')].map(card=>({selector:card.dataset.policySelector,
      open:card.querySelector('details')?.open,kind:card.querySelector('details select')?.value,
      fields:[...card.querySelectorAll('[data-policy-field]')].map(e=>[e.dataset.policyField,e.value]),
      error:card.querySelector('[data-policy-owner-error]')?.textContent||''}));
  }
  function restore(container,saved){
    for(const card of container.querySelectorAll('.policy-owner')){
      const row=saved.find(row=>row.selector===card.dataset.policySelector);if(!row)continue;
      const details=card.querySelector('details'),kind=details?.querySelector('select');
      if(kind&&[...kind.options].some(option=>option.value===row.kind)){kind.value=row.kind;kind.onchange();}
      if(details)details.open=Boolean(row.open);
      for(const e of card.querySelectorAll('[data-policy-field]')){const field=row.fields.find(field=>field[0]===e.dataset.policyField);if(field)e.value=field[1];}
      const error=card.querySelector('[data-policy-owner-error]');if(error)error.textContent=row.error;
    }
  }
  function mount(container,info,options){
    const document=container.ownerDocument;
    function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
    container.replaceChildren();
    // The Inspector sub-panel supplies the heading; explanations live on About pages.
    const ownerName=name=>options.ownerName?options.ownerName(name):name;
    if(info.policy_error)container.append(element('p','Policy needs repair: '+info.policy_error,'policy-error'));
    if(info.has_local_policy){
      const reset=element('button','Reset this block’s local policies');reset.type='button';reset.disabled=options.busy();
      reset.dataset.policyResetBlock='true';
      reset.title='Explicitly remove local trainability and initialization declarations for this block, including shared aliases and obsolete tensors. Referenced definitions and other blocks are unchanged.';
      const error=element('p','','policy-error');error.setAttribute('role','alert');
      reset.onclick=async()=>{if(options.busy())return;try{await options.resetBlock();}catch(e){error.textContent=e.message;}};
      container.append(reset,error);
    }
    if(!info.owners.length){container.append(element('p','This block owns no learned parameters.','inspect-note'));return;}
    for(const owner of info.owners){
      const card=element('div',undefined,'policy-owner');card.dataset.policySelector=owner.selector;
      card.append(element('div',ownerName(owner.name)+' ['+owner.shape.join(', ')+']','policy-owner-title'));
      if(owner.initial_source){const source=owner.initial_source.source,note=element('p','Pinned tensor source: '+source.run.id+' · attempt '+source.attempt,'inspect-note');note.title='Current initial values match this retained receipt. The receipt pins its original model snapshot; this is not proof that the whole current architecture equals that snapshot. '+JSON.stringify(owner.initial_source);note.style.overflowWrap='anywhere';card.append(note);}
      const label=element('label',undefined,'inspect-bias'),toggle=element('input');toggle.type='checkbox';toggle.checked=owner.trainable;toggle.disabled=options.busy()||owner.effective_available===false;
      const status=element('span',owner.effective_available===false?'Unverified policy':owner.trainable?'Trainable':'Frozen');label.append(toggle,status);card.append(label);
      const error=element('p','', 'policy-error');error.setAttribute('role','alert');
      error.dataset.policyOwnerError='true';
      async function commit(changes){
        if(options.busy())return;error.textContent='';
        try{await options.commit(owner.selector,changes);}catch(e){error.textContent=e.message;toggle.checked=owner.trainable;}
      }
      toggle.onchange=()=>commit({trainable:toggle.checked});
      if(owner.shared)card.append(element('p','Shared canonical owner; edit affects every use.','inspect-note'));
      const details=element('details');details.append(element('summary','Initialization · '+owner.initializer_kind));
      const selection=element('select');selection.setAttribute('aria-label','Initializer for '+ownerName(owner.name));
      for(const kind of kinds){const option=element('option',kind);option.value=kind;selection.append(option);}
      selection.value=kinds.includes(owner.initializer_kind)?owner.initializer_kind:'constructor';selection.disabled=options.busy();details.append(selection);
      const fields=element('div',undefined,'policy-fields');details.append(fields);
      let initial=owner.initializer_json?JSON.parse(owner.initializer_json):null;
      function renderFields(){
        fields.replaceChildren();const value=initial?.kind===selection.value?initial:defaultInitializer(selection.value);
        if(selection.value==='values'&&value===null)return;
        if(selection.value==='values'&&!owner.initializer_json&&owner.initializer_kind==='values'){
          fields.append(element('p','Exact values exceed the 64 KiB Inspector limit. Trainability remains editable; use the CLI to replace large arrays.','inspect-note'));
        }
        for(const key of Object.keys(value).filter(k=>!['kind','dtype'].includes(k))){
          const label=element('label',key==='std'?'Standard deviation':key==='values'?'Exact float32 values · JSON':key,'inspect-label');
          const input=element(key==='values'?'textarea':'input');input.dataset.policyField=key;input.setAttribute('aria-label',key+' for '+ownerName(owner.name));
          if(key!=='values'){input.type='number';input.step=key==='seed'?'1':'any';}
          else input.rows=4;
          input.value=key==='values'?J.stringify(value[key]):value[key];input.disabled=options.busy();label.append(input);fields.append(label);
        }
      }
      selection.onchange=renderFields;renderFields();
      const apply=element('button','Apply initialization');apply.type='button';apply.disabled=options.busy();
      apply.onclick=()=>{try{const values={};fields.querySelectorAll('[data-policy-field]').forEach(e=>values[e.dataset.policyField]=e.value);commit({initializer:parse(selection.value,values)});}catch(e){error.textContent=e.message;}};
      details.append(apply);card.append(details);
      if(options.binding&&typeof globalThis.LoomParameterBinding!=='undefined')globalThis.LoomParameterBinding.attach(card,owner,options.binding);
      const reset=element('button','Use inherited/default policy');reset.type='button';reset.disabled=options.busy();reset.onclick=()=>commit({trainable:null,initializer:null});
      card.append(reset);
      if(owner.definition){const open=element('button','Open owner');open.type='button';open.onclick=()=>options.openDefinition(owner.definition);card.append(open);}
      card.append(error);container.append(card);
    }
  }
  return {kinds,defaultInitializer,parse,reconcile,drafts,restore,mount};
});
