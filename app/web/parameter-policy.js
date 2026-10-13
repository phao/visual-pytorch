(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.LoomParameterPolicy=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
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
    const result=JSON.parse(JSON.stringify(snapshot));
    if(Object.hasOwn(authoritative.graph,'parameter_policy'))result.graph.parameter_policy=JSON.parse(JSON.stringify(authoritative.graph.parameter_policy));
    else delete result.graph.parameter_policy;
    return result;
  }
  function mount(container,info,options){
    const document=container.ownerDocument;
    function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
    container.replaceChildren();
    container.append(element('h4','Parameter policy'),element('p','Initial values for a new run; not optimizer resume. Frozen weights still pass input gradients.','inspect-note'));
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
      card.append(element('div',owner.name+' ['+owner.shape.join(', ')+']','policy-owner-title'));
      const label=element('label',undefined,'inspect-bias'),toggle=element('input');toggle.type='checkbox';toggle.checked=owner.trainable;toggle.disabled=options.busy()||owner.effective_available===false;
      const status=element('span',owner.effective_available===false?'Unverified policy':owner.trainable?'Trainable':'Frozen');label.append(toggle,status);card.append(label);
      const error=element('p','', 'policy-error');error.setAttribute('role','alert');
      async function commit(changes){
        if(options.busy())return;error.textContent='';
        try{await options.commit(owner.selector,changes);}catch(e){error.textContent=e.message;toggle.checked=owner.trainable;}
      }
      toggle.onchange=()=>commit({trainable:toggle.checked});
      if(owner.shared)card.append(element('p','Shared canonical owner; edit affects every use.','inspect-note'));
      const details=element('details');details.append(element('summary','Initialization · '+owner.initializer_kind));
      const selection=element('select');selection.setAttribute('aria-label','Initializer for '+owner.name);
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
          const input=element(key==='values'?'textarea':'input');input.dataset.policyField=key;input.setAttribute('aria-label',key+' for '+owner.name);
          if(key!=='values'){input.type='number';input.step=key==='seed'?'1':'any';}
          else input.rows=4;
          input.value=key==='values'?JSON.stringify(value[key]):value[key];input.disabled=options.busy();label.append(input);fields.append(label);
        }
      }
      selection.onchange=renderFields;renderFields();
      const apply=element('button','Apply initialization');apply.type='button';apply.disabled=options.busy();
      apply.onclick=()=>{try{const values={};fields.querySelectorAll('[data-policy-field]').forEach(e=>values[e.dataset.policyField]=e.value);commit({initializer:parse(selection.value,values)});}catch(e){error.textContent=e.message;}};
      details.append(apply);card.append(details);
      const reset=element('button','Use inherited/default policy');reset.type='button';reset.disabled=options.busy();reset.onclick=()=>commit({trainable:null,initializer:null});
      card.append(reset);
      if(owner.definition){const open=element('button','Open owner');open.type='button';open.onclick=()=>options.openDefinition(owner.definition);card.append(open);}
      card.append(error);container.append(card);
    }
    container.append(element('p',info.note,'inspect-note'));
  }
  return {kinds,defaultInitializer,parse,reconcile,mount};
});
