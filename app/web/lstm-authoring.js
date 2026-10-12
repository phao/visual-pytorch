"use strict";
((root,factory)=>{const api=factory();if(typeof module!=="undefined"&&module.exports)module.exports=api;else root.LoomLSTM=api;})(typeof globalThis!=="undefined"?globalThis:this,()=>{
  const roles=['sequence','hidden','cell'];
  function integer(value,name,min=1,max=Number.MAX_SAFE_INTEGER){
    const text=String(value);
    if(!/^\d+$/.test(text)||!Number.isSafeInteger(Number(text))||Number(text)<min||Number(text)>max)
      throw Error(name+' must be an exact integer from '+min+' to '+max+'.');
    return Number(text);
  }
  function options(value,editing=false){
    if(typeof value.label!=='string'||!value.label.trim()||value.label.length>60)throw Error('Choose a block name of 1–60 characters.');
    for(const key of ['bias','bidirectional'])if(typeof value[key]!=='boolean')throw Error(key+' must be boolean.');
    const result={version:1,label:value.label.trim(),hidden_size:integer(value.hidden_size,'Cell width'),
      num_layers:integer(value.num_layers,'Stacked layers',1,64),proj_size:integer(value.proj_size,'Projection width',0),
      bias:value.bias,bidirectional:value.bidirectional};
    if(result.proj_size>=result.hidden_size)throw Error('Projection must be zero or smaller than cell width.');
    if(editing){
      if(!value.inputs||Object.keys(value.inputs).length!==3||roles.some(role=>typeof value.inputs[role]!=='string'||!value.inputs[role].trim()))
        throw Error('Choose existing sequence, initial hidden and initial cell sources.');
      result.inputs=Object.fromEntries(roles.map(role=>[role,value.inputs[role]]));
      if(value.lengths!==null&&(!value.lengths||Object.keys(value.lengths).length!==2||
        typeof value.lengths.source!=='string'||!value.lengths.source.trim()||value.lengths.adapter!=='exact-int64-v1'))
        throw Error('Lengths require an existing source and exact-int64-v1 adapter.');
      result.lengths=value.lengths===null?null:{...value.lengths};
      if(value.share_with!==null&&(typeof value.share_with!=='string'||!value.share_with.trim()))throw Error('Choose a parameter owner or independent parameters.');
      result.share_with=value.share_with;
    }else{
      result.steps=integer(value.steps,'Sequence length');result.input_size=integer(value.input_size,'Input features');
      if(typeof value.lengths!=='boolean')throw Error('Supplied lengths must be boolean.');result.lengths=value.lengths;
      if(result.lengths&&result.steps>2**24)throw Error('Supplied float32 lengths require T ≤ 2²⁴.');
    }
    return result;
  }
  function stateShapes(value){const v=options({...value,steps:'1',input_size:'1',lengths:false}),ld=v.num_layers*(v.bidirectional?2:1),q=v.proj_size||v.hidden_size;
    return {hidden:[ld,q],cell:[ld,v.hidden_size],sequence_features:q*(v.bidirectional?2:1)};}
  function checkToken(info,context,node=null){
    if(!info||!context.id||info.id!==context.id||info.workspace_id!==context.workspace_id||info.panel_scope!==context.panel_scope||
      (node!==null&&info.node!==node)||!['root_graph','root_model'].every(key=>typeof info[key]==='string'&&/^[a-f0-9]{64}$/.test(info[key])))
      throw Error('LSTM response belongs to another block or workspace.');
    return Object.fromEntries(['id','root_graph','root_model',...(node!==null?['node']:[])].map(k=>[k,info[k]]));
  }
  function mount(document,services){
    const $=id=>document.getElementById('lstm-'+id),dialog=$('dialog'),form=$('form');
    let generation=0,pending=false,info=null,context=null,node=null,token=null;
    const current=g=>g===generation&&context&&JSON.stringify(services.context())===JSON.stringify(context)&&!services.busy();
    function lock(value){pending=value;form.querySelectorAll('input,select,button').forEach(el=>el.disabled=value&&el.id!=='lstm-cancel');services.refresh();}
    function close(){generation++;info=null;token=null;lock(false);}
    dialog.addEventListener('close',close);$('cancel').onclick=()=>dialog.close();
    function choices(){const value={label:$('label').value,hidden_size:$('hidden-size').value,num_layers:$('layers').value,
      proj_size:$('projection').value,bias:$('bias').checked,bidirectional:$('bidirectional').checked};
      if(node!==null)return options({...value,inputs:Object.fromEntries(roles.map(r=>[r,$('source-'+r).value])),
        lengths:$('source-lengths').value?{source:$('source-lengths').value,adapter:'exact-int64-v1'}:null,share_with:$('owner').value||null},true);
      return options({...value,steps:$('steps').value,input_size:$('input-size').value,lengths:$('lengths').checked});}
    function preview(){try{const s=stateShapes({label:$('label').value,hidden_size:$('hidden-size').value,num_layers:$('layers').value,
      proj_size:$('projection').value,bias:$('bias').checked,bidirectional:$('bidirectional').checked});
      $('shapes').textContent='Initial/final hidden [B, '+s.hidden.join(', ')+']\nInitial/final cell [B, '+s.cell.join(', ')+']\nSequence output [B, T, '+s.sequence_features+']';
    }catch(error){$('shapes').textContent=error.message;}}
    form.addEventListener('input',preview);form.addEventListener('change',preview);
    function fillSelect(select,rows,selected,optional=false){select.replaceChildren();
      if(optional){const empty=document.createElement('option');empty.value='';empty.textContent='None / independent';select.append(empty);}
      for(const row of rows){const el=document.createElement('option');el.value=row.key;el.textContent=row.name;select.append(el);}
      if(selected&&!rows.some(r=>r.key===selected)){const el=document.createElement('option');el.value=selected;el.textContent='Unavailable: '+selected;select.append(el);}
      select.value=selected||'';
    }
    async function open(nid=null){if(pending||services.busy()||!services.context().id)return;
      const g=++generation;context={...services.context()};node=nid;info=null;token=null;lock(true);
      try{await services.sync();if(!current(g))return;
        info=await services.call(nid===null?'lstmAuthoring':'lstmEditing',{...context,...(nid===null?{}:{node:nid})});
        if(!current(g))return;token=checkToken(info,context,nid);const v=info.options;
        $('title').textContent=nid===null?'Add LSTM branch':'Edit LSTM constructor and bindings';
        $('create-fields').hidden=nid!==null;$('edit-fields').hidden=nid===null;
        $('apply').textContent=nid===null?'Add LSTM branch':'Apply changes';$('error').textContent='';
        $('label').value=v.label;$('hidden-size').value=v.hidden_size;$('layers').value=v.num_layers;$('projection').value=v.proj_size;
        $('bias').checked=v.bias;$('bidirectional').checked=v.bidirectional;
        if(nid===null){$('steps').value=v.steps;$('input-size').value=v.input_size;$('lengths').checked=v.lengths;}
        else{const rows=info.sources.map(s=>({key:s.key,name:s.name+' · '+(s.shape?'[B, '+s.shape.join(', ')+']':'shape unproven')}));
          for(const r of roles)fillSelect($('source-'+r),rows,v.inputs[r]);
          fillSelect($('source-lengths'),rows,v.lengths?.source,true);fillSelect($('owner'),services.owners(nid),v.share_with,true);}
        lock(false);preview();dialog.showModal();dialog.scrollTop=0;
      }catch(error){if(g===generation)services.toast(error.message,true);}
      finally{if(g===generation)lock(false);}
    }
    form.onsubmit=async event=>{event.preventDefault();if(pending||!token)return;
      const g=generation;let value;try{value=choices();}catch(error){$('error').textContent=error.message;return;}
      lock(true);$('error').textContent='';
      try{await services.sync();if(!current(g))throw Error('Workspace changed; reopen LSTM authoring.');
        const result=await services.call(node===null?'addLSTM':'editLSTM',{...context,...token,options_json:JSON.stringify(value)});
        if(!current(g))return;
        if(result.id!==context.id||result.workspace_id!==context.workspace_id||result.panel_scope!==context.panel_scope||
          typeof result.node!=='string'||(node!==null&&result.node!==node))throw Error('LSTM acknowledgement belongs to another block or workspace.');
        await services.accept(result,()=>current(g));if(!current(g))return;
        const editing=node!==null;dialog.close();services.toast(editing?'LSTM changes applied.':'LSTM branch added. Supply initial states and explicit task heads before training.');
      }catch(error){if(g===generation&&dialog.open)$('error').textContent=error.message;}
      finally{if(g===generation)lock(false);}
    };
    return {open,pending:()=>pending};
  }
  return {integer,options,stateShapes,checkToken,mount};
});
