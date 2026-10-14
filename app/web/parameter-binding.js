(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.LoomParameterBinding=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function compatible(owner,row){return row.dtype==='float32'&&JSON.stringify(owner.shape)===JSON.stringify(row.shape);}
  function selection(page,row){
    if(!page?.run||!Number.isSafeInteger(page.attempt)||page.attempt<1||!/^[a-f0-9]{64}$/.test(page.checkpoint_id))throw Error('Choose an exact retained checkpoint.');
    if(!page.parameters.some(p=>p.path===row.path&&p.id===row.id))throw Error('Choose an exact tensor from this checkpoint page.');
    return {run:page.run,attempt:page.attempt,checkpoint_id:page.checkpoint_id,parameter:{path:row.path,id:row.id}};
  }
  function attach(container,owner,services){
    const document=container.ownerDocument,create=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};
    const details=create('details');details.dataset.parameterSource='true';details.append(create('summary','Retained checkpoint tensor'));
    function field(text,input){const label=create('label',text);label.className='inspect-label';label.append(input);details.append(label);return input;}
    const run=field('Exact run revision',create('select'));run.dataset.bindingRun='true';
    const attempt=field('Explicit attempt number',create('input'));attempt.type='number';attempt.min='1';attempt.step='1';attempt.placeholder='Choose explicitly';attempt.dataset.bindingAttempt='true';
    const load=create('button','Read checkpoint tensors');load.type='button';load.dataset.bindingRead='true';details.append(load);
    const tensor=field('Exact source tensor',create('select'));tensor.dataset.bindingTensor='true';
    const prev=create('button','‹ Tensors'),next=create('button','Tensors ›'),bind=create('button','Bind initial values');
    for(const button of [prev,next,bind]){button.type='button';details.append(button);}bind.dataset.bindingApply='true';
    const status=create('p','');status.className='inspect-note';status.setAttribute('role','status');details.append(status);
    const error=create('p','');error.className='policy-error';error.setAttribute('role','alert');details.append(error);container.append(details);
    let page=null,reading=false,generation=0,backs=[],offset=0,started=false;
    const valid=()=>details.isConnected&&services.valid();
    function controls(){
      const blocked=!valid()||reading||services.busy();attempt.disabled=load.disabled=blocked;
      tensor.disabled=blocked||!page;bind.disabled=blocked||!page||!tensor.value;
      prev.disabled=blocked||!backs.length;next.disabled=blocked||page?.next_offset==null;
    }
    function clear(){generation++;page=null;tensor.replaceChildren(create('option','Choose explicitly…'));tensor.value='';backs=[];offset=0;error.textContent='';controls();}
    const choice=globalThis.LoomRevisionChoice.attach(document,run,services,{kind:'run',getReference:()=>run.value?JSON.parse(run.value):null,valid});
    run.onchange=()=>{clear();attempt.value='';};attempt.oninput=clear;tensor.onchange=controls;
    async function readPage(cursor){
      if(!valid()||reading||services.busy())return;
      const number=Number(attempt.value),ref=run.value?JSON.parse(run.value):null;
      if(!ref||attempt.value.trim()===''||!Number.isSafeInteger(number)||number<1){error.textContent='Choose an exact run and explicit positive attempt number.';return;}
      const ticket=++generation;reading=true;page=null;tensor.replaceChildren();error.textContent='';controls();
      try{
        const result=await services.inventory(ref,number,cursor);
        if(!valid()||ticket!==generation)return;
        if(result.scope!=='retained-parameter-page'||result.attempt!==number||result.offset!==cursor||JSON.stringify(result.run)!==JSON.stringify(ref))throw Error('Checkpoint page does not match its exact request.');
        page=result;offset=cursor;tensor.replaceChildren(create('option','Choose explicitly…'));
        result.parameters.forEach((row,index)=>{const option=create('option',row.path+' ['+row.shape.join(', ')+']'+(compatible(owner,row)?'':' · incompatible shape'));option.value=String(index);option.disabled=!compatible(owner,row);option.title=row.id;tensor.append(option);});
        tensor.value='';status.textContent=cursor+'–'+(cursor+result.parameters.length)+' of '+result.total+' · '+result.status+' · '+result.completed_updates+' updates · '+result.checkpoint_id.slice(0,12);
      }catch(e){if(valid()&&ticket===generation)error.textContent=e.message;}
      finally{reading=false;controls();}
    }
    load.onclick=()=>{backs=[];readPage(0);};prev.onclick=()=>{if(backs.length)readPage(backs.pop());};
    next.onclick=()=>{if(page?.next_offset!=null){backs.push(offset);readPage(page.next_offset);}};
    bind.onclick=async()=>{
      if(!valid()||services.busy()||!page||!tensor.value)return;
      try{const row=page.parameters[Number(tensor.value)];if(!row||!compatible(owner,row))throw Error('Source and destination shapes must match exactly.');await services.bind(owner.selector,selection(page,row));}
      catch(e){if(valid())error.textContent=e.message;}
    };
    details.ontoggle=()=>{if(details.open&&!started){started=true;choice.refresh();}};
    clear();return {dispose(){generation++;choice.dispose();details.remove();}};
  }
  return {compatible,selection,attach};
});
