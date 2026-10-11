"use strict";
(() => {
  let choices=[],revisionServices=null;
  const H=()=>window.LoomExperimentAuthoring;
  const key=ref=>ref?.id&&ref?.revision?JSON.stringify({kind:ref.kind,id:ref.id,revision:ref.revision}):'';
  function options(root,select,values,chosen){
    select.replaceChildren();const empty=root.createElement('option');empty.value='';empty.textContent='Choose explicitly…';select.append(empty);
    for(const [value,name] of values){const option=root.createElement('option');option.value=value;option.textContent=name;select.append(option);}
    if(chosen&&![...select.options].some(o=>o.value===chosen)){const missing=root.createElement('option');missing.value=chosen;missing.textContent='Unavailable selected identity · choose another explicitly';select.append(missing);}
    select.value=chosen;
  }
  function refresh(root,document){
    const reads=[];
    root.getElementById('science-experiment-bindings').querySelectorAll('select[data-reference-kind]').forEach(select=>{
      const kind=select.dataset.referenceKind;
      if(select.dataset.selectedModels==='true'){
        const values=document.models.filter(ref=>key(ref)).map(ref=>[key(ref),ref.id+' · '+ref.revision.slice(0,12)+' · selected architecture']);
        options(root,select,values,key(select.getReference()));
      }else{
        if(!select.revisionChoice){select.revisionChoice=window.LoomRevisionChoice.attach(root,select,revisionServices,
          {kind,getReference:select.getReference,valid:revisionServices.valid});choices.push(select.revisionChoice);}
        reads.push(select.revisionChoice.refresh());
      }
    });
    root.getElementById('science-experiment-bindings').querySelectorAll('select[data-data-task-choice]').forEach(select=>
      options(root,select,document.data_tasks.map(p=>[p.id,p.id]),select.getPairing()));
    root.getElementById('science-experiment-context-status').textContent='Stored choices use independent searchable pages · exact selected identities never advance on refresh';
    return Promise.all(reads);
  }
  function render(root,document,services){
    choices.forEach(choice=>choice.dispose());choices=[];revisionServices=services.revisionServices;
    const {el,input,flush,rerender,changed}=services,container=root.getElementById('science-experiment-bindings');container.replaceChildren();
    function button(name,action){const control=el('button',name);control.type='button';control.dataset.authorControl='';control.onclick=()=>{if(flush())action();};return control;}
    function select(label,values,chosen,change){
      const wrapper=el('label',label),control=el('select');control.dataset.authorControl='';
      values.forEach(([value,name])=>{const option=el('option',name);option.value=value;control.append(option);});control.value=chosen;
      control.onchange=()=>{if(!flush()){control.value=chosen;return;}change(control.value);changed();};wrapper.append(control);return wrapper;
    }
    function reference(label,kind,get,set,selectedModels=false){
      const wrapper=el('label',label),control=el('select');control.dataset.authorControl='';control.dataset.referenceKind=kind;
      control.dataset.selectedModels=String(selectedModels);control.getReference=get;
      control.onchange=()=>{if(!flush()){control.value=key(get());return;}set(control.value?JSON.parse(control.value):H().reference(kind));changed();refresh(root,document);};
      wrapper.append(control);return wrapper;
    }
    function pairChoice(get,set){
      const wrapper=el('label','Dataset/task pairing ID'),control=el('select');control.dataset.authorControl='';control.dataset.dataTaskChoice='';control.getPairing=get;
      control.onchange=()=>{if(!flush()){control.value=get();return;}set(control.value);changed();};wrapper.append(control);return wrapper;
    }
    function order(card,items,index){
      const actions=el('div');actions.className='inline-actions';actions.dataset.axisOrder='';
      for(const [name,delta] of [['↑',-1],['↓',1]]){
        const b=button(name,()=>{const moved=window.LoomTaskAuthoring.move(items,index,delta);items.splice(0,items.length,...moved);rerender();});
        b.dataset.orderDisabled=String(delta===-1?index===0:index===items.length-1);b.disabled=b.dataset.orderDisabled==='true';b.title='Move selection '+(delta===-1?'earlier':'later');actions.append(b);
      }
      actions.append(button('Remove entry',()=>{items.splice(index,1);rerender();}));card.append(actions);
    }
    function section(label,count){const details=el('details');details.open=true;details.append(el('summary',label+' · '+count));container.append(details);return details;}
    function card(section,kind,index,label){const result=el('fieldset');result.dataset.experimentAxis=kind;result.dataset.axisIndex=index;result.append(el('legend',label+' '+(index+1)));section.append(result);return result;}
    function grid(card){const result=el('div');result.className='science-author-grid';card.append(result);return result;}
    function numeric(label,object,key,opts={}){return input(label,object[key]===null?'':String(object[key]),value=>{object[key]=H().number(value,opts);},{title:opts.optional?'Blank disables this bound. At least one training bound must remain.':''});}
    const integer={integer:true,minimum:1},seed={integer:true,minimum:0,maximum:2**31-1};
    container.append(input('Research question',document.question,value=>{document.question=value;},{area:true}));
    const models=section('Frozen architecture selections',document.models.length);
    document.models.forEach((model,index)=>{const box=card(models,'model',index,'Architecture selection');box.append(reference('Exact architecture','model',()=>document.models[index],value=>{document.models[index]=value;}));order(box,document.models,index);});
    models.append(button('Add architecture selection',()=>{document.models.push(H().reference('model'));rerender();}));
    const data=section('Dataset/task pairings',document.data_tasks.length);
    document.data_tasks.forEach((pair,index)=>{
      const box=card(data,'data-task',index,'Dataset/task pairing'),g=grid(box);
      g.append(input('Pairing ID',pair.id,value=>{pair.id=value;refresh(root,document);}),reference('Exact dataset','dataset',()=>pair.dataset,value=>{pair.dataset=value;}),reference('Exact task','task',()=>pair.task,value=>{pair.task=value;}));order(box,document.data_tasks,index);
    });
    data.append(button('Add dataset/task pairing',()=>{document.data_tasks.push(H().pairing(crypto.randomUUID()));rerender();}));
    const splits=section('Split selections',document.splits.length);
    splits.append(el('p','A field ID must apply to every relevant dataset. Preview displays incompatible pairs; it does not silently fix their schema.'));
    document.splits.forEach((entry,index)=>{
      const box=card(splits,'split',index,entry.protocol.method+' split'),p=entry.protocol,g=grid(box);
      g.append(input('Split ID',entry.id,value=>{entry.id=value;}));
      if(p.method==='row'||p.method==='group')g.append(numeric('Split seed',p,'seed',seed));
      if(p.method==='group'||p.method==='temporal')g.append(input(p.method==='group'?'Grouping field ID':'Time field ID',p.field,value=>{p.field=value;}));
      if(p.method==='external'){
        for(const role of ['train','validation','test'])g.append(input(role+' observation IDs',p.roles[role].join('\n'),value=>{p.roles[role]=H().ids(value);},{area:true,title:'One literal stable observation ID per line. Spaces/commas are preserved; no automatic sorting or deduplication.'}));
        g.append(input('Excluded observation IDs',p.excluded.join('\n'),value=>{p.excluded=H().ids(value);},{area:true}));
      }else for(const [i,label] of ['Train fraction','Validation fraction','Test fraction'].entries())g.append(input(label,String(p.fractions[i]),value=>{p.fractions[i]=H().number(value,{minimum:0,maximum:1});}));
      box.append(el('p','Method is explicit for this entry. To change it, remove/add a split and specify the intended stable ID. Existing references are not rewritten.'));order(box,document.splits,index);
    });
    const splitAdd=el('div');splitAdd.className='inline-actions';const method=el('select');method.dataset.authorControl='';method.setAttribute('aria-label','Split method to add');
    H().methods.forEach(kind=>{const option=el('option',kind);option.value=kind;method.append(option);});method.value='row';splitAdd.append(method,button('Add split',()=>{document.splits.push(H().split(crypto.randomUUID(),method.value));rerender();}));splits.append(splitAdd);
    const seeds=section('Independent initialization seeds',document.initialization_seeds.length);
    seeds.append(input('Initialization seeds',document.initialization_seeds.join(', '),value=>{document.initialization_seeds=H().seeds(value);},{area:true,title:'Explicit ordered integers in 0..2^31−1, comma/whitespace separated. Duplicate seeds are rejected, not silently removed.'}));
    const protocols=section('Training configurations',document.protocols.length);
    document.protocols.forEach((entry,index)=>{
      const box=card(protocols,'protocol',index,'Training configuration'),p=entry.protocol,g=grid(box);
      g.append(input('Configuration ID',entry.id,value=>{entry.id=value;}),numeric('Batch size',p,'batch_size',integer),numeric('Evaluate every updates',p,'evaluate_every',integer),numeric('Retained curve points',p.history,'max_points',integer));
      box.append(el('p','CPU float32 · Adam · no augmentation · last successful checkpoint · test evaluation only after completed training.'));
      const budgets=grid(box);
      for(const key of ['epochs','updates','examples','seconds'])budgets.append(numeric('Maximum '+key,p.budget,key,{integer:key!=='seconds',minimum:key==='seconds'?Number.MIN_VALUE:1,optional:true}));
      box.append(el('p','Blank disables a bound; at least one is required. Execution stops at the first observed active bound, not their sum.'));
      const opt=grid(box);opt.append(numeric('Learning rate',p.optimizer,'lr',{minimum:Number.MIN_VALUE,maximum:1}),numeric('Adam epsilon',p.optimizer,'eps',{minimum:Number.MIN_VALUE}),numeric('Weight decay',p.optimizer,'weight_decay',{minimum:0}));
      for(let i=0;i<2;i++)opt.append(input('Adam beta '+(i+1),String(p.optimizer.betas[i]),value=>{const n=H().number(value,{minimum:0,maximum:1});if(n===1)throw Error('Adam beta must be less than 1.');p.optimizer.betas[i]=n;}));
      const rng=grid(box);rng.append(numeric('Protocol initialization seed (axis overrides)',p.seeds,'initialization',seed),numeric('Independent shuffle seed',p.seeds,'shuffle',seed));
      const label=el('label','Shuffle training observations '),shuffle=el('input');shuffle.type='checkbox';shuffle.dataset.authorControl='';shuffle.checked=p.shuffle;
      shuffle.onchange=()=>{p.shuffle=shuffle.checked;changed();};label.append(shuffle);rng.append(label);order(box,document.protocols,index);
    });
    protocols.append(button('Add training configuration',()=>{document.protocols.push(H().protocol(crypto.randomUUID()));rerender();}));
    const comparisons=section('Comparison pairing and exclusions',document.selection.pairs.length+document.selection.exclusions.length);
    comparisons.append(select('Comparison mode',[['crossed','Crossed · every architecture × dataset/task'],['paired','Paired · explicit whitelist']],document.selection.kind,value=>{document.selection.kind=value;}));
    comparisons.append(el('p','Crossed requires an empty whitelist; paired requires explicit entries. Changing mode never removes entries. Exclusions keep a visible reason; unlisted paired combinations remain visible as excluded.'));
    for(const [name,excluded] of [['pairs',false],['exclusions',true]]){
      comparisons.append(el('h4',excluded?'Explicit exclusions':'Pair whitelist'));
      document.selection[name].forEach((entry,index)=>{
        const box=card(comparisons,name,index,excluded?'Excluded comparison':'Whitelisted comparison'),g=grid(box);
        g.append(reference('Selected exact architecture','model',()=>entry.model,value=>{entry.model=value;},true),pairChoice(()=>entry.data_task,value=>{entry.data_task=value;}));
        if(excluded)box.append(input('Exclusion reason',entry.reason,value=>{entry.reason=value;},{area:true}));order(box,document.selection[name],index);
      });
      comparisons.append(button(excluded?'Add exclusion':'Add pair whitelist entry',()=>{document.selection[name].push(H().comparison(excluded));rerender();}));
    }
    refresh(root,document);changed();
  }
  function clearPreview(root){root.getElementById('science-experiment-preview-rows').replaceChildren();root.getElementById('science-experiment-preview-status').textContent='Draft preview is not current. Check it explicitly; no training.';}
  window.LoomExperimentForm=Object.freeze({render,refresh,clearPreview,reset(){choices.forEach(choice=>choice.dispose());choices=[];revisionServices=null;}});
})();
