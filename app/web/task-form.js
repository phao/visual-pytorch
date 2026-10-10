"use strict";
(() => {
  const kinds=()=>window.LoomTaskAuthoring;
  function render(root,document,services){
    const {el,input,flush,rerender,changed}=services,container=root.getElementById('science-task-bindings');container.replaceChildren();
    function button(name,action){const control=el('button',name);control.dataset.authorControl='';control.onclick=()=>{if(flush())action();};return control;}
    function select(label,choices,value,change){
      const wrapper=el('label',label),control=el('select');control.dataset.authorControl='';
      choices.forEach(([id,name])=>{const option=el('option',name);option.value=id;control.append(option);});control.value=value;
      control.onchange=()=>{if(!flush()){control.value=value;return;}change(control.value);rerender();};wrapper.append(control);return wrapper;
    }
    function transforms(card,binding){
      const box=el('details');box.open=binding.transforms.length>0;box.append(el('summary','Ordered transforms · '+binding.transforms.length));
      binding.transforms.forEach((step,index)=>{
        const row=el('div');row.className='science-author-grid science-task-transform';row.dataset.transformIndex=index;
        row.append(el('span',(index+1)+'. '+step.kind));
        for(const key of step.kind==='affine'?['scale','offset']:step.kind==='pad'?['length','value']:[]){
          row.append(input(key,String(step[key]),value=>{step[key]=kinds().number(value,{integer:key==='length',positive:key==='length'});}));
        }
        const up=button('↑',()=>{binding.transforms=kinds().move(binding.transforms,index,-1);rerender();});up.dataset.orderDisabled=index===0?'true':'false';up.disabled=index===0;up.title='Move transform earlier';
        const down=button('↓',()=>{binding.transforms=kinds().move(binding.transforms,index,1);rerender();});down.dataset.orderDisabled=index===binding.transforms.length-1?'true':'false';down.disabled=index===binding.transforms.length-1;down.title='Move transform later';
        row.append(up,down,button('Remove transform',()=>{binding.transforms.splice(index,1);rerender();}));box.append(row);
      });
      const add=el('div');add.className='inline-actions';const choice=el('select');choice.dataset.authorControl='';choice.setAttribute('aria-label','Transform to add');
      kinds().kinds.forEach(kind=>{const option=el('option',kind);option.value=kind;choice.append(option);});choice.value='standardize';
      add.append(choice,button('Add transform',()=>{binding.transforms.push(kinds().transform(choice.value));rerender();}));box.append(add);card.append(box);
    }
    function orderButtons(card,items,index){
      const actions=el('div');actions.className='inline-actions';actions.dataset.bindingOrder='';
      for(const [name,delta] of [['↑',-1],['↓',1]]){
        const control=button(name,()=>{const moved=kinds().move(items,index,delta);items.splice(0,items.length,...moved);rerender();});
        const disabled=delta===-1?index===0:index===items.length-1;control.dataset.orderDisabled=disabled?'true':'false';control.disabled=disabled;control.title='Move binding '+(delta===-1?'earlier':'later');actions.append(control);
      }
      actions.append(button('Remove binding',()=>{items.splice(index,1);rerender();}));card.append(actions);
    }
    container.append(el('h4','Named inputs'));
    document.inputs.forEach((binding,index)=>{
      const card=el('fieldset');card.dataset.taskInput=index;card.append(el('legend','Input binding '+(index+1)));
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      grid.append(input('Input port ID',binding.port,value=>{binding.port=value;},{list:'science-task-input-ports'}),input('Source field ID',binding.field,value=>{binding.field=value;},{list:'science-task-fields'}));
      grid.append(select('Derived input',[['length','Sequence length'],['mask','Padding mask'],['none','None · transformed field']],binding.derived||'none',value=>{document.inputs[index]=kinds().derived(binding,value);}));
      if(binding.derived)grid.append(input('Padded length',String(binding.length),value=>{binding.length=kinds().number(value,{integer:true,positive:true});}));
      transforms(card,binding);orderButtons(card,document.inputs,index);container.append(card);
    });
    container.append(button('Add input binding',()=>{document.inputs.push(kinds().input());rerender();}));
    container.append(el('h4','Named output objectives'));
    document.heads.forEach((binding,index)=>{
      const card=el('fieldset');card.dataset.taskHead=index;card.append(el('legend','Loss head '+(index+1)));
      const grid=el('div');grid.className='science-author-grid';card.append(grid);
      grid.append(input('Head ID',binding.id,value=>{binding.id=value;}),input('Output port ID',binding.port,value=>{binding.port=value;},{list:'science-task-output-ports'}),input('Target field ID',binding.field,value=>{binding.field=value;},{list:'science-task-fields'}));
      grid.append(select('Loss',[['cross-entropy','Cross-entropy'],['mae','MAE'],['mse','MSE']],binding.loss,value=>{binding.loss=value;}));
      grid.append(input('Objective weight',String(binding.weight),value=>{binding.weight=kinds().number(value,{positive:true});},{title:'Positive multiplier for this head, separate from observation weights.'}));
      for(const [key,label] of [['mask','Mask field ID (optional)'],['sample_weight','Sample-weight field ID (optional)']])grid.append(input(label,binding[key]||'',value=>{if(value==='')delete binding[key];else binding[key]=value;},{list:'science-task-fields'}));
      transforms(card,binding);orderButtons(card,document.heads,index);container.append(card);
    });
    container.append(button('Add loss head',()=>{document.heads.push(kinds().head(crypto.randomUUID()));rerender();}));
    container.append(input('Task description',typeof document.description==='string'?document.description:document.description===undefined?'':JSON.stringify(document.description),value=>{if(value==='')delete document.description;else document.description=value;},{area:true}));
    changed();
  }
  function context(root,value){
    const create=(tag,text)=>{const el=root.createElement(tag);el.textContent=text;return el;};
    for(const [id,rows] of [['science-task-input-ports',value?.input_ports||[]],['science-task-output-ports',value?.output_ports||[]],['science-task-fields',value?.fields||[]]]){
      const list=root.getElementById(id);list.replaceChildren();
      rows.forEach(row=>{const option=create('option',(row.name||row.id)+' · '+(row.dtype||'float32')+' ['+row.shape.join(', ')+']');option.value=row.id;list.append(option);});
    }
    const status=root.getElementById('science-task-context-status');
    status.textContent=value?'Exact context · '+value.record_count+' observations · inputs '+value.input_ports.map(p=>p.id+' ['+p.shape.join(', ')+']').join('; ')+' · outputs '+value.output_ports.map(p=>p.id+' ['+p.shape.join(', ')+']').join('; '):'Freeze an architecture and create a dataset for suggestions. Manual stable IDs still allow reusable task authoring.';
  }
  window.LoomTaskForm=Object.freeze({render,context});
})();
