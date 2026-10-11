"use strict";
(() => {
  const key=value=>value===null?'':JSON.stringify(window.LoomRevisionChoice.exact(value)||value);
  function attach(root,select,services,{collection,getValue,valid}){
    const el=(tag,text)=>{const n=root.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
    const box=el('div');box.className='revision-choice';box.dataset.localChoice=collection;
    const query=el('input');query.type='search';query.maxLength=200;query.placeholder='Search selected '+collection;
    query.setAttribute('aria-label','Search selected '+collection);
    const search=el('button','Search'),prev=el('button','‹'),next=el('button','›'),status=el('span');
    prev.setAttribute('aria-label','Previous selected '+collection);next.setAttribute('aria-label','Next selected '+collection);
    status.className='tiny';status.setAttribute('role','status');
    for(const [n,action] of [[query,'query'],[search,'search'],[prev,'prev'],[next,'next']]){
      n.dataset.authorControl='';n.dataset.choiceAction=action;if(n!==query)n.type='button';box.append(n);
    }box.append(status);
    const label=select.closest('label'),field=label?el('div'):null;
    if(field){field.className='revision-choice-field';label.before(field);field.append(label,box);}else select.after(box);
    let disposed=false,reading=null,generation=0,offset=0,backs=[],page=null;
    const workspace=services.workspace();
    const current=()=>!disposed&&select.isConnected!==false&&services.workspace()===workspace&&valid();
    const optionKey=value=>collection==='data_tasks'&&typeof value==='string'?value:key(value);
    function controls(){const stop=!current()||services.busy()||reading!==null;
      for(const [n,extra] of [[select,false],[query,false],[search,false],[prev,!backs.length],[next,!page||page.next_offset===null]]){
        n.dataset.orderDisabled=String(stop||extra);n.disabled=stop||extra;
      }services.updateControls?.();
    }
    function populate(result,chosen,loading=false){
      const empty=el('option','Choose explicitly…');empty.value='';select.replaceChildren(empty);
      const rows=[...result.rows];if(result.pinned&&!rows.some(r=>key(r.value)===key(result.pinned.value)))rows.push(result.pinned);
      for(const row of rows){const option=el('option',row.label+(row.valid?'':' · incomplete'));
        option.value=row.valid?optionKey(row.value):'incomplete/'+row.ordinal;option.disabled=!row.valid;select.append(option);
      }
      if(chosen!==null&&!rows.some(r=>r.valid&&key(r.value)===key(chosen))){const option=el('option',loading?'Selected identity · reading…':'Unavailable selected identity · choose another explicitly');
        option.value=optionKey(chosen);select.append(option);
      }select.value=optionKey(chosen);select.title=chosen===null?'No valid selection.':key(chosen);
    }
    async function refresh(){
      if(!current())return;const ticket=++generation,chosen=getValue(),ref=window.LoomRevisionChoice.exact(chosen),pin=collection==='models'?(ref?.kind==='model'?ref:null):
        typeof chosen==='string'&&chosen.length>0?chosen:null;
      const a={...services.arguments(),collection,offset,limit:25,query:query.value,pin};
      reading=ticket;page=null;populate({rows:[],pinned:null},chosen,true);status.textContent='Reading local selections…';controls();
      try{const result=await services.call('scienceDraftExperimentChoices',a);
        if(ticket!==generation||!current()||services.arguments().sequence!==a.sequence||key(getValue())!==key(chosen))return;
        if(result.scope!=='owned-experiment-choices'||result.version!==1||result.session!==a.session||result.sequence!==a.sequence||
          result.workspace_id!==services.workspace()||result.id!==services.id()||result.collection!==collection||result.offset!==a.offset||
          result.limit!==a.limit||result.query!==a.query||key(result.pin)!==key(pin)||
          (result.pinned&&key(result.pinned.value)!==key(pin))||(result.missing_pin!==null&&key(result.missing_pin)!==key(pin))||
          (result.pinned&&result.missing_pin!==null)||result.rows.some(r=>typeof r.valid!=='boolean'||
            (r.valid&&(collection==='models'?window.LoomRevisionChoice.exact(r.value)?.kind!=='model':typeof r.value!=='string'||r.value.length===0))))
          throw Error('Local choices do not match their exact owner, sequence, cursor and pin.');
        page=result;populate(result,chosen);status.textContent=result.offset+'–'+(result.offset+result.rows.length)+' of '+result.total+
          (result.missing_pin!==null?' · selected identity unavailable':result.ambiguous_pin?' · duplicate selected identity':'')+' · browsing only';
      }catch(error){if(ticket===generation&&current()&&services.arguments().sequence===a.sequence&&key(getValue())===key(chosen)){
        status.textContent=error.message;services.toast(error.message,true);}}
      finally{if(reading===ticket)reading=null;controls();}
    }
    function action(fn){if(!current()||reading!==null||services.busy())return Promise.resolve();return services.run(async()=>{if(!current()||reading!==null)return;await fn();controls();});}
    search.onclick=()=>action(async()=>{const old=offset,oldBacks=backs;offset=0;backs=[];await refresh();if(!page){offset=old;backs=oldBacks;}});
    query.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();search.onclick();}};
    prev.onclick=()=>action(async()=>{if(!backs.length)return;const old=offset;offset=backs.at(-1);await refresh();if(page)backs.pop();else offset=old;});
    next.onclick=()=>action(async()=>{if(!page||page.next_offset===null)return;const old=offset;offset=page.next_offset;await refresh();if(page)backs.push(old);else offset=old;});
    controls();return {refresh,controls,dispose(){disposed=true;generation++;reading=null;if(field){field.before(label);field.remove();}else box.remove();}};
  }
  window.LoomExperimentLocalChoice=Object.freeze({attach});
})();
