"use strict";
(() => {
  const exact=value=>value?.kind&&value?.id&&/^[a-f0-9]{64}$/.test(value?.revision||'')?
    {kind:value.kind,id:value.id,revision:value.revision}:null;
  const key=value=>value?JSON.stringify(value):'';
  function attach(root,select,services,options){
    const {kind,getReference,defaultHead=false}=options;
    const workspace=services.workspace(),create=(tag,value)=>{const e=root.createElement(tag);if(value!==undefined)e.textContent=value;return e;};
    const box=create('div');box.className='revision-choice';box.dataset.revisionChoice=kind;
    const query=create('input');query.type='search';query.maxLength=200;query.placeholder='Search names / IDs';query.setAttribute('aria-label','Search '+kind+' revisions');
    const search=create('button','Search'),prev=create('button','‹'),next=create('button','›'),status=create('span');
    search.dataset.choiceAction='search';prev.dataset.choiceAction='prev';next.dataset.choiceAction='next';query.dataset.choiceAction='query';
    prev.setAttribute('aria-label','Previous '+kind+' revisions');next.setAttribute('aria-label','Next '+kind+' revisions');status.className='tiny';status.setAttribute('role','status');
    for(const c of [query,search,prev,next]){c.dataset.authorControl='';if(c!==query)c.type='button';box.append(c);}box.append(status);
    // A label must not acquire a second labelable search input/button. Keep the
    // labelled select and its independently labelled tools as sibling groups.
    const label=select.closest('label'),field=label?create('div'):null;
    if(field){field.className='revision-choice-field';label.before(field);field.append(label,box);}else select.after(box);
    let page=null,offset=0,backs=[],reading=null,generation=0,disposed=false,initial=true;
    const valid=()=>!disposed&&services.workspace()===workspace&&select.isConnected!==false&&(!options.valid||options.valid());
    function controls(){
      const blocked=!valid()||services.busy()||reading!==null;
      for(const [c,extra] of [[select,false],[search,false],[query,false],[prev,!backs.length],[next,!page||page.next_offset===null]]){
        c.dataset.orderDisabled=String(blocked||extra);c.disabled=blocked||extra;
      }
      services.updateControls?.();
    }
    function populate(result,chosen,loading=false){
      const empty=create('option','Choose explicitly…');empty.value='';select.replaceChildren(empty);
      const rows=[...result.rows];if(result.pinned&&!rows.some(row=>key(exact(row))===key(exact(result.pinned))))rows.push(result.pinned);
      for(const row of rows){const option=create('option',row.name+' · '+row.revision.slice(0,12)+(row.selected?' · current head':' · historical')+(result.pinned&&key(exact(row))===key(chosen)&&!result.rows.some(r=>key(exact(r))===key(chosen))?' · pinned selection':''));option.value=key(exact(row));select.append(option);}
      if(chosen&&!rows.some(row=>key(exact(row))===key(chosen))){const option=create('option',loading?'Selected exact revision · reading choices…':'Unavailable exact '+kind+' revision · choose another explicitly');option.value=key(chosen);select.append(option);}
      select.value=key(chosen);
      if(!chosen&&initial&&defaultHead&&!query.value&&offset===0){const head=rows.find(row=>row.selected);if(head)select.value=key(exact(head));}
      const selected=select.value?JSON.parse(select.value):null;
      select.title=selected?'Exact '+kind+' revision: '+selected.id+' @ '+selected.revision:'No exact revision selected.';
    }
    async function refresh(){
      if(!valid())return;
      const ticket=++generation,chosen=exact(getReference()),cursor={offset,query:query.value};
      page=null;reading=ticket;populate({rows:[],pinned:null},chosen,true);status.textContent='Reading exact '+kind+' choices…';controls();
      try{
        const result=await services.call('scienceBrowse',{workspace_id:workspace,kind,offset:cursor.offset,limit:25,query:cursor.query,preferred:chosen,allow_missing_pin:true});
        if(ticket!==generation||!valid())return;
        if(result.scope!=='scientific-metadata-page'||result.kind!==kind||result.offset!==cursor.offset||result.query!==cursor.query)
          throw Error('Revision choices do not match their requested kind and cursor.');
        if((result.pinned&&key(exact(result.pinned))!==key(chosen))||
          (result.missing_pin&&key(exact(result.missing_pin))!==key(chosen))||
          (result.pinned&&result.missing_pin)||result.rows.some(row=>row.kind!==kind||!exact(row)))
          throw Error('Revision choices contain an inconsistent exact selection.');
        if(key(exact(getReference()))!==key(chosen))return;
        page=result;populate(result,chosen);initial=false;
        status.textContent=result.offset+'–'+(result.offset+result.rows.length)+' of '+result.total+
          (result.missing_pin?' · selected revision unavailable':result.pinned&&!result.rows.some(r=>key(exact(r))===key(chosen))?' · exact selection pinned':'')+' · no execution';
      }catch(error){if(ticket===generation&&valid()){status.textContent=error.message;services.toast(error.message,true);}}
      finally{if(reading===ticket)reading=null;controls();}
    }
    function action(fn){Promise.resolve().then(()=>{if(reading===null&&valid()&&!services.busy())return fn();}).catch(error=>{if(valid())services.toast(error.message,true);});}
    search.onclick=()=>action(()=>{offset=0;backs=[];return refresh();});
    query.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();search.onclick();}};
    prev.onclick=()=>action(()=>{if(!backs.length)return;offset=backs.pop();return refresh();});
    next.onclick=()=>action(()=>{if(!page||page.next_offset===null)return;backs.push(offset);offset=page.next_offset;return refresh();});
    controls();
    return {refresh,controls,dispose(){disposed=true;generation++;reading=null;if(field){field.before(label);field.remove();}else box.remove();}};
  }
  window.LoomRevisionChoice=Object.freeze({attach,exact,key});
})();
