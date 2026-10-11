"use strict";
(() => {
  let root,api,identity=null,page=null,offset=0,backs=[],generation=0,reading=null,exporting=false;
  const $=id=>root.getElementById(id),key=h=>h?JSON.stringify([h.scope_id,h.sequence]):null;
  const el=(tag,value)=>{const n=root.createElement(tag);n.textContent=value;return n;};
  const label=d=>d.text+(d.next_offset===null?'':' … [partial; inspect full evidence]');
  function controls(){
    if(!api)return;
    for(const id of ['science-recovery-object','science-recovery-query','science-recovery-search','science-recovery-refresh'])
      $(id).disabled=reading!==null||exporting;
    $('science-recovery-prev').disabled=reading!==null||exporting||!backs.length;
    $('science-recovery-next').disabled=reading!==null||exporting||!page||page.next_offset===null;
    $('science-recovery-show').disabled=reading!==null||exporting||!$('science-recovery-object').value;
    $('science-recovery-export').disabled=api.busy()||reading!==null||exporting||!$('science-recovery-object').value;
  }
  async function refresh(){
    const h=api.recoveryHeader(),ticket=++generation,pin=$('science-recovery-object').value||null;
    if(!h||h.total===0){page=null;$('science-recovery').hidden=true;return false;}
    const request={scope_id:h.scope_id,sequence:h.sequence,offset,limit:25,query:$('science-recovery-query').value,pin};
    reading=ticket;page=null;$('science-recovery').hidden=false;controls();
    try{
      const result=await api.call('scienceRecoveryPage',request);
      if(ticket!==generation||key(api.recoveryHeader())!==key(h))return false;
      if(result.version!==1||result.scope!=='experiment-recovery-page'||result.scope_id!==h.scope_id||
        result.sequence!==h.sequence||result.offset!==request.offset||result.limit!==25||
        result.query!==request.query||result.pin!==pin||(result.pinned&&result.pinned.id!==pin)||
        (result.missing_pin!==null&&result.missing_pin!==pin))throw Error('Recovery page does not match its session, sequence and cursor.');
      page=result;const choices=[...result.rows];
      if(result.pinned&&!choices.some(row=>row.id===result.pinned.id))choices.push(result.pinned);
      const select=$('science-recovery-object');select.replaceChildren();
      choices.forEach(row=>{const option=el('option',label(row.run_id)+' · '+label(row.error)+(row.exported?' · exported':''));
        option.value=row.id;select.append(option);});
      if(result.missing_pin!==null){const option=el('option','Unavailable selected recovery evidence');option.value=result.missing_pin;option.disabled=true;select.append(option);}
      if(pin&&[...select.options].some(o=>o.value===pin))select.value=pin;
      else if(choices.length)select.value=choices[0].id;else select.value='';
      $('science-recovery-page').textContent=result.offset+'–'+(result.offset+result.rows.length)+' of '+result.total+
        (result.pinned&&!result.rows.some(row=>row.id===pin)?' · selected evidence pinned outside page':'')+' · session evidence, not durable jobs';
      return true;
    }catch(error){if(ticket===generation&&key(api.recoveryHeader())===key(h)){
      $('science-recovery-page').textContent=error.message+' · Refresh to retry';api.toast(error.message,true);}return false;
    }finally{if(reading===ticket)reading=null;controls();}
  }
  const act=fn=>Promise.resolve().then(fn).catch(error=>api.toast(error.message,true));
  function navigation(forward){return async()=>{if(reading!==null||exporting)return;
    const owner=identity,old=offset,next=forward?page?.next_offset:backs.at(-1);if(next===null||next===undefined)return;
    offset=next;const ok=await refresh();if(identity!==owner)return;
    if(ok){if(forward)backs.push(old);else backs.pop();}else offset=old;controls();
  };}
  window.LoomScienceRecoveries={
    mount(document,services){root=document;api=services;identity=key(api.recoveryHeader());
      $('science-recovery-prev').onclick=()=>act(navigation(false));$('science-recovery-next').onclick=()=>act(navigation(true));
      $('science-recovery-refresh').onclick=()=>act(()=>reading===null&&!exporting?refresh():undefined);
      $('science-recovery-search').onclick=()=>act(async()=>{if(reading!==null||exporting)return;const owner=identity,old=offset,history=backs;offset=0;backs=[];
        const ok=await refresh();if(identity!==owner)return;if(!ok){offset=old;backs=history;}controls();});
      $('science-recovery-query').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();$('science-recovery-search').onclick();}};
      $('science-recovery-object').onchange=()=>window.LoomScienceDocument.clearRecovery();
      $('science-recovery-show').onclick=()=>{if(reading!==null||exporting||!page)return;const h=api.recoveryHeader(),id=$('science-recovery-object').value;
        if(![...page.rows,page.pinned].some(row=>row?.id===id)||key(h)!==identity)return;
        $('science-document').dataset.projection='recovery-summary';$('science-document').textContent=JSON.stringify({scope:'session-recovery-summary',scope_id:h.scope_id,sequence:h.sequence,id},null,2);
        $('science-details').open=true;window.LoomScienceDocument.selectRecovery({scope_id:h.scope_id,sequence:h.sequence,id});};
      $('science-recovery-export').onclick=()=>act(async()=>{if(api.busy()||reading!==null||exporting||!page)return;
        const h=api.recoveryHeader(),id=$('science-recovery-object').value;
        if(![...page.rows,page.pinned].some(row=>row?.id===id))return;
        exporting=true;controls();try{const result=await api.call('scienceExportRecovery',{id,scope_id:h.scope_id,sequence:h.sequence});
          if(result)api.toast('Recovery evidence exported: '+result.path);
        }finally{exporting=false;this.stateChanged();controls();}await refresh();
      });controls();if(api.recoveryHeader()?.total)act(refresh);
    },
    stateChanged(){if(!api)return;const h=api.recoveryHeader(),next=key(h);if(next!==identity){identity=next;generation++;reading=null;page=null;offset=0;backs=[];
      window.LoomScienceDocument.clearRecovery();$('science-recovery').hidden=!h?.total;if(h?.total)act(refresh);}controls();},
    refresh
  };
})();
