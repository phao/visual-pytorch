/* Read exact saved points; SVG coordinates never recompute training metrics. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomEvidenceCurves=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  const exact=(a,b)=>a?.kind===b?.kind&&a?.id===b?.id&&a?.revision===b?.revision;
  const integer=n=>Number.isSafeInteger(n)&&n>=0;
  let serial=0;
  function check(value,request){
    const fail=()=>{throw Error('Saved curve reply does not match the exact attempt/page.');};
    if(!integer(request.attempt)||request.attempt<1||request.reference?.kind!=='run'||
      !/^[0-9a-f]{64}$/.test(request.reference.revision)||value?.version!==1||value.scope!=='read-saved-curve-only'||value.workspace_id!==request.workspace_id||
      !exact(value.reference,request.reference)||value.attempt?.number!==request.attempt||
      value.execution!=='no-fitting-or-training'||value.scale!=='linear'||value.loss_space!=='transformed-target-space'||
      typeof value.head?.id!=='string'||!['mse','mae','cross-entropy'].includes(value.head.loss)||
      (request.head!==null&&value.head.id!==request.head))fail();
    for(const [name,offset,limit] of [['points',request.offset,request.limit],['heads',request.head_offset,request.head_limit]]){
      const p=value[name];
      if(!p||p.offset!==offset||p.limit!==limit||!integer(p.total)||!Array.isArray(p.rows)||p.rows.length>limit||
        p.rows.length!==Math.max(0,Math.min(limit,p.total-offset))||
        p.next_offset!==(offset+p.rows.length<p.total?offset+p.rows.length:null))fail();
    }
    if(!Array.isArray(value.x_range)||value.x_range.length!==2||value.x_range[0]!==0||!integer(value.x_range[1])||value.x_range[1]<1||
      !Array.isArray(value.y_range)||value.y_range.length!==2||value.y_range[0]!==0||
      typeof value.y_range[1]!=='number'||!Number.isFinite(value.y_range[1])||value.y_range[1]<=0)fail();
    if(value.heads.rows.some(h=>typeof h.id!=='string'||!['mse','mae','cross-entropy'].includes(h.loss))||
      new Set(value.heads.rows.map(h=>h.id)).size!==value.heads.rows.length||
      (request.head===null&&request.head_offset===0&&value.head.id!==value.heads.rows[0]?.id))fail();
    let previous=-1;
    value.points.rows.forEach((p,i)=>{
      if(p.ordinal!==request.offset+i||!integer(p.update)||p.update<previous||p.update>value.x_range[1]||
        !/^[0-9a-f]{64}$/.test(p.checkpoint_id))fail();previous=p.update;
      for(const role of ['train','validation']){
        const m=p.metrics?.[role];
        if(!m||m.population!==role||m.kind!==value.head.loss||m.checkpoint_id!==p.checkpoint_id||
          !['available','unavailable'].includes(m.status)||
          (m.value!==null&&(typeof m.value!=='number'||!Number.isFinite(m.value)||m.value<0||m.value>value.y_range[1]))||
          (m.status==='available')!==(m.value!==null))fail();
      }
    });
    return value;
  }
  function geometry(page){
    const series={train:[],validation:[]};
    for(const role of Object.keys(series)){
      let segment=[];
      for(const point of page.points.rows){
        const y=point.metrics[role].value;
        if(y===null){if(segment.length)series[role].push(segment);segment=[];continue;}
        segment.push({x:80+650*(point.update/page.x_range[1]),y:244-210*(y/page.y_range[1]),point});
      }
      if(segment.length)series[role].push(segment);
    }
    return series;
  }
  function draw(document,parent,page){
    const namespace='http://www.w3.org/2000/svg',id='saved-curve-'+(++serial);
    const node=(tag,attributes={},text)=>{const n=document.createElementNS(namespace,tag);for(const [k,v] of Object.entries(attributes))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=text;return n;};
    const svg=node('svg',{viewBox:'0 0 760 320',width:760,height:320,role:'img','aria-labelledby':id+'-title '+id+'-description'});
    svg.append(node('title',{id:id+'-title'},page.head.id+' · weighted mean '+page.head.loss),
      node('desc',{id:id+'-description'},'Exact saved attempt '+page.attempt.number+'. Train solid circles, validation dashed squares. Linear axes. Missing values break lines. Current retained page only; no smoothing or uncertainty interval.'));
    const format=n=>n===0?'0':n.toPrecision(4);
    for(const f of [0,.25,.5,.75,1]){
      const y=244-210*f;svg.append(node('line',{x1:80,y1:y,x2:730,y2:y,class:'science-curve-grid'}),node('text',{x:72,y:y+5,'text-anchor':'end'},format(page.y_range[1]*f)));
    }
    for(const x of [...new Set([0,.25,.5,.75,1].map(f=>Math.round(page.x_range[1]*f)))])svg.append(node('text',{x:80+650*(x/page.x_range[1]),y:265,'text-anchor':x===0?'start':x===page.x_range[1]?'end':'middle'},String(x)));
    svg.append(node('path',{d:'M80 34 V244 H730',class:'science-curve-axis',fill:'none'}),
      node('text',{x:405,y:291,'text-anchor':'middle'},'Successful optimizer updates · 0 = initial checkpoint'),
      node('text',{x:16,y:140,transform:'rotate(-90 16 140)','text-anchor':'middle'},'Weighted mean '+page.head.loss+' · linear'));
    for(const [role,segments] of Object.entries(geometry(page)))for(const segment of segments){
      svg.append(node('polyline',{points:segment.map(p=>p.x+','+p.y).join(' '),fill:'none','stroke-width':2,class:'science-curve-'+role,...(role==='validation'?{'stroke-dasharray':'6 4'}:{})}));
      for(const {x,y,point} of segment){
        const marker=role==='train'?node('circle',{cx:x,cy:y,r:3,class:'science-curve-train'}):node('rect',{x:x-3,y:y-3,width:6,height:6,class:'science-curve-validation'});
        marker.append(node('title',{},role+' · update '+point.update+' · '+point.metrics[role].value));svg.append(marker);
      }
    }
    parent.append(svg);
  }
  function attach(document,container,view,api){
    const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
    const details=el('details');details.className='science-curve';details.dataset.curveRun=view.reference.id;
    details.append(el('summary','Saved training curves · attempt '+view.attempt.number));container.append(details);
    const actions=el('div');actions.className='inline-actions';details.append(actions);
    const label=el('label','Output head'),choice=el('select');label.append(choice);actions.append(label);
    const button=text=>{const b=el('button',text);b.type='button';actions.append(b);return b;};
    const previousHead=button('‹ Heads'),nextHead=button('Heads ›'),readButton=button('Read / refresh curve'),previous=button('‹ Points'),next=button('Points ›');
    const status=el('p','Open to read this exact saved attempt; never trains.');status.setAttribute('role','status');status.className='tiny';details.append(status);
    const body=el('div');details.append(body);
    let page=null,pending=false,ticket=0,identity=api.workspace(),back=[],headBack=[];
    let acknowledged={head:null,offset:0,limit:100,head_offset:0,head_limit:25};
    const alive=()=>details.isConnected&&identity===api.workspace();
    function controls(){
      choice.disabled=readButton.disabled=pending;
      previous.disabled=pending||!back.length;next.disabled=pending||!page||page.points.next_offset===null;
      previousHead.disabled=pending||!headBack.length;nextHead.disabled=pending||!page||page.heads.next_offset===null;
    }
    function render(){
      choice.replaceChildren();
      const heads=page.heads.rows.some(h=>h.id===page.head.id)?page.heads.rows:[...page.heads.rows,{id:page.head.id,loss:page.head.loss,pin:true}];
      for(const h of heads){const o=el('option',h.id+' · '+h.loss+(h.pin?' · selected outside this page':''));o.value=h.id;choice.append(o);}choice.value=page.head.id;
      body.replaceChildren();
      const caption=el('p','Exact '+page.reference.id+' @ '+page.reference.revision+' · attempt '+page.attempt.number+' ('+page.attempt.status+') · head '+page.head.id+'.');caption.className='tiny';body.append(caption);
      const retention=page.retention;
      body.append(el('p','Retained points '+page.points.offset+'–'+(page.points.offset+page.points.rows.length)+' of '+page.points.total+
        '; head choices '+page.heads.offset+'–'+(page.heads.offset+page.heads.rows.length)+' of '+page.heads.total+'. '+
        (retention?'Observed '+retention.observed_points+'; discarded '+retention.discarded_points+' by '+retention.kind+'.':'No numerical result retained for this attempt. Earlier attempts are not substituted.')));
      body.append(el('p','Train: solid / circles. Validation: dashed / squares. Each point is a weighted population mean evaluated at the saved checkpoint, not the combined objective. Test metrics are final-only and not plotted. Linear axes cover all retained points of this head; only the current page is drawn. Missing values break lines; straight connections are visual guides, not extra evaluations. No smoothing or confidence intervals.'));
      body.append(el('p','Source target unit: '+(page.source_unit??'unspecified')+'. Loss uses transformed target space. Target transforms: '+JSON.stringify(page.head.transforms)+'.'));
      if(page.points.rows.some(p=>Object.values(p.metrics).some(m=>m.value!==null))){const scroll=el('div');scroll.className='science-curve-canvas';body.append(scroll);draw(document,scroll,page);}
      else body.append(el('p','No available curve values in this page; missing evidence is not a zero-valued curve.'));
      const tableDetails=el('details');tableDetails.append(el('summary','Exact plotted rows · checkpoints, denominators and counts'));
      const scroll=el('div');scroll.className='science-evidence-table';const table=el('table');table.append(el('caption','Current retained page; no averaging or display sampling.'));
      const tr=el('tr');for(const h of ['Ordinal','Update','Checkpoint','Population','Status','Value','Effective weight','Records','Numerator','Timing / mode','Attempted batches / examples'])tr.append(el('th',h));const header=el('thead');header.append(tr);table.append(header);
      const tbody=el('tbody');
      for(const point of page.points.rows)for(const role of ['train','validation']){const metric=point.metrics[role],row=el('tr');
        for(const v of [point.ordinal,point.update,point.checkpoint_id,role,metric.status,metric.value===null?'Unavailable':metric.value,metric.denominator,metric.evaluated_records,metric.numerator,point.timing+' / '+point.mode,point.attempted_batches+' / '+point.examples_processed])row.append(el('td',String(v)));tbody.append(row);}
      table.append(tbody);scroll.append(table);tableDetails.append(scroll);body.append(tableDetails);
    }
    async function read(changes={},movement=null){
      if(pending||!alive())return;
      const request={workspace_id:identity,reference:{...view.reference},attempt:view.attempt.number,...acknowledged,...changes};
      const current=++ticket;pending=true;status.textContent='Reading exact saved curve…';controls();
      try{
        const value=await api.call('scienceRunCurve',request);
        if(!alive()||current!==ticket)return;
        check(value,request);
        if(movement==='next')back.push(acknowledged.offset);if(movement==='previous')back.pop();
        if(movement==='next-head')headBack.push(acknowledged.head_offset);if(movement==='previous-head')headBack.pop();
        if(movement==='head')back=[];
        page=value;acknowledged={head:value.head.id,offset:value.points.offset,limit:100,head_offset:value.heads.offset,head_limit:25};
        render();status.textContent='Saved evidence only · no fitting, inference or training.';
      }catch(error){if(alive()&&current===ticket){status.textContent=error.message+' Current accepted page is preserved; Read / refresh retries it.';if(page)choice.value=page.head.id;}}
      finally{if(current===ticket){pending=false;controls();}}
    }
    details.ontoggle=()=>{if(details.open&&!page)read();};readButton.onclick=()=>read();
    choice.onchange=()=>read({head:choice.value,offset:0},'head');
    previous.onclick=()=>{if(back.length)read({offset:back.at(-1)},'previous');};next.onclick=()=>{if(page?.points.next_offset!==null&&page)read({offset:page.points.next_offset},'next');};
    previousHead.onclick=()=>{if(headBack.length)read({head_offset:headBack.at(-1)},'previous-head');};nextHead.onclick=()=>{if(page?.heads.next_offset!==null&&page)read({head_offset:page.heads.next_offset},'next-head');};
    controls();return details;
  }
  return Object.freeze({check,geometry,attach});
});
