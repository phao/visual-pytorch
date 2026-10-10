"use strict";
(() => {
  const H=()=>window.LoomRunEvidence;
  function render(root,container,view,open=null){
    container.replaceChildren();container.dataset.runEvidence=view.reference.revision;
    const el=(tag,value)=>{const node=root.createElement(tag);if(value!==undefined)node.textContent=value;return node;};
    const para=value=>container.append(el('p',value));
    const fields=(parent,values)=>{const list=el('dl');list.className='science-evidence-facts';
      values.forEach(([name,value])=>{list.append(el('dt',name),el('dd',H().literal(value)));});parent.append(list);};
    const detail=(name,values)=>{const d=el('details');d.append(el('summary',name));fields(d,values);container.append(d);};
    const table=(parent,caption,headers,rows)=>{
      const scroll=el('div');scroll.className='science-evidence-table';const t=el('table');t.append(el('caption',caption));
      const head=el('thead'),tr=el('tr');headers.forEach(h=>tr.append(el('th',h)));head.append(tr);t.append(head);
      const body=el('tbody');rows.forEach(values=>{const r=el('tr');values.forEach(value=>r.append(el('td',H().literal(value))));body.append(r);});t.append(body);scroll.append(t);parent.append(scroll);
    };
    const a=view.attempt,e=view.execution;
    container.append(el('h3',view.name+' · attempt '+a.number+' of '+view.attempt_count));
    para(a.status+' · '+a.stage+' · exact run '+view.reference.id+' @ '+view.reference.revision.slice(0,12));
    if(open){const b=el('button','Open exact run');b.onclick=()=>open(view.reference);container.append(b);}
    if(a.error)para(a.error.type+': '+a.error.message);
    if(!e)para('No numerical result retained for this attempt. Counts, checkpoints and metrics are not inferred from earlier attempts.');
    else{
      fields(container,[['Stop reason',e.stop_reason],['Successful updates',e.counts.completed_updates],
        ['Processed / updated examples',e.counts.examples_processed+' / '+e.counts.updated_examples],
        ['Attempted / skipped batches',e.counts.attempted_batches+' / '+e.counts.skipped_batches],
        ['Completed epochs',e.counts.completed_epochs],['Elapsed seconds',e.elapsed_seconds]]);
      if(e.failure)para('Execution diagnostic: '+e.failure);
      if(e.evaluation_error)para('Evaluation diagnostic: '+e.evaluation_error);
      detail('Checkpoint, preprocessing and retention',[
        ['Checkpoint hash',e.checkpoint_id],['Fitted preprocessing hash',e.fitted_state_hash],
        ['Curve observations / discarded',e.history_retention.observed_points+' / '+e.history_retention.discarded_points],
        ['Retention policy',e.history_retention],['Environment',e.environment],['Recorded RAM estimate (not measured peak)',e.memory_admission]]);
    }
    detail('Exact model, dataset, task and selection',Object.entries(view.objects).flatMap(([kind,obj])=>[
      [kind+' name',obj.name],[kind+' identity',obj.reference],
      ...(kind==='model'?[['Trainable parameters',obj.parameters]]:kind==='dataset'?[['Dataset observations',obj.observations]]:[])
    ]).concat([['Experiment/coordinates',view.selection],['Captured split summary',view.split],['Resolved training configuration',view.protocol]]));
    table(container,'Population objectives · available heads only; no weight renormalization',
      ['Population','Status','Objective'],view.populations.map(p=>[p.population,p.status,p.status==='available'?H().value(p.objective):'Unavailable']));
    para('Per-head values are weighted means in transformed target space. Source target units do not imply the physical units of this loss. Accuracy, when present, is weighted. No pooled model ranking or statistical independence is inferred.');
    table(container,'Saved final per-head metrics · values rounded to six significant digits',
      ['Population','Head / loss','Status','Value','Accuracy','Effective weight','Records / unique'],
      view.metrics.rows.map(row=>{const m=row.metric;return [row.population,row.head.id+' / '+m.kind,m.status,
        H().value(m.value),Object.hasOwn(m,'accuracy')?H().value(m.accuracy):'Not applicable',m.denominator,m.evaluated_records+' / '+m.unique_observations];}));
    const ledger=el('details');ledger.append(el('summary','Exact additive ledger, masks and target definitions'));
    view.metrics.rows.forEach(row=>{const box=el('section');box.dataset.metricLedger=row.head.id;box.append(el('h4',row.population+' · '+row.head.id));
      fields(box,[['Head definition (objective weight ≠ sample weight)',row.head],...Object.entries(row.metric)]);ledger.append(box);});container.append(ledger);
    para('Metric rows '+view.metrics.offset+'–'+(view.metrics.offset+view.metrics.rows.length)+' of '+view.metrics.total+
      (view.metrics.next_offset!==null?' · additional rows retained; open/page this exact run.':''));
  }
  window.LoomEvidenceView=Object.freeze({render});
})();
