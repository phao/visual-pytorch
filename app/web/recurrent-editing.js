"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomRecurrent=api;
})(typeof globalThis!=="undefined"?globalThis:this,()=>{
  const multiport=n=>['Recurrent','LSTM','GaussianFilter'].includes(n?.type);
  const multiInput=n=>multiport(n)||n?.type==='CausalLagWindow';
  const bindings=n=>n.type==='CausalLagWindow'?['observations','history','lengths'].map(role=>({role,input:role,source:n.inputs[role]})):n.type==='GaussianFilter'?['observations','initial','transition','means','variances','lengths','emission_mask'].filter(role=>n.inputs[role]).map(role=>({role,input:role,source:n.inputs[role]})):n.type==='LSTM'?[
    ...['sequence','hidden','cell'].map(role=>({role:role==='sequence'?'Sequence':'Initial '+role,input:role,source:n.inputs[role]})),
    ...(n.lengths?[{...n.lengths,role:'lengths'}]:[]),
  ]:[
    ...(n.sequence_inputs||[]).map(row=>({...row,role:'Sequence'})),
    ...(n.states||[]).map(row=>({...row,role:'Initial state'})),
    ...['lengths','mask','reset'].filter(role=>n.controls?.[role]).map(role=>({...n.controls[role],role})),
  ];
  const sources=n=>[...new Set(bindings(n).map(row=>row.source))];
  const ports=n=>n.type==='GaussianFilter'?['filtered','final','step_evidence','sequence_evidence']:n.type==='LSTM'?['sequence','hidden','cell']:[...(n.sequence_outputs||[]),...(n.states||[])].map(row=>row.port);
  function replaceSource(n,oldKey,newKey){
    if(typeof newKey!=='string'||!newKey)throw Error('Choose a named source socket.');
    if(['GaussianFilter','CausalLagWindow'].includes(n.type)){for(const role of Object.keys(n.inputs))if(n.inputs[role]===oldKey)n.inputs[role]=newKey;return;}
    if(n.type==='LSTM'){
      for(const role of ['sequence','hidden','cell'])if(n.inputs[role]===oldKey)n.inputs[role]=newKey;
      if(n.lengths?.source===oldKey)n.lengths.source=newKey;
      return;
    }
    for(const row of [...(n.sequence_inputs||[]),...(n.states||[]),...Object.values(n.controls||{})])
      if(row.source===oldKey)row.source=newKey;
  }
  function portTitle(n,output,key){
    if(output){
      if(n.type==='GaussianFilter')return {filtered:'Filtered log probabilities',final:'Final log probabilities',step_evidence:'Per-step log evidence',sequence_evidence:'Conditional sequence log likelihood'}[key];
      if(n.type==='LSTM')return key==='sequence'?'Sequence output [B,T,D·Hout]':key==='hidden'?'Final hidden [B,LD,Hout]':'Final cell [B,LD,Hcell]';
      const sequence=(n.sequence_outputs||[]).find(row=>row.port===key);
      if(sequence)return 'Sequence '+sequence.output+' → '+key;
      const state=(n.states||[]).find(row=>row.port===key);
      return state?'Final state '+state.input+' → '+key:key;
    }
    return bindings(n).filter(row=>row.source===key).map(row=>row.role+(row.input?' '+row.input:'')+
      (row.adapter?' · '+row.adapter:'')).join(' / ')+' ← '+key;
  }
  const editToken=info=>Object.fromEntries(['id','ref','node','root_graph','root_model','transition'].map(key=>[key,info[key]]));
  function editResponse(info,context){
    for(const key of ['id','ref','node','workspace_id','panel_scope'])
      if(info?.[key]!==context[key])throw Error('Recurrent editing response belongs to another block or workspace.');
    for(const key of ['root_graph','root_model','transition'])
      if(typeof info[key]!=='string'||!/^[0-9a-f]{64}$/.test(info[key]))throw Error('Recurrent editing response lacks an exact closure token.');
    return editToken(info);
  }
  return {multiport,multiInput,bindings,sources,ports,replaceSource,portTitle,editToken,editResponse};
});
