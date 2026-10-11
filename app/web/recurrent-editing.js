"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomRecurrent=api;
})(typeof globalThis!=="undefined"?globalThis:this,()=>{
  const bindings=n=>[
    ...(n.sequence_inputs||[]).map(row=>({...row,role:'Sequence'})),
    ...(n.states||[]).map(row=>({...row,role:'Initial state'})),
    ...['lengths','mask','reset'].filter(role=>n.controls?.[role]).map(role=>({...n.controls[role],role})),
  ];
  const sources=n=>[...new Set(bindings(n).map(row=>row.source))];
  const ports=n=>[...(n.sequence_outputs||[]),...(n.states||[])].map(row=>row.port);
  function replaceSource(n,oldKey,newKey){
    if(typeof newKey!=='string'||!newKey)throw Error('Choose a named source socket.');
    for(const row of [...(n.sequence_inputs||[]),...(n.states||[]),...Object.values(n.controls||{})])
      if(row.source===oldKey)row.source=newKey;
  }
  function portTitle(n,output,key){
    if(output){
      const sequence=(n.sequence_outputs||[]).find(row=>row.port===key);
      if(sequence)return 'Sequence '+sequence.output+' → '+key;
      const state=(n.states||[]).find(row=>row.port===key);
      return state?'Final state '+state.input+' → '+key:key;
    }
    return bindings(n).filter(row=>row.source===key).map(row=>row.role+(row.input?' '+row.input:'')+
      (row.adapter?' · '+row.adapter:'')).join(' / ')+' ← '+key;
  }
  return {bindings,sources,ports,replaceSource,portTitle};
});
