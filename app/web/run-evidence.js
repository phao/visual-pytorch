/* Exact report selection and display formatting; no new metric calculations. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomRunEvidence=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  const key=ref=>JSON.stringify({kind:ref.kind,id:ref.id,revision:ref.revision});
  function add(selected,ref){
    if(!ref||ref.kind!=='run'||typeof ref.id!=='string'||!ref.id||!/^[0-9a-f]{64}$/.test(ref.revision))throw Error('Select an exact saved run revision first.');
    const old=selected.find(r=>r.id===ref.id);
    if(old&&key(old)!==key(ref))throw Error('Another revision of this run is already selected. Remove it explicitly before adding this revision; retries are not extra runs.');
    return old?selected.map(r=>({...r})):[...selected.map(r=>({...r})),{kind:'run',id:ref.id,revision:ref.revision}];
  }
  function value(number){return typeof number==='number'&&Number.isFinite(number)?number.toPrecision(6):'Unavailable';}
  function literal(value){return value===null||value===undefined?'Not retained':typeof value==='object'?JSON.stringify(value):String(value);}
  return Object.freeze({key,add,value,literal});
});
