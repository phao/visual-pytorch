/* Exact bundle admission; no implicit execution or mutable workspace heads. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomNotebookSelection=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  function selection(value){
    if(!value||value.version!==1||value.kind!=='visual-pytorch-notebook-selection'||typeof value.path!=='string'||!value.path||
      !['result-report','rerun'].includes(value.mode)||value.recipe?.kind!=='report'||typeof value.recipe.id!=='string'||!value.recipe.id||
      !/^[0-9a-f]{64}$/.test(value.recipe.revision)||!/^[0-9a-f]{64}$/.test(value.input_index_hash)||!/^[0-9a-f]{64}$/.test(value.requested_sha256))
      throw Error('Inspect an exact notebook bundle before execution.');
    return {version:1,kind:value.kind,path:value.path,mode:value.mode,recipe:{kind:'report',id:value.recipe.id,revision:value.recipe.revision},
      input_index_hash:value.input_index_hash,requested_sha256:value.requested_sha256};
  }
  function request(value,workspace,python,timeout){
    if(typeof workspace!=='string'||!workspace)throw Error('Open a workspace before executing a notebook.');
    if(typeof python!=='string'||!python.trim())throw Error('Choose the report/kernel Python.');
    if(!Number.isSafeInteger(timeout)||timeout<1)throw Error('Cell timeout must be a positive integer.');
    return {selection:selection(value),workspace_id:workspace,python:python.trim(),timeout};
  }
  function behavior(mode){return mode==='rerun'?'Rerun: execution launches the captured fresh CPU fits.':'Saved result: execution reads saved evidence, with no training.';}
  return Object.freeze({selection,request,behavior});
});
