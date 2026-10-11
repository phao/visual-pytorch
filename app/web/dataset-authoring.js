/* Pure typed dataset edits. Field/record order is semantic and never sorted. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomDatasetAuthoring=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  const clone=value=>JSON.parse(JSON.stringify(value));
  const id=value=>typeof value==='string'&&value.length>0&&value.length<=200&&!/[\x00-\x1f]/.test(value);
  function dimensions(value){
    const result=value.trim()===''?[]:value.trim().split(/[\s,]+/).map(Number);
    if(result.length>8||result.some((v,i)=>!Number.isSafeInteger(v)||(v<1&&!(i===0&&v===-1))))throw Error('Shape uses sample dimensions: positive integers, or -1 for the first ragged axis. Leave blank for a scalar.');
    return result;
  }
  function finite(value){
    if(typeof value==='number'&&!Number.isFinite(value))throw Error('Values must be finite.');
    if(Array.isArray(value))value.forEach(finite);
    else if(value&&typeof value==='object')Object.values(value).forEach(finite);
    return value;
  }
  function json(value){
    const result=JSON.parse(value); // Native syntax checking before the key walk.
    const tokens=value.match(/"(?:\\.|[^"\\])*"|[{}\[\],:]|true|false|null|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/g);
    function walk(i){
      if(tokens[i]==='{'){
        const keys=new Set();i++;
        while(tokens[i]!=='}'){
          const key=JSON.parse(tokens[i++]);if(keys.has(key))throw Error('Duplicate JSON metadata key: '+key);keys.add(key);
          i++;i=walk(i);if(tokens[i]===',')i++;
        }return i+1;
      }
      if(tokens[i]==='['){i++;while(tokens[i]!==']'){i=walk(i);if(tokens[i]===',')i++;}return i+1;}
      return i+1;
    }
    walk(0);return finite(result);
  }
  function cell(value,dtype){
    const result=json(value);
    function visit(item){
      if(Array.isArray(item))return item.forEach(visit);
      if(item!==null&&typeof item==='object')throw Error('Tensor cells contain scalars, arrays or null, not objects.');
      if((dtype==='int64'||dtype==='category')&&typeof item==='number'&&!Number.isSafeInteger(item))throw Error('Integer values must be exact JavaScript-safe integers.');
    }
    visit(result);return result;
  }
  function addField(document,field){
    if(!id(field.id)||document.fields.some(f=>f.id===field.id))throw Error('Use a unique, nonempty stable field ID.');
    const result=clone(document);result.fields.push(clone(field));
    result.records.forEach(row=>{row.values={...row.values,[field.id]:null};});return result;
  }
  function removeField(document,fid){
    const result=clone(document);result.fields=result.fields.filter(f=>f.id!==fid);
    result.records.forEach(row=>{delete row.values[fid];});return result;
  }
  function addRecord(document,rid){
    if(!id(rid)||document.records.some(r=>r.id===rid))throw Error('Use a unique, nonempty observation ID.');
    const result=clone(document);result.records.push({id:rid,values:Object.fromEntries(result.fields.map(f=>[f.id,null]))});return result;
  }
  function page(document,rowOffset=0,fieldOffset=0,rowLimit=20,fieldLimit=6){
    for(const value of [rowOffset,fieldOffset])if(!Number.isSafeInteger(value)||value<0)throw Error('Page offsets must be nonnegative integers.');
    if(!Number.isSafeInteger(rowLimit)||rowLimit<1||rowLimit>100||!Number.isSafeInteger(fieldLimit)||fieldLimit<1||fieldLimit>20)throw Error('Page limits are bounded.');
    const fields=document.fields.slice(fieldOffset,fieldOffset+fieldLimit);
    const rows=document.records.slice(rowOffset,rowOffset+rowLimit).map(row=>({id:row.id,values:Object.fromEntries(fields.map(field=>[field.id,row.values[field.id]]))}));
    return clone({rows,fields,rowTotal:document.records.length,fieldTotal:document.fields.length});
  }
  return Object.freeze({dimensions,cell,json,addField,removeField,addRecord,page});
});
