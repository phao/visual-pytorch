/* Reusable task grammar helpers. Order is computation, not presentation. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomTaskAuthoring=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  const kinds=Object.freeze(['affine','impute-mean','onehot','pad','standardize']);
  const clone=value=>JSON.parse(JSON.stringify(value));
  function number(value,{integer=false,positive=false}={}){
    if(typeof value!=='string'||value.trim()==='')throw Error('Enter an explicit number.');
    const result=Number(value);
    if(!Number.isFinite(result)||(integer&&!Number.isSafeInteger(result))||(positive&&result<=0))throw Error('Enter a finite '+(positive?'positive ':'')+(integer?'exact integer.':'number.'));
    return result;
  }
  function transform(kind){
    if(!kinds.includes(kind))throw Error('Unsupported transform.');
    return kind==='affine'?{kind,scale:1,offset:0}:kind==='pad'?{kind,length:1,value:0}:{kind};
  }
  function input(port='',field=''){return {port,field,transforms:[]};}
  function head(id,port='',field=''){return {id,port,field,loss:'mse',weight:1,transforms:[]};}
  function derived(binding,mode){
    if(!['none','length','mask'].includes(mode))throw Error('Unsupported derived input.');
    const result=clone(binding);
    if(mode==='none'){delete result.derived;delete result.length;}
    else{result.derived=mode;if(!Object.hasOwn(result,'length'))result.length=1;}
    // Never silently remove transforms. The validator explains conflicts.
    return result;
  }
  function move(items,index,delta){
    if(!Array.isArray(items)||!Number.isSafeInteger(index)||index<0||index>=items.length||![-1,1].includes(delta))throw Error('Invalid ordered move.');
    const result=clone(items),target=index+delta;
    if(target>=0&&target<result.length)[result[index],result[target]]=[result[target],result[index]];
    return result;
  }
  return Object.freeze({kinds,number,transform,input,head,derived,move});
});
