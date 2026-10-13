(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.LoomJSON=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  function stringify(value){
    const visiting=new Set();
    function encode(value,array=false){
      if(value===null)return 'null';
      if(typeof value==='number'){if(!Number.isFinite(value))throw TypeError('Scientific JSON numbers must be finite.');return Object.is(value,-0)?'-0.0':JSON.stringify(value);}
      if(typeof value==='string'||typeof value==='boolean')return JSON.stringify(value);
      if(['undefined','function','symbol'].includes(typeof value))return array?'null':undefined;
      if(typeof value!=='object')throw TypeError('Unsupported scientific JSON value.');
      if(visiting.has(value))throw TypeError('Circular scientific JSON value.');visiting.add(value);
      let result;
      if(Array.isArray(value))result='['+Array.from(value,item=>encode(item,true)).join(',')+']';
      else{const rows=[];for(const key of Object.keys(value)){const item=encode(value[key]);if(item!==undefined)rows.push(JSON.stringify(key)+':'+item);}result='{'+rows.join(',')+'}';}
      visiting.delete(value);return result;
    }
    return encode(value);
  }
  const clone=value=>JSON.parse(stringify(value));
  return {stringify,clone};
});
