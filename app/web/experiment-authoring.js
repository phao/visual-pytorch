/* Experiment factories/parsers mirror the closed v1 grammar, not a second store. */
(function(root,factory){const value=factory();if(typeof module==='object'&&module.exports)module.exports=value;else root.LoomExperimentAuthoring=value;})(typeof globalThis==='object'?globalThis:this,()=>{
  'use strict';
  const methods=Object.freeze(['external','group','row','temporal']);
  function number(value,{integer=false,minimum=-Infinity,maximum=Infinity,optional=false}={}){
    if(typeof value!=='string')throw Error('Enter an explicit number.');
    if(value.trim()===''){if(optional)return null;throw Error('Enter an explicit number.');}
    const result=Number(value);
    if(!Number.isFinite(result)||(integer&&!Number.isSafeInteger(result))||result<minimum||result>maximum)throw Error('Number is outside the declared finite/exact range.');
    return result;
  }
  function seeds(value){
    if(typeof value!=='string')throw Error('Initialization seeds require text.');
    return value.trim()===''?[]:value.trim().split(/[,\s]+/).map(v=>number(v,{integer:true,minimum:0,maximum:2**31-1}));
  }
  function ids(value){
    if(typeof value!=='string')throw Error('Observation IDs require text.');
    // One literal ID per line; commas/spaces are legal ID characters. Never trim,
    // sort or deduplicate: the authoritative validator rejects duplicate roles.
    return value===''?[]:value.split(/\r?\n/).filter(v=>v!=='');
  }
  function reference(kind){return {kind,id:'',revision:''};}
  function pairing(id){return {id,dataset:reference('dataset'),task:reference('task')};}
  function split(id,method){
    if(!methods.includes(method))throw Error('Unsupported split method.');
    const protocol=method==='external'?{method,roles:{train:[],validation:[],test:[]},excluded:[]}:
      {method,fractions:[.6,.2,.2],...(method==='row'||method==='group'?{seed:3}:{}),...(method==='group'||method==='temporal'?{field:''}:{})};
    return {id,protocol};
  }
  function protocol(id){
    return {id,protocol:{version:1,backend:'cpu-float32-v1',batch_size:16,
      budget:{epochs:20,updates:null,examples:null,seconds:null},
      optimizer:{kind:'Adam',lr:.03,betas:[.9,.999],eps:1e-8,weight_decay:0},
      seeds:{initialization:7,shuffle:11},shuffle:true,evaluate_every:10,augmentation:'none',
      history:{kind:'latest-window',max_points:200},checkpoint_selection:'last-successful-update',test_evaluation:'final-completed-only'}};
  }
  function comparison(excluded=false){return {model:reference('model'),data_task:'',...(excluded?{reason:''}:{})};}
  return Object.freeze({methods,number,seeds,ids,reference,pairing,split,protocol,comparison});
});
