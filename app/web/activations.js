"use strict";
((root,factory)=>{
  const value=factory();
  if(typeof module==='object'&&module.exports)module.exports=value;
  else root.LoomActivations=value;
})(typeof globalThis==='object'?globalThis:this,()=>{
  const info={
    GELU:['h = z · Φ(z)','Smooth gating using the normal CDF. PyTorch uses the exact, non-approximated mode.'],
    LogSoftmax:['hᵢ = zᵢ − m − log Σⱼ exp(zⱼ − m), m = max(z)','Produces stable log probabilities along the selected sample axis; exp(h) sums to 1 along that axis. Shape and batch B are preserved, with no trainable parameters. Batch B is excluded from normalization.'],
    ReLU:['h = max(0, z)','Keeps positive features; zeros negative ones.'],
    Sigmoid:['h = 1 / (1 + exp(−z))','Bounds each feature between 0 and 1. A sigmoid does not choose a classification loss; the task or training interface chooses the loss.'],
    Softmax:['hᵢ = exp(zᵢ − max(z)) / Σⱼ exp(zⱼ − max(z))','Normalizes each slice along the selected sample axis; its values sum to 1. Shape and batch B are preserved, with no trainable parameters. Cross-entropy expects raw logits: do not put Softmax before a cross-entropy task head.'],
    Softplus:['h = log(1 + exp(z))','Smooth elementwise positive transform with fixed beta = 1 and threshold = 20; PyTorch uses the linear branch above the threshold for stability. No positive floor is added; large negative inputs may round to zero. Shape and batch B are preserved, with no trainable parameters.'],
    Tanh:['h = tanh(z)','Smooth, zero-centered output between −1 and 1.'],
  };
  const tex={GELU:'h=z\\,\\Phi(z)',ReLU:'h=\\max(0,z)',Sigmoid:'h=\\frac{1}{1+e^{-z}}',
    LogSoftmax:'h_i=z_i-m-\\log\\sum_j e^{z_j-m},\\quad m=\\max_j z_j',
    Softmax:'h_i=\\frac{e^{z_i-m}}{\\sum_j e^{z_j-m}},\\quad m=\\max_j z_j',
    Softplus:'h=\\log(1+e^z),\\quad \\beta=1,\\;\\mathrm{threshold}=20',Tanh:'h=\\tanh(z)'};
  function axis(text,rank=null){
    const token=String(text).trim();
    if(!/^-?\d+$/.test(token))throw Error('Enter an integer sample axis; omit batch B.');
    const dim=Number(token);
    if(!Number.isSafeInteger(dim)||dim < -8||dim>7 ||
      (rank!==null&&(!Number.isInteger(rank)||rank<1||rank>8||dim < -rank||dim>=rank)))
      throw Error('Softmax/LogSoftmax axis is outside the sample dimensions; omit batch B.');
    return dim;
  }
  // One list for Building blocks and the Inspector. Softmax/LogSoftmax are activations in the
  // broad sense but normalize a whole slice along an axis, so they form their own group.
  const groups=Object.freeze([
    Object.freeze({label:'Elementwise',names:Object.freeze(['GELU','ReLU','Sigmoid','Softplus','Tanh'])}),
    Object.freeze({label:'Normalizing along an axis',names:Object.freeze(['LogSoftmax','Softmax'])}),
  ]);
  const options=(selected,escape=s=>s)=>groups.map(g=>'<optgroup label="'+escape(g.label)+'">'+
    g.names.map(n=>'<option'+(n===selected?' selected':'')+'>'+escape(n)+'</option>').join('')+'</optgroup>').join('');
  return Object.freeze({info,tex,axis,groups,options});
});
