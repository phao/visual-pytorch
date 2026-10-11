"use strict";
((root,factory)=>{
  const value=factory();
  if(typeof module==='object'&&module.exports)module.exports=value;
  else root.LoomActivations=value;
})(typeof globalThis==='object'?globalThis:this,()=>{
  const info={
    GELU:['h = z · Φ(z)','Smooth gating using the normal CDF. PyTorch uses the exact, non-approximated mode.'],
    ReLU:['h = max(0, z)','Keeps positive features; zeros negative ones.'],
    Sigmoid:['h = 1 / (1 + exp(−z))','Bounds each feature between 0 and 1. A sigmoid does not choose a classification loss; the task or training interface chooses the loss.'],
    Softmax:['hᵢ = exp(zᵢ − max(z)) / Σⱼ exp(zⱼ − max(z))','Normalizes each slice along the selected sample axis; its values sum to 1. Shape and batch B are preserved, with no trainable parameters. Cross-entropy expects raw logits: do not put Softmax before a cross-entropy task head.'],
    Tanh:['h = tanh(z)','Smooth, zero-centered output between −1 and 1.'],
  };
  const tex={GELU:'h=z\\,\\Phi(z)',ReLU:'h=\\max(0,z)',Sigmoid:'h=\\frac{1}{1+e^{-z}}',
    Softmax:'h_i=\\frac{e^{z_i-m}}{\\sum_j e^{z_j-m}},\\quad m=\\max_j z_j',Tanh:'h=\\tanh(z)'};
  function axis(text,rank=null){
    const token=String(text).trim();
    if(!/^-?\d+$/.test(token))throw Error('Enter an integer sample axis; omit batch B.');
    const dim=Number(token);
    if(!Number.isSafeInteger(dim)||dim < -8||dim>7 ||
      (rank!==null&&(!Number.isInteger(rank)||rank<1||rank>8||dim < -rank||dim>=rank)))
      throw Error('Softmax axis is outside the sample dimensions; omit batch B.');
    return dim;
  }
  return Object.freeze({info,tex,axis});
});
