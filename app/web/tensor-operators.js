"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomTensorOperators=api;
})(typeof globalThis!=="undefined"?globalThis:this,()=>{
  // UI edits concrete specs; shapes/constructor checks remain backend authority.
  const definitions={
    PositiveVariance:{defaults:{positive_floor:.01},fields:{positive_floor:'positive-scalar'},
      explanation:'Softplus(raw) plus this explicit positive variance floor. It preserves shape and axes, has no learned weights and never clips variance. This is variance, not standard deviation or log variance.',
      equation:'variance = softplus(raw) + ε',tex:'v=\\operatorname{softplus}(r)+\\varepsilon'},
    Scale:{defaults:{coefficient:2},fields:{coefficient:'scalar'},
      explanation:'Multiply every entry by this fixed coefficient. Shapes, axes and batch B are preserved. This block has no learned weights; changing the coefficient is an architecture edit, not training.',
      equation:'y = c x · c fixed',tex:'y=cx,\\quad c\\text{ fixed}'},
    Conv2d:{defaults:{out_channels:2,kernel_size:[3,3],stride:[1,1],padding:[0,0],dilation:[1,1],groups:1,bias:true,padding_mode:"zeros"},
      fields:{out_channels:"positive",kernel_size:"pair",stride:"pair",padding:"padding",dilation:"pair",groups:"positive",bias:"bool"},
      explanation:"NCHW: sample axes C, H, W. Cross-correlation learns local filters, optionally grouped. H/W output = floor((input + 2 padding − dilation (kernel − 1) − 1) / stride) + 1. Zero padding only.",
      equation:"y = Conv2d(x; W, b)",tex:"y=\\operatorname{Conv2d}(x;W,b)"},
    AvgPool2d:{defaults:{kernel_size:[2,2],stride:[2,2],padding:[0,0],ceil_mode:false,count_include_pad:true},
      fields:{kernel_size:"pair",stride:"pair",padding:"padding",count_include_pad:"bool"},
      explanation:"Average spatial windows independently in each channel. NCHW, floor output sizes. Count include pad chooses whether padded zero cells enter the divisor. No learned parameters.",
      equation:"y = spatial window mean(x)",tex:"y=\\operatorname{AvgPool2d}(x)"},
    MaxPool2d:{defaults:{kernel_size:[2,2],stride:[2,2],padding:[0,0],dilation:[1,1],ceil_mode:false},
      fields:{kernel_size:"pair",stride:"pair",padding:"padding",dilation:"pair"},
      explanation:"Maximum over spatial windows independently per channel. NCHW, floor output sizes. Padding is negative infinity, not zero. Indices and ceil mode are not enabled in version 1.",
      equation:"y = spatial window max(x)",tex:"y=\\operatorname{MaxPool2d}(x)"},
    AdaptiveAvgPool2d:{defaults:{output_size:[1,1]},fields:{output_size:"pair"},
      explanation:"Average spatial bins to a specified H, W. Channels and batch are unchanged. Output 1, 1 is global spatial averaging. Bins may overlap when dimensions do not divide evenly.",
      equation:"y = adaptive spatial mean(x)",tex:"y=\\operatorname{AdaptiveAvgPool2d}(x)"},
    Flatten:{defaults:{start_dim:0,end_dim:-1},fields:{start_dim:"axis",end_dim:"axis"},
      explanation:"Axis numbers refer only to sample dimensions; B is excluded. Merge a contiguous range in row-major value order. Negative axes count from the end. Default 0, −1 flattens the whole sample.",
      equation:"y = flatten sample axes(x)",tex:"y=\\operatorname{flatten}(x)"},
    Permute:{defaults:{dims:[2,0,1]},fields:{dims:"permutation"},
      explanation:"Reorder sample axes and their values; B always stays first. For H,W,C input use 2,0,1 to obtain C,H,W. This is not Reshape: the value order changes.",
      equation:"y = permute sample axes(x)",tex:"y=\\operatorname{permute}(x)"},
  };
  function defaults(type,shape){
    if(!definitions[type])throw new Error("Unknown tensor operator.");
    const values=JSON.parse(JSON.stringify(definitions[type].defaults));
    if(type==="Permute" && shape?.length!==3)values.dims=Array.from({length:shape?.length || 1},(_,i)=>i);
    return {op_version:1,...values};
  }
  function parse(type,key,text){
    const kind=definitions[type]?.fields[key];
    if(!kind)throw new Error("Unsupported operator field.");
    if(kind==="bool"){if(typeof text!=="boolean")throw new Error("Use a checkbox value.");return text;}
    if(kind==='scalar'||kind==='positive-scalar'){
      const raw=String(text).trim(),value=Number(raw);
      if(!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw)||!Number.isFinite(value)||Math.abs(value)>3.4028234663852886e38||Number.isInteger(value)&&!Number.isSafeInteger(value))throw new Error('Use a finite float32-range coefficient; integer values must be editor-exact.');
      if(kind==='positive-scalar'&&(!Number.isFinite(Math.fround(value))||Math.fround(value)<=0))throw Error('The floor must remain finite and positive in float32; no clamping.');
      return value;
    }
    const tokens=String(text).trim().split(/[,\s]+/);
    if(!tokens.length || tokens.some(t=>!/^[-]?\d+$/.test(t)))throw new Error("Enter integer values.");
    const values=tokens.map(Number);
    if(values.some(v=>!Number.isSafeInteger(v)))throw new Error("Use exact integer values.");
    if(kind==="pair" || kind==="padding"){
      if(values.length===1)values.push(values[0]);
      if(values.length!==2 || values.some(v=>v<(kind==="padding"?0:1)))throw new Error("Use height, width; positive values (padding may be zero).");
      return values;
    }
    if(kind==="permutation"){
      if(values.length<1 || values.length>8 || new Set(values).size!==values.length || values.some(v=>v<0 || v>=values.length))throw new Error("List each sample axis 0..rank−1 exactly once; omit B.");
      return values;
    }
    if(values.length!==1 || (kind==="positive" && values[0]<1) || (kind==="axis" && (values[0]<-8 || values[0]>7)))throw new Error("Use a valid sample axis or positive integer.");
    return values[0];
  }
  return {definitions,types:Object.keys(definitions).sort(),defaults,parse};
});
