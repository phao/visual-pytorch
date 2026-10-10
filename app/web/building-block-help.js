"use strict";
((root,factory)=>{
  const api=factory(typeof module!=="undefined" && module.exports ? require('./tensor-operators.js') : root.LoomTensorOperators);
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomBlockHelp=api;
})(typeof globalThis!=="undefined"?globalThis:this,T=>{
  const names={
    'add-linear':'Linear','add-activation':'Activation','add-reshape':'Reshape',
    'add-concat':'Concat','add-reference':'Architecture reference','add-input':'Input part',
    'add-output':'Output','add-replicator':'Replicator','add-splitter':'Splitter',
  };
  const activations={
    ReLU:['h = max(0, z)','Keeps positive values and replaces negative values with zero.'],
    Tanh:['h = tanh(z)','Maps each value smoothly into (−1, 1); zero remains zero.'],
    Sigmoid:['h = 1 / (1 + exp(−z))','Maps each value into (0, 1). A sigmoid does not choose a classification loss; the task or training interface chooses the loss.'],
    GELU:['h = z · Φ(z)','Multiplies each value by the standard normal cumulative probability at that value. This app uses PyTorch’s exact, non-approximated GELU.'],
  };
  function describe(id,context={}){
    const type=context.tensorType;
    if(type && T.definitions[type])return {name:type,
      paragraphs:[T.definitions[type].explanation,
        'Adds one '+type+' block on the selected path. Inspect the new block to edit its concrete settings; validation checks the incoming shape.',
        ['Conv2d'].includes(type)?'The filters and optional bias are learned during training. Batch B stays first.':'This operation has no learned weights. Batch B stays first.',
        'Initial settings: '+JSON.stringify(T.definitions[type].defaults)+'. The displayed settings describe this app’s supported version, not every PyTorch option.'],
      equation:T.definitions[type].equation,tex:T.definitions[type].tex};
    if(!names[id])return null;
    let paragraphs,equation=null,tex=null;
    switch(id){
      case 'add-linear':
        paragraphs=['Adds a learned affine map to the selected path. Each output is a weighted sum of the incoming features, plus an optional learned bias.',
          'Only the last axis is mixed: [B, 10, 3] becomes [B, 10, k] when you choose k output features. The same weights are used at every earlier-axis position.',
          'For n incoming features and k outputs, A has shape [k, n] and b has shape [k]. There are k × n weights and, with bias enabled, k more parameters.',
          'The new block initially has bias enabled and uses the incoming last-axis width as its output width. Change output features or bias in the Inspector. Consecutive Linear layers without nonlinear activations still compose to an affine map.'];
        equation='y = x Aᵀ + b';tex='y=xA^{\\mathsf T}+b';break;
      case 'add-activation':{
        const kind=activations[context.activation]?context.activation:'ReLU';
        paragraphs=['Adds the selected '+kind+' activation, applied independently to each tensor value. The shape is unchanged and no weights are learned.',
          activations[kind][1],
          'Place an activation after a Linear or convolution block to introduce nonlinearity. Change the function using the palette selector before adding, or in the Inspector afterwards.'];
        equation=activations[kind][0];break;
      }
      case 'add-reshape':paragraphs=['Adds a box that changes a sample’s dimensions without changing its number of values. Batch B is preserved.',
        'For example, [B, 2, 3] can become [B, 6]. Specify sample dimensions without B; one −1 dimension may be inferred.',
        'Reshape preserves row-major flattened value order. It does not interchange axes: use Permute when you need H,W,C to become C,H,W. No parameters are learned. The new box initially preserves the incoming sample shape.'];break;
      case 'add-concat':paragraphs=['Adds a join for independent branches. Connect each branch to an input socket; the socket order determines the order of output values.',
        'By default each sample is flattened before joining. [B, 2, 3] and [B, 4] then produce [B, 10]. B must match; samples are not joined across the batch.',
        'Disable flattening in the Inspector to join only the last axis. Then all earlier dimensions must match: [B, 10, 3] and [B, 10, 2] produce [B, 10, 5].',
        'Concat does not add branch values together and has no learned parameters. A later Linear layer can learn how to mix the joined features.'];break;
      case 'add-reference':paragraphs=['Adds an instance of the architecture selected in the reusable-architecture list'+(context.referenceName?': '+context.referenceName:'. Choose a reference before adding it')+'.',
        'The referenced graph supplies the computation. Each occurrence has its own learned parameters; this is not weight sharing between occurrences.',
        'Open details from the Inspector to edit that architecture. Updates affect its instances in the workspace; validation rejects cycles or incompatible shapes.',
        'The current prototype connects architecture references through packed tensors. Named inputs and outputs describe the interface; do not assume arbitrary independent multi-input calling conventions.'];break;
      case 'add-input':paragraphs=['Adds a separate named input path with its own Input and initial Output box. It initially carries one float32 feature per sample, [B, 1].',
        'Edit the sample dimensions and name in the Inspector. Different inputs can travel through different layers and meet later at a Concat box.',
        'An Input declares supplied data, not a learned parameter. Data bindings must explicitly match the input identities and shapes; moving boxes must not change column meaning.'];break;
      case 'add-output':paragraphs=['Adds a named output exposing the selected tensor or selected router output port. It does not apply a transformation or create weights.',
        'Its dimensions are inferred from the incoming connection unless you declare a matching shape in the Inspector. Use Reshape to actually change dimensions.',
        'Outputs define what the forward model returns. Adding one is separate from choosing a target, loss or evaluation metric. You can restore a deleted output with this button.'];break;
      case 'add-replicator':paragraphs=['Adds a router that exposes the same entire incoming tensor on two initial output ports. Each port can feed a different downstream path.',
        'Both ports keep the full shape, for example [B, 6] → two ports each [B, 6]. This is logical fan-out, not a physical tensor copy or a partition.',
        'No parameters are learned. Use the Inspector to add ports. Select the tensor-producing box first; router insertion needs an existing outgoing connection.'];break;
      case 'add-splitter':paragraphs=['Adds a router that partitions the selected tensor along its last axis into ordered, non-overlapping slices.',
        'For example, widths 2 and 4 split [B, 10, 6] into [B, 10, 2] and [B, 10, 4]. Every other dimension, including B, is preserved.',
        'Slice widths must be positive and sum to the incoming last-axis width. The button initially creates two nearly equal pieces; edit widths and add ports in the Inspector.',
        'No parameters are learned. Unlike Replicator, each port carries only its assigned slice. Select a tensor-producing box with at least two last-axis features and an existing outgoing connection.'];break;
    }
    return {name:names[id],paragraphs,equation,tex};
  }
  return {describe,names};
});
