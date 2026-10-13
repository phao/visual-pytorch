"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomTypeIcons=api;
})(typeof globalThis!=="undefined" ? globalThis : this,()=>{
  // Trusted SVG fragments only. No workspace strings are interpolated here.
  const icons={
    Constant:'<path d="M4 4h16v16H4zm5 0v16m6-16v16M4 9h16M4 15h16"/>',
    TensorParameter:'<path d="M6 3H3v18h3m12-18h3v18h-3M8 12h8m-4-5c-5 0-5 10 0 10s5-10 0-10"/>',
    Add:'<path d="M3 5h5l4 7-4 7H3m9-7h9M15 5v6m-3-3h6"/>',
    Scale:'<path d="M3 4h18v16H3zm5 4 8 8m0-8-8 8"/>',
    Input:'<path d="M14 4h6v16h-6M2 12h13m-4-4 4 4-4 4"/>',
    Output:'<path d="M10 4H4v16h6m0-8h12m-4-4 4 4-4 4"/>',
    Linear:'<path d="M6 3H3v18h3m12-18h3v18h-3"/><path d="M8 8h1m5 0h1M8 12h1m5 0h1M8 16h1m5 0h1"/>',
    Activation:'<path d="M3 20V4m0 16h18M4 17c5 0 4-10 9-10h7"/>',
    Reshape:'<path d="M3 3h7v5H3zm11 13h7v5h-7zM14 4h6v6m0-6-6 6M4 14v6h6m-6 0 6-6"/>',
    Concat:'<path d="M3 5h4l6 7-6 7H3m10-7h8m-4-4 4 4-4 4"/>',
    Splitter:'<path d="M3 12h6l7-7h5M9 12l7 7h5m-3-17 3 3-3 3m0 8 3 3-3 3"/>',
    Replicator:'<path d="M2 12h5m0 0V6h5m-5 6v6h5"/><path d="M13 3h8v6h-8zm0 12h8v6h-8z"/>',
    Module:'<path d="M3 3h18v18H3zm5 5h8v8H8zM3 12h5m8 0h5"/>',
    Recurrent:'<path d="M5 5h12l4 4-4 4m4-4H5V5m14 14H7l-4-4 4-4m-4 4h16v4M9 9h6v6H9z"/>',
    LSTM:'<path d="M3 4h18v16H3zm0 8h18M8 4v8m8-8v8M7 16h10m-3-2 3 2-3 2"/>',
    Conv2d:'<path d="M3 3h14v14H3zm5 5h13v13H8zM3 8h5M8 3v5m5 5h3m-3 3h3"/>',
    AvgPool2d:'<path d="M3 3h12v12H3zm6 0v12M3 9h12m2 8h4v4h-4zM10 19h5m-3-2 3 2-3 2"/>',
    MaxPool2d:'<path d="M3 3h12v12H3zm3 9 3-6 3 6M17 17h4v4h-4zM10 19h5m-3-2 3 2-3 2"/>',
    AdaptiveAvgPool2d:'<path d="M3 3h14v14H3zm4 0v14m5-14v14M3 9h14m1 9h3v3h-3zM9 20h6"/>',
    Flatten:'<path d="M3 3h8v8H3zm4 0v8M3 7h8M14 6h7m-3-3 3 3-3 3M3 17h18v4H3zm4 0v4m5-4v4m5-4v4"/>',
    Permute:'<path d="M3 5h4l10 14h4M3 19h4L17 5h4m-3-3 3 3-3 3m0 8 3 3-3 3"/>',
  };
  return {types:Object.keys(icons),svg:type=>icons[type] || '<path d="M4 4h16v16H4z"/>'};
});
