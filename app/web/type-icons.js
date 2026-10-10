"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomTypeIcons=api;
})(typeof globalThis!=="undefined" ? globalThis : this,()=>{
  // Trusted SVG fragments only. No workspace strings are interpolated here.
  const icons={
    Input:'<path d="M14 4h6v16h-6M2 12h13m-4-4 4 4-4 4"/>',
    Output:'<path d="M10 4H4v16h6m0-8h12m-4-4 4 4-4 4"/>',
    Linear:'<path d="M6 3H3v18h3m12-18h3v18h-3"/><path d="M8 8h1m5 0h1M8 12h1m5 0h1M8 16h1m5 0h1"/>',
    Activation:'<path d="M3 20V4m0 16h18M4 17c5 0 4-10 9-10h7"/>',
    Reshape:'<path d="M3 3h7v5H3zm11 13h7v5h-7zM14 4h6v6m0-6-6 6M4 14v6h6m-6 0 6-6"/>',
    Concat:'<path d="M3 5h4l6 7-6 7H3m10-7h8m-4-4 4 4-4 4"/>',
    Splitter:'<path d="M3 12h6l7-7h5M9 12l7 7h5m-3-17 3 3-3 3m0 8 3 3-3 3"/>',
    Replicator:'<path d="M2 12h5m0 0V6h5m-5 6v6h5"/><path d="M13 3h8v6h-8zm0 12h8v6h-8z"/>',
    Module:'<path d="M3 3h18v18H3zm5 5h8v8H8zM3 12h5m8 0h5"/>',
  };
  return {types:Object.keys(icons),svg:type=>icons[type] || '<path d="M4 4h16v16H4z"/>'};
});
