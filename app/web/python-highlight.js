"use strict";
// Small Python highlighter for generated, read-only source. Returns escaped HTML
// with tok-* spans; concatenated span text is always exactly the input.
((root,factory)=>{
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.LoomPythonHighlight=api;
})(typeof globalThis==='object'?globalThis:this,()=>{
  const keywords=new Set(('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '));
  const builtins=new Set(('abs all any bool dict enumerate float getattr hasattr int isinstance len list map max min print range reversed set sorted str sum super tuple type zip self ValueError TypeError').split(' '));
  // Order matters: comments, then (prefixed, possibly triple-quoted) strings, numbers, decorators, names.
  const token=/(#[^\n]*)|((?:[rRbBuUfF]{1,2})?(?:'''[\s\S]*?'''|"""[\s\S]*?"""|'(?:\\.|[^'\\\n])*'|"(?:\\.|[^"\\\n])*"))|(\b\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?\b)|(@[A-Za-z_][\w.]*)|([A-Za-z_]\w*)/g;
  const escape=text=>text.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const span=(kind,text)=>'<span class="tok-'+kind+'">'+escape(text)+'</span>';
  function highlight(source){
    let html='',last=0,previous='';
    for(const m of source.matchAll(token)){
      html+=escape(source.slice(last,m.index));
      const [text,comment,string,number,decorator,name]=m;
      if(comment)html+=span('comment',text);
      else if(string)html+=span('string',text);
      else if(number)html+=span('number',text);
      else if(decorator)html+=span('decorator',text);
      else if(['def','class'].includes(previous)&&/^\s+$/.test(source.slice(last,m.index)))html+=span('definition',text);
      else if(keywords.has(name))html+=span('keyword',text);
      else if(builtins.has(name))html+=span('builtin',text);
      else html+=escape(text);
      previous=name||'';last=m.index+text.length;
    }
    return html+escape(source.slice(last));
  }
  return Object.freeze({highlight});
});
