"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomNotifications=api;
})(typeof globalThis!=="undefined" ? globalThis : this,()=>({
  initialize(document,timers=globalThis) {
    const toast=document.getElementById('toast'),message=document.getElementById('toast-message');
    let expires=null,finish=null;
    function clear(){timers.clearTimeout(expires);timers.clearTimeout(finish);expires=finish=null;}
    function close(){
      clear();toast.classList.remove('toast-visible');
      const reduced=timers.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
      finish=timers.setTimeout(()=>{toast.hidden=true;finish=null;},reduced?0:250);
    }
    document.getElementById('toast-close').onclick=close;
    return {close,show(text,error=false){
      clear();message.textContent=text;toast.classList.toggle('error',error);
      const hidden=toast.hidden;toast.hidden=false;
      if(hidden)toast.getBoundingClientRect(); // Commit opacity 0 before transitioning.
      toast.classList.add('toast-visible');expires=timers.setTimeout(close,5500);
    }};
  }
}));
