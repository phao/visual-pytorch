"use strict";
(() => {
  let resolveReady,rejectReady;
  const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
  ready.catch(()=>{});
  // Opt-in containers only: user labels, code and ordinary UI text are not TeX.
  window.MathJax={
    skipStartupTypeset:true,showMathMenu:false,showProcessingMessages:false,messageStyle:'none',
    extensions:['tex2jax.js','Safe.js','AssistiveMML.js'],
    jax:['input/TeX','output/SVG'],
    SVG:{font:'TeX',useGlobalCache:false},
    TeX:{extensions:['AMSmath.js','AMSsymbols.js'],MAXBUFFER:10240,MAXMACROS:1000},
    tex2jax:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]'],['$$','$$']],
      ignoreClass:'.*',processClass:'loom-math',skipTags:['script','noscript','style','textarea','pre','code','svg']},
    Safe:{allow:{URLs:'none',classes:'none',cssIDs:'none',styles:'none',fontsize:'safe',require:'none'}},
    menuSettings:{zoom:'None'},
    AuthorInit(){window.MathJax.Hub.Register.StartupHook('End',resolveReady);},
  };
  let queue=Promise.resolve();
  function enqueue(element,prepare) {
    queue=queue.catch(()=>{}).then(()=>ready).then(()=>new Promise((resolve,reject)=>{
      const hub=window.MathJax.Hub;
      try {
        hub.Queue(()=>{if(prepare)prepare(hub);},['Typeset',hub,element],resolve);
      } catch(error){reject(error);}
    }));
    return queue;
  }
  window.LoomMath={ready,
    typeset(element=document.body){return enqueue(element);},
    set(element,tex,{display=false}={}) {
      return enqueue(element,hub=>{
        hub.getAllJax(element).forEach(jax=>jax.Remove());
        element.classList.add('loom-math');
        element.textContent=(display?'\\[':'\\(')+tex+(display?'\\]':'\\)');
      });
    },
    failed(){rejectReady(new Error('Local MathJax bundle failed to load.'));},
  };
})();
