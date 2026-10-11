'use strict';
class Element{
  constructor(tag='div',text=''){this.tagName=tag;this.textContent=text;this.children=[];this.parent=null;this.dataset={};this.attributes={};this.disabled=false;this.hidden=false;this.value='';}
  append(...nodes){for(const n of nodes){n.remove();n.parent=this;this.children.push(n);}}
  remove(){if(this.parent){this.parent.children=this.parent.children.filter(n=>n!==this);this.parent=null;}}
  before(n){const p=this.parent,i=p.children.indexOf(this);n.remove();n.parent=p;p.children.splice(i,0,n);}
  after(n){const p=this.parent,i=p.children.indexOf(this);n.remove();n.parent=p;p.children.splice(i+1,0,n);}
  replaceChildren(...nodes){this.children.forEach(n=>n.parent=null);this.children=[];this.append(...nodes);}
  setAttribute(k,v){this.attributes[k]=v;}removeAttribute(k){delete this.attributes[k];}
  matches(selector){return selector.split(',').some(s=>{
    if(s.startsWith('.'))return (this.className||'').split(' ').includes(s.slice(1));
    const match=/^\[data-([a-z-]+)(?:="?([^"\]]+)"?)?\]$/.exec(s);
    if(match){const k=match[1].replace(/-([a-z])/g,(_,c)=>c.toUpperCase());return k in this.dataset&&(match[2]===undefined||this.dataset[k]===match[2]);}
    return s===this.tagName;
  });}
  closest(s){return this.matches(s)?this:this.parent?.closest(s)||null;}
  querySelectorAll(s){return this.children.flatMap(n=>[...(n.matches(s)?[n]:[]),...n.querySelectorAll(s)]);}
  querySelector(s){return this.querySelectorAll(s)[0]||null;}contains(n){return this===n||this.children.some(c=>c.contains(n));}
  dispatchEvent(e){return this['on'+e.type]?.(e);}
  get options(){return this.children.filter(c=>c.tagName==='option');}
  get isConnected(){return this.tagName==='body'||Boolean(this.parent?.isConnected);}
}
module.exports={Element,tick:()=>new Promise(r=>setImmediate(r))};
