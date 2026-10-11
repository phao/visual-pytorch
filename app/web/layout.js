"use strict";
((root,factory)=>{
  const api=factory();
  if(typeof module!=="undefined" && module.exports)module.exports=api;
  else root.LoomLayout=api;
})(typeof globalThis!=="undefined"?globalThis:this,()=>{
  const profiles={collapsed:{style:'layered',direction:'LR',layer_gap:48,block_gap:16,rearrange:true,fit:true},
    expanded:{style:'layered',direction:'LR',layer_gap:80,block_gap:32,rearrange:true,fit:true,parameter_sizing:false}};
  const defaults=()=>JSON.parse(JSON.stringify({version:1,...profiles}));
  function validate(p){
    if(!p || typeof p!=='object' || Array.isArray(p) || p.style!=='layered' || !['LR','TB'].includes(p.direction))
      throw new Error('Choose the Layered layout and a supported direction.');
    for(const key of ['layer_gap','block_gap'])
      if(!Number.isInteger(p[key]) || p[key]<8 || p[key]>1000)throw new Error('Layout gaps must be integers from 8 to 1000 model units.');
    for(const key of ['rearrange','fit'])if(typeof p[key]!=='boolean')throw new Error('Layout toggles must be booleans.');
    if(Object.hasOwn(p,'parameter_sizing') && typeof p.parameter_sizing!=='boolean')throw new Error('Parameter sizing must be a boolean.');
    return p;
  }
  function arrange(doc,p,G){
    validate(p);
    const nodes=doc.graph.nodes,byID=new Map(nodes.map(n=>[n.id,n]));
    if(byID.size!==nodes.length)throw new Error('Layout requires unique block IDs.');
    const edges=doc.graph.edges;
    if(edges.some(e=>!byID.has(e.from) || !byID.has(e.to)))throw new Error('Repair missing connection endpoints before arranging.');
    const adjacency=new Map(nodes.map(n=>[n.id,new Set()]));
    edges.forEach(e=>adjacency.get(e.from).add(e.to));
    // Collapse strongly connected components for layout only. Saved cyclic
    // drafts remain cyclic/invalid; arranging must never silently repair them.
    let serial=0;const index=new Map(),low=new Map(),stack=[],onStack=new Set(),components=[];
    function visit(id){
      index.set(id,serial);low.set(id,serial++);stack.push(id);onStack.add(id);
      for(const target of [...adjacency.get(id)].sort()){
        if(!index.has(target)){visit(target);low.set(id,Math.min(low.get(id),low.get(target)));}
        else if(onStack.has(target))low.set(id,Math.min(low.get(id),index.get(target)));
      }
      if(low.get(id)===index.get(id)){
        const members=[];let member;
        do{member=stack.pop();onStack.delete(member);members.push(member);}while(member!==id);
        components.push(members.sort());
      }
    }
    [...byID.keys()].sort().forEach(id=>{if(!index.has(id))visit(id);});
    const componentOf=new Map();components.forEach((members,i)=>members.forEach(id=>componentOf.set(id,i)));
    const successors=components.map(()=>new Set()),incoming=components.map(()=>0),rank=components.map(()=>0);
    for(const e of edges){
      const a=componentOf.get(e.from),b=componentOf.get(e.to);
      if(a!==b && !successors[a].has(b)){successors[a].add(b);incoming[b]++;}
    }
    const queue=incoming.flatMap((count,i)=>count?[]:[i]);
    while(queue.length){const a=queue.shift();for(const b of successors[a]){
      rank[b]=Math.max(rank[b],rank[a]+1);if(!--incoming[b])queue.push(b);
    }}
    const layers=Array.from({length:Math.max(0,...rank)+1},()=>[]);
    nodes.forEach(n=>layers[rank[componentOf.get(n.id)]].push(n.id));
    layers.forEach(layer=>layer.sort());
    const positions=()=>new Map(layers.flatMap(layer=>layer.map((id,i)=>[id,i])));
    const offset=(n,output,key)=>{
      const slots=G.slots(n,output),i=slots.indexOf(key);
      return i<0 || slots.length<2?0:(i/(slots.length-1)-.5)*.6;
    };
    // Four alternating barycentric sweeps. Port offsets preserve recognizable
    // branch order when siblings feed ordered join/router sockets.
    for(let sweep=0;sweep<4;sweep++){
      const forward=sweep%2===0,pos=positions();
      const order=forward?layers:[...layers].reverse();
      for(const layer of order){
        const score=id=>{
          const neighbors=edges.filter(e=>forward?e.to===id:e.from===id)
            .filter(e=>componentOf.get(e.from)!==componentOf.get(e.to));
          if(!neighbors.length)return pos.get(id);
          return neighbors.reduce((sum,e)=>{
            const neighbor=forward?e.from:e.to;
            const key=forward?(e.from_port || (['Splitter','Replicator'].includes(byID.get(e.from).type)?e.to:null)):
              (e.from_port?e.from+'@'+e.from_port:e.from);
            return sum+pos.get(neighbor)+offset(byID.get(neighbor),forward,key);
          },0)/neighbors.length;
        };
        layer.sort((a,b)=>score(a)-score(b) || pos.get(a)-pos.get(b) || a.localeCompare(b));
        layer.forEach((id,i)=>pos.set(id,i));
      }
    }
    const vertical=p.direction==='TB';
    const sizes=new Map(nodes.map(n=>[n.id,{along:vertical?G.height(n):G.width(n),across:vertical?G.width(n):G.height(n)}]));
    if([...sizes.values()].some(s=>!Number.isFinite(s.along) || !Number.isFinite(s.across) || s.along<=0 || s.across<=0))
      throw new Error('Layout requires finite positive block sizes.');
    const extents=layers.map(layer=>layer.reduce((sum,id)=>sum+sizes.get(id).across,0)+Math.max(0,layer.length-1)*p.block_gap);
    const breadth=Math.max(0,...extents),result=new Map();let along=48;
    layers.forEach((layer,i)=>{
      let across=48+(breadth-extents[i])/2;
      const depth=Math.max(0,...layer.map(id=>sizes.get(id).along));
      for(const id of layer){const size=sizes.get(id),a=along+(depth-size.along)/2;
        result.set(id,vertical?{x:across,y:a}:{x:a,y:across});across+=size.across+p.block_gap;
      }
      along+=depth+p.layer_gap;
    });
    if([...result.values()].some(pos=>Math.abs(pos.x)>100000 || Math.abs(pos.y)>100000))
      throw new Error('Layout exceeds workspace coordinate bounds. Reduce block sizes or spacing.');
    nodes.forEach(n=>Object.assign(n,result.get(n.id)));
    return doc;
  }
  function sizeByParameters(doc,counts,G){
    const nodes=doc.graph.nodes;
    if(!counts||typeof counts!=='object'||Array.isArray(counts)||Object.keys(counts).length!==nodes.length)
      throw new Error('Every block needs a validated parameter count before sizing.');
    const values=nodes.map(n=>{
      const value=counts[n.id];
      if(typeof value!=='string'||! /^(0|[1-9][0-9]*)$/.test(value)||value.length>100)
        throw new Error('Use exact nonnegative decimal parameter counts.');
      return BigInt(value);
    });
    const positive=values.filter(v=>v>0n),least=positive.reduce((a,b)=>a<b?a:b,positive[0]??0n),most=positive.reduce((a,b)=>a>b?a:b,0n);
    // Exact integer subtraction precedes the bounded visual ratio. Absolute counts
    // never become model-unit heights, including counts above JS's safe-integer range.
    nodes.forEach((n,i)=>{
      const count=values[i];n.collapsed=count===0n;
      if(count>0n){
        const ratio=most===least?0:Number(count-least)/Number(most-least);
        n.height=G.minimumSize(n).height*(1+3*ratio);
      }
    });
  }
  function apply(doc,collapsed,p,G,counts=null){
    validate(p);
    if(typeof collapsed!=='boolean')throw new Error('Choose a collapsed or expanded state.');
    const result=JSON.parse(JSON.stringify(doc));
    result.graph.nodes.forEach(n=>{n.collapsed=collapsed;});
    if(!collapsed && p.parameter_sizing)sizeByParameters(result,counts,G);
    if(p.rearrange)arrange(result,p,G);
    return result;
  }
  return {defaults,validate,arrange,apply};
});
