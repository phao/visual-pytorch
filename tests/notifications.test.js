'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),N=require('../app/web/notifications.js');
function fixture(){
  const classes=new Set(),toast={hidden:true,getBoundingClientRect:()=>({}),classList:{add:x=>classes.add(x),remove:x=>classes.delete(x),toggle:(x,on)=>on?classes.add(x):classes.delete(x)}},message={},button={},pending=new Map();let id=0;
  const timers={setTimeout:(fn,delay)=>{pending.set(++id,{fn,delay});return id;},clearTimeout:id=>pending.delete(id)};
  const n=N.initialize({getElementById:id=>({'toast':toast,'toast-message':message,'toast-close':button}[id])},timers);
  return {n,toast,message,button,pending,classes};
}
test('Notifications keep 5.5-second duration, then fade for 250 ms',()=>{
  const f=fixture();f.n.show('Saved');assert.equal(f.message.textContent,'Saved');assert.equal(f.toast.hidden,false);
  assert.ok(f.classes.has('toast-visible'));const expiry=[...f.pending.values()][0];assert.equal(expiry.delay,5500);expiry.fn();
  assert.ok(!f.classes.has('toast-visible'));assert.equal(f.toast.hidden,false);
  const fade=[...f.pending.values()][0];assert.equal(fade.delay,250);fade.fn();assert.equal(f.toast.hidden,true);
});
test('Close button fades early; a new notification cancels stale hide timers',()=>{
  const f=fixture();f.n.show('Error',true);assert.ok(f.classes.has('error'));
  f.button.onclick();assert.equal([...f.pending.values()][0].delay,250);
  f.n.show('Recovered');assert.equal(f.pending.size,1);assert.equal([...f.pending.values()][0].delay,5500);
  assert.ok(f.classes.has('toast-visible'));assert.ok(!f.classes.has('error'));assert.equal(f.message.textContent,'Recovered');
});
