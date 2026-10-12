"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),G=require('../app/web/graph-tools.js');
const event=(changes={})=>({key:' ',target:{isContentEditable:false,closest:()=>null},...changes});
test('Plain Space is the selected-block toggle; repeat is consumed by the controller without toggling',()=>{
  assert.equal(G.blockToggleShortcut(event()),true);assert.equal(G.blockToggleShortcut(event({repeat:true})),true);
  for(const key of ['Enter','Space','Spacebar','d','Escape'])assert.equal(G.blockToggleShortcut(event({key})),false);
});
test('Modified or already-handled Space keeps its original behavior',()=>{
  for(const key of ['ctrlKey','metaKey','altKey','shiftKey','defaultPrevented'])
    assert.equal(G.blockToggleShortcut(event({[key]:true})),false);
});
test('Typing and native/accessible control descendants never trigger block folding',()=>{
  assert.equal(G.blockToggleShortcut(event({target:{isContentEditable:true,closest:()=>null}})),false);
  for(const control of ['input','textarea','select','button','a','summary','[contenteditable]',
    '[role="button"]','[role="textbox"]','[role="combobox"]','[role="slider"]','[role="checkbox"]','[role="switch"]']){
    const target={isContentEditable:false,closest:selector=>selector.split(',').includes(control)?{}:null};
    assert.equal(G.blockToggleShortcut(event({target})),false,control);
  }
});
