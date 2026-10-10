"use strict";
const test=require('node:test'),assert=require('node:assert/strict'),{execFileSync}=require('node:child_process');
const E=require('../app/web/experiment-authoring.js');
test('split method browsing is alphabetical; each factory is closed and independent',()=>{
  assert.deepEqual(E.methods,[...E.methods].sort());
  assert.deepEqual(E.split('s','row'),{id:'s',protocol:{method:'row',fractions:[.6,.2,.2],seed:3}});
  assert.deepEqual(E.split('s','group'),{id:'s',protocol:{method:'group',fractions:[.6,.2,.2],seed:3,field:''}});
  assert.deepEqual(E.split('s','temporal'),{id:'s',protocol:{method:'temporal',fractions:[.6,.2,.2],field:''}});
  assert.deepEqual(E.split('s','external'),{id:'s',protocol:{method:'external',roles:{train:[],validation:[],test:[]},excluded:[]}});
  const a=E.split('s','external');a.protocol.roles.train.push('row');assert.deepEqual(E.split('s','external').protocol.roles.train,[]);
  assert.throws(()=>E.split('s','random-guess'));
});
test('training factory matches the authoritative pure Python defaults exactly',()=>{
  const python=JSON.parse(execFileSync('python3',['-c','import json;from app.task_execution import training_protocol;print(json.dumps(training_protocol()))'],{encoding:'utf8'}));
  assert.deepEqual(E.protocol('p'),{id:'p',protocol:python});
  const p=E.protocol('p');p.protocol.optimizer.betas[0]=.2;p.protocol.budget.epochs=null;assert.equal(E.protocol('q').protocol.optimizer.betas[0],.9);
});
test('exact selections start incomplete, never guessed from metadata position',()=>{
  assert.deepEqual(E.reference('model'),{kind:'model',id:'',revision:''});
  assert.deepEqual(E.pairing('pair'),{id:'pair',dataset:E.reference('dataset'),task:E.reference('task')});
  assert.deepEqual(E.comparison(),{model:E.reference('model'),data_task:''});
  assert.deepEqual(E.comparison(true),{model:E.reference('model'),data_task:'',reason:''});
});
test('initialization seeds retain order and duplicates for authoritative validation',()=>{
  assert.deepEqual(E.seeds('13, 7\n3'),[13,7,3]);assert.deepEqual(E.seeds('7,7'),[7,7]);assert.deepEqual(E.seeds(''),[]);
  for(const value of ['true','-1','1.5','2147483648','1e400','7,',true])assert.throws(()=>E.seeds(value));
});
test('external IDs preserve literal spaces, commas, role order and duplicates',()=>{
  assert.deepEqual(E.ids(' row,3 \nrow1\nrow1\n'),[' row,3 ','row1','row1']);
  assert.deepEqual(E.ids('row3\r\nrow1'),['row3','row1']);assert.deepEqual(E.ids(''),[]);
  assert.throws(()=>E.ids(['r0']));
});
test('optional budgets use explicit null, with finite exact range checks',()=>{
  assert.equal(E.number(' ',{optional:true}),null);assert.equal(E.number('3',{integer:true,minimum:1}),3);
  assert.equal(E.number('0',{minimum:0}),0);assert.equal(E.number('2.5',{minimum:Number.MIN_VALUE}),2.5);
  for(const value of ['Infinity','NaN','1e400',true,1])assert.throws(()=>E.number(value));
  for(const value of ['0','-1','1.2','9007199254740992'])assert.throws(()=>E.number(value,{integer:true,minimum:1}));
  assert.throws(()=>E.number('1.1',{maximum:1}));assert.throws(()=>E.number(''));
});
