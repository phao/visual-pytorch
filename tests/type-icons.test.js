'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const I=require('../app/web/type-icons.js');
test('All supported node types have distinct vector icons, not colour-only identity',()=>{
  const types=['Input','Output','Linear','Activation','Reshape','Concat','Splitter','Replicator','Module','Conv2d','AvgPool2d','MaxPool2d','AdaptiveAvgPool2d','Flatten','Permute'];
  assert.deepEqual(new Set(I.types),new Set(types));
  assert.equal(new Set(types.map(I.svg)).size,types.length);
  for(const type of types){assert.match(I.svg(type),/<path /);assert.doesNotMatch(I.svg(type),/script|href|on\w+=/);}
});
