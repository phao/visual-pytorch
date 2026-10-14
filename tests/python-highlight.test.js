"use strict";
const test=require('node:test'),assert=require('node:assert/strict');
const {highlight}=require('../app/web/python-highlight.js');
const text=html=>html.replace(/<[^>]+>/g,'').replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&quot;/g,'"').replace(/&#39;/g,"'").replace(/&amp;/g,'&');
test('Highlighting classifies Python tokens and preserves the exact text',()=>{
  const source="import torch\nclass Net(nn.Module):\n    def forward(self, x):  # a <b> & 'c'\n        return x * 1.5e-3 if x is not None else '''doc\n\"q\"'''\n@torch.no_grad()\ndef f(): return r'\\d' + \"x\"\n";
  const html=highlight(source);
  assert.equal(text(html),source);
  for(const [kind,value] of [['keyword','import'],['keyword','class'],['definition','Net'],['definition','forward'],['builtin','self'],
    ['comment',"# a &lt;b&gt; &amp; &#39;c&#39;"],['number','1.5e-3'],['keyword','None'],['decorator','@torch.no_grad'],['string',"r&#39;\\d&#39;"]])
    assert.ok(html.includes('<span class="tok-'+kind+'">'+value+'</span>'),kind+' '+value);
  assert.ok(html.includes('<span class="tok-string">&#39;&#39;&#39;doc\n&quot;q&quot;&#39;&#39;&#39;</span>'));
  assert.ok(!/<(?!\/?span)/.test(html),'only span markup is produced');
});
test('Identifiers that merely contain keywords stay plain',()=>{
  const html=highlight('format = definitely_not + classy');
  assert.ok(!html.includes('tok-'));
});
