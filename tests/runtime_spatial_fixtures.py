"""Tiny deterministic runtime-spatial source fixtures; no numerical imports."""
import copy
from app.core import default_cnn


def spatial():
    doc = default_cnn(); doc['id'] = 'cnn'
    doc = {k:copy.deepcopy(doc[k]) for k in ('id','name','graph')}
    lookup = {n['id']:n for n in doc['graph']['nodes']}
    lookup['input'].update(shape=[4,5,6],axes=['C','H','W'])
    lookup['conv'].update(out_channels=8,kernel_size=[3,3],padding=[1,1],groups=2)
    lookup['head']['features'] = 3
    return {'cnn':doc},'cnn'


def nested(*, shared=False):
    models,mid = spatial(); child = models[mid]
    child['graph']['nodes'] = [n for n in child['graph']['nodes'] if n['id'] not in {'pool','flatten','head'}]
    child['graph']['edges'] = [{'from':a,'to':b} for a,b in [('input','conv'),('conv','activation'),('activation','output')]]
    root = {'id':'root','name':'Two calls','graph':{'nodes':[
        {'id':'a','type':'Input','shape':[4,5,6],'axes':['C','H','W'],'label':'left'},
        {'id':'b','type':'Input','shape':[4,5,6],'axes':['C','H','W'],'label':'right'},
        {'id':'left','type':'Module','ref':'cnn'},
        {'id':'right','type':'Module','ref':'cnn',**({'share_with':'left'} if shared else {})},
        {'id':'lp','type':'AdaptiveAvgPool2d','output_size':[1,1]},
        {'id':'rp','type':'AdaptiveAvgPool2d','output_size':[1,1]},
        {'id':'join','type':'Concat','sources':['lp','rp'],'flatten':True},
        {'id':'head','type':'Linear','features':3,'bias':True},
        {'id':'out','type':'Output'}],
        'input_order':['a','b'],
        'edges':[{'from':a,'to':b} for a,b in [('a','left'),('b','right'),('left','lp'),('right','rp'),('lp','join'),('rp','join'),('join','head'),('head','out')]]}}
    models['root'] = root
    return models,'root'


def family():
    models,mid = spatial()
    return {'version':1,'id':'spatial-family','name':'Adaptive spatial family','root':mid,
            'models':models,'symbols':[],'assignments':[],'repeats':[]}
