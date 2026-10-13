"""Editable exact two-state HMM graph: real sources, no fabricated unit data."""
def model():
    nodes=[{'id':'observations','type':'Input','shape':[4,1],'axes':['T','D']},
        {'id':'initial_logits','type':'TensorParameter','shape':[2],'axes':['state']},
        {'id':'initial','type':'Activation','activation':'LogSoftmax','dim':0},
        {'id':'transition_logits','type':'TensorParameter','shape':[2,2],'axes':['from_state','to_state']},
        {'id':'transition','type':'Activation','activation':'LogSoftmax','dim':1},
        {'id':'means','type':'TensorParameter','shape':[2,1],'axes':['state','coordinate']},
        {'id':'raw_variances','type':'TensorParameter','shape':[2,1],'axes':['state','coordinate']},
        {'id':'variances','type':'PositiveVariance','positive_floor':.01},
        {'id':'lengths','type':'Input','shape':[1]},
        {'id':'filter','type':'GaussianFilter','inputs':{r:r for r in ('observations','initial','transition','means','variances','lengths')},
         'initial_representation':'log-probabilities','transition_representation':'log-probabilities',
         'transition_mode':'shared','mean_mode':'shared','variance_mode':'shared'}]
    edges=[{'from':'initial_logits','to':'initial'},{'from':'transition_logits','to':'transition'},
           {'from':'raw_variances','to':'variances'}]
    edges.extend({'from':role,'to':'filter'} for role in ('observations','initial','transition','means','variances','lengths'))
    for port in ('filtered','final','step_evidence','sequence_evidence'):
        nodes.append({'id':'out_'+port,'type':'Output','label':port})
        edges.append({'from':'filter','from_port':port,'to':'out_'+port})
    for i,n in enumerate(nodes):n.update(x=40+(i%4)*230,y=60+(i//4)*150)
    graph={'nodes':nodes,'edges':edges,'input_order':['observations','lengths'],
        'parameter_policy':{'version':1,'parameters':{'["means","value"]':{
            'initializer':{'kind':'values','dtype':'float32','values':[[-.4],[.7]]}}}}}
    return {'id':'hmm','name':'Exact two-state Gaussian HMM','graph':graph}
