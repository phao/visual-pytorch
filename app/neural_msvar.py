"""Inspectable causal neural MS-VAR architecture and bounded synthetic study.

No labels or tensor computations are fabricated into authoring. The model learns
marginal sequence likelihood, conditional on its explicit initial observations.
"""
import math
import random


def drawing(*, steps=12, states=5, lags=3, dimensions=3, hidden=32):
    from .core import GraphError
    if any(type(v) is not int or not 1 <= v <= 4096 for v in (steps,states,lags,dimensions,hidden)):
        raise GraphError('Neural MS-VAR dimensions must be explicit positive bounded integers.')
    nodes=[{'id':'observations','type':'Input','label':'Observed sequence','shape':[steps,dimensions],'axes':['T','D']},
           {'id':'history','type':'Input','label':'Initial observed history','shape':[lags,dimensions],'axes':['lag','coordinate']},
           {'id':'lengths','type':'Input','label':'Valid prefix lengths','shape':[1]},
           {'id':'lags','type':'CausalLagWindow','label':'Past observations only','lags':lags,
            'inputs':{'observations':'observations','history':'history','lengths':'lengths'}},
           {'id':'transition_hidden','type':'Linear','label':'Transition features','features':hidden,'bias':True},
           {'id':'transition_gelu','type':'Activation','activation':'GELU','label':'GELU'},
           {'id':'transition_logits','type':'Linear','label':'Previous → next logits','features':states*states,'bias':True},
           {'id':'transition_shape','type':'Reshape','label':'Time × from-state × to-state','shape':[-1,states,states]},
           {'id':'transition','type':'Activation','label':'Next-state log probabilities','activation':'LogSoftmax','dim':2},
           {'id':'mean_linear','type':'Linear','label':'State-wise affine VAR','features':states*dimensions,'bias':True},
           {'id':'means','type':'Reshape','label':'Time × state × coordinate means','shape':[-1,states,dimensions]},
           {'id':'initial_logits','type':'TensorParameter','label':'Initial state-law logits','shape':[states],'axes':['state']},
           {'id':'initial','type':'Activation','label':'Before first transition','activation':'LogSoftmax','dim':0},
           {'id':'raw_variances','type':'TensorParameter','label':'State-wise variance parameters','shape':[states,dimensions],'axes':['state','coordinate']},
           {'id':'variances','type':'PositiveVariance','label':'Positive diagonal variances','positive_floor':.0001},
           {'id':'filter','type':'GaussianFilter','label':'Exact conditional likelihood/filter',
            'inputs':{r:r for r in ('observations','initial','transition','means','variances','lengths')},
            'initial_representation':'log-probabilities','transition_representation':'log-probabilities',
            'transition_mode':'sequence','mean_mode':'sequence','variance_mode':'shared'}]
    edges=[{'from':s,'to':'lags'} for s in ('observations','history','lengths')]
    chains=[('lags','transition_hidden','transition_gelu','transition_logits','transition_shape','transition'),
            ('lags','mean_linear','means'),('initial_logits','initial'),('raw_variances','variances')]
    for chain in chains:edges.extend({'from':a,'to':b} for a,b in zip(chain,chain[1:]))
    edges.extend({'from':r,'to':'filter'} for r in ('observations','initial','transition','means','variances','lengths'))
    for port in ('filtered','final','step_evidence','sequence_evidence'):
        nodes.append({'id':'out_'+port,'type':'Output','label':port})
        edges.append({'from':'filter','from_port':port,'to':'out_'+port})
    for node in ('transition','means','variances'):
        nodes.append({'id':'out_'+node,'type':'Output','label':node})
        edges.append({'from':node,'to':'out_'+node})
    positions={'observations':(30,40),'history':(30,180),'lengths':(30,320),'lags':(260,100),
        'transition_hidden':(490,20),'transition_gelu':(700,20),'transition_logits':(910,20),
        'transition_shape':(1120,20),'transition':(1330,20),'mean_linear':(490,170),
        'means':(720,170),'initial_logits':(490,340),'initial':(720,340),
        'raw_variances':(490,480),'variances':(720,480),'filter':(1570,180)}
    for i,node in enumerate(nodes):
        node.update(zip(('x','y'),positions.get(node['id'],(1810,20+130*(i-16)))))
    graph={'nodes':nodes,'edges':edges,'input_order':['observations','history','lengths'],
           'parameter_policy':{'version':1,'parameters':{'["raw_variances","value"]':{
               'initializer':{'kind':'values','dtype':'float32','values':[[-1.]*dimensions for _ in range(states)]}}}}}
    return {'id':'neural-msvar','name':f'Neural MS-VAR — K={states}, p={lags}, D={dimensions}','graph':graph}


def dataset(*, trajectories=24, steps=12, states=5, lags=3, dimensions=3, seed=23):
    """Independent trajectories, private scalar RNG, no true regime targets.

    Teacher transitions are a bounded nonlinear function of causal lag features;
    the student drawing implements its own learned 9→32→25 transition network.
    This small synthetic study does not establish identifiability or calibration.
    """
    rng=random.Random(seed);records=[]
    for i in range(trajectories):
        history=[[rng.gauss(0,.15) for _ in range(dimensions)] for _ in range(lags)]
        past=[list(row) for row in history];state=rng.randrange(states);values=[]
        valid=steps-i%3
        for t in range(valid):
            features=[v for row in past[-lags:] for v in row]
            logits=[(1.2 if k==state else 0.) + .25*math.tanh(features[(k+state)%len(features)]) for k in range(states)]
            weights=[math.exp(v-max(logits)) for v in logits];point=rng.random()*sum(weights);total=0.
            for k,w in enumerate(weights):
                total+=w
                if point<=total:state=k;break
            center=(state-(states-1)/2)*.15
            value=[center*(1-.15*d)+.25*past[-1][d]-.1*past[-2 if lags>1 else -1][d]+rng.gauss(0,.12+.01*state)
                   for d in range(dimensions)]
            values.append(value);past.append(value)
        records.append({'id':'trajectory-'+str(i),'values':{'observations':values+[[0.]*dimensions for _ in range(steps-valid)],
            'history':history,'lengths':[float(valid)],'trajectory':'subject-'+str(i)}})
    return {'version':1,'id':'msvar-observations','name':'Unlabelled synthetic trajectories',
        'storage':{'kind':'inline-records','version':1},
        'fields':[{'id':'observations','dtype':'float32','shape':[steps,dimensions],'axes':['T','D']},
                  {'id':'history','dtype':'float32','shape':[lags,dimensions],'axes':['lag','coordinate']},
                  {'id':'lengths','dtype':'float32','shape':[1]},{'id':'trajectory','dtype':'string','shape':[]}],
        'records':records,'provenance':{'kind':'bounded-synthetic-check','seed':seed,'unit':'trajectory',
            'regime_labels':'not exposed as targets','conditioning':'explicit oldest-to-newest initial observed history',
            'limitations':'No identifiability, stationarity or calibration claim; teacher and fitted network are distinct.'}}


def likelihood_task():
    return {'version':2,'id':'msvar-likelihood','name':'Conditional marginal sequence likelihood',
        'inputs':[{'port':p,'field':p,'transforms':[]} for p in ('observations','history','lengths')],
        'heads':[{'id':'sequence-nll','port':'out_sequence_evidence','loss':'negative-log-likelihood','weight':1.}],
        'diagnostic_outputs':['out_filtered','out_final','out_step_evidence','out_transition','out_means','out_variances']}
