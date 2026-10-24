"""The five compositional probe tasks and their case generator (plan knowledge-v3.1, Gate B; moved here from the
removed V3 probe script by plan remove-old-generations).

`sample` builds a case in the probes' compact token layout (kind, blueprint, zone, owner, slot, parent, count, flags);
`probe_efficient.sample` converts it into a schema-4 item (grid, context, tables).
"""
import numpy as np

TASKS=('attachment','multiplicity','scope','reference_binding','printed_comparison')

def identities(rng, start, stop, n, heldout):
    # Reserve combinations, not labels or individual blueprints. The split is
    # stable across seeds and independent of option order and target placement.
    while True:
        ids = rng.choice(np.arange(start, stop), n, replace=False)
        key = sum((i+1)*int(bp) for i,bp in enumerate(sorted(ids))) % 4
        if (key == 0) == heldout:
            return ids

def sample(rng, task, heldout, features):
    # Card identities do not encode labels: source placement, scopes and reference
    # routing are randomized. Printed comparisons evaluate unseen blueprint IDs.
    tokens=[];floats=[]
    def row(kind,bp=0,slot=0,parent=0,count=1,flags=0,a=0,b=0):
        tokens.append([kind,bp,0,1,slot,parent,count,flags]);floats.append([a,b,0,0]);return len(tokens)
    head=np.array([4,task+1,0,0]+[0]*18,dtype=np.int32)
    opt=np.array([[1,0,0,0,0,0],[1,0,0,0,0,0]],dtype=np.int32)
    winner=int(rng.integers(2))
    if task in (0,1):
        targets=[row(0,int(bp)) for bp in identities(rng,2,8,2,heldout)]
        opt[:,2]=targets
        if task==0:
            pilot=row(0,8);row(8,slot=3,parent=pilot,a=targets[winner])
            # Same two entities; disjoint nuisance pilots/blueprints in held-out graphs.
            for _ in range(int(rng.integers(1,4 if heldout else 2))):row(0,int(rng.integers(9,14)))
        else:
            low=int(rng.integers(4,7) if heldout else rng.integers(1,3));counts=[low,low];counts[winner]+=int(rng.integers(1,3))
            for target,n in zip(targets,counts):
                for _ in range(n):pilot=row(0,8);row(8,slot=3,parent=pilot,a=target)
    elif task==2:
        a,b=identities(rng,2,14,2,heldout);opt[:,3]=[a,b]
        # Separate normalized questions ask for OUR or PUBLIC membership.
        scope=int(rng.integers(1,3));head[1]=3 if scope==1 else 6
        for j,bp in enumerate((a,b)):
            for sc in (1,2):
                present=(j==winner) if sc==scope else (j!=winner)
                row(2,int(bp),count=int(present),flags=(sc<<1)|1,b=int(present))
    elif task==3:
        refs=identities(rng,2,16,3,heldout);row(9,int(refs[0]),slot=1)
        # Repeated first references cannot solve the question: only second matters.
        for j in range(2):
            row(10,int(refs[1]),slot=1,a=j+1)
            row(10,int(refs[0] if j==winner else refs[2]),slot=2,a=j+1)
    else:
        choices=np.arange(64,128) if heldout else np.arange(2,64)
        a,b=rng.choice(choices,2,replace=False);opt[:,3]=[a,b]
        winner=int(features[b,0]>features[a,0]);row(0)
    # Entity/token row permutation while updating actual source/target routes.
    perm=rng.permutation(len(tokens));inv=np.argsort(perm)
    t=np.array(tokens,dtype=np.int16)[perm];f=np.array(floats,dtype=np.float32)[perm]
    edge=t[:,0]==8;t[edge,5]=inv[t[edge,5]-1]+1;f[edge,0]=inv[f[edge,0].astype(int)-1]+1
    sources=opt[:,2]>0;opt[sources,2]=inv[opt[sources,2]-1]+1
    return dict(tok=t,tokf=f,opt=opt,head=head),winner
