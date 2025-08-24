import numpy as np
from .lfs import ABSTAIN

def majority_vote(L,labels):
    y=np.full(L.shape[0],ABSTAIN,dtype=int)
    for i in range(L.shape[0]):
        row=L[i]
        vals=row[row!=ABSTAIN]
        if vals.size==0:
            continue
        counts=[np.sum(vals==lab) for lab in labels]
        if max(counts)==0 or counts.count(max(counts))>1:
            continue
        y[i]=labels[int(np.argmax(counts))]
    return y

def weighted_vote(L,weights,labels):
    y=np.full(L.shape[0],ABSTAIN,dtype=int)
    for i in range(L.shape[0]):
        s=[0.0 for _ in labels]
        for j in range(L.shape[1]):
            v=L[i,j]
            if v==ABSTAIN:
                continue
            idx=labels.index(int(v))
            s[idx]+=weights[j]
        if max(s)==0 or s.count(max(s))>1:
            continue
        y[i]=labels[int(np.argmax(s))]
    return y

def one_step_em(L,labels):
    y0=majority_vote(L,labels)
    w=np.ones(L.shape[1],dtype=float)
    for j in range(L.shape[1]):
        mask=(L[:,j]!=ABSTAIN)&(y0!=ABSTAIN)
        if np.sum(mask)==0:
            w[j]=1.0
        else:
            acc=np.mean(L[mask,j]==y0[mask])
            eps=1e-3
            acc=max(min(acc,1-eps),eps)
            w[j]=np.log(acc/(1-acc))
    y1=weighted_vote(L,w,labels)
    return y1,w
