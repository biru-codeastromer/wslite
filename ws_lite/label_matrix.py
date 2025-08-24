import numpy as np
from .lfs import ABSTAIN

def apply_lfs(texts,lfs):
    n=len(texts)
    m=len(lfs)
    L=np.full((n,m),ABSTAIN,dtype=int)
    for i,x in enumerate(texts):
        for j,lf in enumerate(lfs):
            L[i,j]=lf(x)
    return L
