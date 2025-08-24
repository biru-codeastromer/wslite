from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np

def _synth(seed=42,n=1000):
    rng=np.random.RandomState(seed)
    autos=['car','engine','vehicle','auto','ford','bmw','honda','toyota','speed','wheel','brake','seat','road','fuel','garage','tyre','steering','clutch','gear','drive']
    med=['health','doctor','disease','medical','patient','treatment','clinic','drug','surgery','blood','nurse','hospital','virus','vaccine','symptom','therapy','scan','xray','diagnosis','medicine']
    def gen(ws):
        k=rng.randint(6,13)
        return " ".join(rng.choice(ws,size=k))
    X=[gen(autos) for _ in range(n//2)]+[gen(med) for _ in range(n//2)]
    y=[0]*(n//2)+[1]*(n//2)
    idx=rng.permutation(len(X))
    X=[X[i] for i in idx]
    y=[int(y[i]) for i in idx]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=seed,stratify=y)
    return Xtr,Xte,ytr,yte

def load_data(seed=42,max_n=2000):
    cats=['rec.autos','sci.med']
    try:
        data=fetch_20newsgroups(subset='train',categories=cats,remove=('headers','footers','quotes'))
        X=data.data
        y=data.target
        if max_n and len(X)>max_n:
            rng=np.random.RandomState(seed)
            idx=rng.choice(len(X),size=max_n,replace=False)
            X=[X[i] for i in idx]
            y=[int(y[i]) for i in idx]
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=seed,stratify=y)
        return Xtr,Xte,ytr,yte
    except Exception:
        return _synth(seed=seed,n=1000 if not max_n else max_n)
