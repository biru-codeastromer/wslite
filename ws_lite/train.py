import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_ws(texts,labels,texts_te,y_te,seed=42):
    mask=[i for i,v in enumerate(labels) if v!=-1]
    Xt=[texts[i] for i in mask]
    yt=[labels[i] for i in mask]
    vec=TfidfVectorizer(max_features=20000,ngram_range=(1,2))
    Xtr=vec.fit_transform(Xt)
    clf=LogisticRegression(max_iter=1000,random_state=seed)
    clf.fit(Xtr,yt)
    Xte=vec.transform(texts_te)
    yp=clf.predict(Xte)
    acc=accuracy_score(y_te,yp)
    return acc

def train_small_gold(texts,y,texts_te,y_te,k=200,seed=42):
    rng=np.random.RandomState(seed)
    idx=rng.choice(len(texts),size=min(k,len(texts)),replace=False)
    Xt=[texts[i] for i in idx]
    yt=[int(y[i]) for i in idx]
    vec=TfidfVectorizer(max_features=20000,ngram_range=(1,2))
    Xtr=vec.fit_transform(Xt)
    clf=LogisticRegression(max_iter=1000,random_state=seed)
    clf.fit(Xtr,yt)
    Xte=vec.transform(texts_te)
    yp=clf.predict(Xte)
    return accuracy_score(y_te,yp)
