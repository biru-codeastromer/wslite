from dataclasses import dataclass
import re

ABSTAIN=-1

@dataclass
class LabelFunction:
    name:str
    f:callable
    def __call__(self,x):
        return self.f(x)

def lf_keywords(label,words):
    pat=re.compile(r"\b("+("|".join(re.escape(w) for w in words))+r")\b",re.I)
    def g(x):
        return label if pat.search(x or "") else ABSTAIN
    return g

def default_lfs():
    autos=['car','engine','vehicle','auto','ford','bmw','honda','toyota','speed','wheel']
    med=['health','doctor','disease','medical','patient','treatment','clinic','drug','surgery','blood']
    lfs=[LabelFunction('autos_kw1',lf_keywords(0,autos[:5])),
         LabelFunction('autos_kw2',lf_keywords(0,autos[5:])),
         LabelFunction('med_kw1',lf_keywords(1,med[:5])),
         LabelFunction('med_kw2',lf_keywords(1,med[5:]))]
    return lfs
