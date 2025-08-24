import streamlit as st, numpy as np, pandas as pd
from ws_lite.data import load_data
from ws_lite.lfs import default_lfs, ABSTAIN
from ws_lite.label_matrix import apply_lfs
from ws_lite.gen_model import majority_vote, one_step_em
from ws_lite.train import train_ws, train_small_gold

st.set_page_config(page_title="WS-Lite", page_icon="🧪", layout="centered")

@st.cache_data
def run_pipeline(seed, gold_k):
    Xtr,Xte,ytr,yte=load_data(seed=seed,max_n=2000)
    lfs=default_lfs()
    Ltr=apply_lfs(Xtr,lfs)
    Lte=apply_lfs(Xte,lfs)
    labels=[0,1]
    mv_te=majority_vote(Lte,labels)
    em_tr,w=one_step_em(Ltr,labels)
    em_te,_=one_step_em(Lte,labels)
    mv_acc=np.mean([int(mv_te[i]==yte[i]) for i in range(len(yte)) if mv_te[i]!=ABSTAIN])
    em_acc=np.mean([int(em_te[i]==yte[i]) for i in range(len(yte)) if em_te[i]!=ABSTAIN])
    ws_acc=train_ws(Xtr,em_tr,Xte,yte,seed=seed)
    gold_small=train_small_gold(Xtr,ytr,Xte,yte,k=gold_k,seed=seed)
    drops=[]
    base_te,_=one_step_em(Lte,labels)
    base_acc=np.mean([int(base_te[i]==yte[i]) for i in range(len(yte)) if base_te[i]!=ABSTAIN])
    for k in range(len(lfs)):
        ll=[lf for i,lf in enumerate(lfs) if i!=k]
        te,_=one_step_em(apply_lfs(Xte,ll),labels)
        acc=np.mean([int(te[i]==yte[i]) for i in range(len(yte)) if te[i]!=ABSTAIN])
        d=float(base_acc-acc if acc==acc and base_acc==base_acc else 0.0)
        drops.append((lfs[k].name,d))
    metrics={"majority_vote_acc":None if mv_acc!=mv_acc else float(mv_acc),
             "weighted_vote_acc":None if em_acc!=em_acc else float(em_acc),
             "ws_classifier_acc":float(ws_acc),
             "small_gold_classifier_acc":float(gold_small)}
    return metrics, pd.DataFrame(drops, columns=["label_function","acc_drop"])

st.title("WS-Lite")
seed=st.sidebar.number_input("seed", value=42, step=1)
gold_k=st.sidebar.number_input("small gold size", value=200, step=50, min_value=50, max_value=1000)
if st.button("run"):
    metrics, ablation = run_pipeline(seed, gold_k)
    st.subheader("metrics")
    st.json(metrics)
    st.subheader("leave-one-LF-out ablation")
    st.dataframe(ablation, hide_index=True)
else:
    st.info("set seed and small-gold size, then click run")
