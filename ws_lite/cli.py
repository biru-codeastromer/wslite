import argparse, json, numpy as np
from .lfs import default_lfs, ABSTAIN
from .label_matrix import apply_lfs
from .gen_model import majority_vote, one_step_em
from .data import load_data
from .train import train_ws, train_small_gold

def run(seed=42):
    Xtr,Xte,ytr,yte=load_data(seed=seed,max_n=2000)
    lfs=default_lfs()
    Ltr=apply_lfs(Xtr,lfs)
    Lte=apply_lfs(Xte,lfs)
    labels=[0,1]
    mv_tr=majority_vote(Ltr,labels)
    mv_te=majority_vote(Lte,labels)
    em_tr,w=one_step_em(Ltr,labels)
    em_te,_=one_step_em(Lte,labels)
    mv_acc=np.mean([int(mv_te[i]==yte[i]) for i in range(len(yte)) if mv_te[i]!=ABSTAIN])
    em_acc=np.mean([int(em_te[i]==yte[i]) for i in range(len(yte)) if em_te[i]!=ABSTAIN])
    ws_acc=train_ws(Xtr,em_tr,Xte,yte,seed=seed)
    gold_small=train_small_gold(Xtr,ytr,Xte,yte,k=200,seed=seed)
    out={"majority_vote_acc":float(mv_acc) if mv_acc==mv_acc else None,
         "weighted_vote_acc":float(em_acc) if em_acc==em_acc else None,
         "ws_classifier_acc":float(ws_acc),
         "small_gold_classifier_acc":float(gold_small)}
    print(json.dumps(out,indent=2))

def run_ablation(seed=42):
    Xtr,Xte,ytr,yte=load_data(seed=seed,max_n=2000)
    lfs=default_lfs()
    labels=[0,1]
    base_te,_=one_step_em(apply_lfs(Xte,lfs),labels)
    base_acc=np.mean([int(base_te[i]==yte[i]) for i in range(len(yte)) if base_te[i]!=-1])
    deltas=[]
    for k in range(len(lfs)):
        l=[lf for i,lf in enumerate(lfs) if i!=k]
        te,_=one_step_em(apply_lfs(Xte,l),labels)
        acc=np.mean([int(te[i]==yte[i]) for i in range(len(yte)) if te[i]!=-1])
        deltas.append((lfs[k].name,float(base_acc-acc if acc==acc and base_acc==base_acc else 0.0)))
    print(json.dumps({"weighted_vote_acc":float(base_acc) if base_acc==base_acc else None,"leave_one_out_drop":deltas},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("cmd",choices=["reproduce","ablation"])
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.cmd=="reproduce":
        run(seed=a.seed)
    else:
        run_ablation(seed=a.seed)

if __name__=="__main__":
    main()
