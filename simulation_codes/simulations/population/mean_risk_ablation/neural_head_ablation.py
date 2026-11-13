"""State-dependent head ablations in E3, matched inherited histories.

4 factorial policy arms: old/new forecast mean (a,beta,chi) x old/new risk forecast v.
Each old/new vector is from saved learned head B checkpoints trained at SAME inherited Q0.
Hybrid arms are OFF-OPTIMUM interventions, not independently trained optimal AI.
A linear OOS external decoder is scored on independent exogenous shock banks.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parent
LIB=ROOT/'vendor'/'phase3_long_validation'
sys.path.insert(0,str(LIB))
sys.path.insert(0,str(ROOT/'vendor'/'task36_src'))
from marketlab.scenarios import economy_8d,diagnostic_policies
from marketlab.learning import load_checkpoint
from marketlab.retraining import stationary_panel
from marketlab.deployment import frozen_price_panel,raw_heads_on_inherited_history
from features import design_matrix


def oos_probe(train,test,L=3,regularization=1e-5):
    """Train augmented and public-only ridge OLS on an independent shock bank."""
    (uTr,uTe)=(design_matrix(train['panel'],L),design_matrix(test['panel'],L))
    yTr=train['panel'].target[L:];yTe=test['panel'].target[L:]
    def fit(X,Z):
        m=X.mean(axis=0);s=np.maximum(X.std(axis=0),1e-8)
        xx=(X-m)/s;zz=(Z-m)/s
        G=xx.T@xx+regularization*len(xx)*np.eye(xx.shape[1]);g=xx.T@(yTr-yTr.mean())
        beta=solve(G,g,assume_a='pos');yhat=yTr.mean()+zz@beta
        return float(np.mean((yTe-yhat)**2))
    baseline=fit(uTr,uTe)
    vals={}
    for arm in ('00','10','01','11'):
        X=np.column_stack([uTr,train[arm]['price'][L:]])
        Z=np.column_stack([uTe,test[arm]['price'][L:]])
        mse=fit(X,Z)
        vals[arm]={'MSE':mse,'gain':1-mse/baseline,'risk_reduction':baseline-mse}
    return vals,baseline


def prepare(eco,tpl,models,high,bank):
    inherited=stationary_panel(eco,tpl,models,bank)
    new_heads=raw_heads_on_inherited_history(high,inherited.public_input)
    old_heads=inherited.head_parameters[:,0,:]
    arms={'panel':inherited}
    for mean in (0,1):
        for var in (0,1):
            combo=old_heads.copy()
            if mean:combo[:,0:3]=np.where(np.isfinite(new_heads[:,0:3]),new_heads[:,0:3],combo[:,0:3])
            if var:combo[:,3]=np.where(np.isfinite(new_heads[:,3]),new_heads[:,3],combo[:,3])
            label=f'{mean}{var}'
            info=frozen_price_panel(eco,tpl,inherited,0,combo)
            if label=='00':
                assert np.max(np.abs(info['price']-inherited.price))<1e-9, 'Old-head factorial baseline must replicate inherited auction'
            beta_contribution=np.mean(tpl[0].mass/(tpl[0].gamma*combo[32:,3])*combo[32:,1])
            info['beta_contribution_type0']=float(beta_contribution)
            info['avg_subjective_variance_type0']=float(np.mean(combo[32:,3]))
            info['std_subjective_variance_type0']=float(np.std(combo[32:,3]))
            # NLL evaluated on the SAME inherited market law, BEFORE price is recomputed.
            mu=combo[32:,0]+combo[32:,1]*inherited.private_signals[32:,0]+combo[32:,2]*inherited.price[32:]
            resid=inherited.future_dividend[32:]-mu
            info['inherited_NLL_type0']=float(.5*np.mean(np.log(2*np.pi*combo[32:,3])+resid**2/combo[32:,3]))
            arms[label]=info
    return arms

def run(architectures=('ssm','gru'),pairs=4,periods=4800,L=3):
    torch.set_num_threads(1)
    eco=economy_8d('E3_state_risk');tpl=diagnostic_policies(eco,'E3_state_risk')
    model_folder=LIB/'models'
    rows=[];squares=[]
    for arch in architectures:
        tag=f'E3_state_risk_{arch}_headB_seed11'
        models=[load_checkpoint(model_folder/f'{tag}_baseline_ep10_type{i}.pt')[0] for i in range(len(tpl))]
        high=load_checkpoint(model_folder/f'{tag}_high_impact.pt')[0]
        print('ARM',arch,'checkpoints loaded',flush=True)
        for banknum in range(pairs):
            trainbank=eco.draw_bank(periods=periods,burn=480,seed=881100+banknum*41)
            testbank=eco.draw_bank(periods=periods,burn=480,seed=982300+banknum*41)
            tr=prepare(eco,tpl,models,high,trainbank)
            te=prepare(eco,tpl,models,high,testbank)
            vals,mse0=oos_probe(tr,te,L=L)
            allrows=[]
            for a in ('00','10','01','11'):
                row={'arch':arch,'bank':banknum,'arm':a,'mean_high':int(a[0]),'variance_high':int(a[1]),
                  'OOS_decoder_gain':vals[a]['gain'],'OOS_decoder_MSE':vals[a]['MSE'],
                  'OOS_public_only_MSE':mse0,'OOS_decoder_risk_reduction':vals[a]['risk_reduction'],
                  'inherited_NLL_type0':te[a]['inherited_NLL_type0'],
                  'demand_private_loading_type0':te[a]['beta_contribution_type0'],
                  'mean_variance_type0':te[a]['avg_subjective_variance_type0'],
                  'std_variance_type0':te[a]['std_subjective_variance_type0'],
                  'max_clearing_residual':te[a]['max_clearing_residual']}
                rows.append(row);allrows.append(row)
            dd={a:vals[a]['gain'] for a in vals}
            shap_mean=.5*((dd['10']-dd['00'])+(dd['11']-dd['01']))
            shap_var=.5*((dd['01']-dd['00'])+(dd['11']-dd['10']))
            sq={'arch':arch,'bank':banknum,'mean_effect':shap_mean,'var_effect':shap_var,
                'interaction':dd['11']-dd['10']-dd['01']+dd['00'],
                'joint_effect':dd['11']-dd['00'],'mean_only_effect':dd['10']-dd['00'],'var_only_effect':dd['01']-dd['00']}
            squares.append(sq)
            print(' ',arch,'bank',banknum,'linear decoder', {k:round(100*v,3) for k,v in dd.items()},
                  'shapley p.p.',round(100*shap_mean,3),round(100*shap_var,3),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'neural_head_arms.csv',index=False)
    pd.DataFrame(squares).to_csv(ROOT/'neural_factorial_effects.csv',index=False)
    result={'protocol':'Cross-swapped saved population-approximate neural heads; inherited Q0; independent shock banks; every trial uses the same baseline and frozen public history.',
            'econ':'8d E3 state-dependent volatility with 3 atomistic CARA types; risk head depends on pre-auction public history',
            'arms':'00=old mean old variance,10=new mean old variance,01=old mean new variance,11=new mean new variance',
            'learned_heads_are_not_retrained_after_swap':True,
            'metrics':'Out-of-sample incremental linear decoder gain using pre-auction macro+3 lag inputs; NOT mutual information.',
            'summary':{arch:{col:{'mean':float(g[col].mean()),'sd':float(g[col].std())} for col in ('mean_effect','var_effect','interaction','joint_effect','mean_only_effect','var_only_effect')} for arch,g in pd.DataFrame(squares).groupby('arch')}}
    (ROOT/'neural_results.json').write_text(json.dumps(result,indent=2))
    return rows,squares

if __name__=='__main__':run()
