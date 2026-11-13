"""Two-stage factorial E3 ablations: one-shot inherited histories vs endogenous frozen-market stationary adjustment.
No retraining after mean/variance swaps. The learned B-heads are fixed, not reoptimized.
"""
from __future__ import annotations
import copy, json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'phase3_long_validation'))
from marketlab.scenarios import economy_8d,diagnostic_policies
from marketlab.learning import load_checkpoint
from marketlab.fast_inference import FastFiniteContextEncoder
from marketlab.retraining import model_policy
from marketlab.policies import InvestorPolicy
from marketlab.simulation import simulate
from marketlab.deployment import PrecomputedTorchHead
from neural_head_ablation import oos_probe

class HybridEncoder:
    def __init__(self,lo,hi,mean_high,variance_high):
        self.lo_model=lo;self.hi_model=hi;self.mean_high=mean_high;self.variance_high=variance_high
        self.reset()
    def reset(self):
        self.lo=FastFiniteContextEncoder(self.lo_model)
        self.hi=FastFiniteContextEncoder(self.hi_model)
    def __deepcopy__(self,memo):
        return HybridEncoder(self.lo_model,self.hi_model,self.mean_high,self.variance_high)
    def step(self,u):
        low=self.lo.step(u)
        if not (self.mean_high or self.variance_high):return low
        high=self.hi.step(u)
        result=low.copy()
        if self.mean_high:result[:3]=high[:3]
        if self.variance_high:result[3]=high[3]
        return result

def run(periods=3200,pairs=2):
    torch.set_num_threads(1)
    eco=economy_8d('E3_state_risk')
    tpl=diagnostic_policies(eco,'E3_state_risk')
    saved=ROOT/'vendor'/'phase3_long_validation'/'models'
    rows=[]
    for arch in ('ssm','gru'):
        tag=f'E3_state_risk_{arch}_headB_seed11'
        lo=[load_checkpoint(saved/f'{tag}_baseline_ep10_type{i}.pt')[0] for i in range(3)]
        hi=load_checkpoint(saved/f'{tag}_high_impact.pt')[0]
        pol_rest=[model_policy(tpl[i],lo[i]) for i in (1,2)]
        for bankid in range(pairs):
            banks=[eco.draw_bank(periods=periods,burn=550,seed=614010+bankid*113),eco.draw_bank(periods=periods,burn=550,seed=719010+bankid*113)]
            panels={}
            for arm in ('00','10','01','11'):
                enc=HybridEncoder(lo[0],hi,arm[0]=='1',arm[1]=='1')
                type0=InvestorPolicy(tpl[0].name,tpl[0].mass,tpl[0].gamma,tpl[0].signal_loading,tpl[0].signal_noise_sd,enc,PrecomputedTorchHead())
                pp=[simulate(eco,[type0]+pol_rest,bank) for bank in banks]
                for p in pp:
                    assert np.max(np.abs(p.demand_residual))<2e-8
                    assert np.min(p.clearing_slope)>0
                panels[arm]=pp
            # oos_probe receives arm dictionary, and expects panel to be 'panel'; rebuild each candidate with panel as inherited info.
            from features import design_matrix
            from scipy.linalg import solve
            gains={}
            for arm in panels:
                tr,te=panels[arm];L=3
                Xtr=design_matrix(tr,L);Xte=design_matrix(te,L)
                ytr=tr.target[L:]; yte=te.target[L:]
                def risk(X,Z):
                    mu=X.mean(axis=0);sd=np.maximum(X.std(axis=0),1e-8)
                    xx=(X-mu)/sd;zz=(Z-mu)/sd
                    b=solve(xx.T@xx+1e-5*len(xx)*np.eye(xx.shape[1]),xx.T@(ytr-ytr.mean()),assume_a='pos')
                    return np.mean((yte-ytr.mean()-zz@b)**2)
                mse_pub=risk(Xtr,Xte)
                mse_join=risk(np.c_[Xtr,tr.price[L:]],np.c_[Xte,te.price[L:]])
                gain=1-mse_join/mse_pub
                rows.append({'arch':arch,'bank':bankid,'arm':arm,'OOS_decoder_gain':gain,
                    'OOS_joint_MSE':mse_join,'OOS_public_MSE':mse_pub,
                    'private_demand_beta_type0':np.mean(tpl[0].mass*te.head_parameters[:,0,1]/(tpl[0].gamma*te.head_parameters[:,0,3])),
                    'mean_var_type0':np.mean(te.head_parameters[:,0,3]),
                    'sd_var_type0':np.std(te.head_parameters[:,0,3]),
                    'maximum_clearing_error':float(np.max(np.abs(te.demand_residual))),'min_demand_slope':float(np.min(te.clearing_slope))})
                gains[arm]=gain
            print(arch,bankid,'frozen stationary gain %',{k:round(100*v,3) for k,v in gains.items()},'joint shift pp',round(100*(gains['11']-gains['00']),3),flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'frozen_stationary_arms.csv',index=False)
    return rows
if __name__=='__main__':run()
