"""Matched inherited-law comparison: independently optimized restricted heads
versus post-training component swaps out of the jointly optimized high model.
"""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.linalg import solve
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from marketlab.scenarios import diagnostic_policies
from marketlab.simulation import simulate
from marketlab.learning import ResourceForecaster,Normalization
from marketlab.deployment import raw_heads_on_inherited_history,frozen_price_panel
from features import design_matrix
from experiment import calibrated_economy,combine_heads

def restore(arch,cal,seed,arm):
    data=torch.load(ROOT/'models'/f'{arch}_v{int(cal*100):03d}_seed{seed}_arm{arm}.pt',map_location='cpu',weights_only=True)
    norm=Normalization.from_dict(data['norm'])
    model=ResourceForecaster(norm.u_mean.size,data['hidden'],data['arch'],data['head_type'],norm,data['window'])
    model.load_state_dict(data['state']);model.eval()
    return model

def evaluate(eco,tpl,teacher,optimized,bank):
    panel=simulate(eco,tpl,bank)
    old=raw_heads_on_inherited_history(teacher,panel.public_input)
    hs={a:raw_heads_on_inherited_history(model,panel.public_input) for a,model in optimized.items()}
    out={'panel':panel}
    combos={'base':old,
            'opt_mean':combine_heads('10',old,hs['10']),
            'opt_risk':combine_heads('01',old,hs['01']),
            'swap_mean':combine_heads('10',old,hs['11']),
            'swap_risk':combine_heads('01',old,hs['11']),
            'joint':hs['11']}
    for arm,v in combos.items():
        info=frozen_price_panel(eco,tpl,panel,0,v)
        ii=slice(32,None)
        mu=v[ii,0]+v[ii,1]*panel.private_signals[ii,0]+v[ii,2]*panel.price[ii]
        error=panel.future_dividend[ii]-mu
        info['old_market_NLL']=float(.5*np.mean(np.log(2*np.pi*v[ii,3])+error**2/v[ii,3]))
        out[arm]=info
    return out

def mse_fit(X,Z,yt,yv,regularization=1e-5):
    mx=X.mean(0);sx=np.maximum(X.std(0),1e-7)
    X=(X-mx)/sx;Z=(Z-mx)/sx
    co=solve(X.T@X+regularization*len(X)*np.eye(X.shape[1]),X.T@(yt-yt.mean()),assume_a='pos')
    return float(np.mean((yv-yt.mean()-Z@co)**2))

def run(calibrations=(.7,1.,1.3),architectures=('ssm','gru'),seeds=(11,23,37),pairs=2,periods=4200):
    torch.set_num_threads(1); rows=[]
    for cal in calibrations:
        eco=calibrated_economy(cal);tpl=diagnostic_policies(eco,'E3_state_risk')
        for arch in architectures:
            for seed in seeds:
                lo=restore(arch,cal,seed,'00')
                hi={a:restore(arch,cal,seed,a) for a in ('10','01','11')}
                for bank in range(pairs):
                    x=evaluate(eco,tpl,lo,hi,eco.draw_bank(periods,burn=500,seed=200000+int(cal*1000)*127+bank*727))
                    z=evaluate(eco,tpl,lo,hi,eco.draw_bank(periods,burn=500,seed=300000+int(cal*1000)*127+bank*727))
                    L=3;ut=design_matrix(x['panel'],L);uv=design_matrix(z['panel'],L)
                    yt=x['panel'].target[L:];yv=z['panel'].target[L:]
                    pb=mse_fit(ut,uv,yt,yv)
                    for arm in ('base','opt_mean','opt_risk','swap_mean','swap_risk','joint'):
                        score=1-mse_fit(np.c_[ut,x[arm]['price'][L:]],np.c_[uv,z[arm]['price'][L:]],yt,yv)/pb
                        rows.append({'calibration':cal,'arch':arch,'seed':seed,'bank':bank,'arm':arm,
                                     'decoder_gain':score,'NLL_old_market':z[arm]['old_market_NLL'],
                                     'max_clearing_error':z[arm]['max_clearing_residual']})
                print('COMPARE',cal,arch,seed,flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results'/'swap_vs_reoptimized.csv',index=False)
    diffs=[]
    for key,g in df.groupby(['calibration','arch','seed','bank']):
        q=g.set_index('arm')
        diffs.append(dict(zip(['calibration','arch','seed','bank'],key))|
           {col:float(100*(q.loc[a,'decoder_gain']-q.loc['base','decoder_gain']))
            for col,a in [('opt_mean_pp','opt_mean'),('swap_mean_pp','swap_mean'),
                           ('opt_risk_pp','opt_risk'),('swap_risk_pp','swap_risk'),('joint_pp','joint')]})
    o=pd.DataFrame(diffs);o.to_csv(ROOT/'results'/'swap_vs_reoptimized_effects.csv',index=False)
    print(o.groupby('arch').mean(numeric_only=True).round(5).to_string(),flush=True)
    return df
if __name__=='__main__': run()
