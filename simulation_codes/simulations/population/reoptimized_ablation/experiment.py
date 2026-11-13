"""Independently reoptimized mean-only, risk-only, and joint heads.

All comparisons within a cell share frozen inherited Q0 training/evaluation laws.
Each restrictive training arm optimizes its own Gaussian NLL, holding the
other forecast component equal to the LOW-capacity forecaster output.
Not a population optimum; old market remains a diagnostic incumbent Q0.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys,copy,json,time,math,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.linalg import solve
from dataclasses import replace
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'))
from marketlab.scenarios import economy_8d, diagnostic_policies
from marketlab.simulation import simulate
from marketlab.learning import (ResourceForecaster,Normalization,prepare_panel,gaussian_loss,
    _init_ols,set_seed,fit_forecaster,make_windows)
from marketlab.deployment import raw_heads_on_inherited_history, frozen_price_panel
from features import design_matrix

ARMS={'00':(False,False),'10':(True,False),'01':(False,True),'11':(True,True)}

def calibrated_economy(multiplier):
    src=economy_8d('E3_state_risk'); assert src.volatility_loading is not None
    return replace(src,name=f'E3_vol{int(round(multiplier*100))}',
        volatility_loading=np.asarray(src.volatility_loading)*float(multiplier))

def frozen_teacher_predictions(teacher,data,subset_indices=None,chunk=256):
    teacher.eval()
    n=len(data['y']); out_mu=np.empty(n,dtype=np.float32);out_lv=np.empty(n,dtype=np.float32)
    with torch.no_grad():
        for a in range(0,n,chunk):
            j=slice(a,min(a+chunk,n))
            mu,lv=teacher(data['windows'][j],data['s'][j],data['p'][j])
            out_mu[j]=mu.numpy();out_lv[j]=lv.numpy()
    return torch.from_numpy(out_mu),torch.from_numpy(out_lv)

def _predict_selected(model,data,teacher_mu,teacher_lv,ids,learn_mean,learn_risk):
    mu,lv=model(data['windows'][ids],data['s'][ids],data['p'][ids])
    if not learn_mean:mu=teacher_mu[ids]
    if not learn_risk:lv=teacher_lv[ids]
    return mu,lv

def train_restricted(panel,teacher,arch,hidden,arm,seed,steps=240,window=32,
                     batch=128,patience=5,lr=.006):
    assert arm in ('10','01','11')
    set_seed(seed)
    data=prepare_panel(panel,0,window)
    assert data['norm'].to_dict()==teacher.norm.to_dict(), 'teacher and student must share normalization'
    high=ResourceForecaster(panel['public_input'].shape[1],hidden,arch,'B',data['norm'],window)
    _init_ols(high,data)
    teacher_mu,teacher_lv=frozen_teacher_predictions(teacher,data)
    learn_mean,learn_risk=ARMS[arm]
    # No weight update for unused output blocks; encoder is trained on the active loss.
    opt=torch.optim.Adam(high.parameters(),lr=lr,weight_decay=1e-5)
    rng=np.random.default_rng(seed+1103)
    best=np.inf; best_state=None; stagnation=0; curve=[]; t0=time.monotonic()
    tr=data['train_idx'];va=data['val_idx']; va=va[::max(1,len(va)//450)]
    for step in range(1,steps+1):
        ids=rng.choice(tr,size=min(batch,len(tr)),replace=True)
        high.train();opt.zero_grad(set_to_none=True)
        mu,lv=_predict_selected(high,data,teacher_mu,teacher_lv,ids,learn_mean,learn_risk)
        loss=gaussian_loss(mu,lv,data['y'][ids]);loss.backward()
        nn.utils.clip_grad_norm_(high.parameters(),3.)
        opt.step()
        if step%30==0 or step==steps:
            high.eval()
            with torch.no_grad():
                mm,ll=_predict_selected(high,data,teacher_mu,teacher_lv,va,learn_mean,learn_risk)
                v=float(gaussian_loss(mm,ll,data['y'][va]))
            curve.append([step,v])
            if v<best-1e-5:
                best=v;best_state=copy.deepcopy(high.state_dict());stagnation=0
            else:
                stagnation+=1
                if stagnation>=patience:break
    assert best_state is not None
    high.load_state_dict(best_state);high.eval()
    def evaluate(ids):
        high.eval()
        pr=[];lv=[]
        with torch.no_grad():
            for i in range(0,len(ids),256):
                ix=ids[i:i+256]
                m,v=_predict_selected(high,data,teacher_mu,teacher_lv,ix,learn_mean,learn_risk)
                pr.append(m.numpy());lv.append(v.numpy())
        mu=np.concatenate(pr);var=np.exp(np.concatenate(lv))*high.norm.y_sd**2
        yr=data['y'][ids].numpy()*high.norm.y_sd+high.norm.y_mean
        pred=mu*high.norm.y_sd+high.norm.y_mean
        nll=.5*np.mean(np.log(2*np.pi*var)+(yr-pred)**2/var)
        return {'NLL':float(nll),'MSE':float(np.mean((yr-pred)**2)),
                'mean_variance':float(np.mean(var)), 'risk_ratio':float(np.mean((yr-pred)**2/var))}
    return high,{'train':evaluate(tr),'validation':evaluate(data['val_idx']),
                 'test':evaluate(data['test_idx']), 'curve':curve,'steps':step,'sec':time.monotonic()-t0}

def combine_heads(arm,low_heads,high_heads):
    x=low_heads.copy()
    if arm[0]=='1': x[:,0:3]=high_heads[:,0:3]
    if arm[1]=='1': x[:,3]=high_heads[:,3]
    assert np.all(x[31:,3]>0)
    return x

def oos_probe(tr,te,L=3,regularization=1e-5):
    a=design_matrix(tr['panel'],L);b=design_matrix(te['panel'],L)
    yt=tr['panel'].target[L:];yv=te['panel'].target[L:]
    def mse(X,Z):
        mx=X.mean(0);sx=np.maximum(X.std(0),1e-7)
        X=(X-mx)/sx;Z=(Z-mx)/sx
        co=solve(X.T@X+regularization*len(X)*np.eye(X.shape[1]),X.T@(yt-yt.mean()),assume_a='pos')
        return float(np.mean((yv-yt.mean()-Z@co)**2))
    pub=mse(a,b);out={}
    for arm in ARMS:
        q=mse(np.c_[a,tr[arm]['price'][L:]],np.c_[b,te[arm]['price'][L:]])
        out[arm]={'gain':1-q/pub,'MSE':q,'public_MSE':pub}
    return out

def evaluate_arms(eco,tpl,low,students,bank):
    inherited=simulate(eco,tpl,bank)
    original=raw_heads_on_inherited_history(low,inherited.public_input)
    high={arm:raw_heads_on_inherited_history(students[arm],inherited.public_input)
          for arm in ('10','01','11')}
    out={'panel':inherited}
    for arm in ARMS:
        v=combine_heads(arm,original, high[arm] if arm!='00' else original)
        info=frozen_price_panel(eco,tpl,inherited,0,v)
        ii=slice(32,None);s=inherited.private_signals[ii,0];p=inherited.price[ii]
        mu=v[ii,0]+v[ii,1]*s+v[ii,2]*p
        error=inherited.future_dividend[ii]-mu
        info['inherited_NLL']=float(.5*np.mean(np.log(2*np.pi*v[ii,3])+error**2/v[ii,3]))
        info['mean_private_loading']=float(np.mean(tpl[0].mass*v[ii,1]/(tpl[0].gamma*v[ii,3])))
        info['subjective_variance']=float(np.mean(v[ii,3]))
        out[arm]=info
    return out

def one_cell(arch,cal,seed,periods=2200,evalperiods=4200,steps=240,pairs=2):
    eco=calibrated_economy(cal);tpl=diagnostic_policies(eco,'E3_state_risk')
    marketbank=eco.draw_bank(periods=periods,burn=420,seed=88000+int(cal*1000)*37+seed*101)
    inherited=simulate(eco,tpl,marketbank)
    panel=vars(inherited)
    low,lowdetail=fit_forecaster(panel,'E3_restricted',arch,4,'B',
             seed=seed*97+17, type_index=0,window=32,steps=steps,batch_size=128,patience=5)
    students={};stats={}
    for j,arm in enumerate(('10','01','11')):
        high,details=train_restricted(panel,low,arch,12,arm,seed=seed*97+23+j*13,
                   steps=steps,window=32,batch=128)
        students[arm]=high;stats[arm]=details
    armmodels={'00':low,**students}
    evalrows=[];trainbankscores=[]
    for bank in range(pairs):
        shockA=eco.draw_bank(periods=evalperiods,burn=500,seed=200000+int(cal*1000)*127+bank*727)
        shockB=eco.draw_bank(periods=evalperiods,burn=500,seed=300000+int(cal*1000)*127+bank*727)
        tr=evaluate_arms(eco,tpl,low,students,shockA)
        te=evaluate_arms(eco,tpl,low,students,shockB)
        score=oos_probe(tr,te,L=3)
        for arm in ARMS:
            row={'arch':arch,'calibration':cal,'seed':seed,'bank':bank,'arm':arm,
             'train_NLL':lowdetail['scores']['train']['NLL'] if arm=='00' else stats[arm]['train']['NLL'],
             'inherited_NLL':te[arm]['inherited_NLL'],
             'linear_decoder_gain':score[arm]['gain'],
             'linear_decoder_MSE':score[arm]['MSE'],'public_MSE':score[arm]['public_MSE'],
             'private_demand_loading':te[arm]['mean_private_loading'],
             'subjective_variance':te[arm]['subjective_variance'],
             'max_clearing_error':te[arm]['max_clearing_residual'],
             'min_slope':te[arm]['min_delta'],
             'train_steps':lowdetail['steps_completed'] if arm=='00' else stats[arm]['steps']}
            evalrows.append(row)
    for arm in ARMS:
        model=armmodels[arm]
        sav=ROOT/'models';sav.mkdir(exist_ok=True)
        torch.save({'state':model.state_dict(),'norm':model.norm.to_dict(),
            'arch':arch,'hidden':model.hidden,'head_type':model.head_type,'window':model.window,
            'calibration':cal,'seed':seed,'arm':arm,
            'training_info':lowdetail if arm=='00' else stats[arm]},
            sav/f'{arch}_v{int(cal*100):03d}_seed{seed}_arm{arm}.pt')
    return evalrows,{'arch':arch,'calibration':cal,'seed':seed,'low':lowdetail['scores'],
                     'restricted':stats}

def run(architectures=('ssm','gru'),calibrations=(.7,1.,1.3),seeds=(11,23,37),
        periods=2200,evalperiods=4200,steps=240,pairs=2):
    torch.set_num_threads(1)
    ROOT.joinpath('results').mkdir(exist_ok=True)
    rows=[];details=[]; t0=time.monotonic()
    for cal in calibrations:
        for arch in architectures:
            for seed in seeds:
                print('BEGIN',cal,arch,seed,flush=True)
                new,det=one_cell(arch,cal,seed,periods,evalperiods,steps,pairs)
                rows.extend(new);details.append(det)
                pd.DataFrame(rows).to_csv(ROOT/'results'/'all_arms_checkpoint.csv',index=False)
                (ROOT/'results'/'training_details_checkpoint.json').write_text(json.dumps(details,indent=2))
                x=pd.DataFrame(new).groupby('arm')['linear_decoder_gain'].mean()
                print('DONE',cal,arch,seed,'sec',round(time.monotonic()-t0,1),'gain-pp',
                    {a:round(100*(x[a]-x['00']),3) for a in ('10','01','11')},flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results'/'independently_optimized_arms.csv',index=False)
    (ROOT/'results'/'training_details.json').write_text(json.dumps(details,indent=2))
    diffs=[]
    for (cal,arch,seed,bank),x in df.groupby(['calibration','arch','seed','bank']):
        q=x.set_index('arm')
        v={a:q.loc[a,'linear_decoder_gain'] for a in ARMS}
        N={a:q.loc[a,'inherited_NLL'] for a in ARMS}
        diffs.append({'calibration':cal,'arch':arch,'seed':seed,'bank':bank,
            'mean_effect_pp':100*(v['10']-v['00']),
            'variance_effect_pp':100*(v['01']-v['00']),
            'joint_effect_pp':100*(v['11']-v['00']),
            'interaction_pp':100*(v['11']-v['10']-v['01']+v['00']),
            'NLL_mean_gain':N['00']-N['10'],
            'NLL_var_gain':N['00']-N['01'],
            'NLL_joint_gain':N['00']-N['11']})
    df2=pd.DataFrame(diffs);df2.to_csv(ROOT/'results'/'paired_effects.csv',index=False)
    summary={'n_cells':len(details),'n_independent_eval_pairs_per_cell':pairs,'seconds':time.monotonic()-t0,
             'by_architecture':{a:{k:{'mean':float(g[k].mean()),'sd':float(g[k].std())} for k in ['mean_effect_pp','variance_effect_pp','joint_effect_pp','NLL_mean_gain','NLL_var_gain','NLL_joint_gain']} for a,g in df2.groupby('arch')},
             'n_positive_effects':{c:int((df2[c]>0).sum()) for c in ['mean_effect_pp','variance_effect_pp','joint_effect_pp']}}
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    return df,df2

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--steps',type=int,default=240)
    args=ap.parse_args()
    if args.quick:run(architectures=('ssm',),calibrations=(1.,),seeds=(11,),periods=900,evalperiods=1000,steps=60,pairs=1)
    else:run(steps=args.steps)
