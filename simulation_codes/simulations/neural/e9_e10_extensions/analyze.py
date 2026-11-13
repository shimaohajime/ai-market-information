import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from common import *
from economies import MODES
OUT=ROOT/'summary';OUT.mkdir(exist_ok=True)
METRICS=['J','low_iota','high_iota','other_iota']
def cycle_stats(a):
 a=np.asarray(a,float);scaled=a/np.maximum(abs(a[0]),1e-12);step=np.linalg.norm(np.diff(scaled,axis=0),axis=1);ret=np.linalg.norm(scaled[2:]-scaled[:-2],axis=1);dj=np.diff(a[:,0]);alternations=int(np.sum(dj[:-1]*dj[1:]<0));ratio=float(ret.mean()/max(step.mean(),1e-15));amp=float(np.median(step))
 last=np.abs(np.diff(a,axis=0)[-3:]/np.maximum(abs(a[:-1][-3:]),1e-12));stable=bool(np.all(last<.005))
 tail=scaled[-5:];ts=np.linalg.norm(np.diff(tail,axis=0),axis=1);tr=np.linalg.norm(tail[2:]-tail[:-2],axis=1)
 return dict(period2_tail_ratio=float(tr.mean()/max(ts.mean(),1e-15)),tail_median_step_pct=float(100*np.median(ts)),period2_ratio=ratio,median_step_pct=100*amp,J_sign_reversals=alternations,period2_screen=bool(ratio<.25 and amp>.005 and alternations>=4),last3_update_max_pct=100*last.max(),stable_05pct_last3=stable,J_late_block_pct=100*(a[-3:,0].mean()/a[-6:-3,0].mean()-1),J_total_extension_pct=100*(a[-1,0]/a[0,0]-1))
def intervals(df,groups,metrics):
 rows=[]
 for key,g in df.groupby(groups):
  if not isinstance(key,tuple):key=(key,)
  for m in metrics:
   a=g[m].to_numpy();mean=a.mean();hw=t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a));rows.append(dict(zip(groups,key),metric=m,mean=mean,ci_low=mean-hw,ci_high=mean+hw,n=len(a)))
 return pd.DataFrame(rows)
def e9():
 states=[];diagnostics=[];noise=[];checks=[]
 for s in SEEDS:
  for x in XS:
   for mode in (['fresh','common_shocks'] if x==.5 else ['fresh']):
    p=ROOT/'results'/'E9'/tag(s,x)/mode;assert (p/'complete.json').exists();a=pd.read_csv(p/'states.csv');assert len(a)==7
    for n in range(6,13):
     z=p/f'round{n:02d}';m,o=load_profile(z);fp=e.fingerprint(m,o);c=json.loads((z/'complete.json').read_text());assert c['profile_sha256']==fp
     st=json.loads((z/'state.json').read_text());ss=stats(m,o,x)
     for k in ss:assert abs(st[k]-ss[k])<1e-10
     if n<12:
      gen=json.loads((z/'generation.json').read_text());assert all(g['profile_sha256']==fp for g in gen);assert [g['bank'] for g in gen]==[1,50];key=60000+n if mode=='fresh' else 95000;assert all(g['round_index']==key for g in gen)
      tr=json.loads((z/'training.json').read_text());assert all(v['updates']==4096 and v['optimization_windows']==1048576 for v in tr.values());checks+=gen
    old=[]
    for n in range(6):
     st=json.loads((ROOT/'source_e6'/(tag(s,x)+'_both')/f'round{n:02d}'/'market.json').read_text());old.append(dict(seed=s,x=x,mode=mode,round=n,**{k:st[k] for k in ['A','J','delta','low_iota','high_iota','other_iota']}))
    states.append(pd.concat([pd.DataFrame(old),a],ignore_index=True));dd=cycle_stats(a[METRICS].to_numpy());active=['J','other_iota']+(['low_iota'] if x<1 else [])+(['high_iota'] if x>0 else []);av=a[active].to_numpy();dd['stable_active_05pct_last3']=bool(np.all(abs(np.diff(av,axis=0)[-3:]/av[:-1][-3:])<.005));diagnostics.append(dict(seed=s,x=x,mode=mode,**dd))
    if mode=='fresh' and x==.5:
     audits=pd.read_csv(p/'audit.csv');base=a.iloc[-1]
     for metric in METRICS:
      changes=100*(audits[metric]/base[metric]-1);mean=changes.mean();sd=changes.std(ddof=1);hw=t.ppf(.975,3)*sd/2
      noise.append(dict(seed=s,metric=metric,mean_pct=mean,sd_pct=sd,ci_low=mean-hw,ci_high=mean+hw,within_1pct=bool(abs(mean)+hw<1),last6_step_sd_pct=100*a[metric].diff().dropna().std(ddof=1)/base[metric]))
     for rep in range(4):
      z=p/f'audit{rep}';nm,no=load_profile(z);saved=json.loads((z/'state.json').read_text());actual=stats(nm,no,x)
      for k in actual:assert abs(saved[k]-actual[k])<1e-10
      tr=json.loads((z/'training.json').read_text());assert all(v['updates']==4096 and v['optimization_windows']==1048576 for v in tr.values())
      c=json.loads((z/'complete.json').read_text());assert c['inherited_sha256']==e.fingerprint(*load_profile(p/'round12'));g=json.loads((z/'generation.json').read_text());assert all(v['profile_sha256']==c['inherited_sha256'] for v in g);checks+=g
 functional=[]
 for s in SEEDS:
  for x in XS:
   for mode in (['fresh','common_shocks'] if x==.5 else ['fresh']):
    p=ROOT/'results'/'E9'/tag(s,x)/mode;functional.append(pd.read_csv(p/'functional_check.csv'));meta=json.loads((p/'anchor_generation.json').read_text());assert meta['bank']==60 and meta['round_index']==94000
 pd.concat(functional).to_csv(OUT/'E9_functional_changes.csv',index=False)
 df=pd.concat(states);dg=pd.DataFrame(diagnostics);ns=pd.DataFrame(noise);df.to_csv(OUT/'E9_states.csv',index=False);dg.to_csv(OUT/'E9_diagnostics.csv',index=False);ns.to_csv(OUT/'E9_fixed_law_audit.csv',index=False)
 intervals(dg,['mode','x'],['J_total_extension_pct','J_late_block_pct']).to_csv(OUT/'E9_intervals.csv',index=False)
 print('E9');print(dg.to_string(index=False));print('FIXED LAW AUDIT');print(ns.to_string(index=False))
 return dict(trajectories=12,added_joint_updates=72,independent_one_step_refits=12,max_clearing_error=max(g['clearing_error'] for g in checks))
def e10():
 rows=[];ce=[];impact=[];audit=[];boot=[]
 for mode in MODES:
  for seed in SEEDS:
   base=ROOT/'results'/'E10'/mode/f'seed{seed}';p=base/'inherited_fit';m,o=load_profile(p);fp=e.fingerprint(m,o);c=json.loads((p/'complete.json').read_text());assert c['fitted_sha256']==fp
   ig=json.loads((p/'generation.json').read_text());assert [g['bank'] for g in ig]==[1,50];assert all(g['profile_sha256']==c['inherited_sha256'] and g['round_index']==101000 and g['economy']==mode for g in ig)
   mix=json.loads((p/'inherited_mixing.json').read_text());assert mix['error']<1e-5 and mix['profile_sha256']==c['inherited_sha256']
   tr=json.loads((p/'training.json').read_text());assert all(v['updates']==4096 for v in tr.values());a=pd.read_csv(p/'impact.csv');a['J_impact_pct']=100*(a.J/a[a.x==0].J.iloc[0]-1);impact.append(a)
   for x in XS:
    p=base/f'x{round(x*100):03d}';c=json.loads((p/'complete.json').read_text());assert c['market_sha256']==fp and c['mixing_error']<1e-5
    gen=json.loads((p/'generation.json').read_text());assert all(g['profile_sha256']==fp for g in gen);assert [g['bank'] for g in gen][-4:]==[20,21,22,23];audit+=gen
    tr=json.loads((p/'training.json').read_text());assert all(v['updates']==4096 and v['optimization_windows']==1048576 for v in tr.values())
    training_gen=json.loads((p/'training_generation.json').read_text());assert [g['bank'] for g in training_gen]==[1,50];assert all(g['round_index']==102000 for g in training_gen)
    b=pd.read_csv(p/'banks.csv');cc=pd.read_csv(p/'CE.csv');cc['seed']=seed;cc['x']=x;cc['economy']=mode;ce.append(cc);cc=cc.set_index('model')
    row=dict(seed=seed,x=x,economy=mode,J_impact_pct=a[a.x==x].J_impact_pct.iloc[0],**stats(m,o,x))
    for cap in ['low','high','nested']:
     q=b[(b.kind=='E3')&(b.model==cap)]
     for k in ['gain','MSE_no_price','MSE_price']:row[f'E3_{cap}_{k}']=q[k].mean()
    for cap in ['high','nested']:row[f'E3_T_{cap}']=row['E3_low_gain']-row[f'E3_{cap}_gain']
    for hd,cap in [('neural','high'),('ols','high'),('ols','nested')]:row[f'E4_{hd}_{cap}_upgrade']=cc.loc[f'E4_{hd}_{cap}','CE']-cc.loc[f'E4_{hd}_low','CE']
    paths=pd.read_csv(p/'paths.csv');aa=paths[paths.model=='E4_neural_high'].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256);bb=paths[paths.model=='E4_neural_low'].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256)
    rng=np.random.default_rng(seed+14883);bs=[]
    for j in range(500):
     ix=rng.integers(0,256,(4,256));bs.append(-(logsumexp(np.take_along_axis(aa,ix,1))-logsumexp(np.take_along_axis(bb,ix,1)))/e.GAMMA[0])
    lo,hi=np.quantile(bs,[.025,.975]);boot.append(dict(seed=seed,x=x,economy=mode,estimate=row['E4_neural_high_upgrade'],eval_low=lo,eval_high=hi));rows.append(row)
 v=pd.DataFrame(rows);metrics=['J_impact_pct','E3_T_high','E3_T_nested','E4_neural_high_upgrade','E4_ols_high_upgrade','E4_ols_nested_upgrade']
 v.to_csv(OUT/'E10_seed_results.csv',index=False);pd.concat(impact).to_csv(OUT/'E10_impact.csv',index=False);pd.concat(ce).to_csv(OUT/'E10_CE_diagnostics.csv',index=False);pd.DataFrame(boot).to_csv(OUT/'E10_CE_evaluation_uncertainty.csv',index=False)
 intervals(v,['economy','x'],metrics).to_csv(OUT/'E10_levels.csv',index=False);changes=[]
 for (mode,seed),g in v.groupby(['economy','seed']):
  b=g[g.x==0].iloc[0]
  for x in [.5,1.]:
   a=g[g.x==x].iloc[0];changes.append(dict(economy=mode,seed=seed,x=x,**{k:a[k]-b[k] for k in metrics}))
 ch=pd.DataFrame(changes);ch.to_csv(OUT/'E10_seed_changes.csv',index=False);intervals(ch,['economy','x'],metrics).to_csv(OUT/'E10_changes.csv',index=False)
 sensitivity=[]
 for _,r in ch.iterrows():
  if r.economy=='control':continue
  b=ch[(ch.economy=='control')&(ch.seed==r.seed)&(ch.x==r.x)].iloc[0];sensitivity.append(dict(economy=r.economy,seed=r.seed,x=r.x,**{k:r[k]-b[k] for k in metrics}))
 intervals(pd.DataFrame(sensitivity),['economy','x'],metrics).to_csv(OUT/'E10_paired_difference_from_control.csv',index=False)
 print('E10 endpoint changes');print(ch[ch.x==1].groupby('economy')[metrics].agg(['mean','min','max']).to_string())
 return dict(economies=3,frozen_laws=27,initial_profile_refits=9,candidate_refits=27,max_clearing_error=max(g['clearing_error'] for g in audit))
if __name__=='__main__':
 for f,sha in json.loads((ROOT/'SOURCE_HASHES.json').read_text()).items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==sha
 a=e9();b=e10();e.dump(OUT/'VALIDATION.json',dict(all_pass=True,E9=a,E10=b,independent_initial_seeds=3))
