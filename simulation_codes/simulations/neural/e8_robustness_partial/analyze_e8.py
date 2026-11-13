from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import logsumexp
from run_e8 import ROOT,BASE,e,tag,load_market
OUT=ROOT/'summary';OUT.mkdir(exist_ok=True)
VARIANTS=['original_1x','original_2x','original_4x','newrep1_1x','newrep2_1x']

def ci_rows(df,groups,metrics):
 rows=[]
 for key,g in df.groupby(groups):
  if not isinstance(key,tuple):key=(key,)
  for metric in metrics:
   vals=g[metric].to_numpy();mean=vals.mean();hw=t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
   rows.append(dict(zip(groups,key),metric=metric,mean=mean,ci_low=mean-hw,ci_high=mean+hw,n=len(vals),min=vals.min(),max=vals.max()))
 return pd.DataFrame(rows)

def main():
 ms=json.loads((BASE/'markets.json').read_text());learning=[];dec=[];ces=[];banks=[];audits=[];boot=[];selection=[]
 for m in ms:
  name=tag(m);sourcefp=e.fingerprint(*load_market(m));assert sourcefp==m['profile_sha256']
  for variant in VARIANTS:
   p=ROOT/'results'/name/variant;done=json.loads((p/'complete.json').read_text());assert done['market_sha256']==sourcefp
   gen=json.loads((p/'generation.json').read_text());assert [q['bank'] for q in gen]==[3,20,21,22,23]
   assert all(q['profile_sha256']==sourcefp for q in gen);assert all(q['observations']==262144 for q in gen)
   a=pd.read_csv(p/'banks.csv');a['seed']=m['seed'];a['x']=m['x'];a['variant']=variant;banks.append(a)
   c=pd.read_csv(p/'CE.csv');c['seed']=m['seed'];c['x']=m['x'];c['variant']=variant;ces.append(c);c=c.set_index('model')
   q=dict(seed=m['seed'],x=m['x'],variant=variant)
   for cap in ['low','high','nested']:
    sub=a[(a.kind=='E3')&(a.model==cap)]
    for col in ['gain','MSE_no_price','MSE_price']:q[f'E3_{cap}_{col}']=sub[col].mean()
   for cap in ['high','nested']:
    q[f'E3_T_{cap}']=q['E3_low_gain']-q[f'E3_{cap}_gain']
    assert abs(q[f'E3_T_{cap}']-((q['E3_low_MSE_no_price']-q[f'E3_{cap}_MSE_no_price'])-(q['E3_low_MSE_price']-q[f'E3_{cap}_MSE_price'])))<1e-12
   for mode,cap in [('neural','high'),('ols','high'),('ols','nested')]:q[f'E4_{mode}_{cap}_upgrade']=c.loc[f'E4_{mode}_{cap}','CE']-c.loc[f'E4_{mode}_low','CE']
   # Within-market uncertainty conditional on model; paired bank-stratified path bootstrap.
   paths=pd.read_csv(p/'paths.csv');aa=paths[paths.model=='E4_neural_high'].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256);bb=paths[paths.model=='E4_neural_low'].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256)
   rng=np.random.default_rng(48567+m['seed']);draw=[]
   for j in range(500):
    ix=rng.integers(0,256,(4,256));draw.append(-(logsumexp(np.take_along_axis(aa,ix,1))-logsumexp(np.take_along_axis(bb,ix,1)))/e.GAMMA[0])
   lo,hi=np.quantile(draw,[.025,.975]);boot.append(dict(seed=m['seed'],x=m['x'],variant=variant,estimate=q['E4_neural_high_upgrade'],eval_low=lo,eval_high=hi))
   learning.append(q);audits.append(dict(seed=m['seed'],x=m['x'],variant=variant,clearing_error=max(q['clearing_error'] for q in gen)))
  # Training budgets and bank independence.
  for rep in [0,1,2]:
   p=ROOT/'results'/name/f'train_rep{rep}';details=json.loads((p/'complete.json').read_text())['training'];assert len(details)==(6 if rep==0 else 2);assert all(v['updates']==4096 and v['optimization_windows']==1048576 for v in details)
   gen=json.loads((p/'generation.json').read_text());assert all(v['profile_sha256']==sourcefp for v in gen);assert not any(v['bank'] in [20,21,22,23] for v in gen)
  p=ROOT/'results'/name/'decoders';done=json.loads((p/'complete.json').read_text());assert done['market_sha256']==sourcefp
  gen=json.loads((p/'generation.json').read_text());assert [v['bank'] for v in gen]==[3,4,20,21,22,23]
  a=pd.read_csv(p/'banks.csv');heads=json.loads((p/'heads.json').read_text())
  for k,h in heads.items():selection.append(dict(seed=m['seed'],x=m['x'],decoder=k,**{key:h[key] for key in ['ridge','validation_MSE','features','fit_observations']}))
  for size in [0,64,128]:
   q=dict(seed=m['seed'],x=m['x'],size=size)
   for cap in ['low','high','nested']:
    ss=a[(a['size']==size)&(a.model==cap)]
    for col in ['gain','MSE_no_price','MSE_price']:q[f'E3_{cap}_{col}']=ss[col].mean()
   for cap in ['high','nested']:q[f'E3_T_{cap}']=q['E3_low_gain']-q[f'E3_{cap}_gain']
   dec.append(q)
 l=pd.DataFrame(learning);d=pd.DataFrame(dec);l.to_csv(OUT/'learning_seed_results.csv',index=False);d.to_csv(OUT/'decoder_seed_results.csv',index=False)
 metric=[c for c in l if c.startswith('E')];dm=[c for c in d if c.startswith('E')]
 old=pd.read_csv(BASE/'summary/seed_results.csv');new=l[l.variant=='original_1x'];fresh=old.merge(new,on=['seed','x'],suffixes=('_old','_fresh'))
 fresh.to_csv(OUT/'fresh_evaluation_check.csv',index=False)
 ci_rows(l,['variant','x'],metric).to_csv(OUT/'learning_levels.csv',index=False);ci_rows(d,['size','x'],dm).to_csv(OUT/'decoder_levels.csv',index=False)
 def changes(df,groups,metrics):
  rr=[]
  for key,g in df.groupby(groups):
   if not isinstance(key,tuple):key=(key,)
   z=g[g.x==0].iloc[0]
   for x in [.5,1.]:
    q=g[g.x==x].iloc[0];rr.append(dict(zip(groups,key),x=x,**{c:q[c]-z[c] for c in metrics}))
  return pd.DataFrame(rr)
 lc=changes(l,['seed','variant'],metric);dc=changes(d,['seed','size'],dm);lc.to_csv(OUT/'learning_seed_changes.csv',index=False);dc.to_csv(OUT/'decoder_seed_changes.csv',index=False)
 ci_rows(lc,['variant','x'],metric).to_csv(OUT/'learning_changes.csv',index=False);ci_rows(dc,['size','x'],dm).to_csv(OUT/'decoder_changes.csv',index=False)
 # Sensitivity of adoption contrasts to learning procedure, paired within original seeds.
 sensitivity=[]
 for (seed,variant),g in lc.groupby(['seed','variant']):
  for _,r in g.iterrows():
   b=lc[(lc.seed==seed)&(lc.variant=='original_1x')&(lc.x==r.x)].iloc[0];sensitivity.append(dict(seed=seed,variant=variant,x=r.x,**{m:r[m]-b[m] for m in metric}))
 ci_rows(pd.DataFrame(sensitivity),['variant','x'],metric).to_csv(OUT/'contrast_sensitivity.csv',index=False)
 pd.concat(ces).to_csv(OUT/'CE_diagnostics.csv',index=False);pd.concat(banks).to_csv(OUT/'learning_bank_results.csv',index=False);pd.DataFrame(boot).to_csv(OUT/'CE_evaluation_uncertainty.csv',index=False);pd.DataFrame(selection).to_csv(OUT/'decoder_selection.csv',index=False)
 audit=dict(all_pass=True,frozen_markets=9,independent_market_seeds=3,learning_evaluations=len(learning),decoder_evaluations=len(dec),training_pairs=45,evaluation_observations_per_case=1048576,max_clearing_error=max(a['clearing_error'] for a in audits),market_unchanged=True)
 e.dump(OUT/'VALIDATION.json',audit)
 print('LEARNING ADOPTION0->1');print(lc[lc.x==1].groupby('variant')[['E3_T_high','E4_neural_high_upgrade','E4_ols_nested_upgrade']].agg(['mean','min','max']).to_string())
 print('DECODER ADOPTION0->1');print(dc[dc.x==1].groupby('size')[['E3_T_high','E3_T_nested']].agg(['mean','min','max']).to_string())
if __name__=='__main__':main()
