"""Audit all E6 branches and compute paired, baseline-adjusted E6/E7 effects."""
import json,hashlib,platform
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
import engine as e
import run_e6 as r

def ci(v):
 v=np.asarray(v,dtype=float);mean=float(v.mean());se=float(v.std(ddof=1)/np.sqrt(len(v)));half=float(t.ppf(.975,len(v)-1)*se)
 return dict(n=len(v),mean=mean,lower=mean-half,upper=mean+half,min=float(v.min()),max=float(v.max()))

def main():
 torch.set_num_threads(1);out=e.ROOT/'summary';out.mkdir(exist_ok=True)
 states=[];models=[];steps=[];fitted=[];checks=[];start_hash={};first_hash={}
 for seed in r.CFG['seeds']:
  base,bo=r.load_baseline(seed);fp=e.fingerprint(base,bo);bgh=r.group_hashes(base,bo)
  for x in r.CFG['adoption']:
   for arm,(g1,g2) in r.ARMS.items():
    root=e.ROOT/'results'/r.name(seed,x,arm);last=0 if arm=='none' else r.CFG['slow_updates']
    assert (root/'complete.json').exists()
    for n in range(last+1):
     p=root/f'round{n:02d}';meta=json.loads((p/'complete.json').read_text());gen=json.loads((p/'generation.json').read_text())
     mm={c:e.load_checkpoint(p/(c+'_deployed.pt'))[0] for c in ['low','high']};oo=json.loads((p/'other_deployed.json').read_text())
     gh=r.group_hashes(mm,oo);assert meta['profile_sha256']==e.fingerprint(mm,oo);assert meta['baseline_sha256']==fp
     assert meta['group_hashes']==gh
     if n==0:assert meta['profile_sha256']==fp
     if not g1:assert gh['low']==bgh['low'] and gh['high']==bgh['high']
     if not g2:assert gh['other']==bgh['other']
     for z in gen:assert z['profile_sha256']==meta['profile_sha256'] and z['x']==x
     assert meta['mixing_error']<1e-5 and max(z['clearing_error'] for z in gen)<1e-10
     expected=[1,50,60] if n<last and g1 else ([1,60] if n<last else [60])
     assert [z['bank'] for z in gen]==expected
     v=pd.read_csv(p/'deployed_metrics.csv');m=json.loads((p/'market.json').read_text());dd=r.describe(mm,oo,x)
     for key,value in dd.items():assert abs(m[key]-value)<1e-10
     assert abs(m['J']-(m['A_group1']+m['A_group2'])**2/e.ZETA2)<1e-10
     states.append(dict(seed=seed,x=x,arm=arm,round=n,simulated_state=True,**m))
     for row in v.to_dict('records'):models.append(dict(seed=seed,x=x,arm=arm,round=n,**row))
     if n==0:start_hash[(seed,x,arm)]=meta['profile_sha256']
     if n==1:first_hash[(seed,x,arm)]=gh
     if n<last:
      assert meta['training_started']>=meta['generation_finished']>meta['generation_started']
      assert meta['train_observations']==1048576
      if g1:assert all(z['updates']==4096 and z['optimization_windows']==1048576 for z in meta['updates'].values())
      else:assert meta['updates']=={}
      new={c:e.load_checkpoint(p/(c+'_next.pt'))[0] for c in ['low','high']};no=json.loads((p/'other_next.json').read_text())
      nx=root/f'round{n+1:02d}'
      nextm={c:e.load_checkpoint(nx/(c+'_deployed.pt'))[0] for c in ['low','high']};nexto=json.loads((nx/'other_deployed.json').read_text())
      assert e.fingerprint(new,no)==e.fingerprint(nextm,nexto)
      dec=pd.read_csv(p/'E7_decomposition.csv');sw=pd.read_csv(p/'E7_swaps.csv').set_index('swap')
      assert np.max(abs(dec.mean_contribution+dec.variance_contribution-dec.total))<1e-10
      for metric in ['J','A','low_iota','high_iota','other_iota','A_group1','A_group2']:
       assert abs(sw.loc['old',metric]-m[metric])<1e-10
       assert abs(sw.loc['joint',metric]-r.describe(new,no,x)[metric])<1e-10
      dec['seed']=seed;dec['x']=x;dec['arm']=arm;dec['update']=n+1;steps.append(dec)
      f=pd.read_csv(p/'refitted_on_old_law.csv');f['seed']=seed;f['x']=x;f['arm']=arm;f['update']=n+1
      for _,row in f.iterrows():
       old=v.set_index('capacity').loc[row.capacity,'MSE'];f.loc[f.capacity==row.capacity,'MSE_improvement_pct']=100*(1-row.MSE/old)
      fitted.append(f)
     checks.append(dict(seed=seed,x=x,arm=arm,round=n,clearing_error=max(z['clearing_error'] for z in gen),mixing_error=meta['mixing_error'],delta=m['delta']))
    if arm=='none':
     m=json.loads((root/'round00/market.json').read_text());v=pd.read_csv(root/'round00/deployed_metrics.csv')
     for n in range(1,r.CFG['slow_updates']+1):
      states.append(dict(seed=seed,x=x,arm=arm,round=n,simulated_state=False,**m))
      for row in v.to_dict('records'):models.append(dict(seed=seed,x=x,arm=arm,round=n,**row))
  for x in r.CFG['adoption']:
   assert len({start_hash[(seed,x,a)] for a in r.ARMS})==1
   # Common law and fitting randomness: first independently fitted blocks match.
   for c in ['low','high']:assert first_hash[(seed,x,'group1')][c]==first_hash[(seed,x,'both')][c]
   assert first_hash[(seed,x,'group2')]['other']==first_hash[(seed,x,'both')]['other']
 s=pd.DataFrame(states);mod=pd.DataFrame(models);st=pd.concat(steps,ignore_index=True)
 s.to_csv(out/'states.csv',index=False);mod.to_csv(out/'model_metrics.csv',index=False);st.to_csv(out/'E7_steps.csv',index=False)
 pd.concat(fitted,ignore_index=True).to_csv(out/'same_law_learning.csv',index=False)
 sidx=s.set_index(['seed','x','arm','round']);effects=[];factorial=[]
 for seed in r.CFG['seeds']:
  baseJ=sidx.loc[(seed,0.,'none',0),'J']
  for x in r.CFG['adoption']:
   for n in range(r.CFG['slow_updates']+1):
    for metric in ['J','A','A_group1','A_group2']:
     direct=sidx.loc[(seed,x,'none',0),metric]-sidx.loc[(seed,0.,'none',0),metric]
     values={}
     for arm in r.ARMS:
      Y=sidx.loc[(seed,x,arm,n),metric];Y0=sidx.loc[(seed,0.,arm,n),metric]
      R=Y-sidx.loc[(seed,x,'none',0),metric];R0=Y0-sidx.loc[(seed,0.,'none',0),metric]
      T=R-R0;values[arm]=T
      effects.append(dict(seed=seed,x=x,round=n,arm=arm,metric=metric,baseline_J=baseJ,
       level=Y,raw_retraining=R,zero_adoption_retraining=R0,extra_retraining=T,direct_adoption=direct,total_adoption=Y-Y0,
       raw_retraining_pct_baseJ=100*R/baseJ if metric=="J" else np.nan,extra_retraining_pct_baseJ=100*T/baseJ if metric=="J" else np.nan,total_adoption_pct_baseJ=100*(Y-Y0)/baseJ if metric=="J" else np.nan))
     ix=(sidx.loc[(seed,x,'both',n),metric]-sidx.loc[(seed,x,'group1',n),metric]-sidx.loc[(seed,x,'group2',n),metric]+sidx.loc[(seed,x,'none',n),metric])
     i0=(sidx.loc[(seed,0.,'both',n),metric]-sidx.loc[(seed,0.,'group1',n),metric]-sidx.loc[(seed,0.,'group2',n),metric]+sidx.loc[(seed,0.,'none',n),metric])
     extraI=ix-i0;assert abs(values['both']-values['group1']-values['group2']-extraI)<1e-10
     factorial.append(dict(seed=seed,x=x,round=n,metric=metric,baseline_J=baseJ,direct=direct,group1=values['group1'],group2=values['group2'],interaction=extraI,both=values['both'],total=direct+values['both'],group1_given_group2=values['both']-values['group2'],group2_given_group1=values['both']-values['group1'],
      **{k+'_pct_baseJ':100*v/baseJ if metric=='J' else np.nan for k,v in [('direct',direct),('group1',values['group1']),('group2',values['group2']),('interaction',extraI),('both',values['both']),('total',direct+values['both']),('group1_given_group2',values['both']-values['group2']),('group2_given_group1',values['both']-values['group1'])]}))
 eff=pd.DataFrame(effects);fac=pd.DataFrame(factorial);eff.to_csv(out/'E6_effects.csv',index=False);fac.to_csv(out/'E6_factorial.csv',index=False)
 intervals=[]
 for (x,n,metric),g in fac.groupby(['x','round','metric']):
  for key in ['direct','group1','group2','interaction','both','total','group1_given_group2','group2_given_group1']:
   intervals.append(dict(x=x,round=n,metric=metric,component=key,units='raw',**ci(g[key])))
   if metric=='J':intervals.append(dict(x=x,round=n,metric=metric,component=key,units='percent_of_pre_adoption_J',**ci(g[key+'_pct_baseJ'])))
 pd.DataFrame(intervals).to_csv(out/'E6_intervals.csv',index=False)
 # Distinguish interaction from the square J=A^2/supply_variance and interaction
 # in learned aggregate responses. At the first fit, A is exactly additive.
 channels=[]
 for seed in r.CFG['seeds']:
  baseJ=sidx.loc[(seed,0.,'none',0),'J']
  for n in range(r.CFG['slow_updates']+1):
   cc0={}
   for x in r.CFG['adoption']:
    a0=sidx.loc[(seed,x,'none',n),'A'];a1=sidx.loc[(seed,x,'group1',n),'A'];a2=sidx.loc[(seed,x,'group2',n),'A'];ab=sidx.loc[(seed,x,'both',n),'A']
    da=ab-a1-a2+a0
    mechanical=2*(a1-a0)*(a2-a0)/e.ZETA2
    feedback=(ab**2-(a1+a2-a0)**2)/e.ZETA2
    rawJ=(ab**2-a1**2-a2**2+a0**2)/e.ZETA2
    assert abs(mechanical+feedback-rawJ)<1e-10
    if n==1:assert abs(da)<1e-10 and abs(feedback)<1e-10
    cc0[x]=dict(A_interaction=da,J_mechanical=mechanical,J_feedback=feedback,J_interaction=rawJ)
   for x,q in cc0.items():
    row=dict(seed=seed,x=x,round=n,**q)
    for key in q:
     row[key+'_adjusted']=q[key]-cc0[0.][key]
     if key.startswith('J_'):row[key+'_adjusted_pct_baseJ']=100*row[key+'_adjusted']/baseJ
    channels.append(row)
 pd.DataFrame(channels).to_csv(out/'interaction_channels.csv',index=False)
 # Cumulative E7: only structural quantities telescope across market laws.
 cumulative=[]
 structural=['J','A','low_iota','high_iota','other_iota','A_group1','A_group2']
 for seed in r.CFG['seeds']:
  for x in r.CFG['adoption']:
   for arm in r.ARMS:
    for n in range(r.CFG['slow_updates']+1):
     for metric in structural:
      q=st[(st.seed==seed)&(st.x==x)&(st.arm==arm)&(st['update']<=n)&(st.metric==metric)]
      m=q.mean_contribution.sum();v=q.variance_contribution.sum()
      actual=sidx.loc[(seed,x,arm,n),metric]-sidx.loc[(seed,x,arm,0),metric]
      assert abs(m+v-actual)<1e-9
      cumulative.append(dict(seed=seed,x=x,arm=arm,round=n,metric=metric,mean_contribution=m,variance_contribution=v,total=actual))
 cum=pd.DataFrame(cumulative);cum.to_csv(out/'E7_cumulative.csv',index=False);cx=cum.set_index(['seed','x','arm','round','metric'])
 adoption=pd.read_csv(e.ROOT/'adoption_e7/decomposition.csv').set_index(['seed','x','metric'])
 adj=[]
 for row in eff[eff.metric=='J'].itertuples():
  q=cx.loc[(row.seed,row.x,row.arm,row.round,'J')];z=cx.loc[(row.seed,0.,row.arm,row.round,'J')]
  a=adoption.loc[(row.seed,row.x,'J')];assert abs(a.total-row.direct_adoption)<1e-9
  mean=q.mean_contribution-z.mean_contribution;risk=q.variance_contribution-z.variance_contribution
  assert abs(mean+risk-row.extra_retraining)<1e-9
  adj.append(dict(seed=row.seed,x=row.x,arm=row.arm,round=row.round,baseline_J=row.baseline_J,
   adoption_mean_pct=100*a.mean_contribution/row.baseline_J,adoption_variance_pct=100*a.variance_contribution/row.baseline_J,
   extra_retraining_mean_pct=100*mean/row.baseline_J,extra_retraining_variance_pct=100*risk/row.baseline_J,
   total_mean_pct=100*(a.mean_contribution+mean)/row.baseline_J,total_variance_pct=100*(a.variance_contribution+risk)/row.baseline_J,
   total_pct=100*row.total_adoption/row.baseline_J))
 adj=pd.DataFrame(adj);adj.to_csv(out/'E7_adoption_adjusted.csv',index=False)
 cc=pd.DataFrame(checks);cc.to_csv(out/'state_validation.csv',index=False)
 expected=3*3*(1+3*7);assert len(cc)==expected and len(s)==3*3*4*7
 audit=dict(passed=True,simulated_states=len(cc),reported_state_horizons=len(s),none_reused_horizons=len(s)-len(cc),
  seeds=r.CFG['seeds'],arms=list(r.ARMS),adoption=r.CFG['adoption'],slow_updates=6,training_banks=3*3*3*6,
  maximum_clearing_residual=float(cc.clearing_error.max()),maximum_initial_price_coupling_error=float(cc.mixing_error.max()),minimum_clearing_slope=float(cc.delta.min()),
  frozen_groups_exact=True,initial_profiles_identical=True,first_update_blocks_match=True,checkpoint_continuity_exact=True,
  generation_before_learning=True,E6_factorial_identities=True,E7_shapley_and_telescoping_identities=True,
  interpretation='Finite six-update comparisons from E5 candidate baselines, not equilibrium comparative statics; intervals use three paired seed-level contrasts.')

 for filename,sha in json.loads((e.ROOT/'SOURCE_HASHES.json').read_text()).items():assert hashlib.sha256((e.ROOT/filename).read_bytes()).hexdigest()==sha
 audit['source_hashes_verified']=True
 audit['first_update_A_additivity_and_J_square_identity']=True
 audit['versions']=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__)
 e.dump(out/'VALIDATION.json',audit)
 print(json.dumps(audit,indent=2));print(fac[(fac.metric=='J')&(fac['round']==6)].groupby('x')[[k+'_pct_baseJ' for k in ['direct','group1','group2','interaction','both','total']]].mean().to_string())
 print(adj[(adj.arm=='both')&(adj['round']==6)].groupby('x').mean(numeric_only=True).to_string())
if __name__=='__main__':main()
