import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.special import logsumexp
import engine as e
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'summary';OUT.mkdir(exist_ok=True)

def main():
 markets=json.loads((ROOT/'markets.json').read_text());allbanks=[];ce=[];values=[];bootrows=[];audits=[]
 for market in markets:
  seed=market['seed'];x=market['x'];p=ROOT/'results'/f'seed{seed}_x{round(x*100):03d}';done=json.loads((p/'complete.json').read_text());assert done['market_sha256']==market['profile_sha256']
  banks=pd.read_csv(p/'banks.csv');paths=pd.read_csv(p/'paths.csv');cc=pd.read_csv(p/'CE.csv');allbanks.append(banks);ce.append(cc)
  generation=json.loads((p/'generation.json').read_text());train=json.loads((p/'training.json').read_text())
  assert [v['bank'] for v in train['generation']]==[1,2]
  assert [v['bank'] for v in generation][-5:]==[3,10,11,12,13]
  assert all(v['profile_sha256']==market['profile_sha256'] for v in generation+train['generation'])
  assert all(v['updates']==4096 and v['optimization_windows']==1048576 for v in train['details'].values())
  # Verify immutable market files and repeated x-invariant candidate initialization provenance.
  source=ROOT/market['path'];models={c:e.load_checkpoint(source/(c+'_deployed.pt'))[0] for c in ['low','high']};other=json.loads((source/'other_deployed.json').read_text());assert e.fingerprint(models,other)==market['profile_sha256']
  audits.append(dict(seed=seed,x=x,clearing_error=max(v['clearing_error'] for v in generation),profile_matches=True))
  row=dict(seed=seed,x=x)
  for c in ['low','high','nested']:
   sub=banks[(banks.kind=='E3')&(banks.model==c)]
   row['E3_'+c+'_gain']=sub.gain.mean();row['E3_'+c+'_MSE_no_price']=sub.MSE_no_price.mean();row['E3_'+c+'_MSE_price']=sub.MSE_price.mean();row['E3_'+c+'_relative_gain']=1-sub.MSE_price.mean()/sub.MSE_no_price.mean()
  for c in ['high','nested']:
   row['E3_T_'+c]=row['E3_low_gain']-row['E3_'+c+'_gain']
   assert abs(row['E3_T_'+c]-((row['E3_low_MSE_no_price']-row['E3_'+c+'_MSE_no_price'])-(row['E3_low_MSE_price']-row['E3_'+c+'_MSE_price'])))<1e-12
  cs=cc.set_index('model')
  for c in cs.index:row[c+'_CE']=cs.loc[c,'CE']
  for kind,cap in [('neural','high'),('ols','high'),('ols','nested')]:
   prefix='E4_'+kind;label=prefix+'_'+cap+'_upgrade';row[label]=cs.loc[prefix+'_'+cap,'CE']-cs.loc[prefix+'_low','CE']
   # Same path indices for both models; independent paths within each bank.
   a=paths[paths.model==prefix+'_'+cap].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256)
   b=paths[paths.model==prefix+'_low'].sort_values(['bank','path']).logmeanweight.to_numpy().reshape(4,256)
   rng=np.random.default_rng(20261029+seed);boots=[]
   for rep in range(1000):
    ix=rng.integers(0,256,(4,256));aa=np.take_along_axis(a,ix,1);bb=np.take_along_axis(b,ix,1)
    boots.append(-(logsumexp(aa)-logsumexp(bb))/e.GAMMA[0])
   lo,hi=np.quantile(boots,[.025,.975]);upbanks=banks[(banks.kind=='E4')&(banks.model==prefix+'_'+cap)].set_index('bank').CE-banks[(banks.kind=='E4')&(banks.model==prefix+'_low')].set_index('bank').CE
   bootrows.append(dict(seed=seed,x=x,metric=label,estimate=row[label],evaluation_boot_low=lo,evaluation_boot_high=hi,bank_min=upbanks.min(),bank_max=upbanks.max(),first_half_upgrade=cs.loc[prefix+'_'+cap,'CE_first_half']-cs.loc[prefix+'_low','CE_first_half'],last_half_upgrade=cs.loc[prefix+'_'+cap,'CE_last_half']-cs.loc[prefix+'_low','CE_last_half']))
  values.append(row)
 v=pd.DataFrame(values);v.to_csv(OUT/'seed_results.csv',index=False);pd.concat(allbanks).to_csv(OUT/'bank_results.csv',index=False);pd.concat(ce).to_csv(OUT/'CE_diagnostics.csv',index=False);pd.DataFrame(bootrows).to_csv(OUT/'CE_evaluation_uncertainty.csv',index=False)
 intervals=[]
 for x,group in v.groupby('x'):
  for metric in v.columns.drop(['seed','x']):
   for contrast in ['level','change_from_x0']:
    a=group.set_index('seed')[metric]
    if contrast=='change_from_x0':a=a-v[v.x==0].set_index('seed')[metric]
    mean=a.mean();hw=t.ppf(.975,2)*a.std(ddof=1)/np.sqrt(3)
    intervals.append(dict(x=x,metric=metric,contrast=contrast,mean=mean,ci_low=mean-hw,ci_high=mean+hw,min=a.min(),max=a.max(),n_seeds=3))
 pd.DataFrame(intervals).to_csv(OUT/'intervals.csv',index=False)
 e.dump(OUT/'VALIDATION.json',dict(markets=9,market_audits=audits,all_pass=True,total_evaluation_observations=9*4*256*1024,independent_training_seeds=3,tests='TEST_RESULTS.txt'))
 print(v[['seed','x','E3_low_gain','E3_high_gain','E3_T_high','E4_neural_high_upgrade','E4_ols_nested_upgrade']].to_string(index=False))
if __name__=='__main__':main()
