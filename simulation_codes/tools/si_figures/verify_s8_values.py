"""Independent numeric/provenance checks for S8 manuscript.
Reads archived CSV/JSON; it does not replay neural training.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent
P=R/'recovered'/'phase2_market_lab'/'results'/'phase2_models.csv'
E5=R/'recovered'/'e5_continuation'/'e5_summary'
E6=R/'recovered'/'e6_group_retraining'/'summary'
checks=0

def close(a,b,tol=6e-3,name=''):
    global checks
    a=float(a);b=float(b)
    if not np.isclose(a,b,rtol=0,atol=tol): raise AssertionError(f'{name}: {a} != {b}')
    checks+=1

def yes(cond,name):
    global checks
    if not cond:raise AssertionError(name)
    checks+=1

p=pd.read_csv(P)
y=p[(p.scenario=='E1_aligned')&(p.head_type=='A')]
for arch,width,nll,J in [('ssm',4,.94043,.27184),('ssm',12,.93946,.26559),('gru',4,.95134,.27605),('gru',12,.94584,.27392)]:
    d=y[(y.architecture==arch)&(y.hidden==width)]
    yes(len(d)==3,f'Phase2 3 seeds {arch}{width}')
    close(d.NLL.mean(),nll,0.00001,f'Phase2 NLL {arch}{width}')
    close(d.fresh_precision_E1.mean(),J,0.00001,f'Phase2 J {arch}{width}')
for arch in ['ssm','gru']:
    d=y[y.architecture==arch].pivot(index='seed',columns='hidden',values='fresh_precision_E1')
    v=d[12]-d[4]
    yes((v<0).sum()==2 and (v>0).sum()==1,f'Seed sign variation {arch}')
q=p[p.scenario=='E3_state_risk']
for arch,va,vb in [('ssm',1.03551,1.03053),('gru',1.04616,1.03900)]:
    close(q[(q.architecture==arch)&(q.head_type=='A')].NLL.mean(),va,0.00002,'Model A risk NLL')
    close(q[(q.architecture==arch)&(q.head_type=='B')].NLL.mean(),vb,0.00002,'Model B risk NLL')

z=pd.read_csv(E5/'endpoints.csv')
for x,JJ,gain,iota in [(0,3.3489,3.73,-8.36),(.5,3.1737,2.78,-6.40),(1,3.0647,2.08,-4.90)]:
    d=z[(z.x==x)&(z['round']==12)]
    yes(len(d)==3,f'E5 rows x={x}')
    close(d.J.mean(),JJ,.00011,'E5 J')
    close(d.MSE_gain_pct.mean(),gain,.012,'E5 MSE gain')
    close(d.iota_high_minus_low_pct.mean(),iota,.012,'E5 iota pct')
status=json.load(open(E5/'BASELINE_STATUS.json'))
yes(status['strict_original_passes']==0 and status['strict_original_total']==9,'E5 strict convergence 0/9')
yes(not status['operational_mechanism_baseline_supported'],'E5 baseline not certified')
same=pd.read_csv(E5/'same_law_across_seeds.csv');jr=same[same.metric=='J'].iloc[0]
for a,b in [('mean',-.3828529768789557),('lower',-1.886115274986323),('upper',1.1204093212284116)]:close(jr[a],b,1e-9,'E5 same law '+a)

f=pd.read_csv(E6/'E6_factorial.csv');v=pd.read_csv(E6/'E6_intervals.csv');steps=pd.read_csv(E6/'states.csv');e7=pd.read_csv(E6/'E7_adoption_adjusted.csv')
for x,expected in [(.5,[-3.4031,-.48,-1.83,.48,-5.2314]),(1,[-6.7473,.6322,-3.0037,.6513,-8.4675])]:
    a=f[(f.metric=='J')&(f.x==x)&(f['round']==6)]
    yes(len(a)==3,'E6 3 seeds')
    actual=[a[c].mean() for c in ['direct_pct_baseJ','group1_pct_baseJ','group2_pct_baseJ','interaction_pct_baseJ','total_pct_baseJ']]
    for got,exp in zip(actual,expected):close(got,exp,.008,f'E6 factorial x={x}')
    close(sum(actual[:4]),actual[4],1e-10,'E6 factorial identity')
    for key in ['direct','group1','group2','interaction','total']:
        q=v[(v.x==x)&(v['round']==6)&(v.metric=='J')&(v.units=='percent_of_pre_adoption_J')&(v.component==key)]
        yes(len(q)==1 and int(q.iloc[0].n)==3,f'E6 interval {x} {key}')
        close(q.iloc[0]['mean'],a[dict(direct='direct_pct_baseJ',group1='group1_pct_baseJ',group2='group2_pct_baseJ',interaction='interaction_pct_baseJ',total='total_pct_baseJ')[key]].mean(),1e-9,'E6 CI mean')
for x,arm,vals in [(.5,'none',(-4.90,1.50,-3.40)),(.5,'both',(-2.60,.77,-1.83)),(1,'none',(-9.72,2.97,-6.75)),(1,'group2',(-4.09,1.09,-3.00)),(1,'both',(-2.40,.68,-1.72))]:
    a=e7[(e7.x==x)&(e7.arm==arm)&(e7['round']==6)];yes(len(a)==3,f'E7 rows {x} {arm}')
    keys=('adoption_mean_pct','adoption_variance_pct') if arm=='none' else ('extra_retraining_mean_pct','extra_retraining_variance_pct')
    mm,vv=[a[k].mean() for k in keys]
    close(mm,vals[0],.012,'E7 mean');close(vv,vals[1],.012,'E7 variance');close(mm+vv,vals[2],.015,'E7 net')

yes(len(steps)==252 and int(steps.simulated_state.sum())==198,'198 true states and 54 reused none states')
# structural likelihood precision should have the SAME supply variance across the archive
err=(steps.J-(steps.A**2)/.81).abs().max()
close(err,0,1e-11,'J from A and supply')
va=json.load(open(E6/'VALIDATION.json'))
yes(va['passed'] and va['maximum_clearing_residual']<3e-15,'E6 clearing')
yes(va['generation_before_learning'] and va['frozen_groups_exact'] and va['checkpoint_continuity_exact'],'E6 chronology')
print(f'S8 crosschecks PASSED: {checks}; max |J-A^2/0.81|={err:.2e}')
