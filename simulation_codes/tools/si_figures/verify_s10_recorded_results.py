"""Independent, light-weight checks of recorded E8/E9/E10 aggregate results.
Never retrains neural networks; preserves the saved protocol and source lineage.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
R=Path(__file__).resolve().parent/'data'
E8=R/'E8_learning_changes.csv'
checks=[]
def check(name, x, expected, atol=6e-5):
    if not np.isfinite(x) or abs(x-expected)>atol:
        raise AssertionError((name,x,expected,atol))
    checks.append(name)
# E8 reported adoption 0->1 levels in the original V6 CSV
v=pd.read_csv(E8)
variants=[('original_1x',.00814,-.00943),('original_2x',.00813,-.00942),('original_4x',.00809,-.00943),('newrep1_1x',.00810,-.00942),('newrep2_1x',.00809,-.00941)]
for name,t,u in variants:
    for metric,goal in [('E3_T_high',t),('E4_neural_high_upgrade',u)]:
        sub=v[(v.variant==name)&(v.x==1.0)&(v.metric==metric)];assert len(sub)==1
        check('E8_'+name+'_'+metric,float(sub['mean'].iloc[0]),goal)
# E9 exact interval and zero/period2 screens
a=pd.read_csv(R/'E9_intervals.csv');d=pd.read_csv(R/'E9_diagnostics.csv')
for mode,x,mean in [('fresh',0,.352),('fresh',.5,.388),('fresh',1,.424),('common_shocks',.5,.169)]:
    q=a[(a['mode']==mode)&(a.x==x)&(a.metric=='J_total_extension_pct')]
    assert len(q)==1
    check(f'E9_{mode}_{x}_J',float(q['mean'].iloc[0]),mean,6e-4)
    rows=d[(d['mode']==mode)&(d.x==x)]; assert len(rows)==3
    check(f'E9_{mode}_{x}_settling',int(rows.stable_05pct_last3.sum()),3 if mode=='common_shocks' else 0,0)
    check(f'E9_{mode}_{x}_period2',int(rows.period2_screen.sum()),0,0)
# Four of twelve one-step fixed-law checks fall inside ±1%
audit=pd.read_csv(R/'E9_fixed_law_audit.csv')
assert len(audit)==12
check('E9_fixed_law_4of12',int(audit.within_1pct.sum()),4,0)
check('E9_other_group_negative_CI',int((audit[audit.metric=='other_iota'].ci_high<0).sum()),3,0)
# held-out policy functional change benchmark (last three updates, seed x .5)
f=pd.read_csv(R/'E9_functional_changes.csv')
for mode,cap,metric,goal in [('fresh','low','forecast_RMS_change_pct_payoffSD',1.3832),('fresh','high','demand_RMS_change_pct_initialRMS',5.0188),('common_shocks','low','forecast_RMS_change_pct_payoffSD',.0523),('common_shocks','other','demand_RMS_change_pct_initialRMS',.0730)]:
    w=f[(f['mode']==mode)&(f.x==.5)&(f.capacity==cap)&f['round'].isin([10,11,12])]
    check(f'E9_function_{mode}_{cap}_{metric}',float(w[metric].mean()),goal,1e-4)
# E10 estimate at full adoption pooled over original seeds
chg=pd.read_csv(R/'E10_changes.csv');lev=pd.read_csv(R/'E10_levels.csv');s=pd.read_csv(R/'E10_seed_changes.csv')
for econ,j,t,u in [('control',-6.645,.00962,-.01113),('nonaligned',-4.386,.01135,-.01140),('heteroskedastic',-6.637,.00968,-.01095)]:
    for metric,goal in [('J_impact_pct',j),('E3_T_high',t),('E4_neural_high_upgrade',u)]:
        w=chg[(chg.economy==econ)&(chg.x==1.0)&(chg.metric==metric)]
        assert len(w)==1
        check(f'E10_{econ}_{metric}',float(w['mean'].iloc[0]),goal,7e-4 if metric=='J_impact_pct' else 1e-5)
    w=s[(s.economy==econ)&(s.x==1.0)];assert len(w)==3
    check(f'E10_{econ}_J_sign',int((w.J_impact_pct<0).sum()),3,0)
    check(f'E10_{econ}_T_sign',int((w.E3_T_high>0).sum()),3,0)
    check(f'E10_{econ}_U_sign',int((w.E4_neural_high_upgrade<0).sum()),3,0)
    for x,goal in [(0,dict(control=.02134,nonaligned=.02835,heteroskedastic=.02110)[econ]),(1,dict(control=.01021,nonaligned=.01695,heteroskedastic=.01015)[econ])]:
        w=lev[(lev.economy==econ)&(lev.x==x)&(lev.metric=='E4_neural_high_upgrade')];assert len(w)==1
        check(f'E10_{econ}_{x}_U_level',float(w['mean'].iloc[0]),goal,1e-5)
# Stocked data fingerprints
manifest=json.loads((R/'VALIDATION.json').read_text())
check('E9_trajectories',manifest['E9']['trajectories'],12,0)
check('E10_frozen_laws',manifest['E10']['frozen_laws'],27,0)
check('E9_max_clearing_error',manifest['E9']['max_clearing_error'],0,1e-12)
check('E10_max_clearing_error',manifest['E10']['max_clearing_error'],0,1e-12)
print('S10 lightweight checks passed:',len(checks))
for x in checks:print('PASS',x)
