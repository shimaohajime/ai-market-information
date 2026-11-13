"""Independent arithmetic and provenance checks of the population S7 printed moments.
Checks recorded outputs, not a fresh full end-to-end computation of the 147-model sweep.
"""
import csv,json,math
from pathlib import Path
base=Path(__file__).resolve().parent.parent/'data'
get=lambda name:json.loads((base/name).read_text())
with open(base/'baseline_finiteK3.csv',newline='') as f: r=list(csv.DictReader(f))
p=get('population_results.json'); a=get('ablation_population_results.json');d=get('baseline_derivative_decomposition.json'); s=get('sweep_results.json')
checks=[]
def near(label,value,expected,tol=1e-6):
 assert abs(value-expected)<tol,(label,value,expected)
 checks.append(label)
getrow=lambda x,protocol:next(z for z in r if abs(float(z['x'])-x)<1e-10 and z['protocol']==protocol)
cap=p['capacities'][0]
near('J baseline',cap['J_baseline'],2.469506627264)
near('J initial',cap['J_impact'],2.262529331203)
near('J retrained',cap['J_equilibrium'],2.073514101203)
near('Impact percentage',100*(cap['J_impact']/cap['J_baseline']-1),-8.38132175)
near('Long-run percentage',100*(cap['J_equilibrium']/cap['J_baseline']-1),-16.03528906)
near('Baseline x0 frozen J',float(getrow(0,'frozen')['J']),cap['J_baseline'])
for k,v in [('S',0.357054528261),('F',0.178380476926),('V',0.023256970273),('P_L',0.422094934686),('P_H',0.173595122372)]:near('Baseline '+k,float(getrow(0,'retrained')[k]),v)
for k,v in [('S',0.544236344255),('F',0.165813133444),('V',0.006082728458),('P_H',0.173801969691)]:near('Frozen endpoint '+k,float(getrow(1,'frozen')[k]),v)
for k,v in [('S',0.483883),('F',0.154406),('V',0.010213),('P_H',0.174489)]:near('Retrained endpoint '+k,float(getrow(1,'retrained')[k]),v,tol=2e-6)
for key,expected in [('mean_old_risk_old',2.469506627264),('mean_new_risk_old',2.108367918595),('mean_old_risk_new',2.735404208171),('mean_new_risk_new',2.262529331203)]:near('Factorial '+key,a['factorial'][key]['J'],expected)
near('Mean Shapley',a['shapley']['mean'],-.417006792819)
near('Variance Shapley',a['shapley']['risk'],.210029496757)
near('Shapley consistency',a['shapley']['mean']+a['shapley']['risk'],a['factorial']['mean_new_risk_new']['J']-a['factorial']['mean_old_risk_old']['J'])
for key,v in [('none',2.262529331203),('only_upgraded',2.272432427948),('only_other',1.994345807115),('both',2.073514101203)]:near('Retraining permission '+key,a['retraining_arms'][key]['J'],v)
assert len(r)==42 and len({(v['x'],v['protocol']) for v in r})==42; checks.append('21 adoption shares and two protocols')
assert s['total']==180 and s['valid']==147 and s['invalid']==33 and len(s['records'])==147;checks.append('Screen denominator 180 = 147+33')
with open(base/'aligned_condition_audit.csv') as f: audits=list(csv.DictReader(f))
sel=[v for v in audits if v['span']=='p25_to_p75']
assert len(sel)==294;checks.append('147 calibration endpoint contrasts for both protocols')
for prot in ['frozen','retrained']:
 rr=[v for v in sel if v['protocol']==prot];assert len(rr)==147
 for flag in ['positive_spillover','negative_fresh','strategic_substitution']:assert sum(v[flag]=='True' for v in rr)==147
 counts=(sum(v['prior_increase']=='True' for v in rr),sum(v['bridge_weak']=='True' for v in rr))
 assert counts==(76,140) if prot=='frozen' else counts==(108,145),(prot,counts)
 checks.append('Screen endpoint signs '+prot)
with open(base/'aligned_local_derivative_audit.csv') as f: loc=list(csv.DictReader(f))
assert len(loc)==294 and all(float(v['sharp_margin'])>0 for v in loc);checks.append('Local sharp condition all 294')
# S5 derivative components in source
contrib=d['retrained']['components']
near('S5 bridge components sum',sum(contrib[k] for k in ('low_spillover','high_fresh','low_prior','high_prior')),contrib['sum'],tol=1e-9)
near('CE derivative',contrib['sum'],-.01246170655)
near('Source proxy SD 0.43 squared',.43**2,.1849)
print(f'PASS: {len(checks)} S7 source-derived value checks')
for i,c in enumerate(checks,1):print(f'{i:02d} {c}')
