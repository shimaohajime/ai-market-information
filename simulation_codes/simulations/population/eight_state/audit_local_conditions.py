import sys,json,time,csv,numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'prior'/'src'),str(HERE/'prior'/'vendor')]
from local_decomp import deriv
from adoption_spillovers import make_b
records=json.loads((HERE/'prior'/'vendor'/'sweep_results.json').read_text())['records']
rows=[];t=time.time()
for n,rec in enumerate(records):
 b=make_b(rec['factors'],'aligned')
 for prot in ['frozen','retrained']:
  x=deriv(b,prot,.5,step=1e-4)
  d=x['d'];m=x['v'];w=x['weights']
  # sharp threshold with PF constant
  threshold=2*d['F']-2*w['wL']/w['wH']*d['S']
  actual=d['P_H']/m['P_H']
  rows.append(dict(index=rec['index'],protocol=prot,
                   short_gain=d['S']>0,fresh_loss=d['F']<0,prior_nondec=d['P_H']>=0,
                   high_post_nondec=d['H']>=0,strategic_sub=d['V']<0,
                   sharp_margin=actual-threshold,threshold=threshold,actual=actual,
                   dS=d['S'],dF=d['F'],dPH=d['P_H'],dL=d['L'],dH=d['H'],dCE=d['V']))
 if n%50==0: print(n,round(time.time()-t,1),flush=True)
with open(HERE/'aligned_local_derivative_audit.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
for prot in ['frozen','retrained']:
 subs=[r for r in rows if r['protocol']==prot]
 print(prot)
 for k in ['short_gain','fresh_loss','prior_nondec','high_post_nondec','strategic_sub']:
  print(k,sum(s[k] for s in subs),'/',len(subs))
 print('sharp margin >0',sum(s['sharp_margin']>0 for s in subs),'min',min(s['sharp_margin'] for s in subs),'median',np.median([s['sharp_margin'] for s in subs]))
print('elapsed',time.time()-t)
