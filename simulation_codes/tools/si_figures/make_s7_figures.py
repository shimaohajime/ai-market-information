"""Regenerate Science SI S7 figures from verified exact Gaussian population outputs.
This script does not refit models or draw stochastic samples.
"""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
DATA = HERE/'data'
OUT = HERE/'figures'
OUT.mkdir(exist_ok=True)
with open(DATA/'population_results.json') as f: p=json.load(f)
with open(DATA/'baseline_finiteK3.csv',newline='') as f: rows=list(csv.DictReader(f))

# Figure S7.1: structural private-information precision; initial vs population-retrained
arr=p['adoption']
xx=np.array([r['adoption'] for r in arr]); jj=p['capacities'][0]['J_baseline']
fig,ax=plt.subplots(figsize=(7.4,3.75))
ax.plot(xx,[r['J_fixedlaw'] for r in arr],linestyle='--',linewidth=2,label='First deployment / incumbents fixed')
ax.plot(xx,[r['J_longrun'] for r in arr],linestyle='-',linewidth=2,label='Population retraining equilibrium')
ax.axhline(jj,linewidth=1,linestyle=':',label='Before adoption')
ax.set(xlabel='Share of investor group 1 using high-capacity forecasting',ylabel='Fresh-price likelihood precision $J$')
ax.set_xlim(0,1);ax.grid(alpha=.18);ax.legend(loc='best',frameon=False,fontsize=8.2)
fig.tight_layout();fig.savefig(OUT/'S7_population_precision.pdf',bbox_inches='tight');plt.close(fig)

# Figure S7.2: each observer's incremental price information at the same K=3 field
fig,ax=plt.subplots(figsize=(7.4,3.85))
for k,style in [('retrained','-'),('frozen','--')]:
 z=[r for r in rows if r['protocol']==k]
 ax.plot([float(r['x']) for r in z],[float(r['S']) for r in z],linestyle=style,linewidth=2,label=f'Limited observer · {k}')
 ax.plot([float(r['x']) for r in z],[float(r['F']) for r in z],linestyle=style,linewidth=2,label=f'High-capacity observer · {k}')
ax.set(xlabel='Share of investor group 1 using high-capacity forecasting',ylabel='Incremental price information (nats)');ax.grid(alpha=.18);ax.set_xlim(0,1)
ax.legend(frameon=False,ncol=2,fontsize=7.6,loc='upper center',bbox_to_anchor=(.5,1.02))
fig.tight_layout();fig.savefig(OUT/'S7_price_information.pdf',bbox_inches='tight');plt.close(fig)

# Figure S7.3: marginal candidate CE upgrade value, same law low/high refits
fig,ax=plt.subplots(figsize=(7.4,3.6))
for k,sty in [('retrained','-'),('frozen','--')]:
 z=[r for r in rows if r['protocol']==k]
 ax.plot([float(r['x']) for r in z],[float(r['V']) for r in z],linestyle=sty,linewidth=2,label=k.capitalize())
ax.set(xlabel='Share of investor group 1 using high-capacity forecasting',ylabel='Marginal candidate upgrade value (CE units)')
ax.grid(alpha=.18);ax.set_xlim(0,1);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(OUT/'S7_computation_value.pdf',bbox_inches='tight');plt.close(fig)

print('Created',*[str(x) for x in sorted(OUT.glob('*.pdf'))],sep='\n')
