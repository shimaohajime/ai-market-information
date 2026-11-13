import sys,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'prior'/'src'),str(HERE/'prior'/'vendor')]
from audit_theorem2 import statistics
from benchmark import Benchmark
from run_redqueen import stationary_population,train_cells,map_shares
b=Benchmark(K=3)
q0,_=stationary_population(b,(0,0));trained=train_cells(b,q0)
q_last=q0.copy();rows=[]
for x in np.linspace(0,1,21):
 qf=map_shares(b,None,(x,0),((0,3),(0,3)),trained)
 qr,_=stationary_population(b,(x,0),init=q_last);q_last=qr.copy()
 for protocol,q in [('frozen',qf),('retrained',qr)]:
  row=statistics(b,q);row.update(x=x,protocol=protocol);rows.append(row)
with open(HERE/'baseline_finiteK3.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(2,2,figsize=(10,6.9),layout='constrained')
for p,sty in [('frozen','--'),('retrained','-')]:
 xx=np.array([r['x'] for r in rows if r['protocol']==p]);rr=[r for r in rows if r['protocol']==p]
 axs[0,0].plot(xx,[r['S'] for r in rr],linestyle=sty,label=f'Short info · {p}',lw=2)
 axs[0,0].plot(xx,[r['F'] for r in rr],linestyle=sty,label=f'Fresh info · {p}',lw=2)
 axs[0,1].plot(xx,[r['P_H'] for r in rr],linestyle=sty,label=f'High prior · {p}',lw=2)
 axs[0,1].plot(xx,[r['P_L'] for r in rr],linestyle=sty,label=f'Low prior · {p}',lw=2)
 axs[1,0].plot(xx,[r['L'] for r in rr],linestyle=sty,label=f'Low postprice · {p}',lw=2)
 axs[1,0].plot(xx,[r['H'] for r in rr],linestyle=sty,label=f'High postprice · {p}',lw=2)
 axs[1,1].plot(xx,[r['V'] for r in rr],linestyle=sty,label=f'CE premium · {p}',lw=2)
for ax,title,yl in zip(axs.flat,['Price information, finite K=3','Pre-price uncertainty','Post-price uncertainty','Computation upgrade value'],['Nats','Conditional variance','Conditional variance','CARA CE units']):
 ax.set(title=title,xlabel='High-capacity adoption share x',ylabel=yl);ax.grid(alpha=.25);ax.legend(fontsize=7,frameon=False)
fig.suptitle('Same 8-state microfounded market: exact population Gaussian conditional moments',fontsize=13)
fig.savefig(HERE/'figure_bridge_8d.pdf',bbox_inches='tight')
fig.savefig(HERE/'figure_bridge_8d.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print('done')
