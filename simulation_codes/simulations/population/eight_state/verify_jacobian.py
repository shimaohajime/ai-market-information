from pathlib import Path
import sys,numpy as np,json
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'prior'/'vendor'),str(HERE/'prior'/'src')]
from audit_theorem2 import statistics
from benchmark import Benchmark
from run_redqueen import stationary_population,train_cells,map_shares
from sensitivity import analytic_map_jac
b=Benchmark(K=3)
x=.5
q=stationary_population(b,(x,0))[0]
J=analytic_map_jac(b,q,adoption=x,lo=0,hi=3)
assert np.max(abs(J))<1e5
p=train_cells(b,q,(0,3))
lo,hi=p[0,0],p[0,3];pi=b.temps[0].mass
G=np.r_[pi*(hi['beta']/(hi['gamma']*hi['v'])-lo['beta']/(lo['gamma']*lo['v'])), pi*(hi['b']/(hi['gamma']*hi['v'])-lo['b']/(lo['gamma']*lo['v'])),pi*((b.eco.risk_free_gross-hi['chi'])/(hi['gamma']*hi['v'])-(b.eco.risk_free_gross-lo['chi'])/(lo['gamma']*lo['v']))]
dq=np.linalg.solve(np.eye(len(q))-J,G)
h=1e-5
qp=stationary_population(b,(x+h,0),init=q)[0]
qm=stationary_population(b,(x-h,0),init=q)[0]
dq_fd=(qp-qm)/(2*h)
print('Jac radius',max(abs(np.linalg.eigvals(J))))
print('dq derivative max absolute err',max(abs(dq-dq_fd)))
print('dq max component',max(abs(dq)))
for field in ['P_L','P_H','L','H','S','F','V','J']:
 grad=np.empty(len(q))
 for k in range(len(q)):
  e=np.zeros(len(q));e[k]=1e-5*max(1,abs(q[k]))
  grad[k]=(statistics(b,q+e,checks=False)[field]-statistics(b,q-e,checks=False)[field])/(2*e[k])
 der=grad@dq
 fd=(statistics(b,qp)[field]-statistics(b,qm)[field])/(2*h)
 print(field,'IFT',der,'FD',fd,'abs diff',abs(der-fd))
