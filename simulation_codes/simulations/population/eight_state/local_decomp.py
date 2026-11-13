import sys,json,numpy as np,math,csv
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from audit_theorem2 import statistics
sys.path[:0]=[str(HERE/'prior'/'src'),str(HERE/'prior'/'vendor')]
from adoption_spillovers import Benchmark,make_b
from run_redqueen import stationary_population,train_cells,map_shares

def compute(b,x,protocol,trained=None,init=None):
    if protocol=='frozen':return map_shares(b,None,(x,0.),((0,3),(0,3)),trained)
    else:return stationary_population(b,(x,0.),((0,3),(0,3)),init=init)[0]

def deriv(b,protocol,x=.5,step=1e-4):
    q0=stationary_population(b,(0,0))[0]
    trained=train_cells(b,q0)
    q=compute(b,x,protocol,trained,q0)
    qm=compute(b,x-step,protocol,trained,q)
    qp=compute(b,x+step,protocol,trained,q)
    a,m,z=(statistics(b,w) for w in (qm,q,qp))
    D={k:(z[k]-a[k])/(2*step) for k in ['P_L','P_H','L','H','S','F','V','vL','vH','J']}
    rL=m['L'];rH=m['H'];nu=m['nu'];lam=m['lam'];gamma=m['gamma']
    def weight(r,v):return lam**2*nu**2*r/(v*(nu+r)**2)
    wL=weight(rL,m['vL']);wH=weight(rH,m['vH'])
    kL=D['P_L']/m['P_L']-2*D['S'];kH=D['P_H']/m['P_H']-2*D['F']
    c_short_MI=(-2*wL*D['S'])/(2*gamma)
    c_high_MI=(+2*wH*D['F'])/(2*gamma)
    c_low_prior=(wL*D['P_L']/m['P_L'])/(2*gamma)
    c_high_prior=(-wH*D['P_H']/m['P_H'])/(2*gamma)
    dd={'v':m,'d':D,'weights':{'wL':wL,'wH':wH},'components':{'low_spillover':c_short_MI,'high_fresh':c_high_MI,'low_prior':c_low_prior,'high_prior':c_high_prior,'sum':sum([c_short_MI,c_high_MI,c_low_prior,c_high_prior])},'dx_log_pH':D['P_H']/m['P_H'],'dx_log_H':kH}
    return dd

if __name__=='__main__':
 b=Benchmark(K=3);out={}
 for protocol in ('frozen','retrained'):
  d=deriv(b,protocol)
  print('\n',protocol,'x=.5')
  for key in ('P_L','P_H','L','H','S','F','V','J'):print(key,'value',d['v'][key],'derivative',d['d'][key])
  print('weights',d['weights'],'contributions',d['components'])
  out[protocol]=d
 (HERE/'baseline_derivative_decomposition.json').write_text(json.dumps(out,indent=2))
 # examples where P_H drops and H drops but V still falls
 r=list(csv.DictReader(open(HERE/'aligned_condition_audit.csv')))
 for protocol in ('frozen','retrained'):
  group=[rr for rr in r if rr['protocol']==protocol and rr['span']=='p25_to_p75']
  negH=[x for x in group if float(x['H']) < -1e-10]
  print('negative H examples',protocol,[(e['index'],e['P_H'],e['L'],e['H'],e['S'],e['F'],e['V']) for e in negH]);
