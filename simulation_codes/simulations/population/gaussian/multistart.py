"""Local basin and orbit checks (not global uniqueness claims)."""
import json
from dataclasses import replace
import numpy as np
from benchmark import Benchmark
from pathlib import Path
OUT=Path(__file__).resolve().parent

def multistart(sd,caps=(3,0),repeats=12,iterations=2500,seed=215):
    rng=np.random.default_rng(seed);b=Benchmark()
    b.eco=replace(b.eco,supply_noise_sd=sd);b.sigma_supply=sd
    base=b.seed()
    outs=[]
    for i in range(repeats):
        q=base.copy()
        q[0]*=np.exp(rng.uniform(-.1,.1));q[-1]*=np.exp(rng.uniform(-.1,.1))
        q[1:-1]+=rng.normal(0,.005,len(q)-2)
        try:
            for _ in range(iterations):q=b.update(q,caps)
            a=b.J(q); bb=b.J(b.update(q,caps));
            outs.append({'id':i,'J_even':a,'J_odd':bb,'distance_two':float(max(abs(q-b.update(b.update(q,caps),caps))))})
        except Exception as exc:outs.append({'id':i,'error':str(exc)})
    return outs
if __name__=='__main__':
    out={'sd09':multistart(.9,iterations=250), 'sd0715':multistart(.715,iterations=2500)}
    print(json.dumps(out,indent=2))
    with open(OUT/'multistart.json','w') as f:json.dump(out,f,indent=2)
