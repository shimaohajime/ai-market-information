import torch
from common import *
from economies import MODES,simulate
if __name__=='__main__':
 torch.set_num_threads(1)
 for mode in MODES:
  for seed in SEEDS:
   m,o=load_profile(ROOT/'baselines'/f'seed{seed}');out=ROOT/'results'/'E10'/mode/f'seed{seed}'/'inherited_fit'
   a,ma=simulate(mode,m,o,0,seed,101000,80,16,64,initial_price=-10,return_trace=True);b,mb=simulate(mode,m,o,0,seed,101000,80,16,64,initial_price=10,return_trace=True)
   error=float(np.max(abs(a['trace'][512:]-b['trace'][512:])));assert error<1e-5
   e.dump(out/'inherited_mixing.json',dict(error=error,profile_sha256=e.fingerprint(m,o),generation=[ma,mb]))
 print('All nine inherited-law coupling checks passed')
