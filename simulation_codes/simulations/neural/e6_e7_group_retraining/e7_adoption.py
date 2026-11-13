"""Mean/risk swaps for the initial low-to-high adoption comparison."""
import json
import pandas as pd
import torch
import engine as e
from run_e6 import load_baseline,e7,CFG

def main():
 torch.set_num_threads(1);out=e.ROOT/'adoption_e7';out.mkdir(exist_ok=True)
 rows=[];parts=[];meta=[]
 for seed in CFG['seeds']:
  models,other=load_baseline(seed)
  p,g=e.simulate(models,other,0.,seed,70000,60,CFG['evaluation_paths'],CFG['evaluation_periods'],CFG['burn']);meta.append(g)
  # The upgrading cell initially uses the SAME low model as non-upgraders.
  before={'low':models['low'],'high':models['low']}
  for x in CFG['adoption']:
   swaps,dec=e7(before,other,models,other,x,p)
   for a in [swaps,dec]:a['seed']=seed;a['x']=x
   rows.append(swaps);parts.append(dec)
 pd.concat(rows).to_csv(out/'swaps.csv',index=False)
 pd.concat(parts).to_csv(out/'decomposition.csv',index=False)
 e.dump(out/'generation.json',meta)
 print('Initial adoption swaps complete',flush=True)
if __name__=='__main__':main()
