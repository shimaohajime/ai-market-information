"""Two prespecified departures from aligned, homoskedastic dividends."""
import numpy as np
from common import e
MODES=['control','nonaligned','heteroskedastic']
S=e.ECO.stationary_cov;ell=e.ECO.target_loading
q=np.eye(8)[0];q=q-ell*(q@S@ell)/(ell@S@ell);PSI=q/np.sqrt(q@S@q)
qvar=float(e.W[0]@S@e.W[0]);PUBLIC_SD=np.sqrt(qvar+12*qvar*qvar+60*qvar**3+.04)

def draw(mode,seed,key,bank,paths,steps):
 rng=np.random.default_rng(np.random.SeedSequence([20261028,seed,key,bank]));f=rng.multivariate_normal(np.zeros(8),S,size=paths)
 shocks=rng.standard_normal((steps,paths,e.ECO.G.shape[1]));states=np.empty((steps,paths,8))
 for t in range(steps):f=f@e.ECO.F.T+shocks[t]@e.ECO.G.T;states[t]=f
 old=states@ell;eta=rng.normal(0,np.sqrt(e.EQ['prior']),old.shape);theta=np.sqrt(.8)*old+np.sqrt(.2)*eta
 q=states@e.W.T;public=q+2*q**3+.2*rng.standard_normal(q.shape);eps=rng.standard_normal(theta.shape)
 supply=.9*rng.standard_normal(theta.shape);private=theta[:,:,None]+np.sqrt(e.EQ['nu'])*rng.standard_normal((*theta.shape,2))
 mean=e.LAM*theta;nv=np.full(theta.shape,e.SIG2)
 if mode=='nonaligned':mean=e.LAM*(np.sqrt(.8)*theta+np.sqrt(.2*e.EQ['prior'])*(states@PSI))
 elif mode=='heteroskedastic':nv=e.SIG2*(1+.6*np.tanh(public[:,:,0]/PUBLIC_SD))
 elif mode!='control':raise ValueError(mode)
 y=mean+np.sqrt(nv)*eps
 return dict(public=public,theta=theta,y=y,supply=supply,private=private,payoff_mean=mean,noise_variance=nv)

def simulate(mode,m,o,x,seed,key,bank,paths,periods,burn=512,initial_price=0.,return_trace=False):
 shocks=draw(mode,seed,key,bank,paths,burn+periods+e.WINDOW-1)
 p,meta=e.simulate(m,o,x,seed,key,bank,paths,periods,burn,initial_price,shocks,return_trace)
 for k in ['payoff_mean','noise_variance']:p[k]=np.swapaxes(shocks[k][burn:],0,1).copy()
 meta['economy']=mode;return p,meta
