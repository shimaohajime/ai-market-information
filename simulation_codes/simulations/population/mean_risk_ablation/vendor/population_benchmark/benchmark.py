"""Exact population Gaussian benchmark for microfounded retraining in an 8D aligned market.

Law coordinate q=(A, C[0:2+4K], Delta), price p=(A theta+C z-zeta)/Delta.
Analytic Gaussian stationary covariance via discrete Lyapunov. Investor subjective
Gaussian log score optimized globally by population OLS + residual variance. CARA
trading and atomistic competitive clearing determine next q. No sampled data or SGD.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys, os, json
from pathlib import Path
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.optimize import root

OUT=Path(__file__).resolve().parent
from marketlab.scenarios import economy_8d
from marketlab.policies import InvestorPolicy
from marketlab.scenarios import diagnostic_policies

@dataclass
class Benchmark:
    K: int=3
    scenario: str='E1_aligned'
    dimension: int=8
    chi_bound: float|None=None
    # regularization is zero: exact Gaussian population log score is optimized.
    def __post_init__(self):
        eco=economy_8d(self.scenario,self.dimension)
        temps=diagnostic_policies(eco,self.scenario)
        assert all(np.allclose(a.signal_loading,eco.target_loading) for a in temps)
        self.eco=eco; self.temps=temps
        self.m=eco.dim; self.npub=eco.n_public; self.nobs=2+self.npub
        self.zdim=self.npub+self.nobs*self.K
        self.nstate=self.m+self.nobs*self.K
        self.ne=self.npub+2+self.m # pub noise, payoff eps, supply eps, next latent eta
        self.e_pub=slice(0,self.npub)
        self.e_div=self.npub;self.e_supply=self.npub+1;self.e_eta=slice(self.npub+2,self.ne)
        self.rho_exo=max(abs(np.linalg.eigvals(eco.F)))
        self.sigma_supply=eco.supply_noise_sd
        # Construct z_t=[m_t, (p_{t-1},D_t,m_{t-1}), (p_{t-2},D_{t-1},m_{t-2}), ... ]
        self.Zh=np.zeros((self.zdim,self.nstate));self.Ze=np.zeros((self.zdim,self.ne))
        self.Zh[:self.npub,:self.m]=eco.public_loading
        self.Ze[:self.npub,self.e_pub]=np.diag(eco.public_noise_sd)
        self.Zh[self.npub:,self.m:]=np.eye(self.nobs*self.K)
        # cov target,theta,private,feature representations filled by complete law cov

    def law(self,q,verify=True):
        eco=self.eco
        q=np.asarray(q,dtype=float); assert q.shape==(self.zdim+2,)
        A=q[0]; C=q[1:-1]; Delta=q[-1]
        if not Delta>1e-5: raise ValueError(f'non-positive price denominator {Delta}')
        # y_t=(p_t,D_{t+1},m_t) = Yh h_t+Ye e_t
        Yh=np.zeros((self.nobs,self.nstate)); Ye=np.zeros((self.nobs,self.ne))
        theta_h=np.zeros(self.nstate); theta_h[:self.m]=eco.target_loading
        Yh[0]=(A*theta_h+C@self.Zh)/Delta
        Ye[0]=(C@self.Ze)/Delta
        Ye[0,self.e_supply]-=eco.supply_noise_sd/Delta
        Yh[1,:self.m]=eco.payoff_loading
        Ye[1,self.e_div]=eco.payoff_noise_sd
        Yh[2:,:self.m]=eco.public_loading
        Ye[2:,self.e_pub]=np.diag(eco.public_noise_sd)
        # h_(t+1)=[F f_t+G eta_(t+1), y_t, y_(t-1), ...]
        M=np.zeros((self.nstate,self.nstate));N=np.zeros((self.nstate,self.ne))
        M[:self.m,:self.m]=eco.F
        N[:self.m,self.e_eta]=eco.G
        M[self.m:self.m+self.nobs,:]=Yh
        N[self.m:self.m+self.nobs,:]=Ye
        if self.K>1:
            M[self.m+self.nobs:, self.m:-self.nobs]=np.eye(self.nobs*(self.K-1))
        rad=max(abs(np.linalg.eigvals(M)))
        if rad>=1-1e-10:raise ValueError(f'nonstationary fast law radius={rad}')
        P=solve_discrete_lyapunov(M,N@N.T)
        if verify and not (np.min(np.linalg.eigvalsh(P))>-1e-10):
            raise ValueError('bad covariance')
        covhe=np.zeros((self.nstate+self.ne,self.nstate+self.ne))
        covhe[:self.nstate,:self.nstate]=P
        covhe[self.nstate:,self.nstate:]=np.eye(self.ne)
        Zh_full=np.hstack([self.Zh,self.Ze])
        Theta=np.zeros(self.nstate+self.ne);Theta[:self.m]=eco.target_loading
        Ph=np.hstack([Yh[0],Ye[0]])
        Dh=np.hstack([Yh[1],Ye[1]])
        # double-check public features / target timed correctly
        return {'q':q,'Z':Zh_full,'theta':Theta,'p':Ph,'d':Dh,'cov':covhe,'fast_radius':float(rad),'M':M,'N':N,'Yh':Yh,'Ye':Ye,'P':P}

    def fit_one(self, law, type_index, lag_capacity):
        eco=self.eco
        if lag_capacity<0 or lag_capacity>self.K:raise ValueError('capacity invalid')
        # z indices for macro_t, +lag features up to lag_capacity
        inds=np.arange(self.npub+self.nobs*lag_capacity)
        L=law['Z'][inds]
        # independent private observation noise, appended as last feature on top of given (h,e) covariance
        s=law['theta']
        p=law['p']; d=law['d']
        rows=np.vstack([L,s,p])
        cov=rows@law['cov']@rows.T
        var_s=self.temps[type_index].signal_noise_sd**2
        cov[len(inds),len(inds)]+=var_s
        cross=rows@law['cov']@d
        beta=np.linalg.solve(cov,cross)
        predvar=float(d@law['cov']@d-beta@cross)
        # Optional exact population optimum with the same bounded price slope
        # used by the finite-sample SSM/GRU heads. A binding constraint
        # requires reoptimizing ALL other regression coefficients.
        if self.chi_bound is not None and abs(beta[-1])>self.chi_bound:
            chi=float(np.clip(beta[-1],-self.chi_bound,self.chi_bound))
            residual_target=d-chi*p
            X=rows[:-1]
            XX=cov[:-1,:-1]
            XY=X@law['cov']@residual_target
            other=np.linalg.solve(XX,XY)
            predvar=float(residual_target@law['cov']@residual_target-other@XY)
            beta=np.r_[other,chi]
        if predvar<=0 or not np.isfinite(predvar):raise ValueError(f'negative conditional variance {predvar}')
        bb=np.zeros(self.zdim);bb[inds]=beta[:len(inds)]
        return {'b':bb,'beta':float(beta[len(inds)]),'chi':float(beta[len(inds)+1]),
                'v':predvar,'NLL':float(.5*np.log(2*np.pi*np.e*predvar)),
                'pi':self.temps[type_index].mass, 'gamma':self.temps[type_index].gamma,
                'lagcap':int(lag_capacity)}

    def update(self, q, capacities=(1,1), full=False):
        law=self.law(q,verify=False)
        preds=[self.fit_one(law,c,k) for c,k in enumerate(capacities)]
        A=0.; C=np.zeros(self.zdim); Delta=0.
        for d in preds:
            w=d['pi']/(d['gamma']*d['v'])
            A+=w*d['beta'];C+=w*d['b'];Delta+=w*(self.eco.risk_free_gross-d['chi'])
        if Delta<=0:raise ValueError('negative auction slope')
        new=np.r_[A,C,Delta]
        if full:return new,{'preds':preds,'law':law,'J':self.J(new),'updated_q':new}
        return new

    def J(self,q):return float(q[0]**2/self.sigma_supply**2)
    def local_jac(self,q,caps,step=1e-5):
        q=np.asarray(q,dtype=float)
        jac=np.empty((len(q),len(q)))
        for j in range(len(q)):
            dq=np.zeros_like(q); dq[j]=step*max(1.,abs(q[j]))
            jac[:,j]=(self.update(q+dq,caps)-self.update(q-dq,caps))/(2*dq[j])
        return jac
    def equilibrium(self,caps,start=None,maxiter=1000,tol=1e-11):
        q0=self.seed() if start is None else np.array(start,float)
        path=[];q=q0.copy()
        for n in range(maxiter):
            qnext=self.update(q,caps)
            path.append([n,self.J(qnext),float(np.max(abs(qnext-q)))])
            if np.max(abs(qnext-q))<tol:
                q=qnext;break
            q=qnext
        err=np.max(abs(q-self.update(q,caps)))
        # root solve only to identify alternative fixed point when natural dynamics do not converge
        if err>1e-8:
            sol=root(lambda qq:qq-self.update(qq,caps),q0,method='hybr',tol=1e-10)
            if sol.success and np.max(abs(sol.x-self.update(sol.x,caps)))<1e-7:
                q=sol.x;err=np.max(abs(q-self.update(q,caps)))
        jac=self.local_jac(q,caps)
        eig=np.linalg.eigvals(jac)
        return q,{'iterations':len(path),'residual':float(err),'rho_jac':float(max(abs(eig))),
                  'fast_rho':self.law(q)['fast_radius'],'J':self.J(q),'path':np.asarray(path),
                  'eigenvalues':[(float(x.real),float(x.imag)) for x in eig]}
    def seed(self):
        return np.r_[1.0,np.zeros(self.zdim),1.75]


def run_main():
    t=Benchmark(K=3)
    print('E1 dimension',t.m,'nstate',t.nstate,'zdim',t.zdim,'exogenous radius',t.rho_exo,flush=True)
    q0,stat0=t.equilibrium((1,1));print('low',np.around(q0,4),'J',stat0['J'],'jac',stat0['rho_jac'],'fast',stat0['fast_rho'],'res',stat0['residual'],'iter',stat0['iterations'],flush=True)
    new,md=t.update(q0,(3,1),True)
    print('impact J',t.J(new),'delta',t.J(new)-t.J(q0),'NLL lo hi',md['preds'][0]['NLL'],flush=True)
    q1,stat1=t.equilibrium((3,1),start=new)
    print('hi',np.around(q1,4),'J',stat1['J'],'jac',stat1['rho_jac'],'fast',stat1['fast_rho'],'res',stat1['residual'],'iter',stat1['iterations'],flush=True)
    print('path',np.around(stat1['path'][:30,1],4),flush=True)
if __name__=='__main__':run_main()
