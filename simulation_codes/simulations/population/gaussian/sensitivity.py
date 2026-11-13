"""Analytic population retraining Jacobian: exact Lyapunov sensitivities + OLS implicit differentiation."""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from benchmark import Benchmark


def analytic_map_jac(b:Benchmark,q,caps=(0,0),adoption=None,lo=0,hi=3):
    """J_q d best-response clearing coefficients/d inherited-market coefficients.

    Frozen conditional Gaussian log scores are optimized globally by linear
    population least squares. All derivatives of the invariant covariance
    solve exact discrete Lyapunov derivative equations.
    """
    L=b.law(q)
    if adoption is None:
        spec=[(c,k,b.temps[c].mass) for c,k in enumerate(caps)]
    else:
        spec=[(0,lo,b.temps[0].mass*(1-adoption)),(0,hi,b.temps[0].mass*adoption),
              (1,lo,b.temps[1].mass)]
    models=[]
    for c,k,mass in spec:
        if mass<1e-15:continue
        n=b.npub+b.nobs*k; inds=np.arange(n)
        X=np.vstack([L['Z'][inds],L['theta'],L['p']])
        SX=X@L['cov']@X.T
        SX[n,n]+=b.temps[c].signal_noise_sd**2
        XY=X@L['cov']@L['d']
        beta=np.linalg.solve(SX,XY)
        v=L['d']@L['cov']@L['d']-XY@beta
        assert v>0
        models.append((c,k,mass,n,inds,X,SX,XY,beta,v))
    nQ=b.zdim+2; Jac=np.zeros((nQ,nQ))
    P=L['P']; M=L['M'];N=L['N']
    S=L['cov'];theta=L['theta'];Z=L['Z'];p=L['p'];d=L['d']
    for j in range(nQ):
        dp=np.zeros(b.nstate+b.ne)
        if j==0: dp[:b.nstate]=theta[:b.nstate]/q[-1]
        elif j==nQ-1:dp=-p/q[-1]
        else:dp=Z[j-1]/q[-1]
        dM=np.zeros_like(M); dN=np.zeros_like(N)
        dM[b.m,:]=dp[:b.nstate];dN[b.m,:]=dp[b.nstate:]
        rhs=dM@P@M.T+M@P@dM.T+dN@N.T+N@dN.T
        dP=solve_discrete_lyapunov(M,rhs)
        dS=np.zeros_like(S);dS[:b.nstate,:b.nstate]=dP
        col=np.zeros(nQ)
        for c,k,mass,n,inds,X,SX,XY,beta,v in models:
            dX=np.zeros_like(X);dX[-1]=dp
            dXX=dX@S@X.T+X@dS@X.T+X@S@dX.T
            dXY=dX@S@d+X@dS@d
            dBeta=np.linalg.solve(SX,dXY-dXX@beta)
            dv=float(d@dS@d-2*beta@dXY+beta@dXX@beta)
            w=mass/(b.temps[c].gamma*v);dw=-w*dv/v
            col[0]+=dw*beta[n]+w*dBeta[n]
            col[-1]+=dw*(b.eco.risk_free_gross-beta[-1])-w*dBeta[-1]
            col[1:1+n]+=dw*beta[:n]+w*dBeta[:n]
        Jac[:,j]=col
    return Jac


def derivative_check():
    b=Benchmark();q,d=b.equilibrium((3,0))
    n=analytic_map_jac(b,q,(3,0));fn=b.local_jac(q,(3,0))
    print('analytical jac vs centered diff max',np.max(abs(n-fn)))
    assert np.max(abs(n-fn))<1e-7
    print('rho analytic',max(abs(np.linalg.eigvals(n))),'finite diff',max(abs(np.linalg.eigvals(fn))))
    from dataclasses import replace
    b.eco=replace(b.eco,supply_noise_sd=.715);b.sigma_supply=.715
    from scipy.optimize import root
    fixed=root(lambda x:x-b.update(x,(3,0)),q,tol=1e-11).x
    jac=analytic_map_jac(b,fixed,(3,0))
    print('rho near flip',max(abs(np.linalg.eigvals(jac))))

if __name__=='__main__':derivative_check()
