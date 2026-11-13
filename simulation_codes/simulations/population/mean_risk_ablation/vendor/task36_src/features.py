"""Causally timed pre-auction public-history matrices shared by external decoder probes."""
import numpy as np

def design_matrix(panel,L):
    """[macro_t, u_{t-1},...,u_{t-L}], u_t=(p_{t-1}, D_t, macro_t).
    Never include contemporaneous price p_t or target D_(t+1) in the public features.
    """
    u=panel.public_input
    n=len(u); mx=panel.public.shape[1]
    X=np.empty((n-L,mx+len(u[0])*L),dtype=np.float64)
    X[:,:mx]=u[L:,2:]
    d=len(u[0]); off=mx
    for k in range(1,L+1):
        X[:,off:off+d]=u[L-k:n-k,:]
        off+=d
    return X
