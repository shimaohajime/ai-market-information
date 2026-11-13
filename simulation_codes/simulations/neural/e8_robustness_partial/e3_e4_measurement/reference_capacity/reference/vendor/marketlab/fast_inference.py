"""Numerically equivalent fast CPU inference of finite-context SSM/GRU from Phase 2.

This is a speed-up, NOT a different learning architecture. All 32 or 48 past
observations are reprocessed from a zero hidden state every auction exactly as
in supervised training. A genuine streaming GRU would change the model class.
"""
from __future__ import annotations
from collections import deque
import numpy as np
from numba import njit

@njit(cache=True)
def gru_encode(U,wi,wh,bi,bh):
    n=wi.shape[0]//3
    h=np.zeros(n,np.float64)
    for t in range(U.shape[0]):
        r=np.empty(n,np.float64);z=np.empty(n,np.float64);a=np.empty(n,np.float64)
        for j in range(n):
            ir=bi[j]+bh[j]; iz=bi[n+j]+bh[n+j]
            inn=bi[2*n+j]; hn=bh[2*n+j]
            for k in range(U.shape[1]):
                ir+=wi[j,k]*U[t,k]; iz+=wi[n+j,k]*U[t,k]
                inn+=wi[2*n+j,k]*U[t,k]
            for k in range(n):
                ir+=wh[j,k]*h[k];iz+=wh[n+j,k]*h[k]
                hn+=wh[2*n+j,k]*h[k]
            r[j]=1.0/(1.0+np.exp(-ir))
            z[j]=1.0/(1.0+np.exp(-iz))
            a[j]=np.tanh(inn+r[j]*hn)
        h=(1.0-z)*a+z*h
    return h

@njit(cache=True)
def ssm_encode(U,decay,proj):
    h=np.zeros(decay.shape[0],np.float64)
    for t in range(U.shape[0]):
        for k in range(len(h)):
            h[k]*=decay[k]
            for j in range(U.shape[1]):
                h[k]+=proj[k,j]*U[t,j]
    return h


class FastFiniteContextEncoder:
    """Drop-in training-exact replacement for TorchRollingWindowEncoder.
    Returns [physical intercept, private-signal slope, current-price slope, variance].
    """
    def __init__(self,model):
        self.arch=model.architecture;self.model=model
        self.window=model.window;self.input_dim=model.input_dim
        self.norm=model.norm
        self.pmean=model.public_mean.weight.detach().numpy().reshape(-1).astype('float64').copy()
        self.pbias=float(model.public_mean.bias.detach().item())
        self.beta=float(model.raw_beta.detach().item())
        self.chi=float(model.raw_chi.detach().item())
        self.logv=float(model.log_variance.detach().item())
        self.head_type=model.head_type
        self.betastate=self.chistate=self.varstate=np.zeros(model.hidden)
        if self.head_type=='B':
            self.betastate=model.beta_state.weight.detach().numpy().flatten().astype('float64').copy()
            self.chistate=model.chi_state.weight.detach().numpy().flatten().astype('float64').copy()
            self.varstate=model.logvar_state.weight.detach().numpy().flatten().astype('float64').copy()
        if self.arch=='ssm':
            self.decay=(.95*np.tanh(model.raw_decay.detach().numpy())).astype('float64')
            self.proj=model.input_projection.detach().numpy().astype('float64').copy()
        else:
            self.wi=np.ascontiguousarray(model.gru.weight_ih_l0.detach().numpy(),dtype='float64')
            self.wh=np.ascontiguousarray(model.gru.weight_hh_l0.detach().numpy(),dtype='float64')
            self.bi=np.ascontiguousarray(model.gru.bias_ih_l0.detach().numpy(),dtype='float64')
            self.bh=np.ascontiguousarray(model.gru.bias_hh_l0.detach().numpy(),dtype='float64')
        self.reset()

    def reset(self):
        self.buffer=deque(maxlen=self.window)

    def __deepcopy__(self,memo):
        return FastFiniteContextEncoder(self.model)

    def step(self,u):
        self.buffer.append(np.asarray(u,dtype=np.float32).copy())
        seq=np.zeros((self.window,self.input_dim),dtype='float32')
        past=np.asarray(self.buffer,dtype=np.float32)
        seq[-len(past):]=past
        U=np.asarray((seq-self.norm.u_mean)/self.norm.u_sd,dtype='float64')
        if self.arch=='ssm':
            z=ssm_encode(U,self.decay,self.proj)
        else:
            z=gru_encode(U,self.wi,self.wh,self.bi,self.bh)
        off=float(self.pmean@z+self.pbias)
        beta_norm=self.beta+float(self.betastate@z)
        chi_phys=.45*np.tanh(self.chi+float(self.chistate@z))
        logvar=np.clip(self.logv+float(self.varstate@z),-4.,3.)
        beta_phys=beta_norm*(self.norm.y_sd/self.norm.s_sd)
        intercept=(self.norm.y_mean+self.norm.y_sd*off
                   -beta_phys*self.norm.s_mean-chi_phys*self.norm.p_mean)
        var=self.norm.y_sd**2*np.exp(logvar)
        return np.array([intercept,beta_phys,chi_phys,var],dtype='float64')
