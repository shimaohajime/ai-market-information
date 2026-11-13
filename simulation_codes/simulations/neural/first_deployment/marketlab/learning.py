"""Train resource-bounded Gaussian forecasters on a FROZEN inherited market law.

Inputs: U_t=(p_{t-1},D_t,m_t) for window ending at t, plus individual
s_ct and candidate p_t (only in the PRICE-AFFINE prediction head).
Targets: realized D_{t+1}. Gaussian likelihood learns mean and variance.
No latent factor / future dividend enters any training input.

Architectures: diagonal stable linear state-space filter and finite-context GRU.
Heads: constant slope/variance (A) or public-state-dependent slopes/variance (B).
Bounded price slope ensures downward-sloping demand, not rational expectations.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import numpy as np
import torch
from torch import nn


def set_seed(seed:int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


@dataclass
class Normalization:
    u_mean:np.ndarray
    u_sd:np.ndarray
    s_mean:float
    s_sd:float
    p_mean:float
    p_sd:float
    y_mean:float
    y_sd:float

    def to_dict(self):
        return {k:(v.tolist() if isinstance(v,np.ndarray) else float(v)) for k,v in vars(self).items()}

    @classmethod
    def from_dict(cls,d):
        vals=d.copy()
        for k in ('u_mean','u_sd'):
            vals[k]=np.asarray(vals[k],dtype=np.float32)
        return cls(**vals)


class ResourceForecaster(nn.Module):
    def __init__(self, input_dim, hidden, architecture, head_type, norm:Normalization, window=48):
        super().__init__()
        assert architecture in ('ssm','gru')
        assert head_type in ('A','B')
        self.architecture=architecture; self.head_type=head_type
        self.input_dim=input_dim; self.hidden=hidden; self.window=window
        self.norm=norm
        if architecture=='ssm':
            self.raw_decay=nn.Parameter(torch.linspace(-0.4,1.1,hidden))
            self.input_projection=nn.Parameter(torch.randn(hidden,input_dim)*.04)
        else:
            self.gru=nn.GRU(input_size=input_dim,hidden_size=hidden,batch_first=True)
        self.public_mean=nn.Linear(hidden,1)
        nn.init.zeros_(self.public_mean.weight)
        nn.init.zeros_(self.public_mean.bias)
        self.raw_beta=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.raw_chi=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.log_variance=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        if head_type=='B':
            self.beta_state=nn.Linear(hidden,1,bias=False)
            self.chi_state=nn.Linear(hidden,1,bias=False)
            self.logvar_state=nn.Linear(hidden,1,bias=False)
            for layer in (self.beta_state,self.chi_state,self.logvar_state):
                nn.init.zeros_(layer.weight)

    def encode(self,u):
        """u is normalized rolling window [batch,window,input_dim]."""
        if self.architecture=='gru':
            return self.gru(u)[0][:,-1,:]
        a=.95*torch.tanh(self.raw_decay)
        L=u.shape[1]
        # Efficient exact FIR representation of stable diagonal recurrence.
        # Equivalent to z_t = a*z_{t-1} + B*u_t initialized L steps ago at zero.
        powers=a[None,:]**torch.arange(L-1,-1,-1,device=u.device,dtype=u.dtype)[:,None]
        projected=torch.einsum('bld,hd->blh',u,self.input_projection)
        return (projected*powers[None,:,:]).sum(dim=1)

    def coefficients_normalized(self,z):
        offset=self.public_mean(z).squeeze(-1)
        beta=self.raw_beta.expand_as(offset)
        chi_raw=self.raw_chi.expand_as(offset)
        logv=self.log_variance.expand_as(offset)
        if self.head_type=='B':
            beta=beta+self.beta_state(z).squeeze(-1)
            chi_raw=chi_raw+self.chi_state(z).squeeze(-1)
            logv=logv+self.logvar_state(z).squeeze(-1)
        # Convert physical price slope -> normalized slope.
        chi_phys=.45*torch.tanh(chi_raw)
        chi_norm=chi_phys*(self.norm.p_sd/self.norm.y_sd)
        return offset,beta,chi_norm,torch.clamp(logv,-4.,3.),chi_phys

    def forward(self,u,s,p):
        z=self.encode(u)
        offset,beta,chi_norm,logv,_=self.coefficients_normalized(z)
        mu=offset+beta*s+chi_norm*p
        return mu,logv

    def heads_from_z(self,z):
        """Returns physical [intercept,beta,chi,var], to submit demand schedules."""
        offset,beta,_,logv,chi_phys=self.coefficients_normalized(z)
        beta_phys=beta*(self.norm.y_sd/self.norm.s_sd)
        intercept=(self.norm.y_mean+self.norm.y_sd*offset
                   -beta_phys*self.norm.s_mean-chi_phys*self.norm.p_mean)
        var=(self.norm.y_sd**2)*torch.exp(logv)
        return torch.stack((intercept,beta_phys,chi_phys,var),dim=-1)

    def physical_heads(self,u_normalized, batch_size=512):
        self.eval()
        pieces=[]
        with torch.no_grad():
            for i in range(0,len(u_normalized),batch_size):
                uu=torch.as_tensor(u_normalized[i:i+batch_size],dtype=torch.float32)
                z=self.encode(uu)
                pieces.append(self.heads_from_z(z).numpy())
        return np.vstack(pieces)

    def infer_from_window(self,u_raw):
        """Exact same rolling-window computation as the training model."""
        arr=np.asarray(u_raw,dtype=np.float32)
        assert arr.shape==(self.window,self.input_dim)
        x=np.asarray((arr-self.norm.u_mean)/self.norm.u_sd,dtype=np.float32)
        with torch.no_grad():
            z=self.encode(torch.from_numpy(x[None,:,:]))
            return self.heads_from_z(z).numpy()[0]

    def resource_stats(self):
        h=self.hidden;d=self.input_dim;L=self.window
        # MACs for forward encoder, counted approximately from architecture equation.
        if self.architecture=='ssm':
            macs=L*h*(d+2)
        else:
            macs=L*3*(d*h+h*h+h)
        macs+=h*(1+(3 if self.head_type=='B' else 0))
        return {'params':sum(p.numel() for p in self.parameters()),
                'forward_macs_approx':int(macs),
                'roll_buffer_bytes_fp32':int(L*d*4),
                'hidden_bytes_fp32':int(h*4),
                'training_context_periods':int(L),
                'inference_method':'recompute exact finite-context encoder'}


def make_windows(u,window):
    from numpy.lib.stride_tricks import sliding_window_view
    arr=np.ascontiguousarray(u,dtype=np.float32)
    w=sliding_window_view(arr,window,axis=0).transpose(0,2,1)
    assert w.shape==(len(u)-window+1,window,u.shape[1])
    return w


def prepare_panel(panel,type_index,window):
    """Split chronologically; normalize using training portion only."""
    u=np.asarray(panel['public_input'],dtype=np.float32)
    p=np.asarray(panel['price'],dtype=np.float32)
    y=np.asarray(panel['future_dividend'],dtype=np.float32)
    s=np.asarray(panel['private_signals'][:,type_index],dtype=np.float32)
    n=len(y);a=int(.65*n);b=int(.82*n)
    def moments(v):
        mu=np.mean(v,axis=0)
        sd=np.std(v,axis=0)
        return mu,np.maximum(sd,1e-4)
    um,us=moments(u[:a]);sm,ss=moments(s[:a]);pm,ps=moments(p[:a]);ym,ys=moments(y[:a])
    norm=Normalization(um,us,float(sm),float(ss),float(pm),float(ps),float(ym),float(ys))
    windows=make_windows((u-um)/us,window)
    t=np.arange(window-1,n)
    data={'windows':torch.as_tensor(windows.copy()),
          's':torch.as_tensor((s[t]-sm)/ss),
          'p':torch.as_tensor((p[t]-pm)/ps),
          'y':torch.as_tensor((y[t]-ym)/ys),
          'time_index':t,
          'train_idx':np.flatnonzero(t<a),
          'val_idx':np.flatnonzero((t>=a)&(t<b)),
          'test_idx':np.flatnonzero(t>=b),
          'norm':norm}
    return data


def gaussian_loss(mu,logv,y):
    return .5*(logv+(y-mu).square()*torch.exp(-logv)).mean()


def _init_ols(model,data):
    inds=data['train_idx']
    s=data['s'][inds].numpy(); p=data['p'][inds].numpy(); y=data['y'][inds].numpy()
    x=np.column_stack([np.ones(len(s)),s,p])
    intercept,beta,coef=np.linalg.lstsq(x,y,rcond=None)[0]
    # χ is expressed in physical units, with bound 0.45.
    chi_phys=float(coef*model.norm.y_sd/model.norm.p_sd)
    chi_phys=float(np.clip(chi_phys,-.42,.42))
    with torch.no_grad():
        model.public_mean.bias.fill_(float(intercept))
        model.raw_beta.fill_(float(beta))
        model.raw_chi.fill_(float(np.arctanh(chi_phys/.45)))
        mu=x@np.array([intercept,beta,coef])
        model.log_variance.fill_(float(np.log(max(np.mean((y-mu)**2),.04))))


def score_model(model,data,subset='test'):
    inds=data[subset+'_idx']
    model.eval();pred=[];logvs=[]
    with torch.no_grad():
        for j in range(0,len(inds),512):
            ix=inds[j:j+512]
            mean,lvar=model(data['windows'][ix],data['s'][ix],data['p'][ix])
            pred.append(mean.numpy()); logvs.append(lvar.numpy())
    pred=np.concatenate(pred);logv=np.concatenate(logvs)
    y=data['y'][inds].numpy(); residual=y-pred
    nll=.5*np.mean(math.log(2*math.pi)+logv+residual**2*np.exp(-logv))+np.log(model.norm.y_sd)
    mse=np.mean(residual**2)*model.norm.y_sd**2
    var=np.exp(logv)*model.norm.y_sd**2
    return {'n':int(len(y)),'NLL':float(nll),'MSE':float(mse),
            'variance_mean':float(var.mean()),'residual_variance':float(np.mean(residual**2)*model.norm.y_sd**2),
            'variance_calibration_ratio':float(np.mean(residual**2*np.exp(-logv))),
            'NLL_minus_constant':float(nll-np.log(model.norm.y_sd))}


def fit_forecaster(panel, scenario, architecture, hidden, head_type='A', seed=0,
                   window=48, steps=240,batch_size=96,lr=.005,patience=5):
    set_seed(seed)
    data=prepare_panel(panel,0,window)
    model=ResourceForecaster(panel['public_input'].shape[1],hidden,architecture,head_type,data['norm'],window)
    _init_ols(model,data)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)
    trainidx=data['train_idx']; validx=data['val_idx']
    generator=np.random.default_rng(seed+116)
    best=float('inf');best_state=None;stale=0; log=[]
    checkpoint_every=30
    for step in range(1,steps+1):
        ix=generator.choice(trainidx,size=batch_size,replace=True)
        model.train();optimizer.zero_grad()
        mu,lv=model(data['windows'][ix],data['s'][ix],data['p'][ix])
        loss=gaussian_loss(mu,lv,data['y'][ix])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training NLL')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        optimizer.step()
        if step%checkpoint_every==0 or step==steps:
            model.eval()
            with torch.no_grad():
                vidx=validx[::max(1,len(validx)//512)]
                vm,vv=model(data['windows'][vidx],data['s'][vidx],data['p'][vidx])
                vloss=float(gaussian_loss(vm,vv,data['y'][vidx]))
            log.append({'step':step,'train_loss':float(loss.detach().item()),'validation_loss':vloss})
            if vloss<best-1e-5:
                best=vloss;best_state=copy.deepcopy(model.state_dict());stale=0
            else:
                stale+=1
                if stale>=patience:break
    assert best_state is not None
    model.load_state_dict(best_state);model.eval()
    scores={'train':score_model(model,data,'train'),
            'validation':score_model(model,data,'val'),
            'test':score_model(model,data,'test')}
    details={'scenario':scenario,'architecture':architecture,'hidden':hidden,
             'head_type':head_type,'seed':seed,'steps_completed':int(step),
             'best_validation_normalized_NLL':float(best),
             'scores':scores,'resources':model.resource_stats(),'learning_curve':log}
    return model,details


def checkpoint(model,details,path):
    torch.save({'model_state':model.state_dict(),
                'normalization':model.norm.to_dict(),
                'config':{'input_dim':model.input_dim,'hidden':model.hidden,
                          'architecture':model.architecture,'head_type':model.head_type,
                          'window':model.window},'metrics':details},path)


def load_checkpoint(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    norm=Normalization.from_dict(d['normalization'])
    model=ResourceForecaster(norm.u_mean.size,d['config']['hidden'],
         d['config']['architecture'],d['config']['head_type'],norm,d['config']['window'])
    model.load_state_dict(d['model_state']); model.eval()
    return model,d['metrics']
