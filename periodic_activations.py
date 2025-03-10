import torch
from torch import nn
import numpy as np
import math
from einops import rearrange, repeat, reduce, pack, unpack

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class TrigonometricActivation(nn.Module):
    def __init__(self, in_features, out_features, f):
        super(TrigonometricActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = f

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):    
    def __init__(self, time_intervals, out_features):
        super(Time2Vec, self).__init__()
        in_features = len(time_intervals)
        #self.sin_acts = TrigonometricActivation(
        #    in_features, out_features // 2, f=torch.sin)
        self.cos_acts = TrigonometricActivation(
            in_features, out_features, f=torch.cos)
        self.time_intervals = torch.from_numpy(time_intervals)
        self.scale = nn.Parameter(torch.ones(1) * out_features ** -0.5)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, tau):
        tau_vec = rearrange(tau, "b i -> b i 1")      
        time_intervals = rearrange(self.time_intervals.to(self.device), 'l -> 1 1 l')
        time_reminds = (tau_vec % time_intervals).float() / time_intervals.float()
        return self.cos_acts(time_reminds) * self.scale
        #return torch.concat([
        #    self.sin_acts(time_reminds), self.cos_acts(time_reminds)],
        #    dim=2)