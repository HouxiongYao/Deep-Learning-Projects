import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self,input_dim,bias=True):
        super().__init__()
        self.linear=nn.Linear(input_dim,input_dim*2,bias=bias)
    def forward(self,x):
        out=self.linear(x)
        out,gate=out.chunk(2,dim=-1)
        gate=torch.sigmoid(gate)
        return out*gate
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x*torch.sigmoid(x)
