'''
credits:
    Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). 
    Spectral normalization for generative adversarial networks. 
    arXiv preprint arXiv:1802.05957.

    Liu, X., & Hsieh, C. J. (2019). 
    Rob-gan: Generator, discriminator, and adversarial attacker. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 11234-11243).

'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _downsample(x):
    return F.avg_pool2d(x, 2)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,
            ksize=3, pad=1, activation=F.relu, downsample=False, bn=False,sn=False):
        super(DownBlock, self).__init__()
        self.activation = activation
        self.bn = bn
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c1.weight)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c2.weight)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight)
        if sn:
            from torch.nn.utils import spectral_norm
            self.c1 = spectral_norm(self.c1)
            self.c2 = spectral_norm(self.c2)
            if self.learnable_sc:
                self.c_sc = spectral_norm(self.c_sc)
        if self.bn:
            self.b1 = nn.BatchNorm2d(hidden_channels)
            self.b2 = nn.BatchNorm2d(out_channels)
            if self.learnable_sc:
                self.b_sc = nn.BatchNorm2d(out_channels)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.b1(self.c1(h)) if self.bn else self.c1(h)
        h = self.activation(h)
        h = self.b2(self.c2(h)) if self.bn else self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.b_sc(self.c_sc(x)) if self.bn else self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, pad=1, activation=F.relu, bn=False,sn=False):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.bn = bn
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c1.weight)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, 1, pad)
        nn.init.xavier_uniform_(self.c2.weight)
        self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        nn.init.xavier_uniform_(self.c_sc.weight)
        if sn:
            from torch.nn.utils import spectral_norm
            self.c1 = spectral_norm(self.c1)
            self.c2 = spectral_norm(self.c2)
            self.c_sc = spectral_norm(self.c_sc)
        if self.bn:
            self.b1 = nn.BatchNorm2d(out_channels)
            self.b2 = nn.BatchNorm2d(out_channels)
            self.b_sc = nn.BatchNorm2d(out_channels)

    def residual(self, x):
        h = x
        h = self.b1(self.c1(h)) if self.bn else self.c1(h)
        h = self.activation(h)
        h = self.b2(self.c2(h)) if self.bn else self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.b_sc(self.c_sc(_downsample(x))) if self.bn else self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


def _upsample(x):
    h, w = x.shape[2:]
    return F.interpolate(x, size=(h * 2, w * 2))

def upsample_conv(x, conv):
    return conv(_upsample(x))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, ksize=3, \
            pad=1, activation=F.relu, upsample=False, n_classes=0,bn=False,sn=False,bias=True):
        super(UpBlock, self).__init__()
        self.bn = bn
        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_channels != out_channels or upsample
        hidden_channels = out_channels if hidden_channels is None else hidden_channels
        self.n_classes = n_classes
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad,bias=bias)
        nn.init.xavier_uniform_(self.c1.weight)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad,bias=bias)
        nn.init.xavier_uniform_(self.c2.weight)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,bias=bias)
            nn.init.xavier_uniform_(self.c_sc.weight)
        if sn:
            from torch.nn.utils import spectral_norm
            self.c1 = spectral_norm(self.c1)
            self.c2 = spectral_norm(self.c2)
            if self.learnable_sc:
                self.c_sc = spectral_norm(self.c_sc)
        if self.bn:
            if n_classes > 0:
                self.b1 = CatCondBatchNorm2d(in_channels, n_cat=n_classes,bias=bias)
                self.b2 = CatCondBatchNorm2d(hidden_channels, n_cat=n_classes,bias=bias)
            else:
                self.b1 = nn.BatchNorm2d(in_channels,affine=bias)
                self.b2 = nn.BatchNorm2d(hidden_channels,affine=bias)


    def residual(self, x, y=None):
        h = x
        if self.bn:
            h = self.b1(h, y) if self.n_classes>0 else self.b1(h)
        h = self.activation(h)
        h = upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        if self.bn:
            h = self.b2(h, y) if self.n_classes>0 else self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        f1 = self.residual(x, y)
        f2 = self.shortcut(x)
        return f1 + f2



class CatCondBatchNorm2d(nn.Module):
    def __init__(self, size, n_cat, decay=0.9, eps=2.0e-5, bias=True):
        super(CatCondBatchNorm2d, self).__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer('avg_mean', torch.zeros(size))
        self.register_buffer('avg_var', torch.ones(size))
        self.register_buffer('gamma_', torch.ones(size))
        self.register_buffer('beta_', torch.zeros(size))
        self.gammas = nn.Embedding(n_cat, size)
        nn.init.constant_(self.gammas.weight, 1)
        self.bias = bias
        if self.bias:
            self.betas = nn.Embedding(n_cat, size)
            nn.init.constant_(self.betas.weight, 0)
    def forward(self, x, c):
        c = torch.argmax(c,1).view(-1)
        gamma_c = self.gammas(c).view(c.size(0), -1, 1, 1)
        if self.bias:
            beta_c = self.betas(c).view(c.size(0), -1, 1, 1)
        feature = F.batch_norm(x, self.avg_mean, self.avg_var, None, None, self.training, (1-self.decay), self.eps)
        if self.bias:
            out = gamma_c * feature + beta_c
        else:
            out = gamma_c * feature
        return out



class Attention(nn.Module):
    def __init__(self, ch, sn):
        super(Attention, self).__init__()
        # Channel multiplier
        bias = False
        self.ch = ch
        self.theta = nn.Conv2d(in_channels = ch, out_channels = ch//8 , kernel_size= 1, bias=bias)
        nn.init.xavier_uniform_(self.theta.weight)
        self.phi = nn.Conv2d(in_channels = ch, out_channels = ch//8 , kernel_size= 1, bias=bias)
        nn.init.xavier_uniform_(self.phi.weight)
        self.g = nn.Conv2d(in_channels = ch , out_channels = ch//2 , kernel_size= 1, bias=bias)
        nn.init.xavier_uniform_(self.g.weight)
        self.o = nn.Conv2d(in_channels = ch//2 , out_channels = ch , kernel_size= 1, bias=bias)
        nn.init.xavier_uniform_(self.o.weight)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        if sn:
            from torch.nn.utils import spectral_norm
            self.theta = spectral_norm(self.theta)
            self.phi = spectral_norm(self.phi)
            self.g = spectral_norm(self.g)
            self.o = spectral_norm(self.o)
    def forward(self,x):
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# class Attention(nn.Module):
#     def __init__(self, ch, sn):
#         super(Attention, self).__init__()
#         # Channel multiplier
#         bias = False
#         self.ch = ch
#         self.f = nn.Conv2d(in_channels = ch, out_channels = ch//8 , kernel_size= 1, bias=bias)
#         self.h = nn.Conv2d(in_channels = ch, out_channels = ch//8 , kernel_size= 1, bias=bias)
#         self.g = nn.Conv2d(in_channels = ch , out_channels = ch//2 , kernel_size= 1, bias=bias)
#         self.o = nn.Conv2d(in_channels = ch//2 , out_channels = ch , kernel_size= 1, bias=bias)
#         self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
#         if sn:
#             from torch.nn.utils import spectral_norm
#             self.f = spectral_norm(self.f)
#             self.h = spectral_norm(self.h)
#             self.g = spectral_norm(self.g)
#             self.o = spectral_norm(self.o)
#     def forward(self,x):
#         # Apply convs
#         f = self.f(x)
#         h = F.max_pool2d(self.h(x), [2,2])
#         g = F.max_pool2d(self.g(x), [2,2])    
#         # Perform reshapes
#         f = f.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
#         h = h.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
#         g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
#         # Matmul and softmax to get attention maps
#         attn = F.softmax(torch.bmm(f.transpose(1, 2), f), -1)
#         # Attention map times g path
#         o = self.o(torch.bmm(g, attn.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
#         o = self.gamma * o + x
#         return o




# class Attention(nn.Module):
#     def __init__(self, ch, sn):
#         super(Attention, self).__init__()
#         # Channel multiplier
#         bias = True
#         self.K = 8
#         self.ch = ch
#         self.f = nn.Conv2d(in_channels = ch, out_channels = ch//self.K , kernel_size= 1, bias=bias)
#         self.g = nn.Conv2d(in_channels = ch, out_channels = ch//self.K , kernel_size= 1, bias=bias)
#         self.h = nn.Conv2d(in_channels = ch , out_channels = ch//self.K , kernel_size= 1, bias=bias)
#         self.o = nn.Conv2d(in_channels = ch//self.K  , out_channels = ch , kernel_size= 1, bias=bias)
#         self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
#         if sn:
#             from torch.nn.utils import spectral_norm
#             self.f = spectral_norm(self.f)
#             self.g = spectral_norm(self.g)
#             self.h = spectral_norm(self.h)
#             self.o = spectral_norm(self.o)
#     def forward(self,x):
#         # Apply convs
#         f = self.f(x) # f(x)
#         h = self.h(x) # h(x)
#         g = self.g(x) # g(x)
#         # Perform reshapes
#         f = f.view(-1, self.ch // self.K, x.shape[2] * x.shape[3])
#         h = h.view(-1, self.ch // self.K, x.shape[2] * x.shape[3])
#         g = g.view(-1, self.ch // self.K, x.shape[2] * x.shape[3])
#         # Matmul and softmax to get attention maps
#         attn = F.softmax(torch.bmm(f.transpose(1, 2), h), -1) # attention map: softmax((f(x)^T)*h(x))
#         # Attention map times g path
#         o = self.o(torch.bmm(g, attn.transpose(1,2)).view(-1, self.ch // self.K, x.shape[2], x.shape[3])) # o(x): o(g(x)*(attn^T))
#         o = self.gamma * o + x # with learnabel gamma back to input: gamma*o + x
#         return o
