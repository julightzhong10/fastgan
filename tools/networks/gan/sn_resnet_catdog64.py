'''
credits:
    Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). 
    Spectral normalization for generative adversarial networks. 
    arXiv preprint arXiv:1802.05957.

    Liu, X., & Hsieh, C. J. (2019). 
    Rob-gan: Generator, discriminator, and adversarial attacker. 
    In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 11234-11243).

paras:
    w/o sa: netD:10,156,683, netG:12,272,515
    w/  sa: netD:10,303,201, netG:13,038,595
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sn_resblocks import DownBlock, OptimizedBlock, UpBlock, Attention
from tools.metric import Metric_Types

class SN_ResNet_CatDog64_Dis(nn.Module):
    def __init__(self, img_c=3,fm_base=64, n_classes=0, bn=False, sn=False,sa=False,c_metric='',wide=False):
        super(SN_ResNet_CatDog64_Dis, self).__init__()
        self.wide = wide
        self.activation = F.relu
        self.n_classes = n_classes
        self.c_metric = c_metric
        self.sa = sa
        self.block1 = OptimizedBlock(img_c, fm_base,bn=bn, sn=sn)
        if self.sa:
            self.attn1 = Attention(ch=fm_base,sn=sn)
        self.block2 = DownBlock(fm_base, fm_base * 2, (fm_base*2 if self.wide else fm_base*1), activation=self.activation, downsample=True, bn=bn, sn=sn)
        self.block3 = DownBlock(fm_base * 2, fm_base * 4, (fm_base*4 if self.wide else fm_base*2), activation=self.activation, downsample=True, bn=bn, sn=sn)
        self.block4 = DownBlock(fm_base * 4, fm_base * 8, (fm_base*8 if self.wide else fm_base*4), activation=self.activation, downsample=True, bn=bn, sn=sn)
        self.block5 = DownBlock(fm_base * 8, fm_base * 16, (fm_base*16 if self.wide else fm_base*8), activation=self.activation, downsample=True, bn=bn, sn=sn)
        # bin
        self.o_b = nn.Linear(fm_base * 16, 1)
        nn.init.xavier_uniform_(self.o_b.weight)
        # class 
        if n_classes > 0:
            if c_metric in [Metric_Types[0],Metric_Types[2]]:
                self.o_c = nn.Linear(fm_base * 16, n_classes)
            elif c_metric == Metric_Types[1]:
                self.o_c = nn.Embedding(n_classes, fm_base * 16)
            nn.init.xavier_uniform_(self.o_c.weight)
        if sn:
            from torch.nn.utils import spectral_norm
            self.o_b = spectral_norm(self.o_b)
            if n_classes > 0 and (c_metric in Metric_Types):
                self.o_c = spectral_norm(self.o_c)

    def forward(self, x, y_c=None):
        h = x
        h = self.block1(h)
        if self.sa:
            h = self.attn1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        output_b = self.o_b(h)
        output_c = None
        if self.n_classes > 0:
            if self.c_metric in [Metric_Types[0],Metric_Types[2]]:
                output_c = self.o_c(h)
            elif self.c_metric == Metric_Types[1]:
                output_c = torch.sum(self.o_c(y_c) * h, dim=1, keepdim=True)

        return output_b, output_c


class SN_ResNet_CatDog64_Gen(nn.Module):
    def __init__(self, dim_z=128,img_c=3,fm_base=64, bottom_width=4, n_classes=0, bn=False, sn=False,sa=False):
        super(SN_ResNet_CatDog64_Gen, self).__init__()
        self.bn = bn
        self.bottom_width = bottom_width
        self.activation = F.relu
        self.dim_z = dim_z
        self.n_classes = n_classes
        self.sa = sa
        self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * fm_base * 16)
        nn.init.xavier_uniform_(self.l1.weight)
        self.block2 = UpBlock(fm_base * 16, fm_base * 8, activation=self.activation, upsample=True, n_classes=n_classes,bn=bn, sn=sn)
        self.block3 = UpBlock(fm_base * 8, fm_base * 4, activation=self.activation, upsample=True, n_classes=n_classes,bn=bn, sn=sn)
        self.block4 = UpBlock(fm_base * 4, fm_base * 2, activation=self.activation, upsample=True, n_classes=n_classes,bn=bn, sn=sn)
        if self.sa:
            self.attn1 = Attention(ch=fm_base * 2,sn=sn)
        self.block5 = UpBlock(fm_base * 2, fm_base * 1, activation=self.activation, upsample=True, n_classes=n_classes,bn=bn, sn=sn)
        if bn:
            self.finalbn = nn.BatchNorm2d(fm_base)
        self.l6 = nn.Conv2d(fm_base, img_c, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.l6.weight)
        if sn:
            from torch.nn.utils import spectral_norm
            self.l1 = spectral_norm(self.l1)
            self.l6 = spectral_norm(self.l6)


    def forward(self, z, y):
        h = z
        h = self.l1(h)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        if self.sa:
            h = self.attn1(h)
        h = self.block5(h, y)
        if self.bn:
            h = self.finalbn(h)
        h = self.activation(h)
        h = self.l6(h)
        h = torch.tanh(h)
        return h

