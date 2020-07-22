'''
credit:
  Brock, A., Donahue, J., & Simonyan, K. (2018). 
  Large scale gan training for high fidelity natural image synthesis. 
  arXiv preprint arXiv:1809.11096.

  github: https://github.com/ajbrock/BigGAN-PyTorch

paras:
  ch=96, AC-clsss=1000:
    256*256: netD:99,226,890, netG:82,108,548,
    128*128: netD:87,983,370, netG:70,433,988
      64*64: netD:45,512,970, netG:70,377,348
      32*32: netD:10,575,213, netG: 9,650,436

  ch=64, AC-clsss=1000
    256*256: netD:44,446,890, netG:37,240,580
    128*128: netD:39,449,258, netG:31,978,628
      64*64: netD:20,572,842, netG:31,969,540
      32*32: netD:4,789,485,  netG:4,567,556

  ch=48, AC-clsss=10
      32*32: netD:2,553,423,  netG:2,615,172
  ch=32, AC-clsss=10
      32*32: netD:1,137,039,  netG:1,266,692
  ch=16, AC-clsss=10
      32*32: netD:285,903,  netG:395,396

'''
import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import tools.networks.gan.biggan_blocks as layers
from tools.metric import Metric_Types


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class BigGAN_Gen(nn.Module):
  def __init__(self, dim_z=128, G_ch=64, bottom_width=4, resolution=128,n_classes=1000, bn=False,sn=False,sa=False,
                G_attn ='32', shared_dim=128, G_activation=nn.ReLU(inplace=False),G_shared=True,G_hier=True,G_init='',**kwargs):
    super(BigGAN_Gen, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = G_hier
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # use bn?
    self.bn = bn
    # use sn?
    self.sn = sn
    # use sa?
    self.sa = sa
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.sn:
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1)
      self.which_linear = functools.partial(layers.SNLinear)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    if self.bn:
      self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes))
    else:
      self.which_bn = functools.partial(layers.identity())


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]] and self.sa:
        #print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    if self.bn:
      self.output_layer = nn.Sequential(nn.BatchNorm2d(self.arch['out_channels'][-1]),
                                                      self.activation,
                                                      self.which_conv(self.arch['out_channels'][-1], 3))
    else:
      self.output_layer = nn.Sequential(self.activation,
                                        self.which_conv(self.arch['out_channels'][-1], 3))      

    # Initialize weights. Optionally skip init for testing.
    self.init_weights()

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        elif self.init in ['kaiming', 'he']:
          init.kaiming_uniform_(module.weight,nonlinearity='relu')

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # If hierarchical, concatenate zs and ys
    y = torch.argmax(y,1).view(-1)
    y = self.shared(y)
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class BigGAN_Dis(nn.Module):

  def __init__(self, D_ch=64, resolution=128, n_classes=1000, bn=False, sn=False, sa=False, c_metric='',
               D_attn='32', wide=True, D_activation=nn.ReLU(inplace=False),D_init='', **kwargs):
    super(BigGAN_Dis, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = wide
    # Resolution
    self.resolution = resolution
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.sn = sn
    # use bn?
    self.bn = bn
    # use sa?
    self.sa = sa
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]
    self.c_metric = c_metric

    # Which convs, batchnorms, and linear layers to use
    if self.sn:
      self.which_conv = functools.partial(layers.SNConv2d,
                          kernel_size=3, padding=1)
      self.which_linear = functools.partial(layers.SNLinear)
      if c_metric == Metric_Types[1]:
        self.which_embedding = functools.partial(layers.SNEmbedding)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      self.which_embedding = nn.Embedding
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       bn = self.bn,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]] and self.sa:
        #print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output

    self.o_b = self.which_linear(self.arch['out_channels'][-1], 1)
    # Embedding for projection discrimination
    if self.n_classes > 0:
      if c_metric in [Metric_Types[0],Metric_Types[2]]:
        self.o_c = self.which_linear(self.arch['out_channels'][-1], self.n_classes)
      elif c_metric == Metric_Types[1]:
        self.o_c = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])


    # Initialize weights
    self.init_weights()

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        elif self.init in ['kaiming', 'he']:
          init.kaiming_uniform_(module.weight,nonlinearity='relu')

  def forward(self, x, y_c=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    h = self.activation(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])

    output_b = self.o_b(h)
    output_c = None
    if self.n_classes > 0:
      if self.c_metric in [Metric_Types[0],Metric_Types[2]]:
        output_c = self.o_c(h)
      elif self.c_metric == Metric_Types[1]:
        output_c = torch.sum(self.o_c(y_c) * h, dim=1, keepdim=True)
    return output_b, output_c
