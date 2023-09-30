import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .layersUtils import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

from functools import partial
from copy import deepcopy

class Miner(nn.Module):
  def __init__(self, code_dim=512):
    super().__init__()
    self.transform = nn.Sequential(
      nn.Linear(code_dim, code_dim),
      nn.ReLU(),
      nn.Linear(code_dim, code_dim),
    )

  def forward(self, input):
    if type(input) not in (list, tuple):
      input = [input]

    output = [self.transform(i) for i in input]
    return output

  
class MineGAN_G(nn.Module):
  '''
  Fewshot learning with MineGAN

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    self.G = pretrained_trainers.model.G
    self.G_ema = deepcopy(self.G)
    self.in_noise_dim = self.G.in_noise_dim
    self.time_steps = self.G.time_steps
    self.miner = Miner(self.in_noise_dim)

    self.modules = ['concat_encoder', 'concat_encoder2',
                    'audio_encoder', 'pose_encoder', 'text_encoder'] 
    requires_grad(self.G, False)
    requires_grad(self.G_ema, False)
    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    if kwargs.get('step') is None or kwargs.get('step') == 1:
      requires_grad(self.G, False)
    elif kwargs.get('step') == 2:
      requires_grad(self.G, True)
      for mod in self.modules:
        requires_grad(getattr(self.G, mod), False)

    G_wrapper = self.G_ema if kwargs['sample_flag'] else self.G  ## if sampling use the ema model
    if noise is None: 
      if kwargs['sample_flag']:
        n_time_steps = y.shape[1] // self.time_steps 
        noise = torch.rand(n_time_steps, self.in_noise_dim)
      else:
        noise = torch.rand(y.shape[0], self.in_noise_dim)

    noise = noise.to(y.device)
    noise = self.miner(noise)[0]
    
    out, internal_losses = G_wrapper(x, y, time_steps, noise, **kwargs)
    return out, internal_losses

class MineGAN2_G(nn.Module):
  '''
  Fewshot learning with MineGAN

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    self.G = pretrained_trainers.model.G
    self.in_noise_dim = self.G.in_noise_dim
    self.time_steps = self.G.time_steps
    self.miner = UNet1D(in_channels, in_channels)
    requires_grad(self.G, False)
    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    x, labels, internal_losses = self.G.forward_enc(x, y, time_steps=time_steps, noise=noise, **kwargs)
    x = self.miner(x)
    out, internal_losses = self.G.forward_dec(x, y, labels, noise, internal_losses, **kwargs)
    return out, internal_losses

class MineGAN3_G(nn.Module):
  '''
  Fewshot learning with MineGAN

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    self.G = pretrained_trainers.model.G
    self.G_ema = deepcopy(self.G)
    self.in_noise_dim = self.G.in_noise_dim
    self.time_steps = self.G.time_steps
    self.miner = nn.Sequential(ConvNormRelu(in_channels, in_channels, leaky=True),
                               nn.Conv1d(in_channels, in_channels, 3, 1, 1))
    requires_grad(self.G, False)
    requires_grad(self.G_ema, False)
    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    if kwargs.get('step') is None or kwargs.get('step') == 1:
      requires_grad(self.G, False)
    elif kwargs.get('step') == 2:
      requires_grad(self.G, True)

    G_wrapper = self.G_ema if kwargs['sample_flag'] else self.G ## if sampling use the ema model
    x, labels, internal_losses = G_wrapper.forward_enc(x, y, time_steps=time_steps, noise=noise, **kwargs)
    x = self.miner(x)
    out, internal_losses = G_wrapper.forward_dec(x, y, labels, noise, internal_losses, **kwargs)
    return out, internal_losses

