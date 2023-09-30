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

class AdaptiveGAN_G(nn.Module):
  '''
  Fewshot learning with by just adapting the decoder 

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    self.G = pretrained_trainers.model.G

    self.modules = ['decoder']

    requires_grad(self.G, False)
    for mod in self.modules: ## Fix the encoding modules
      requires_grad(getattr(self.G, mod), True)

    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps, noise, **kwargs)

    return out, internal_losses

AdaptiveMineGAN_G = AdaptiveGAN_G

class AdaptiveGAN2_G(nn.Module):
  '''
  Fewshot learning with by just adapting the decoder 

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    self.G = pretrained_trainers.model.G

    self.modules = ['decoder.3']

    requires_grad(self.G, False)
    for mod in self.modules: ## Fix the encoding modules
      for name, m in self.G.named_modules():
        if name == mod:
          requires_grad(m, True)
    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps, noise, **kwargs)

    return out, internal_losses

class StyleAdaptiveGAN3_G(nn.Module):
  '''
  Adaptive GAN for Mix-Stage Models

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    if 'GAN' in str(type(pretrained_trainers.model.G)):
      self.G = pretrained_trainers.model.G.G
    else:
      self.G = pretrained_trainers.model.G

    self.modules = ['decoder', 'pose_style_encoder', 'logits', 'classify_cluster', 'style_emb']

    requires_grad(self.G, False)
    for mod in self.modules: ## Fix the encoding modules
      requires_grad(getattr(self.G, mod), True)

    
  def forward(self, x, y, time_steps=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps, **kwargs)

    return out, internal_losses

class StyleAdaptiveGAN4_G(nn.Module):
  '''
  Adaptive GAN for Mix-Stage Models

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    if 'GAN' in str(type(pretrained_trainers.model.G)):
      self.G = pretrained_trainers.model.G.G
    else:
      self.G = pretrained_trainers.model.G
    self.G_src = deepcopy(self.G)
    self.G_src.eval()
    
    self.modules = ['decoder', 'pose_style_encoder', 'logits', 'classify_cluster', 'style_emb']

    requires_grad(self.G, False)
    requires_grad(self.G_src, False)
    for mod in self.modules: ## Fix the encoding modules
      requires_grad(getattr(self.G, mod), True)

    
  def forward(self, x, y, time_steps=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps, **kwargs)

    return out, internal_losses
