import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
import warnings

from .layers import *
from .layersUtils import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

from functools import partial
from copy import deepcopy

class DiffGAN_G(nn.Module):
  '''
  Fewshot learning with by just adapting only the relevant grounding and gesture rules

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

    try:
      self.in_noise_dim = self.G.in_noise_dim
    except:
      warnings.warn("'self.G' object does not have attribute 'in_noise_dim'")

    try:
      self.time_steps = self.G.time_steps
    except:
      warnings.warn("'self.G' object does not have attribute 'time_steps'")

    self.sfm = nn.Softmax(dim=1)
    self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    self.kl_wt = 1000
    self.sim = nn.CosineSimilarity()

    self.modules = ['decoder.3']
    self.big_modules = ['decoder']

    requires_grad(self.G, False)

  def activate_decoder(self, choice='small'):
    if choice == 'small':
      modules = self.modules
    elif choice == 'big':
      modules = self.big_modules
    for mod in modules:
      for name, m in self.G.named_modules():
        if name == mod:
          requires_grad(m, True)

  def deactivate_decoder(self, choice='small'):
    if choice == 'small':
      modules = self.modules
    elif choice == 'big':
      modules = self.big_modules
    for mod in modules:
      for name, m in self.G.named_modules():
        if name == mod:
          requires_grad(m, False)
    
      
  def get_layers(self):
    layers = []
    #layers.append('concat_encoder2')
    layers += ['unet.conv1.{}'.format(i) for i in range(5)]
    layers += ['unet.conv2.{}'.format(i) for i in range(5)]
    layers += ['decoder.{}'.format(i) for i in range(4)]
    return layers

  def estimate_z_l(self, x, y, time_steps=None, noise=None, **kwargs):
    z_l_1, internal_losses = self.G.enc1(x, y, time_steps, noise, **kwargs)
    x, internal_losses = self.G.enc2(z_l_1, y, internal_losses, **kwargs)
    return x, z_l_1

  def forward_z_l(self, x, y, z_l, internal_losses, **kwargs):
    return self.G.enc3(x, y, internal_losses, z_l=z_l, **kwargs)

  def forward_z_l_1(self, x, y, z_l_1, internal_losses, **kwargs):
    z_l, internal_losses = self.G.enc2(z_l_1, y, internal_losses, **kwargs)
    x, internal_losses = self.G.enc3(x, y, internal_losses, z_l=z_l, **kwargs)
    return x, z_l, internal_losses
  
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps=time_steps, noise=noise, **kwargs)
    return out, internal_losses


class DiffGAN2_G(DiffGAN_G):
  '''
  Fewshot learning with by just adapting only the relevant grounding and gesture rules
  The complete model is modifiable

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__(time_steps=time_steps,
                     in_channels=in_channels,
                     out_feats=out_feats,
                     p=p,
                     num_clusters=num_clusters,
                     cluster=cluster,
                     pretrained_trainers=pretrained_trainers,
                     **kwargs)
    self.big_modules = [''] ## The complete model

class StyleDiffGAN3_G(DiffGAN_G):
  '''
  Fewshot learning with by just adapting only the relevant grounding and gesture rules
  Most of the model is modifiable except language and audio encoders

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__(time_steps=time_steps,
                     in_channels=in_channels,
                     out_feats=out_feats,
                     p=p,
                     num_clusters=num_clusters,
                     cluster=cluster,
                     pretrained_trainers=pretrained_trainers,
                     **kwargs)
    self.big_modules = ['decoder', 'pose_style_encoder', 'logits', 'classify_cluster', 'style_emb']



class StyleDiffGAN4_G(DiffGAN_G):
  '''
  Fewshot learning with by just adapting only the relevant grounding and gesture rules
  Most of the model is modifiable except language and audio encoders

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__(time_steps=time_steps,
                     in_channels=in_channels,
                     out_feats=out_feats,
                     p=p,
                     num_clusters=num_clusters,
                     cluster=cluster,
                     pretrained_trainers=pretrained_trainers,
                     **kwargs)
    self.big_modules = ['decoder', 'pose_style_encoder', 'logits', 'classify_cluster', 'style_emb']

    self.G_src = deepcopy(self.G)
    self.G_src.eval()
    requires_grad(self.G, False)
    requires_grad(self.G_src, False)
