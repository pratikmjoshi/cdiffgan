import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
import random

import torch
import torch.nn as nn

class NN_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, **kwargs):
    super().__init__()
    self.dummy = nn.Parameter(torch.zeros(1))
    
  def forward(self, x, y, time_steps=None, **kwargs):
    audio = kwargs['audio']
    pose = kwargs['pose']

    min_mse, idx = ((audio.unsqueeze(1) - x[0].mean(1).unsqueeze(0))**2).mean(-1).min(dim=0)

    out = pose[idx]
    internal_losses = [self.dummy.sum()]
    return out, internal_losses

class Rand_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, **kwargs):
    super().__init__()
    self.dummy = nn.Parameter(torch.zeros(1))
    
  def forward(self, x, y, time_steps=None, **kwargs):
    audio = kwargs['audio']
    pose = kwargs['pose']
    
    idx = torch.randint(low=0, high=pose.shape[0], size=(y.shape[0],))
    
    out = pose[idx]
    internal_losses = [self.dummy.sum()]
    return out, internal_losses

class Mean_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, **kwargs):
    super().__init__()
    self.dummy = nn.Parameter(torch.zeros(1))
    
  def forward(self, x, y, time_steps=None, **kwargs):
    out = torch.zeros_like(y)
    internal_losses = [self.dummy.sum()]
    return out, internal_losses
