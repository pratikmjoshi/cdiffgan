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

class ewcGAN_G(nn.Module):
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

  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    out, internal_losses = self.G(x, y, time_steps, noise, **kwargs)

    return out, internal_losses
