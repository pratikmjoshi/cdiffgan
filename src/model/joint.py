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

Joint_D = Speech2Gesture_D

class Joint_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, **kwargs):
    super(Joint_G, self).__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)for i in range(4)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

  def forward(self, x, y, time_steps=None, **kwargs):
    if x.dim() == 3:
      x = x.unsqueeze(dim=1)

    #x needs to be defined by pose or audio accordingly.
    if random.random() > 0.5 and self.training:
        x = self.pose_encoder(y, time_steps)
    else:
        x = self.audio_encoder(x, time_steps)

    x = self.unet(x)
    x = self.decoder(x)
    x = self.logits(x)
    internal_losses = []
    return x.transpose(-1, -2), internal_losses
