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

VanillaCNNLate_D = Speech2Gesture_D

class VanillaCNNLate_G(nn.Module):
  '''
  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, **kwargs):
    super().__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    text_key = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'text':
        text_key=key
    if text_key:
      text_channels = kwargs['shape'][text_key][-1]
      self.text_encoder = TextEncoder1D(output_feats = time_steps,
                                        input_channels = text_channels,
                                        p=p)
    else:
      self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
#    self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)
                                  for i in range(4)]))
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

  def forward(self, x, y, time_steps=None, **kwargs):
    # Late Fusion
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        x[i] = self.text_encoder(x[i], time_steps)
      if modality.split('/')[0] == 'audio':
        if x[i].dim() == 3:
          x[i] = x[i].unsqueeze(dim=1)
        x[i] = self.audio_encoder(x[i], time_steps)

    if len(x) >= 2:
      x = torch.cat(tuple(x),  dim=1)
      x = self.concat_encoder(x)
    else:
      x = torch.cat(tuple(x),  dim=1)

    x = self.decoder(x)
    x = self.logits(x)
    internal_losses = []
    return x.transpose(-1, -2), internal_losses
