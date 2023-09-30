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

JointLate_D = Speech2Gesture_D

class JointLate_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
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

    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)for i in range(4)]))
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

  def forward(self, x, y, time_steps=None, **kwargs):
    #pdb.set_trace()
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    if random.random() > 0.5 and self.training:
        x = self.pose_encoder(y, time_steps)
    else:
        #pdb.set_trace()
        for i, modality in enumerate(kwargs['input_modalities']):
            # UNSQUEEZE?
            #if x[i].dim() == 3:
            #    x[i] = x[i].unsqueeze(dim=1)
            if modality.split('/')[0] == "text":
                x[i] = self.text_encoder(x[i], time_steps)
            if modality.split('/')[0] == 'audio':
                if x[i].dim() == 3:
                    x[i] = x[i].unsqueeze(dim=1)
                x[i] = self.audio_encoder(x[i], time_steps)
        #pdb.set_trace()
        if len(x) >= 2:
            x = torch.cat(tuple(x),  dim=1)
            x = self.concat_encoder(x)
        else:
            x = torch.cat(tuple(x),  dim=1)

    #list no tuple
    x = self.unet(x)
    x = self.decoder(x)
    x = self.logits(x)
    internal_losses = []
    return x.transpose(-1, -2), internal_losses
