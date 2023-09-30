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

JointRes_D = Speech2Gesture_D

class JointRes_G(nn.Module):
  '''
  Baseline: http://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/

  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, input_channels = 256, output_channels = 256,  out_feats=104, p=0, max_depth = 5, kernel_size=None, stride=None, **kwargs):
    super().__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)
                                  for i in range(4)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

    '''
    UNET INITS
    '''
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv_a = nn.ModuleList([])
    self.conv_p = nn.ModuleList([])
    self.conv_up = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p))
    for i in range(self.max_depth):
      self.conv_a.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p))

    for i in range(self.max_depth):
      self.conv_p.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p))

    for i in range(self.max_depth):
      self.conv_up.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p))

  def forward(self, x, y, time_steps=None, **kwargs):
    input_size = x.shape[-1]
    #WHAT DOES THIS DO WHAT WHY. WHAT IS THIS
    if x.dim() == 3:
      x = x.unsqueeze(dim=1)
    '''
    x = nn.Sequential(*self.pre_downsampling_conv)(x)
    '''
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)


    #x needs to be defined by pose or audio accordingly.
    if random.random() > 0.5 and self.training:
        residuals = []
        x = self.pose_encoder(y, time_steps)
        x = nn.Sequential(*self.pre_downsampling_conv)(x)
        residuals.append(x)
        for i, conv_a in enumerate(self.conv_a):
          x = conv_a(x)
          if i < self.max_depth - 1:
            residuals.append(x)
    else:
        residuals = []
        x = self.audio_encoder(x, time_steps)
        x = nn.Sequential(*self.pre_downsampling_conv)(x)
        residuals.append(x)
        for i, conv_p in enumerate(self.conv_p):
          x = conv_p(x)
          if i < self.max_depth - 1:
            residuals.append(x)
    #upsample
    for i, conv_up in enumerate(self.conv_up):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv_up(x)

    x = self.decoder(x)
    x = self.logits(x)
    internal_losses = []
    return x.transpose(-1, -2), internal_losses
