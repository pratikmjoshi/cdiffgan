import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
from pycasper.torchUtils import some_grad

import torch
import torch.nn as nn

#JointLateClusterSoftPhase_D = Speech2Gesture_D

class TransformerEncoderLayerConvNormRelu(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
    super().__init__()
    self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
    self.conv = ConvNormRelu(d_model, d_model)

  def forward(self, x, src_mask=None, src_key_padding_mask=None):
    x = self.encoder_layer(x, src_mask, src_key_padding_mask)
    x = self.conv(x.transpose(-2, -1)).transpose(-2, -1)
    return x

class JointLateClusterSoftPhase_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, gest_len=2, **kwargs):
    super().__init__()
    self.gest_len = gest_len
    self.lin_downsample = nn.Linear(self.gest_len*out_feats, in_channels)
    self.lin_upsample = nn.Linear(in_channels, self.gest_len*out_feats)

    ## Transformer Encoder
    t_kwargs = dict(
      d_model = in_channels,
      nhead = 8,
      num_layers = 2)
    t_kwargs.update(kwargs)

    encoder_layer = TransformerEncoderLayerConvNormRelu(t_kwargs['d_model'],
                                                        t_kwargs['nhead'],
                                                        t_kwargs['d_model'])
    self.encoder = nn.TransformerEncoder(encoder_layer,
                                         num_layers=t_kwargs['num_layers'],
                                         norm=nn.LayerNorm(t_kwargs['d_model']))
    
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    ## split pose data 16 ways
    y_shape = y.shape
    x = y.reshape(y_shape[0], y.shape[1]//self.gest_len, -1) # (B, 16, 4* feats)
    x = self.lin_downsample(x) # (B, 16, in_channels)
    x = self.encoder(x)
    x = self.lin_upsample(x) # (B, 16, 4*feats)
    x = x.reshape(y_shape[0], y.shape[1], y.shape[2])

    return x, internal_losses
