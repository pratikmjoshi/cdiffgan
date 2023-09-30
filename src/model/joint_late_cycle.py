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

JointLateCycle_D = Speech2Gesture_D

class JointLateCycle_G(nn.Module):
  '''
  Audio Only Model

  x: audio (N, time, frequency)
  y: pose (output): (N, time, pose_feats)

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
      
#    self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)for i in range(4)]))
    self.audio_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(96, 128,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)

  def forward(self, x, y, time_steps=None, **kwargs):

    internal_losses = []
    #pdb.set_trace()
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.


    x = x[0]
    if x.dim() == 3:
      x = x.unsqueeze(dim=1)

#POSE - Loss: audio or pose to pose hat
    randval1 = random.random()
    if randval1 > 0.5 and self.training:
        p_hat = self.pose_encoder(y, time_steps)
    if randval1 <= 0.5:
        p_hat = self.audio_encoder(x, time_steps)
    p_hat = self.unet(p_hat)
    p_hat = self.decoder(p_hat)
    p_hat = self.logits(p_hat)
    #for later p hat to a hat
    p_hat_loss = self.audio_decoder(p_hat)
    if p_hat_loss.dim() == 3:
      p_hat_loss = p_hat_loss.unsqueeze(dim=1)

    p_hat = p_hat.transpose(-1, -2)
    internal_losses.append(nn.functional.mse_loss(p_hat, y))

#Audio - Loss: audio or pose to audio hat
    randval2 = random.random()
    if randval2 > 0.5 and self.training:
        a_hat = self.pose_encoder(y, time_steps)
    if randval2 <= 0.5 and self.training:
        a_hat = self.audio_encoder(x, time_steps)
    a_hat = self.unet(a_hat)
    a_hat = self.decoder(a_hat)
    a_hat = self.logits(a_hat)
    #for later a hat to p hat
    a_hat_loss = torch.transpose(a_hat, 1, 2)
    a_hat = self.audio_decoder(a_hat)
    if a_hat.dim() == 3:
      a_hat = a_hat.unsqueeze(dim=1)


    a_hat = a_hat.transpose(-1,-2)
    internal_losses.append(nn.functional.mse_loss(a_hat, x))

    #pdb.set_trace()
    # p_hat to a_hat, vice versa -> which way to decode?
    p_hat_loss = p_hat_loss.transpose(-1, -2)
    internal_losses.append(nn.functional.mse_loss(a_hat_loss, p_hat)) #prior to decoding a_hat
    internal_losses.append(nn.functional.mse_loss(p_hat_loss, a_hat)) #after decoding a_hat

    #pdb.set_trace()
    #list no tuple

    #WHAT LOSS FOR NN.FUNCTIONAL?

    # x = p_hat

    return p_hat, internal_losses
