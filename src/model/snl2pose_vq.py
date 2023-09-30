import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

SnL2PoseLateVQ_D = Speech2Gesture_D

class SnL2PoseLateVQ_G(nn.Module):
  '''
  Late Fusion

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, **kwargs):
    super().__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)
                                  for i in range(4)]))
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)
    #self.vq = VQLayer(num_embeddings=512, num_features=in_channels)
    self.vq = VQLayerSG(num_embeddings=512, num_features=in_channels)

  def forward(self, x, y, time_steps=None, **kwargs):
    # Late Fusion
    internal_losses = []
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

    x = self.unet(x)
    x = self.decoder(x)
    #x = self.vq(x)
    x, i_losses = self.vq(x)
    internal_losses += i_losses
    #pdb.set_trace()
    x = self.logits(x)

    return x.transpose(-1, -2), internal_losses

