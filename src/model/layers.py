import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb
import copy
from typing import Optional, Any

import torch
import torch.nn as nn
from transformers import BertModel
#from functions import vq, vq_st
import logging

from soft_dtw_cuda import SoftDTW, SoftDTW_DBS
#from fastdtw_mod import fastdtw

logging.getLogger('transformers').setLevel(logging.CRITICAL)

FLOAT = torch.float # torch.float | torch.double

def num_powers_of_two(x):
  num_powers = 0
  while x>1:
    if x % 2 == 0:
      x /= 2
      num_powers += 1
    else:
      break
  return num_powers

def next_multiple_power_of_two(x, power=5):
  curr_power = num_powers_of_two(x)
  if curr_power < power:
    x = x * (2**(power-curr_power))
  return x

class ConvNormRelu(nn.Module):
  def __init__(self, in_channels, out_channels,
               type='1d', leaky=False,
               downsample=False, kernel_size=None, stride=None,
               padding=None, p=0, groups=1):
    super(ConvNormRelu, self).__init__()
    if kernel_size is None and stride is None:
      if not downsample:
        kernel_size = 3
        stride = 1
      else:
        kernel_size = 4
        stride = 2

    if padding is None:
      if isinstance(kernel_size, int) and isinstance(stride, tuple):
        padding = tuple(int((kernel_size - st)/2) for st in stride)
      elif isinstance(kernel_size, tuple) and isinstance(stride, int):
        padding = tuple(int((ks - stride)/2) for ks in kernel_size)
      elif isinstance(kernel_size, tuple) and isinstance(stride, tuple):
        assert len(kernel_size) == len(stride), 'dims in kernel_size are {} and stride are {}. They must be the same'.format(len(kernel_size), len(stride))
        padding = tuple(int((ks - st)/2) for ks, st in zip(kernel_size, kernel_size))
      else:
        padding = int((kernel_size - stride)/2)


    in_channels = in_channels*groups
    out_channels = out_channels*groups
    if type == '1d':
      self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm1d(out_channels)
      self.dropout = nn.Dropout(p=p)
    elif type == '2d':
      self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            groups=groups)
      self.norm = nn.BatchNorm2d(out_channels)
      self.dropout = nn.Dropout2d(p=p)
    if leaky:
      self.relu = nn.LeakyReLU(negative_slope=0.2)
    else:
      self.relu = nn.ReLU()

  def forward(self, x, **kwargs):
    return self.relu(self.norm(self.dropout(self.conv(x))))

class UNet1D(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super(UNet1D, self).__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, return_bottleneck=False, return_feats=False, feats=[]):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    bn = x

    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)
      if return_feats:
        feats.append(x)

    if return_feats:
      return x, feats
    elif return_bottleneck:
      return x, bn
    else:
      return x

class UNet1DEncoder(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.pre_downsampling_conv = nn.ModuleList([])
    self.conv1 = nn.ModuleList([])
    self.max_depth = max_depth
    self.groups = groups

    ## pre-downsampling
    self.pre_downsampling_conv.append(ConvNormRelu(input_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.pre_downsampling_conv.append(ConvNormRelu(output_channels, output_channels,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    for i in range(self.max_depth):
      self.conv1.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=True,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))


  def forward(self, x):
    input_size = x.shape[-1]
    assert input_size/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(input_size, 2**(self.max_depth - 1))
    #assert np.log2(input_size) == int(np.log2(input_size)), 'Input size is {}. It must be a power of 2.'.format(input_size)
    assert num_powers_of_two(input_size) >= self.max_depth, 'Input size is {}. It must be a multiple of 2^(max_depth) = 2^{} = {}'.format(input_size, self.max_depth, 2**self.max_depth)

    x = nn.Sequential(*self.pre_downsampling_conv)(x)

    residuals = []
    residuals.append(x)
    for i, conv1 in enumerate(self.conv1):
      x = conv1(x)
      if i < self.max_depth - 1:
        residuals.append(x)

    return x, residuals

class UNet1DDecoder(nn.Module):
  '''
  UNet Decoder for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    max_depth (int, optional): depth of the UNet (default: ``5``).
    kernel_size (int, optional): size of the kernel for each convolution (default: ``None``)
    stride (int, optional): stride of the convolution layers (default: ``None``)

  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, output_channels, max_depth=5, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.Upsample(scale_factor=2, mode='nearest')
    self.max_depth = max_depth
    self.groups = groups

    for i in range(self.max_depth):
      self.conv2.append(ConvNormRelu(output_channels, output_channels,
                                     type='1d', leaky=True, downsample=False,
                                     kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, residuals, return_feats=False, feats=[], residual_mask=None):
    if residual_mask is None:
      residual_mask = [1]*len(residuals)
    for i, conv2 in enumerate(self.conv2):
      x = self.upconv(x) + residual_mask[self.max_depth - i - 1] * residuals[self.max_depth - i - 1]
      #x = self.upconv(x) + residuals[self.max_depth - i - 1]
      x = conv2(x)
      if return_feats:
        feats.append(x)

    if return_feats:
      return x, feats
    else:
      return x

class AudioEncoder(nn.Module):
  '''
  input_shape:  (N, C, time, frequency)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=1, kernel_size=None, stride=None, p=0, groups=1):
    super(AudioEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(128, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='2d', leaky=True, downsample=False,
                                  kernel_size=(3,8), stride=1, p=p, groups=groups))

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear', align_corners=False)
    x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='nearest')
    x = x.squeeze(dim=-1)
    return x

class PoseEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1):
    super(PoseEncoder, self).__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None, return_feats=False, feats=[]):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    if return_feats:
      for conv in self.conv:
        x = conv(x)
        feats.append(x)
    else:
      x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)

    if return_feats:
      return x, feats
    else:
      return x

    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class PoseStyleEncoder(nn.Module):
  '''
  input_shape:  (N, time, pose_features: 104) #changed to 96?
  output_shape: (N, 256, t)
  '''
  def __init__(self, output_feats=64, input_channels=96, kernel_size=None, stride=None, p=0, groups=1, num_speakers=4):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(256, num_speakers, type='1d', leaky=True, downsample=True,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))



    ## deprecated, but kept only for older models
    # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
    ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

  def forward(self, x, time_steps=None):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.mean(-1)
    #x = x.squeeze(dim=-1)
    return x

class PoseDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, style_dim=10, num_clusters=8, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.style_dim = style_dim
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels+style_dim,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(4)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    style = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])[:, -self.style_dim:]
    for i, model in enumerate(self.pose_decoder):
      #x = torch.split(x, int(x.shape[1]/self.num_clusters), dim=1)
      #x = torch.cat([torch.cat([x_, kwargs['style']], dim=1) for x_ in x], dim=1)
      x = model(x)
      if i < len(self.pose_decoder) - 1: ## ignore last layer
        x = x.view(x.shape[0], -1, self.num_clusters, x.shape[-1])
        x = torch.cat([x, style], dim=1).view(x.shape[0], -1, x.shape[-1])
    return self.pose_logits(x)

class StyleDecoder(nn.Module):
  '''
  input_shape:  (N, channels, time)
  output_shape: (N, 256, time)
  '''
  def __init__(self, input_channels=256, num_clusters=10, out_feats=96, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.num_clusters = num_clusters
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(input_channels,
                                                                   input_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=num_clusters)
                                                      for i in range(2)]))
    self.pose_logits = nn.Conv1d(input_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

  def forward(self, x, **kwargs):
    x = self.pose_decoder(x)
    return self.pose_logits(x)


#TODO Unify Encoders via input_channel size?
class TextEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, text_features: 300)
  output_shape: (N, 256, time)
  '''
  def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])

    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None, **kwargs):
    x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x

class MultimodalMultiscaleEncoder(nn.Module):
  '''
  Encoding language and audio jointly with unsupervised alignment
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
    super().__init__()
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        text_key=key
    if text_key:
      self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
    self.pos_encoder = PositionalEncoding(256, p)

    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                      p=p)]))
    self.norm = nn.LayerNorm(256)

  def forward(self, x, y, time_steps=None, **kwargs):
    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
    mod_map = {}

    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        mod_map['text'] = i
        #.set_trace() #self.training FALSE

        for te in self.text_encoder:
          x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

      if modality.split('/')[0] == 'audio':
        mod_map['audio'] = i
        if x[i].dim() == 3:
          x[i] = x[i].unsqueeze(dim=1)
        x[i] = self.audio_encoder(x[i], time_steps)

    if len(x) >= 2:
      memory = x[mod_map['text']]
      tgt = x[mod_map['audio']]
      tgt = self.pos_encoder(tgt.permute(2, 0, 1)).permute(1, 2, 0)
      x_ = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, **kwargs)
      x = torch.cat([x_, x[mod_map['audio']]], dim=1)
      x = self.concat_encoder2(x)
    else:
      x = torch.cat(tuple(x),  dim=1)

    x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
    return x

class JointEncoder(nn.Module):
  def __init__(self, m1, m2, num_iters):
    super().__init__()
    self.m1 = m1
    self.m2 = m2
    self.thresh = Curriculum(0, 1, num_iters)

  def forward(self, x, y, time_steps=None, **kwargs):

    if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      return self.m1(y, time_steps) # pose encoder
    else:
      return self.m2(x, y, time_steps, **kwargs) # audio/language encoder

class MixGANDecoder(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters

      ## no residual connections by default
      self.resnet = kwargs.get('resnet') if kwargs.get('resnet') is not None else False

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, input_channels=in_channels)
      self.classify_loss = nn.CrossEntropyLoss()
      self.decoder = nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                 type='1d', leaky=True, downsample=False,
                                                 p=p, groups=self.num_clusters)
                                    for i in range(4)])
      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

      self.labels_cap_soft = None

    def index_select_outputs(self, x, labels):
      '''
      x: (B, num_clusters*out_feats, T)
      labels: (B, T, num_clusters)
      '''
      x = x.transpose(2, 1)
      x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
      labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling

      x = (x * labels.unsqueeze(-1)).sum(dim=-2)
      return x

    def forward(self, x, labels, **kwargs):
      internal_losses = []

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      for dec in self.decoder:
        if self.resnet:
          x = dec(x) + x
        else:
          x = dec(x)

      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)

      return x, internal_losses

#Positional Encoding missing in vanilla Transformer
#source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class PositionalEncoding(nn.Module):
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class FramesPositionalEncoding(nn.Module):
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000, batch_size = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x, text_duration, train):
        text_duration_long = text_duration.long()
        sample_flag = 0 if x.shape[1] != text_duration.shape[0] else 1
        if sample_flag:
            with torch.no_grad():
                for i, interval in enumerate(text_duration_long):
                    new_pos = 0
                    for j, word_dur in enumerate(interval):
                        try:
                            x[new_pos: new_pos + word_dur, i,:] += self.pe[0: word_dur,:]
                            new_pos += word_dur
                        except:
                            pdb.post_mortem()
        else:
            text_duration_long = text_duration.long()
            text_collapsed = text_duration_long.reshape(-1)
            new_pos = 0
            for i, word_dur in enumerate(text_collapsed):
                x[new_pos:new_pos + word_dur,0,:] += self.pe[0: word_dur,:]
                new_pos += word_dur
        return self.dropout(x)


class RepeatWordPositionalEncoding(nn.Module): #word repeats
    def __init__(self, input_channels=300, dropout=0.1, max_len=5000, batch_size = 32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, input_channels)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-math.log(10000.0) / input_channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x, text_duration, train):
        text_duration_long = text_duration.long()
        if train:
            with torch.no_grad():
                for i, interval in enumerate(text_duration_long):
                    new_pos = 0
                    for j, word_dur in enumerate(interval):
                        x[new_pos: new_pos + word_dur, i,:] += self.pe[j: j+1,:]
                        new_pos += word_dur
        else:
            text_duration_long = text_duration.long()
            text_collapsed = text_duration_long.reshape(-1)
            new_pos = 0
            for i, word_dur in enumerate(text_collapsed):
                x[new_pos:new_pos + word_dur,0,:] += self.pe[0: word_dur,:]
                new_pos += word_dur
        return self.dropout(x)


class TransfomerEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    ## Linear Layers
    self.tlinear_enc = nn.Linear(in_channels, E)
    self.ptlinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)

    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)

  def forward(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    #pdb.set_trace()
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    text_duration = kwargs['text/token_duration']
    if src_key_padding_mask.shape[1] != x.shape[1]:
      src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
      text_duration = text_duration.view(x.shape[0], x.shape[1])
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    memory = self.pos_encoder(memory)
    memory = self.transformer_text_encoder(memory, src_key_padding_mask=src_key_padding_mask)
    if output_repeat:
      assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
      batch_memory_list = []
      for b in range(memory.shape[1]): # batch
        memory_list = []
        for i in range(memory.shape[0]): # word
            repeats = int(text_duration[b, i].item())
            if (repeats != 0):
                memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) ]
                memory_list.append(torch.cat(memory_list_, dim=0))
        sec_memory = torch.cat(memory_list, dim=0)
        batch_memory_list.append(sec_memory)
      final_memory = torch.cat(batch_memory_list, dim=1)
    return final_memory.transpose(0, 1) ## (B, time, channels)

class TransfomerEncoder2(TransfomerEncoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()

  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs): #repeated text, kwargs token_count, duration not given
    #src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count']) #not needed
    # text_duration = kwargs['text/token_duration'] #not given
    # if src_key_padding_mask.shape[1] != x.shape[1]:
    #   src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    #   text_duration = text_duration.view(x.shape[0], x.shape[1])
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    memory = self.pos_encoder(memory)
    #memory = self.transformer_text_encoder(memory, src_key_padding_mask=src_key_padding_mask)
    # if repeat_text:
    #   #assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)


class TransfomerEncoder_WordPOS(TransfomerEncoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=2, **kwargs):
    super().__init__()
    ## Linear Layers
    self.tlinear_enc = nn.Linear(in_channels, E)
    self.ptlinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)
    self.frames_pos_encoder = FramesPositionalEncoding(self.ninp, p)
    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)


  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs):
    text_duration = kwargs['text/token_duration']
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    train = True
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    if src_key_padding_mask.shape[0] != x.shape[0]:
      train = False
    memory = self.frames_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory) #mask unneeded for source
    #pdb.set_trace()
    # if repeat_text:
    #   assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)



class TransfomerEncoder_Multi(TransfomerEncoder_WordPOS): #Word Level + Frame Level Pos Encoding
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    self.word_pos_encoder = RepeatWordPositionalEncoding(self.ninp, p)


  def forward(self, x, y, input_repeat = 1,  output_repeat=1, **kwargs):
    text_duration = kwargs['text/token_duration']
    memory = x.transpose(0, 1)
    memory = self.tlinear_enc(memory)
    train = True
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'])
    if src_key_padding_mask.shape[0] != x.shape[0]:
      train = False
    memory = self.word_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory)
    memory = self.frames_pos_encoder(memory, text_duration, train)
    memory = self.transformer_text_encoder(memory) #mask unneeded for source


    #pdb.set_trace()
    # if repeat_text:
    #   assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    #   memory_list = []
    #   for b in range(memory.shape[1]):
    #     memory_list_ = [memory[i, b:b+1].repeat(int(text_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
    #     memory_list.append(torch.cat(memory_list_, dim=0))
    #   memory = torch.cat(memory_list, dim=1)
    return memory.transpose(0, 1) ## (B, time, channels)



class TransfomerDecoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    ## Linear Layers
    self.plinear_enc = nn.Linear(out_feats, E)
    self.linear_decoder = nn.Linear(E, out_feats)
    self.decoder_emb = nn.Linear(E, out_feats)

    ## Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, p)

    #Decoder
    decoder_layers = nn.TransformerDecoderLayer(E, self.nhead, self.nhid)
    decoder_norm = torch.nn.LayerNorm(self.ninp)
    self.transformer_text_decoder = nn.TransformerDecoder(decoder_layers, self.nhid, decoder_norm) # Norm\

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)

  def forward(self, memory, y, time_steps=None, **kwargs):
    tgt_mask = self._generate_square_subsequent_mask(y.shape[1]).to(y.device).float()
    if time_steps is None:
      time_steps = y.shape[1]
    memory = memory.transpose(0, 1)
    y = y.transpose(0, 1)

    if self.training:
      y = self.plinear_enc(y)
      output = self.transformer_text_decoder(y, memory, tgt_mask = tgt_mask)
    else:
      batch_size = y.shape[1]
      output = torch.zeros(time_steps+1, batch_size, self.ninp).float().to(y.device)
      for t in range(1, time_steps+1):
        tgt_emb = output[:t]
        decoder_output = self.transformer_text_decoder(tgt_emb, memory)
        output[t] = decoder_output[-1]
        #output[1:t+1] = decoder_output ## output of decoder updated at every step
      output = output[1:] ## removing the starting zero
    output = self.linear_decoder(output)
    return output.transpose(0, 1)

class TransfomerDecoderRand(TransfomerDecoder):
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__(time_steps, in_channels, out_feats, p, E, nhead, nhid, **kwargs)

  def forward(self, memory, y, time_steps=None, **kwargs):
    tgt_mask = self._generate_square_subsequent_mask(y.shape[1]).to(y.device).float()
    if time_steps is None:
      time_steps = y.shape[1]
    memory = memory.transpose(0, 1)
    y = y.transpose(0, 1)

    y = torch.rand_like(y)
    y = self.plinear_enc(y)
    output = self.transformer_text_decoder(y, memory, tgt_mask = tgt_mask)
    output = self.linear_decoder(output)
    return output.transpose(0, 1)


class TextEncoderTransformer_d(nn.Module):
      '''
      input_shape:  (N, time, text_features: 300)
      output_shape: (N, 256, output_feats)
      '''
      def __init__(self, ntokens = 30, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0):
        super().__init__()
        self.n_heads = 12
        self.n_layers = 3
        self.ntoken = ntokens#TODO: needs to be found
        self.input_channels = input_channels
        self.pos_encoder = PositionalEncoding(input_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(input_channels, self.n_heads, 256)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, self.n_layers)
        self.encoder = nn.Embedding(self.ntoken, input_channels) #token
        self.conv = ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                 kernel_size=kernel_size, stride=stride, p=p)


      def forward(self, x, time_steps=None):
        if time_steps is None:
          time_steps = x.shape[-2]

        x = self.encoder(x) * math.sqrt(self.input_channels)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.transpose(1, 0)).transpose(1, 0).transpose(2, 1)
        x = self.conv(x)
        return x

class TextEncoderTransformer(nn.Module):
  '''
  input_shape:  (N, time, text_features: 300)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=300, kernel_size=None, stride=None, p=0):
    super().__init__()
    self.n_heads = 12
    self.n_layers = 3
    self.encoder_layer = nn.TransformerEncoderLayer(input_channels, self.n_heads, 256)
    self.encoder = nn.TransformerEncoder(self.encoder_layer, self.n_layers)

    self.conv = ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                             kernel_size=kernel_size, stride=stride, p=p)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2]
    x = self.encoder(x.transpose(1, 0)).transpose(1, 0).transpose(2, 1)
    x = self.conv(x)
    return x

    #TODO Unify Encoders via input_channel size?

class BertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=3, **kwargs):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.linear = torch.nn.Linear(768, out_feats)

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  def chunk(self, x, pad, max_len=512):
    x_len = x.shape[-1]
    batch = (x_len - 1) // max_len + 1
    if batch > 1:
      new_len = max_len * batch
      x = torch.cat([x, torch.zeros(1, new_len-x_len).float().to(x.device)], dim=-1)
      pad = torch.cat([pad, torch.zeros(1, new_len-x_len).bool().to(x.device)], dim=-1)
      x = x.view(batch, -1)
      pad = pad.view(batch, -1)

    return x, pad, x_len, batch

  def forward(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    token_type_ids = None
    # if len(x.shape) == 3:
    #   sample_flag = True
    # else:
    #   sample_flag = False
    sample_flag = kwargs["sample_flag"]

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    if sample_flag:
      x = x.view(1, -1)
      src_key_padding_mask = src_key_padding_mask.view(1, -1)
      x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())

    memory = self.linear(memory)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
      token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)

    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    return memory.transpose(-1, -2)

  def forward_archive(self, x, y, input_repeat = 0, output_repeat=0, **kwargs):
    token_type_ids = None
    if len(x.shape) == 3:
      sample_flag = True
      x = x.squeeze(0)
    else:
      sample_flag = False

    ## Create Masks
    assert 'text/token_duration' in kwargs, 'Could not find text/token_duration'
    token_duration = kwargs['text/token_duration']
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['text/token_count'], mask_val=1)
    #if src_key_padding_mask.shape[1] != x.shape[1]:
    # if sample_flag:
    #   x = x.view(1, -1)
    #   src_key_padding_mask = src_key_padding_mask.view(1, -1)
    #   x, src_key_padding_mask, orig_len, batch = self.chunk(x, src_key_padding_mask)

      #src_key_padding_mask = src_key_padding_mask.view(x.shape[0], x.shape[1])
    memory, pooled_output = self.bert(x.long(), token_type_ids, src_key_padding_mask.long())
    memory = self.linear(memory)

    # if sample_flag:
    #   memory = memory.view(1, -1, memory.shape[-1])[:, :orig_len]
    #   token_duration = token_duration.view(memory.shape[0], memory.shape[1])[:, :orig_len]

    if 'pos_encoder' in kwargs: ## add positional embedding before repeating -- Useful is used in conjunction with another transformer
      memory = kwargs['pos_encoder'](memory.transpose(1, 0)).transpose(1, 0) ## needs input in the form of (T, B, C)

    if output_repeat:
      memory = self.output_repeat_text(memory.transpose(1, 0), token_duration).transpose(1, 0)

    if sample_flag:
      memory = memory.view(1, -1, memory.shape[-1])

    return memory.transpose(-1, -2)


class MultiScaleBertEncoder(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, **kwargs):
    super().__init__()
    self.word_encoder = BertEncoder(out_feats=out_feats)

    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.ninp = out_feats
    self.pos_encoder = PositionalEncoding(self.ninp, p)
    self.frame_pos_encoder = FramesPositionalEncoding(input_channels=E, dropout=0)

    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    encoder_norm = torch.nn.LayerNorm(self.ninp)
    self.frame_encoder = nn.TransformerEncoder(encoder_layers, self.nhid, encoder_norm) # Norm

  def forward(self, x, y, input_repeat=0, output_repeat=1, **kwargs):
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False
    memory = self.word_encoder(x, y, input_repeat=0, output_repeat=1, pos_encoder=self.pos_encoder, **kwargs).transpose(-1, -2) ## (B, T) -> (B, T, C)
    memory = self.frame_pos_encoder(memory.transpose(1, 0), kwargs['text/token_duration'], is_train) # (T, B, C) as input -> (T, B, C)
    memory = self.frame_encoder(memory)
    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)

class MultimodalTransformerFusion(nn.Module):
  '''
  tgt: audio signal (T, B, C)
  src: text signal (L, B, C), if input_repeat == 0 => L!=T and if input_repeat == 1 => L==T
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, p = 0, E = 256, nhead=8, nhid=256, nlayer=2,**kwargs):
    super().__init__()
    ## Frame Encoder
    self.nhead = nhead
    self.nhid = nhid
    self.nlayer = nlayer
    self.ninp = out_feats

    decoder_layers = nn.TransformerDecoderLayer(self.ninp, self.nhead, self.nhid)
    decoder_norm = torch.nn.LayerNorm(self.ninp)
    self.memory_decoder = nn.TransformerDecoder(decoder_layers, self.nlayer, decoder_norm) # Norm

  def _generate_source_key_padding_mask(self, token_count, mask_val=0):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count) - mask_val
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = mask_val
    return mask.bool().to(token_count.device)

  def _generate_source_mask(self, token_duration, tgt_len, bsz, input_repeat):
    if input_repeat == 0:
      mask = torch.ones(bsz*self.nhead, tgt_len, token_duration.shape[-1]) # (B, T, L)
    else:
      mask = torch.ones(bsz*self.nhead, tgt_len, tgt_len) # (B, T, T)
    for b in range(token_duration.shape[0]):
      pos = 0
      for i in range(token_duration.shape[1]):
        duration = int(token_duration[b, i].item())
        if input_repeat == 0:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, i] = 0
        else:
          mask[b*self.nhead:(b+1)*self.nhead, pos:pos+duration, pos:pos+duration] = 0
        pos = pos + duration
    #mask = mask.float().masked_fill(mask==1, float('-inf')).masked_fill(mask==0, float(0.0)).to(token_duration.device)
    #return mask
    return mask.bool().to(token_duration.device)


  def output_repeat_text(self, memory, token_duration):
    memory_list = []
    for b in range(memory.shape[1]):
      memory_list_ = [memory[i, b:b+1].repeat(int(token_duration[b, i].item()), 1, 1) for i in range(memory.shape[0])]
      memory_list.append(torch.cat(memory_list_, dim=0))
    memory = torch.cat(memory_list, dim=1)
    return memory

  '''
  tgt: (B, C, T) -> (T, B, C)
  memory: (B, C, L) -> (L, B, C)
  '''
  def forward(self, tgt, memory, y, input_repeat=0, output_repeat=1, src_mask=True, query_text=False, **kwargs):
    tgt = tgt.permute(2, 0, 1)
    memory = memory.permute(2, 0, 1)
    if kwargs['description'] == 'train':
      is_train = True
    else:
      is_train = False

    token_duration = kwargs['text/token_duration']
    token_count = kwargs['text/token_count']
    if token_duration.shape[0] != tgt.shape[1]: ## sample_loop
      token_duration = token_duration.view(1, -1)
      sample_flag = True
    else:
      sample_flag = False

    if src_mask:
      src_mask = self._generate_source_mask(token_duration, tgt.shape[0], tgt.shape[1], input_repeat)
    else:
      src_mask = None
    if input_repeat == 0:
      src_key_padding_mask = self._generate_source_key_padding_mask(token_count)
      if sample_flag:
        src_key_padding_mask = src_key_padding_mask.view(1, -1)
    else:
      src_key_padding_mask = None

    if not query_text:
      ## memory(~key and value) is text, tgt (~query) is audio
      #TODO: what is this?????
      memory = self.memory_decoder(tgt, memory, memory_key_padding_mask=src_key_padding_mask, memory_mask=src_mask)
    else:
      memory = self.memory_decoder(memory, tgt, tgt_key_padding_mask=src_key_padding_mask, tgt_mask=src_mask)

    return memory.transpose(1, 0).transpose(-1, -2) # (B, C, T)


class Transpose(nn.Module):
  def __init__(self, idx):
    super().__init__()
    self.param = torch.nn.Parameter(torch.ones(1))
    self.idx = idx

  def forward(self, x, *args, **kwargs):
    return x.transpose(*self.idx)

class AudioEncoder1D(nn.Module):
  '''
  input_shape:  (N, time, audio_features: 128)
  output_shape: (N, 256, output_feats)
  '''
  def __init__(self, output_feats=64, input_channels=128, kernel_size=None, stride=None, p=0, groups=1):
    super().__init__()
    self.conv = nn.ModuleList([])
    self.conv.append(ConvNormRelu(input_channels, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(64, 64, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

    self.conv.append(ConvNormRelu(64, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(128, 128, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))


    self.conv.append(ConvNormRelu(128, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv.append(ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))

  def forward(self, x, time_steps=None):
    #x = torch.transpose(x, 1, 2)
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    #x = self.upconv(x)
    #x = torch.nn.functional.interpolate(x, size=(time_steps, 1), mode='bilinear')
    x = x.squeeze(dim=-1)
    return x


        ## deprecated, but kept only for older models
        # self.upconv = nn.Upsample(size=(output_feats, 1), mode='bilinear')
        ## TODO maybe the size should be (output_feats,1) instead, as we want to upsample along the time dimension

class LatentEncoder(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels=2, p=0):
    super().__init__()
    enc1 = nn.ModuleList([ConvNormRelu(in_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    enc2 = nn.ModuleList([ConvNormRelu(hidden_channels, hidden_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(2)])
    enc3 = nn.ModuleList([ConvNormRelu(hidden_channels, out_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=1)
                          for i in range(1)])
    self.enc = nn.Sequential(*enc1, *enc2, *enc3)

  def forward(self, x):
    x = self.enc(x)
    return x


class VQLayer(nn.Module):
  '''
  VQ Layer without Stop gradient
  '''
  def __init__(self, num_embeddings=8, num_features=96, weight=None):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.emb = nn.Embedding(self.num_embeddings, self.num_features, _weight=weight)

  def forward(self, x):
    x = x.transpose(-1, -2) ## (B, T, num_features)
    x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) ## (B, T, num_features)
    centers = self.emb.weight.transpose(1, 0) ## (num_features, num_embeddings)
    centers = centers.view(1, 1, centers.shape[0], centers.shape[1]) ## (1, 1, num_features, num_embeddings)
    dist = ((x-centers)**2).sum(dim=-2) ## (B, T, num_embeddings)
    idxs = torch.argmin(dist, dim=-1) ## (B, T)

    return self.emb(idxs).transpose(-1, -2), dist

class VQLayerSG(nn.Module):
  '''
  VQ layer with stop gradient
  '''
  def __init__(self, num_embeddings=8, num_features=96, weight=None,
               beta=1, expavg=False, gamma=0.99, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.emb = nn.Embedding(self.num_embeddings, self.num_features, _weight=weight)
    self.emb.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)
    #self.emb.weight.data.uniform_(-2., 2.)
    self.beta = beta
    self.EMA = expavg ## exponential moving average
    if self.EMA:
      self.expavg = ExpAvg(self.emb.weight.clone())
    self.gamma = gamma

  def forward(self, x):
    internal_losses = []
    x = x.permute(0, 2, 1) ## (B, T, C)

    dist = torch.sum(x ** 2, dim=-1, keepdim=True) + \
           torch.sum(self.emb.weight ** 2, dim=-1).unsqueeze(0).unsqueeze(0) - \
           2 * x @ self.emb.weight.t() ## (B, T, num_embeddings)

    idxs = torch.argmin(dist, dim=-1)
    x_q = self.emb(idxs)

    ## update Embeddings
    if self.EMA:
      internal_losses.append(torch.zeros(1)[0].to(x.device))
      if self.training:
        self.expavg(x_q.view(-1, x_q.shape[-1]), idxs.view(-1))
        self.emb.weight.data = self.expavg.get_weights()
    else:
      internal_losses.append(((x.detach() - x_q)**2).mean()) ## vq loss

    ## Update Encoder
    internal_losses.append(((x - x_q.detach())**2).mean() * self.beta) ## commit loss

    ## preserve gradients to update Decoder and Encoder
    x_q = x + (x_q - x).detach()

    return x_q.permute(0, 2, 1), idxs, internal_losses
  # def get_dist(self, x, y):
  #   return ((x - y)**2).mean(dim=-2)

  # def get_min_dist(self, dist):
  #   return dist.min(dim=-1)[0].mean()

  # def forward(self, x):
  #   x = x.transpose(-1, -2) ## (B, T, num_features)
  #   x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) ## (B, T, num_features)
  #   centers = self.emb.weight.transpose(1, 0) ## (num_features, num_embeddings)
  #   centers = centers.view(1, 1, centers.shape[0], centers.shape[1]) ## (1, 1, num_features, num_embeddings)
  #   dist = self.get_dist(x, centers.detach()) ## (B, T, num_embeddings)
  #   dist, idxs = dist.min(dim=-1) ## (B, T)

  #   internal_losses = []
  #   beta = 0.1
  #   internal_losses.append(dist.mean()*beta)
  #   internal_losses.append(self.get_min_dist(self.get_dist(x.detach(), centers)))

  #   ## get the output
  #   with torch.no_grad():
  #     out = self.emb(idxs.detach())

  #   # idxs_shape = list(idxs.shape)
  #   # idxs = idxs.view(-1)
  #   # out = torch.index_select(self.emb.weight.detach(), dim=0, index=idxs)
  #   # out_shape = idxs_shape + [out.shape[-1]]
  #   # out = out.view(*out_shape)
  #   return out.transpose(-1, -2), internal_losses

# class VQEmbedding(nn.Module):
#   def __init__(self, K, D):
#     super().__init__()
#     self.embedding = nn.Embedding(K, D)
#     self.embedding.weight.data.uniform_(-1./K, 1./K)

#   def forward(self, z_e_x):
#     z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
#     latents = vq(z_e_x_, self.embedding.weight)
#     return latents

#   def straight_through(self, z_e_x):
#     z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
#     z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
#     z_q_x = z_q_x_.permute(0, 2, 1).contiguous()
    
#     z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
#                                            dim=0, index=indices)
#     z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
#     z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()

#     return z_q_x, z_q_x_bar

class ExpAvg(nn.Module):
  '''
  Esponential average to update Weights of a Vector Quantized Layer
  weight: torch.float (num_embeddings, emdedding_dim)
  gamma: float (default:0.99)
  
  __call__ args
  E: torch.float (B, embedding_dim)
  idxs: torch.long (B,), 0 <= value < num_embeddings
  '''
  def __init__(self, weight=None, gamma=0.99):
    super().__init__()
    assert weight is not None, 'provide embedding weight to the Exponential Averaging Class'
    self.N = torch.nn.Parameter(torch.zeros(weight.shape[0], 1), requires_grad=False)
    self.m = torch.nn.Parameter(weight, requires_grad=False)
    self.gamma = gamma
    
  def forward(self, E, idxs):
    with torch.no_grad():
      idxs_onehot = torch.nn.functional.one_hot(idxs, num_classes=self.N.shape[0]).to(FLOAT)
      n_i_t = idxs_onehot.sum(0).unsqueeze(1)
      m_i_t = idxs_onehot.t() @ E

      zero_mask = (n_i_t > 0).to(FLOAT)

      ## update number of vectors
      N_ = self.N * self.gamma + n_i_t * (1-self.gamma)
      self.N.data = zero_mask * N_ + (1-zero_mask) * self.N ## do not update the indices not available in the batch

      ## update embeddings
      m_ = self.m * self.gamma + m_i_t * (1-self.gamma)
      self.m.data = zero_mask * m_ + (1-zero_mask) * self.m ## do not update the indices not available in the batch
  
  def get_weights(self):
    ## edge case: self.N[i] is zero. So the averager has yet to see any updates to the `i`th embedding
    ## we will return the original embedding in that case
    
    ## add 1 to the indices not seen before
    #N_ = self.N + (self.N == 0).to(FLOAT)

    ## Laplace Smoothing
    eps = 1
    N_sum = self.N.data.sum()
    with torch.no_grad():
      self.N.data = ((self.N + eps)/(N_sum + self.N.shape[0] + eps) * N_sum).data
    return (self.m / self.N).data

class ClusterClassify(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, num_clusters, T)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x

class ClusterClassifyGAN(nn.Module):
  '''
  input_shape: (B, C, T)
  output_shape: (B, T, num_clusters)
  '''
  def __init__(self, num_clusters=8, kernel_size=None, stride=None, p=0, groups=1, input_channels=256):
    super().__init__()
    self.conv = nn.ModuleList()
    self.conv.append(ConvNormRelu(input_channels, 256, type='1d', leaky=True, downsample=False,
                                  kernel_size=kernel_size, stride=stride, p=p, groups=groups))
    self.conv += nn.ModuleList([ConvNormRelu(256, 256, type='1d', leaky=True, downsample=False,
                                             kernel_size=kernel_size, stride=stride, p=p, groups=groups) for i in range(5)])

    self.logits = nn.Conv1d(256*groups, num_clusters*groups, kernel_size=1, stride=1, groups=groups)

  def forward(self, x, y, time_steps=None):
    if time_steps is None:
      time_steps = x.shape[-2] ## assume it is same as the input time steps

    x = nn.Sequential(*self.conv)(x)
    x = self.logits(x)
    return x.transpose(-1, -2), []


class Confidence(nn.Module):
  '''
  0 < confidence <= 1
  '''
  def __init__(self, beta=0.1, epsilon=1e-8):
    super().__init__()
    self.beta = beta
    self.epsilon = epsilon

  def forward(self, y, y_cap, confidence):
    if isinstance(confidence, int):
      confidence = torch.ones_like(y)
    sigma = self.get_sigma(confidence)
    P_YCAP_Y = self.p_ycap_y(y, y_cap, sigma)
    sigma_ycap = self.get_sigma(P_YCAP_Y)
    return self.get_entropy(sigma_ycap)

  def p_ycap_y(self, y, y_cap, sigma):
    diff = -(y-y_cap)**2
    diff_normalized = diff/(2*sigma**2)
    prob = torch.exp(diff_normalized)
    prob_normalized = prob*(1/(2*math.pi*sigma))
    return prob_normalized

  def get_sigma(self, confidence):
    mask = (confidence < self.epsilon).float()
    confidence = (1 - mask) * confidence + torch.ones_like(confidence)*self.epsilon*mask
    sigma = 1/(2*math.pi*confidence)
    return sigma

  ## entropy of a guassian
  def get_entropy(self, sigma):
    return 0.5*(torch.log(2*math.pi*math.e*(sigma**2)))*self.beta

class Repeat(nn.Module):
  def __init__(self, repeat, dim=-1):
    super().__init__()
    self.dim = dim
    self.repeat = repeat
    #self.temp = torch.nn.Parameter(torch.zeros(1))

  def forward(self, x):
    return x.repeat_interleave(self.repeat, self.dim)


class BatchGroup(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models

  Inputs:
    x: list of list of inputs; x[group][batch], len(x) == groups, and len(x[0]) == batches
    labels: uses these labels to give a soft attention on the outputs. labels[batch], len(labels) == batches
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if not isinstance(x, list):
      raise 'x must be a list'
    if not isinstance(x[0], list):
      raise 'x must be a list of lists'
    if labels is not None:
      assert isinstance(labels, list), 'labels must be a list'

    groups = len(x)
    assert self.groups == groups, 'input groups should be the same as defined groups'
    batches = len(x[0])

    x = [torch.cat(x_, dim=0) for x_ in x] # batch
    x = torch.cat(x, dim=1)  # group

    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)

    is_tuple = isinstance(x, tuple)
    if labels is not None:
      assert not is_tuple, 'labels is not None does not work with is_tuple=True'
      labels = torch.cat(labels, dim=0) # batch
      x = [self.index_select_outputs(x, labels).transpose(-1, -2)]
    else: # separate the groups
      if is_tuple:
        channels = [int(x[i].shape[1]/groups) for i in range(len(x))]
        x = [torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]
        #x = list(zip(*[torch.split(x_, channels[i], dim=1) for i, x_ in enumerate(x)]))
        #x = [tuple([x_[:, start*channels[i]:(start+1)*channels[i]] for i, x_ in enumerate(x)]) for start in range(groups)]
      else:
        channels = int(x.shape[1]/groups)
        x = list(torch.split(x, channels, dim=1))
        #x = [x[:, start*channels:(start+1)*channels] for start in range(groups)]

    if is_tuple:
      channels = int(x[0][0].shape[0]/batches)
      x = tuple([[torch.split(x__, channels, dim=0) for x__ in x_] for x_ in x])
      #x = [[tuple([x__[start*channels:(start+1)*channels] for x__ in x_]) for start in range(batches)] for x_ in x]
    else:
      channels = int(x[0].shape[0]/batches)
      x = [list(torch.split(x_, channels, dim=0)) for x_ in x]
      #x = [[x_[start*channels:(start+1)*channels] for start in range(batches)] for x_ in x]
    return x


class Group(nn.Module):
  '''
  Group conv networks to run in parallel
  models: list of instantiated models
  groups: groups of inputs
  dim: if dim=0, use batch a set of inputs along batch dimension (group=1 always)
       elif dim=1, combine the channel dimension (group=num_inputs)

  Inputs:
    x: list of inputs
    labels: uses these labels to give a soft attention on the outputs. Use only with dim=1.
            if labels is None, return a list of outputs
    transpose: if true, model needs a transpose of the input
  '''
  def __init__(self, models, groups=1, dim=1):
    super().__init__()
    if not isinstance(models, list):
      models = [models]
    self.models = nn.ModuleList(models)
    self.groups = groups
    self.dim = dim

  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward(self, x, labels=None, transpose=True, **kwargs):
    if self.dim == 0:
      self.groups = len(x)
    if isinstance(x, list):
      x = torch.cat(x, dim=self.dim) ## concatenate along channels
    if transpose:
      x = x.transpose(-1, -2)
    for model in self.models:
      if kwargs:
        x = model(x, **kwargs)
      else:
        x = model(x)
    if labels is not None:
      x = self.index_select_outputs(x, labels).transpose(-1, -2) ## only for dim=1
      return x
    else:
      channels = int(x.shape[self.dim]/self.groups)
      dim = self.dim % len(x.shape)
      if dim == 2:
        x = [x[:, :, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 1:
        x = [x[:, start*channels:(start+1)*channels] for start in range(self.groups)]
      elif dim == 0:
        x = [x[start*channels:(start+1)*channels] for start in range(self.groups)]
      return x

class EmbLin(nn.Module):
  def __init__(self, num_embeddings, embedding_dim):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.emb = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, x, mode='lin'):
    if mode == 'lin':
      return x.matmul(self.emb.weight)
    elif mode == 'emb':
      return self.emb(x)

class EmbLin2(nn.Module):
  def __init__(self, num_embeddings, embedding_dim, groups=1):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.groups = groups
    self.emb = nn.Embedding(num_embeddings, embedding_dim*self.groups)

  def forward(self, x, mode='lin'):
    if mode == 'lin':
      return x.matmul(self.emb.weight)
    elif mode == 'emb':
      return self.emb(x)


def get_roll_value(x, num_styles):
  if isinstance(x, list):
    shape = x[0].shape[0]
  else:
    shape = x.shape[0]

  roll_value = torch.arange(0, shape)
  if num_styles > 1:
    roll_value = roll_value[roll_value%num_styles!=0]
  else:
    roll_value = roll_value[1:]
  roll_value = roll_value[torch.randint(0, len(roll_value), size=(1,))].item()
  return roll_value

def roll(x, roll_value):
  if isinstance(x, list):
    return [torch.roll(x_, roll_value, dims=0) for x_ in x]
  else:
    return torch.roll(x, roll_value, dims=0)

class Style(nn.Module):
  '''
  input_shape: (B, )
  output_shape: (B, )
  '''
  def __init__(self, num_speakers=1):
    self.style_emb = nn.Embedding(num_embeddings=num_speakers, embedding_dim=256)

  def forward(self, x):
    pass

class Curriculum():
  def __init__(self, start, end, num_iters):
    self.start = start
    self.end = end
    self.num_iters = num_iters
    self.iters = 0
    self.diff = (end-start)/num_iters
    self.value = start

  def step(self, flag=True):
    if flag:
      value_temp = self.value
      if self.iters < self.num_iters:
        self.value += self.diff
        self.iters += 1
        return value_temp
      else:
        return self.end
    else:
      return self.value

class UNet1D_stackedResidual(nn.Module):
  '''
  UNet model for 1D inputs
  (cite: ``https://arxiv.org/pdf/1505.04597.pdf``)

  Arguments
    input_channels (int): input channel size
    output_channels (int): output channel size (or the number of output features to be predicted)
    kernel_size (int, optional): size of the kernel for each convolution (default: ``3``)
    start_channels (int, optional): channel size after the first convolution (default: ``64``)
    max_depth (int, optional): depth of the UNet (default: ``5``).


  Shape
    Input: :math:`(N, C_{in}, L_{in})`
    Output: :math:`(N, C_{out}, L_{out})` where
      .. math::
        assert L_{in} >= 2^{max_depth - 1}
        L_{out} = L_{in}
        C_{out} = output_channels

  Inputs
    x (torch.Tensor): speech signal in form of a 3D Tensor

  Outputs
    x (torch.Tensor): input transformed to a lower frequency
      latent vector

  '''
  def __init__(self, input_channels, output_channels, kernel_size=3, start_channels=64, max_depth=5):
    super(UNet1D_stackedResidual, self).__init__()
    self.conv1 = nn.ModuleList([])
    ## TODO maybe store the indices for the unpooling operation
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=False)
    self.conv2 = nn.ModuleList([])
    self.upconv = nn.ModuleList([])
    self.max_depth = max_depth

    prev_input_channels = input_channels
    i_size_list = [start_channels*(2**i) for i in range(self.max_depth)]
    for i, i_size in enumerate(i_size_list):
      ## create the first set of convolutions as a list
      module_list = nn.ModuleList([])
      module_list.append(nn.Conv1d(prev_input_channels, i_size, kernel_size, padding=1))
      module_list.append(nn.Conv1d(i_size, i_size, kernel_size, padding=1))

      ## appending the set of convolutions to self.conv
      self.conv1.append(module_list)

      prev_input_channels = i_size

    prev_input_channels = i_size_list[-1]
    i_size_list_rev = reversed(i_size_list[:-1])
    for i, i_size in enumerate(i_size_list_rev):
      ## adding up-convs
      self.upconv.append(nn.ConvTranspose1d(prev_input_channels, i_size, kernel_size=2, stride=2))

      ## create the first set of convolutions as a list
      module_list = nn.ModuleList([])
      module_list.append(nn.Conv1d(prev_input_channels, i_size, kernel_size, padding=1))
      module_list.append(nn.Conv1d(i_size, i_size, kernel_size, padding=1))

      ## appending the set of convolutions to self.conv
      self.conv2.append(module_list)

      prev_input_channels = i_size

    ## fully connected conv layer
    self.final_conv = nn.Conv1d(i_size, output_channels, kernel_size=1)

  def forward(self, x):
    assert x.shape[-1]/(2**(self.max_depth - 1)) >= 1, 'Input size is {}. It must be >= {}'.format(x.shape[-1], 2**(self.max_depth - 1))

    residuals = []
    for i, conv1 in enumerate(self.conv1):
      x = nn.Sequential(*conv1)(x)
      if i < self.max_depth - 1:
        residuals.append(x)
        x = self.pool(x)

    num_residuals = self.max_depth - 1
    for i, (conv2, upconv) in enumerate(zip(self.conv2, self.upconv)):
      x = upconv(x)
      x = torch.cat([residuals[num_residuals - i - 1], x], dim=1) ## concat residuals
      x = nn.Sequential(*conv2)(x)

    x = self.final_conv(x)
    return x

'''
Layers for Gesture LM+Segementation+Decoder
'''

class TransformerLM(nn.Module):
  def __init__(self, out_feats=96, ninp=256, nhead=8, nhid=256, nlayers=6, dropout=0, mlm=True, clm=False):
    super().__init__()
    self.mlm = mlm
    self.clm = clm
    assert self.mlm or self.clm, 'atleast one kind of LM, either causal or masked LM'

    ## Positional Encoding
    self.pe = PositionalEncoding(ninp, dropout=dropout)

    ## Encoder
    self.lin_in = nn.Linear(out_feats, ninp)

    ## Transformer Encoder
    transformer_encoder_layer = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout=dropout)
    self.encoder = nn.TransformerEncoder(transformer_encoder_layer, nlayers)

    ## Decoder
    self.lin_out = nn.Linear(ninp, out_feats)

  def get_src_key_padding_mask(self, mask_percent, x, chunk_len):
    T, B, device = x.shape[0], x.shape[1], x.device
    num_chunks = T//chunk_len
    assert num_chunks * chunk_len == T, 'Time dimension should be divisible by chunk_len'
    num_masked_chunks = int(mask_percent*num_chunks)

    mask = torch.zeros(B, num_chunks, chunk_len, 1)
    for b in range(B):
      idx = torch.from_numpy(np.random.choice(range(num_chunks), size=(num_masked_chunks,), replace=False))
      mask[b, idx] = 1
    mask = mask.view(B, T).to(device)

    return mask.bool()

  def get_clm_mask(self, x):
    mask = torch.triu(torch.ones(x.shape[0], x.shape[0])).transpose(1, 0).to(x.device)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def forward(self, x, mask_percent=0.5, chunk_len=4):
    if self.mlm:
      src_key_padding_mask = self.get_src_key_padding_mask(mask_percent, x, chunk_len)
      mask_float = src_key_padding_mask.float().transpose(1, 0).unsqueeze(-1)
      x = x*(1-mask_float) + -15*mask_float ## mask for x, assign -15 to masked points
    else:
      src_key_padding_mask = None

    if self.clm:
      clm_mask = self.get_clm_mask(x)
    else:
      clm_mask = None

    x = torch.relu(self.lin_in(x))
    x = self.pe(x)
    x = self.encoder(x, src_key_padding_mask=src_key_padding_mask, mask=clm_mask)
    x = self.lin_out(x)

    src_key_padding_mask = src_key_padding_mask.float() if src_key_padding_mask is not None else None
    return x, src_key_padding_mask

'''
Contrastive Learning
'''
class MoCo(nn.Module):
  '''
  in_channels: feature size
  time_steps: temporal dimension
  K: queue size
  m: momentum for key encoder updates
  T: temperature
  bn_splits: number of splits for the batch norm
  symmetric: flag for symmetric loss
  '''
  def __init__(self, enc, in_channels=128, time_steps=64, K=4096, m=0.99, T=0.1, bn_splits=1, symmetric=True, **kwargs):
    super().__init__()

    self.K = K
    self.m = m
    self.T = T
    self.symmetric = symmetric

    # create the encoders
    self.encoder_q = enc
    self.encoder_k = copy.deepcopy(enc)

    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
      #param_k.data.copy_(param_q.data)  # initialize
      param_k.requires_grad = False  # not update by gradient

    # create the queue
    self.register_buffer("queue", torch.randn(in_channels, K, time_steps))
    self.queue = nn.functional.normalize(self.queue, dim=0)

    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

  @torch.no_grad()
  def _momentum_update_key_encoder(self):
    """
    Momentum update of the key encoder
    """
    for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
      param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

  @torch.no_grad()
  def _dequeue_and_enqueue(self, keys):
    batch_size = keys.shape[0]

    ptr = int(self.queue_ptr)
    #assert self.K % batch_size == 0  # for simplicity

    # replace the keys at ptr (dequeue and enqueue)
    if ptr + batch_size > self.queue.shape[1]:
      chunk1 = self.queue.shape[1] - ptr
      #chunk2 = (ptr + batch_size) - self.queue.shape[1]
      self.queue[:, ptr:ptr+chunk1] = keys.transpose(0,1)[:, :chunk1]
      self.queue[:, :batch_size-chunk1] = keys.transpose(0,1)[:, chunk1:]
    else:
      self.queue[:, ptr:ptr + batch_size] = keys.transpose(0,1)  # transpose
    ptr = (ptr + batch_size) % self.K  # move pointer

    self.queue_ptr[0] = ptr

  @torch.no_grad()
  def _batch_shuffle_single_gpu(self, x):
    """
    Batch shuffle, for making use of BatchNorm.
    """
    # random shuffle index
    idx_shuffle = torch.randperm(x.shape[0]).cuda()

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    return x[idx_shuffle], idx_unshuffle

  @torch.no_grad()
  def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
    """
    Undo batch shuffle.
    """
    return x[idx_unshuffle]

  def contrastive_loss(self, im_q, im_k):
    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1xT
    l_pos = torch.einsum('nct,nct->nt', [q, k]).unsqueeze(1)
    # negative logits: NxKxT
    l_neg = torch.einsum('nct,ckt->nkt', [q, self.queue.clone().detach()])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= self.T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda()

    loss = nn.CrossEntropyLoss().cuda()(logits, labels)


    return loss, q, k

  def forward(self, im1, im2):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        loss
    """

    # update the key encoder
    with torch.no_grad():  # no gradient to keys
      self._momentum_update_key_encoder()

    # compute loss
    if self.symmetric:  # asymmetric loss
      loss_12, q1, k2 = self.contrastive_loss(im1, im2)
      loss_21, q2, k1 = self.contrastive_loss(im2, im1)
      loss = loss_12 + loss_21
      k = torch.cat([k1, k2], dim=0)
      q = q1
    else:  # asymmetric loss
      loss, q, k = self.contrastive_loss(im1, im2)

    self._dequeue_and_enqueue(k)


    return q, [loss]

  def sample(self, im1):
    with torch.no_grad():
      q = self.encoder_q(im1)
      q = nn.functional.normalize(q, dim=1)  # already normalized

    return q, [torch.zeros(1)[0].to(im1.device)]


class PatchNCELoss(nn.Module):
  '''
  PatchNCE adapted from https://github.com/taesungp/contrastive-unpaired-translation/blob/afdc8fb027/models/patchnce.py
  '''
  def __init__(self, nce_T=0.07, negs_from_minibatch=False):
    super().__init__()
    self.nce_T = nce_T
    self.negs_from_minibatch = negs_from_minibatch
    self.ce = nn.CrossEntropyLoss()

  def forward(self, feat_q, feat_k):
    ## Positive Logits
    l_pos = torch.einsum('bctw,bctw->bw', feat_q, feat_k).view(-1, 1)

    ## Negative Logits
    if self.negs_from_minibatch:
      l_neg = torch.einsum('wct,xct->wx', feat_q.reshape(-1, feat_q.shape[1], feat_q.shape[2]), feat_k.reshape(-1, feat_k.shape[1], feat_k.shape[2]))[None, :, :]
    else:
      l_neg = torch.einsum('bctw,bctx->bwx', feat_q, feat_k)
    diagonal = torch.eye(l_neg.shape[-1], device=l_neg.device).bool()[None, :, :]
    l_neg.masked_fill_(diagonal, -10.0)
    l_neg = l_neg.flatten(0, 1)

    ## Get Loss
    logits = torch.cat([l_pos, l_neg], dim=-1)/self.nce_T
    labels = torch.zeros(logits.shape[0]).long().to(l_neg.device)
    loss = self.ce(logits, labels)
    return logits, loss

class TemporalPatches(nn.Module):
  '''
  Get Temporal Patches from a 1d sequence
  input: list of feats of the shape B, C, T
  output: list of feats of the shape B, C, 1, W (window_size)
          list of patch_ids to help extract the exact same patches for other feats

  '''
  def __init__(self, num_patches=5, window_size=8, window_hop=5):
    super().__init__()
    self.num_patches = num_patches
    self.window_size = window_size
    self.window_hop = window_hop

    self.mlp_init = False

  def create_mlp(self, feats):
    #m4(m3(m2(m1(feat_qs[0].permute(0, 3, 1, 2).flatten(0, 1)))).squeeze(-1)).shape
    for mlp_id, feat in enumerate(feats):
      C = feat.shape[1]
      T = self.window_size
      num_layers = int(np.log2(T))
      mlp = nn.ModuleList([])

      for l in range(num_layers):
        mlp.append(ConvNormRelu(C, C, downsample=True))
      if num_layers == 0:
        mlp.append(ConvNormRelu(C, C, kernel_size=1, stride=1)) ##
      mlp.append(Transpose([2, 1]))
      mlp.append(nn.Linear(C, C))
      mlp.append(Transpose([2, 1]))

      setattr(self, 'mlp_{}'.format(mlp_id), nn.Sequential(*mlp))
      getattr(self, 'mlp_{}'.format(mlp_id)).to(feat.device)
    self.mlp_init = True

  def filter_feats(self, feats):
    feats_ = []
    for feat in feats:
      B, C, T = feat.shape[0], feat.shape[1], feat.shape[2]
      if T - self.window_size > self.window_hop:
        feats_.append(feat)
    return feats_

  def forward(self, feats, patch_ids=None):
    return_feats = []
    return_ids = []

    feats = self.filter_feats(feats)

    if not self.mlp_init:
      self.create_mlp(feats)

    for idx, feat in enumerate(feats):
      B, C, T = feat.shape[0], feat.shape[1], feat.shape[2]
      starts = torch.arange(0, T-self.window_size, self.window_hop)
      ends = starts + self.window_size

      ## get random patches
      if patch_ids is None:
        patch_id = torch.randperm(starts.shape[0])[:self.num_patches]
      else:
        patch_id = patch_ids[idx]

      feat_patches = torch.zeros(feat.shape[0], feat.shape[1], self.window_size, patch_id.shape[-1]).to(feat.device) # B x C x T (window) x W (num_patches)
      for w, (s, e) in enumerate(zip(starts[patch_id], ends[patch_id])):
        feat_patches[:, :, :, w] = feat[:, :, s:e]

      feat_patches = getattr(self, 'mlp_{}'.format(idx))(feat_patches.permute(0, 3, 1, 2).flatten(0, 1)) ## BxW, C, T
      feat_patches = feat_patches.view(B, -1, feat_patches.shape[-2], feat_patches.shape[-1]).permute(0, 2, 3, 1) # B, C, T, W

      return_feats.append(feat_patches)
      return_ids.append(patch_id)

    return return_feats, return_ids



class MoCo_DTW(MoCo):
  '''
        TopK clustering based on DTW
  
  '''
  def __init__(self, enc, in_channels=128, time_steps=64, DTW = True, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__(enc, in_channels, time_steps, K, m, T, bn_splits, symmetric, **kwargs)
    self.softdtw_DBS = SoftDTW_DBS(use_cuda = True)
    self.softdtw = SoftDTW(use_cuda = True)
    self.clusterdict = dict()
    self.margin = margin
    self.select_idx = None
    self.DTW_true = DTW
    #self.SoftDTW = SoftDTW(use_cuda=True, gamma=0.1)


    #online-kmeans inits
    self.cluster_centers = None
    self.w_star = None
    self.r = 1
    self.q = 0
    self.f = None

  def cluster_DTW(self, q, k, labels, iter, top_k = 5): #anchor: 1 X T X E

    with torch.no_grad():
        if self.select_idx == None:
            sample_indices = torch.nonzero(labels)
            select_idx = torch.randperm(len(sample_indices))[:1]
            anchor = q[select_idx]
        else:
            select_idx = self.select_idx[0]
            anchor = q[select_idx]

        labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
        sample_indices = torch.nonzero(labels) #update sample_indices to be used later
        remain_size = sample_indices.shape[0]



        if self.DTW_true:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores, path = self.softdtw(anchor,q[sample_indices[:,0]].clone()) # calculate sim scores via dtw
            # for i in range(sample_indices.shape[0]):
            #     try:
            #         distance, path = fastdtw(anchor.squeeze(), q[sample_indices[i,0]].clone())
            #     except Exception as e:
            #         print(e)
            #         pdb.set_trace()
        if self.DTW_true == False:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores = torch.einsum('nct,nct->n', [anchor, q[sample_indices[:,0]].clone()])
        scores_size = sim_scores.shape[0]

        if (top_k + 1) >= scores_size:
            best_values, top_indices = torch.topk(sim_scores, scores_size)

            self.clusterdict[iter] = sample_indices[top_indices]
            self.clusterdict[iter] = dict()

            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx])
            self.clusterdict[iter]["vals"] = best_values.cuda()
            labels[sample_indices[top_indices]] = 0

        else:
            best_values, top_indices = torch.topk(sim_scores, top_k) # TODO: calculate mean of sim scores, kind of wasting medium values of DTW
            worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
            self.clusterdict[iter] = dict()
            #try pdb post mortem
            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx]) #last elem is anchor seq.
            self.clusterdict[iter]["vals"] = best_values.cuda()
            self.select_idx = sample_indices[bot_indices]
            labels[sample_indices[top_indices]] = 0
            labels[sample_indices[bot_indices]] = 0

    return labels


  def lifted_embedding_loss(self, im_q, im_k):
    # https://arxiv.org/pdf/1703.07737.pdf
    '''
      losses for TopK clustering, the im_k is not used at all, only q is used
    '''

    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)


    batch_size = q.shape[0]
    not_all_assigned = True
    labels = torch.ones(batch_size)
    iter = 0
    self.select_idx = None
    while not_all_assigned:
        iter += 1

        # labels = self.cluster_DTW(q.transpose(1,2),  k.transpose(1,2), labels, iter, top_k = 5)
        labels = self.cluster_DTW(q.transpose(1,2).clone().detach(),  q.transpose(1,2).clone().detach(), labels, iter, top_k = batch_size//8)

        if torch.sum(labels) == 0:
            not_all_assigned = False


    loss = torch.zeros(1).cuda()


    for cluster, v in self.clusterdict.items():
        try:
            pos_labels = torch.zeros(batch_size, dtype=torch.bool)
            pos_labels[v["idx"][:-1]] = True
            neg_labels = ~(pos_labels.clone())

            l_pos = torch.einsum('nct,nct->nt', [q[pos_labels], q[pos_labels].detach()]).unsqueeze(1)
            # negative logits: NxKxT
            l_neg = torch.einsum('nct, kct->nkt', [q[pos_labels],q[neg_labels]])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)
            #stability
            logits = logits - torch.max(logits)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda() # TODO: mean???

            loss += nn.CrossEntropyLoss().cuda()(logits, labels)
        except Exception:
            pdb.post_mortem()

    loss = loss/iter
    self.clusterdict = dict()

    return loss.squeeze(), q, k




  def forward(self, im1, im2):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        loss
    """

    # update the key encoder
    with torch.no_grad():  # no gradient to keys
      self._momentum_update_key_encoder()

    # compute loss
    if self.symmetric:  # asymmetric loss
      loss_12, q1, k2 = self.lifted_embedding_loss(im1, im2)
      loss_21, q2, k1 = self.lifted_embedding_loss(im2, im1)
      loss = loss_12 + loss_21
      k = torch.cat([k1, k2], dim=0)
      q = q1
    else:  # asymmetric loss
      loss, q, k = self.lifted_embedding_loss(im1, im2)

    self._dequeue_and_enqueue(k)

    return q, [loss]


class KMeansContr(MoCo):
  '''
    Class for online kmeans
    - online_kmeans lead to memory issues
    - online_kmeans_practical works
    all adapted from  https://arxiv.org/pdf/1412.5721.pdf


  '''

  def __init__(self, enc, in_channels=128, time_steps=64, DTW = True, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__(enc, in_channels, time_steps, K, m, T, bn_splits, symmetric, **kwargs)
    self.softdtw_DBS = SoftDTW_DBS(use_cuda = True)
    self.softdtw = SoftDTW(use_cuda = True)
    self.clusterdict = dict()
    self.margin = margin
    self.select_idx = None
    self.DTW_true = DTW
    self.select_idx_list = []
    #self.SoftDTW = SoftDTW(use_cuda=True, gamma=0.1)


    #online-kmeans inits
    self.kmeans_clusterdict = dict()
    self.cluster_centers = []
    self.cluster_centers_indices = []
    self.w_star = None
    self.r = 1
    self.q = 0
    self.f = None
    self.n = 0
    self.pdist = nn.PairwiseDistance(p=2)



  def cluster_DTW(self, q, k, labels, iter, top_k = 5): #anchor: 1 X T X E

    with torch.no_grad():
        if self.select_idx == None:
            sample_indices = torch.nonzero(labels)
            select_idx = torch.randperm(len(sample_indices))[:1]
            anchor = q[select_idx]
        else:
            select_idx = self.select_idx[0]
            anchor = q[select_idx]

        labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
        sample_indices = torch.nonzero(labels) #update sample_indices to be used later
        remain_size = sample_indices.shape[0]
        self.cluster_centers.append(anchor)
        self.cluster_centers_indices.append(select_idx)
        self.kmeans_clusterdict[iter] = anchor


        if self.DTW_true:
            # anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            #sim_scores, path = self.softdtw(anchor,q[sample_indices[:,0]].clone()) # calculate sim scores via dtw
            for i in range(sample_indices.shape[0]):
                try:
                    distance, path = fastdtw(anchor.squeeze(), q[sample_indices[i,0]].clone())
                except Exception as e:
                    print(e)
                    pdb.set_trace()
        if self.DTW_true == False:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores = torch.einsum('nct,nct->n', [anchor, q[sample_indices[:,0]].clone()])
        scores_size = sim_scores.shape[0]

        if (top_k + 1) >= scores_size:
            best_values, top_indices = torch.topk(sim_scores, scores_size)

            self.clusterdict[iter] = sample_indices[top_indices]
            self.clusterdict[iter] = dict()

            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx])
            self.clusterdict[iter]["vals"] = best_values.cuda()
            labels[sample_indices[top_indices]] = 0

        else:
            best_values, top_indices = torch.topk(sim_scores, top_k) # TODO: calculate mean of sim scores, kind of wasting medium values of DTW
            worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
            self.clusterdict[iter] = dict()
            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(), select_idx]) #last elem is anchor seq.
            self.clusterdict[iter]["vals"] = best_values.cuda()
            self.select_idx = sample_indices[bot_indices]
            labels[sample_indices[top_indices]] = 0
            labels[sample_indices[bot_indices]] = 0

    return labels


  def online_kmeans(self, V , k, labels):
      # https://arxiv.org/pdf/1412.5721.pdf
      # standard kmeans -> led to memory issues

    for idx, v in enumerate(V):
      self.n += 1
      dist = torch.sum(torch.norm(v.T - self.cluster_centers, dim =2)**2, dim = 1)
      D = torch.clamp((torch.min(dist))/self.f, max = 1)

      if torch.rand(1) <= D:
        self.cluster_centers = torch.cat((self.cluster_centers, v.T.unsqueeze(0)))
        self.q += 1
        print("NEW CLUSTER SIZE : " + str(self.cluster_centers.shape[0]))

      if self.q >= 3*k*(1+math.log(k+1)):

        self.r += 1
        self.q = 0
        self.f = 2* self.f
        print("making threshold bigger!")
        print(self.f)

      dist = torch.sum(torch.norm(v.T - self.cluster_centers, dim =2)**2, dim = 1)
      labels[idx] = torch.argmin(dist)


    return labels


  def online_kmeans_practical(self, V, k, labels):
      # https://arxiv.org/pdf/1412.5721.pdf
      # practical implementation


    for idx, v in enumerate(V):
      self.n += 1
      dist = torch.sum(torch.norm(v.T - self.cluster_centers, dim =2)**2, dim = 1)
      D = torch.clamp((torch.min(dist))/self.f, max = 1)

      if torch.rand(1) <= D:
        self.cluster_centers = torch.cat((self.cluster_centers, v.T.unsqueeze(0)))
        self.q += 1
        print("NEW CLUSTER SIZE : " + str(self.cluster_centers.shape[0]))

      if self.q >= k:

        self.r += 1
        self.q = 0
        self.f = 10* self.f
        print("making threshold bigger!")
        print(self.f)

      dist = torch.sum(torch.norm(v.T - self.cluster_centers, dim =2)**2, dim = 1)
      labels[idx] = torch.argmin(dist)


    return labels



  def lifted_embedding_loss_cluster(self, im_q, im_k, batch_iter, OG = False):
      # https://arxiv.org/pdf/1703.07737.pdf
    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)



      batch_size = q.shape[0]
      V_indices = torch.ones(batch_size)
      V = q
      k = batch_size//4
      kmean_labels = torch.zeros(batch_size)

      #initialization of first k cluster centroids using DTW -> realized i dont need this:
      # https://arxiv.org/pdf/1412.5721.pdf
      if batch_iter == 0:
        not_all_assigned = True
        labels = torch.ones(batch_size)
        iter = 0
        self.select_idx = None
        while not_all_assigned:
            iter += 1

            # labels = self.cluster_DTW(q.transpose(1,2),  k.transpose(1,2), labels, iter, top_k = 5)
            labels = self.cluster_DTW(q.transpose(1,2).clone().detach(),  q.transpose(1,2).clone().detach(), labels, iter, top_k = k)

            if torch.sum(labels) == 0:
                not_all_assigned = False

        self.cluster_centers_indices = torch.Tensor(self.cluster_centers_indices).long()
        self.cluster_centers = torch.stack(self.cluster_centers).squeeze()
        V_indices[self.cluster_centers_indices] = 0
        V = V[V_indices.bool()]
        kmean_labels[self.cluster_centers_indices] = torch.arange(self.cluster_centers_indices.shape[0]).float() #self.cluster_centers_indices - kinda hacky


        curr_min = []
        for idx, val in enumerate(q[self.cluster_centers_indices]):
          curr_centers = torch.cat([self.cluster_centers[:idx], self.cluster_centers[idx+1:]])
          dist = torch.sum(torch.norm(val.T - curr_centers, dim = 2)**2, dim = 1)
          curr_min.append(dist/2)
        curr_min = torch.stack(curr_min)
        curr_min = curr_min.reshape(-1)

        # self.w_star = (self.pdist(q[self.cluster_centers_indices])**2)/2
        # self.w_star = torch.sum(self.w_star, axis = 1)

        if OG == True:

          self.w_star = torch.min(curr_min) #should return number of clusters k
          self.r = 1
          self.q = 0
          self.f = self.w_star/(k)
          self.n = self.cluster_centers.shape[0] + 1

        elif OG == False:
          values, indices = torch.topk(curr_min, 10, largest = False )
          self.w_star = torch.sum(values) #should return number of clusters k
          self.r = 1
          self.q = 0
          self.f = self.w_star/(k)
          self.n = self.cluster_centers.shape[0] + 1

    # kmeans for rest
    if OG == True:
      kmean_labels = self.online_kmeans(V, k, kmean_labels)

    elif OG == False:

      kmean_labels = self.online_kmeans_practical(V, k, kmean_labels)



    loss = torch.zeros(1).cuda()
    iter = 0

        #we have first cluster centers here with DTW


    for idx, cluster in enumerate(self.cluster_centers):
        iter += 1

        pos_labels = torch.where(kmean_labels == idx, torch.tensor(1) , torch.tensor(0))
        neg_labels = ~(pos_labels.clone())

        l_pos = torch.einsum('nct,nct->nt', [q[pos_labels], q[pos_labels].detach()]).unsqueeze(1)

        # negative logits: NxKxT
        l_neg = torch.einsum('nct, kct->nkt', [q[pos_labels],q[neg_labels]])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        #stability
        logits = logits - torch.max(logits)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda() # TODO: mean???

        loss += nn.CrossEntropyLoss().cuda()(logits, labels)

    loss = loss/iter

    return loss.squeeze(), q, k


  def forward(self, im1, im2, batch_iter):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        loss
    """

    # update the key encoder
    with torch.no_grad():  # no gradient to keys
      self._momentum_update_key_encoder()

    # compute loss
    if self.symmetric:  # asymmetric loss
      loss_12, q1, k2 = self.lifted_embedding_loss_cluster(im1, im2, batch_iter)
      loss_21, q2, k1 = self.lifted_embedding_loss_cluster(im2, im1, batch_iter)
      loss = loss_12 + loss_21
      k = torch.cat([k1, k2], dim=0)
      q = q1
    else:  # asymmetric loss
      loss, q, k = self.lifted_embedding_loss_cluster(im1, im2, batch_iter)

    #self._dequeue_and_enqueue(k)

    return q, [loss]

class BeamSearch():
  def __init__(self, beam_width=10, beams=None, beam_scores=None):
    self.beam_width = beam_width
    self.beams = beams # (beam_width, output_length)
    self.beam_scores = beam_scores # (beam_width,)
    
  def __call__(self, x):
    '''
    x: (beam_width, num_embeddings, T)
    '''
    num_embeddings = x.shape[1]
    neg_log_prob = -F.log_softmax(x[..., -1], dim=1) # (beam_width, num_embeddings)
    if self.beam_scores is not None:
      neg_log_prob += self.beam_scores.unsqueeze(-1)
    neg_log_prob_flattened = neg_log_prob.flatten()
    val, indices = torch.topk(neg_log_prob_flattened, self.beam_width, largest=False)
    
    ## Update Beam scores
    self.beam_scores = val
    
    ## Update Beams
    rows = (indices // num_embeddings).long() # beam_width
    cols = (indices - rows*num_embeddings).long().unsqueeze(-1) # beam_width, 1
    if self.beams is not None:
      self.beams = self.beams[rows]
      self.beams = torch.cat([self.beams, cols], dim=-1)
    else:
      self.beams = cols
    
    return self.beams
  
  def get_best_beam(self):
    return self.beams[0], self.beam_scores[0]
  
class TransformerMod(nn.Transformer):
  r"""Mod of a transformer model with added functionality for decoding like Greedy Decoding, Beam Search, etc. 
  Rest of the module is exactly the same.

  User is able to modify the attributes as needed. The architecture
  is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
  Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
  Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
  Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
  model with corresponding parameters.

  Args:
      d_model: the number of expected features in the encoder/decoder inputs (default=512).
      nhead: the number of heads in the multiheadattention models (default=8).
      num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
      num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
      dim_feedforward: the dimension of the feedforward network model (default=2048).
      dropout: the dropout value (default=0.1).
      activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
      custom_encoder: custom encoder (default=None).
      custom_decoder: custom decoder (default=None).

  Examples::
      >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
      >>> src = torch.rand((10, 32, 512))
      >>> tgt = torch.rand((20, 32, 512))
      >>> out = transformer_model(src, tgt)

  Note: A full example to apply nn.Transformer module for the word language model is available in
  https://github.com/pytorch/examples/tree/master/word_language_model
  """

  def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
               num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
               activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
    super().__init__(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                     num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                     dropout=dropout, activation=activation, custom_encoder=custom_encoder,
                     custom_decoder=custom_decoder)
    pass
  
