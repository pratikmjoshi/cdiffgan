import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

from functools import partial

JointLateClusterNoise2Pose_D = Speech2Gesture_D
JointLateClusterNoise2Pose2_D = Speech2Gesture_D

'''
Noise -> Pose
'''

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=False, p=0):
    super().__init__()

    self.upsample = upsample
    self.conv1 = ConvNormRelu(in_channels, in_channels,
                              type='1d', leaky=True, downsample=False,
                              p=p)
    self.upconv = nn.Upsample(scale_factor=2)
    
    self.conv2 = ConvNormRelu(in_channels, out_channels,
                              type='1d', leaky=True, downsample=False,
                              p=p)
    self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

  def forward(self, input):
    if self.upsample:
      input = self.upconv(input)
    out = self.conv1(input)
    out = self.conv2(out)
    skip = self.skip(input)
    out = (out + skip) / math.sqrt(2)
    
    return out

    
class JointLateClusterNoise2Pose_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256,
                 in_noise_dim=100, out_noise_dim=256, **kwargs):
      super().__init__()

      self.time_steps = time_steps
      self.scale = 3
      self.in_noise_dim = in_noise_dim
      self.out_noise_dim = out_noise_dim
      
      #self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
      self.noise_dec = nn.Sequential(nn.Linear(self.in_noise_dim, self.in_noise_dim),
                                     nn.BatchNorm1d(self.in_noise_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(self.in_noise_dim, time_steps*self.out_noise_dim//(2**self.scale)))

      self.upconv = nn.Sequential(*nn.ModuleList([ResBlock(self.out_noise_dim,
                                                           self.out_noise_dim, upsample=True)
                                                  for _ in range(self.scale)]))
      ## decoder with noise and language concatenated
      self.decoder = nn.ModuleList([])
      for i in range(4):
        self.decoder.append(ConvNormRelu(in_channels, in_channels,
                                         type='1d', leaky=True, downsample=False,
                                         p=p))
      self.decoder = nn.Sequential(*self.decoder)
      self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)
      

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

    def forward(self, x, y, time_steps=None, noise=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs

      ## decode
      if noise is None: 
        if kwargs['sample_flag']:
          n_time_steps = y.shape[1] // self.time_steps 
          noise = torch.rand(n_time_steps, self.in_noise_dim)
        else:
          noise = torch.rand(y.shape[0], self.in_noise_dim)

      noise = noise.to(y.device)
      ## (B, self.out_noise_dim, self.time_steps)          
      noise = self.noise_dec(noise).view(-1, self.out_noise_dim, self.time_steps//(2**self.scale))
      if kwargs['sample_flag']:
        noise = noise.permute(1, 0, 2).reshape(noise.shape[1], -1).unsqueeze(0) ## (1, out_noise_dim, time_steps*n_time_steps)


      x = self.upconv(noise)
      x = self.decoder(x)
      x = self.logits(x).permute(0, 2, 1)

      return x, internal_losses


class JointLateClusterNoise2Pose2_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256,
                 in_noise_dim=100, out_noise_dim=256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters

      self.time_steps = time_steps
      self.in_noise_dim = in_noise_dim
      self.out_noise_dim = out_noise_dim
      
      #self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
      self.noise_dec = nn.Sequential(nn.Linear(self.in_noise_dim, self.in_noise_dim),
                                     nn.BatchNorm1d(self.in_noise_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(self.in_noise_dim, time_steps*self.out_noise_dim))

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()

      ## decoder with noise and language concatenated
      self.decoder = nn.ModuleList([])
      for i in range(4):
        self.decoder.append(ConvNormRelu(in_channels, in_channels,
                                         type='1d', leaky=True, downsample=False,
                                         p=p, groups=self.num_clusters))
      self.decoder = nn.Sequential(*self.decoder)

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

    def forward(self, x, y, time_steps=None, noise=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs

      ## decode
      if noise is None: 
        if kwargs['sample_flag']:
          n_time_steps = y.shape[1] // self.time_steps 
          noise = torch.rand(n_time_steps, self.in_noise_dim)
        else:
          noise = torch.rand(y.shape[0], self.in_noise_dim)

      noise = noise.to(y.device)
      ## (B, self.out_noise_dim, self.time_steps)          
      noise = self.noise_dec(noise).view(-1, self.out_noise_dim, self.time_steps)
      if kwargs['sample_flag']:
        noise = noise.permute(1, 0, 2).reshape(noise.shape[1], -1).unsqueeze(0) ## (1, out_noise_dim, time_steps*n_time_steps)

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(noise).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      #repeat inputs before decoder
      x = torch.cat([noise]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)

      return x, internal_losses


class JointLateClusterNoise2Pose3_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256,
                 in_noise_dim=512, out_noise_dim=256, **kwargs):
      super().__init__()

      self.time_steps = time_steps
      self.scale = 4
      self.in_noise_dim = in_noise_dim
      self.out_noise_dim = out_noise_dim
      
      #self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
      self.noise_dec = nn.Sequential(nn.Linear(self.in_noise_dim, self.in_noise_dim),
                                     nn.BatchNorm1d(self.in_noise_dim),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(self.in_noise_dim, time_steps*self.out_noise_dim//(2**self.scale)))

      # self.upconv = nn.Sequential(*nn.ModuleList([ResBlock(self.out_noise_dim,
      #                                                      self.out_noise_dim, upsample=True)
      #                                             for _ in range(self.scale)]))
      ## decoder with noise and language concatenated
      self.decoder = nn.ModuleList([])
      scale_count = 0
      for i in range(4):
        self.decoder.append(ConvNormRelu(in_channels, in_channels,
                                         type='1d', leaky=True, downsample=False,
                                         p=p))
        if scale_count < self.scale:
          #self.decoder.append(nn.Upsample(scale_factor=2))
          upconv = nn.Sequential(nn.ConvTranspose1d(in_channels, in_channels,
                                                    kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm1d(in_channels),
                                 nn.LeakyReLU(0.2))
          self.decoder.append(upconv)
          scale_count += 1
      self.decoder = nn.Sequential(*self.decoder)
      self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)
      

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

    def forward(self, x, y, time_steps=None, noise=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs

      ## decode
      if noise is None: 
        if kwargs['sample_flag']:
          n_time_steps = y.shape[1] // self.time_steps 
          noise = torch.rand(n_time_steps, self.in_noise_dim)
        else:
          noise = torch.rand(y.shape[0], self.in_noise_dim)
          
      noise = noise.to(y.device)
      ## (B, self.out_noise_dim, self.time_steps)          
      noise = self.noise_dec(noise).view(-1, self.out_noise_dim, self.time_steps//(2**self.scale))
      if kwargs['sample_flag']:
        noise = noise.permute(1, 0, 2).reshape(noise.shape[1], -1).unsqueeze(0) ## (1, out_noise_dim, time_steps*n_time_steps)

      #x = self.upconv(noise)
      x = self.decoder(noise)
      x = self.logits(x).permute(0, 2, 1)

      return x, internal_losses
