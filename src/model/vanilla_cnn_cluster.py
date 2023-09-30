import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

VanillaCNNLateCluster_D = Speech2Gesture_D

class VanillaCNNLateCluster_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
  
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p, groups=self.num_clusters)
                                                 for i in range(4)]))
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    
  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T)
    '''
    labels_shape = list(labels.shape) + [self.num_clusters] 
    labels = labels.view(-1)
    idxs = torch.index_select(self.eye, 0, labels).view(*labels_shape) ## (B, T, num_clusters)
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
    x = (x * idxs.unsqueeze(-1)).sum(dim=-2)
    return x

  
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    # Late Fusion
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        x[i] = self.text_encoder(x[i], time_steps)
      if modality.split('/')[0] == 'audio':
        if x[i].dim() == 3:
          x[i] = x[i].unsqueeze(dim=1)
        x[i] = self.audio_encoder(x[i], time_steps)

    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]
    if len(x) >= 2:
      x = torch.cat(tuple(x),  dim=1)
      x = self.concat_encoder(x)
    else:
      x = torch.cat(tuple(x),  dim=1)
  
    ## Classify clusters
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    _, labels_cap = labels_score.max(dim=-1)
    labels_cap = labels_cap.detach()
    if self.training:
      labels_cap = labels ## use ground truth while training

    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap)
    x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)
    
    return x, internal_losses

