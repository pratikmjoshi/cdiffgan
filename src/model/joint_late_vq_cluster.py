import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

JointLateClusterSoftVQ_D = Speech2Gesture_D

class JointLateClusterSoftVQ_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
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
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
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
    self.cluster = cluster
    self.vq = VQLayer(num_embeddings=self.num_clusters,
                      num_features=256)

    self.thresh = Curriculum(0, 1, 1000)
    
  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x
  
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    # Late Fusion with Joint
    if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
    #if True:
      x = self.pose_encoder(y, time_steps)
    else:
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
    ## Classify clusters using audio/text
    _, labels_score = self.vq(x.detach())
    #x, labels_score = self.vq(x)
    #labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score.detach(), dim=-1) ## TODO, might be okay to use label_score.detach()
    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    return x, internal_losses

