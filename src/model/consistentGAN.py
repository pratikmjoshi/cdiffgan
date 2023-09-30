import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .layersUtils import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

from functools import partial
from copy import deepcopy

class ConsistentGAN_G(nn.Module):
  '''
  Fewshot learning with ConsistentGAN

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''

  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_clusters=8, cluster=None, pretrained_trainers=None, **kwargs):
    super().__init__()
    G = pretrained_trainers.model.G
    self.G_ema = deepcopy(G) ## Final Model 

    ## Feature Extractors
    self.layers = self.get_layers()
    self.G_src = FeatureExtractor(deepcopy(G), self.layers) ## Distance consistency Loss
    self.G_tgt = FeatureExtractor(G, self.layers) ## Distance consistency Loss
    self.G = self.G_tgt.model  ## Train Generator
    
    self.in_noise_dim = self.G.in_noise_dim
    self.time_steps = self.G.time_steps

    self.sfm = nn.Softmax(dim=1)
    self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    self.kl_wt = 1000
    self.sim = nn.CosineSimilarity()

    self.modules = ['concat_encoder', 'concat_encoder2',
                    'audio_encoder', 'pose_encoder', 'text_encoder'] 

    requires_grad(self.G_tgt, True)
    for mod in self.modules: ## Fix the encoding modules
      requires_grad(getattr(self.G_tgt.model, mod), False)

    requires_grad(self.G_ema, False)
    requires_grad(self.G_src, False)


  def get_layers(self):
    layers = []
    #layers.append('concat_encoder2')
    layers += ['unet.conv1.{}'.format(i) for i in range(5)]
    layers += ['unet.conv2.{}'.format(i) for i in range(5)]
    layers += ['decoder.{}'.format(i) for i in range(4)]
    return layers


  def get_subset(self, x, y, consistency_loss_bsz, kwargs):
    consistency_loss_bsz = min(consistency_loss_bsz, y.shape[0])
    subset_idx = torch.randperm(y.shape[0])[:consistency_loss_bsz]
    y_subset = y[subset_idx]
    kwargs_subset = {}
    for key in kwargs:
      if key in ['text/token_count', 'text/token_duration']:
        kwargs_subset[key] = kwargs[key][subset_idx]
      else:
        kwargs_subset[key] = kwargs[key]

    x_subset = []
    for i, x_ in enumerate(x):
      if i < len(kwargs['input_modalities']):
        if kwargs['input_modalities'][i] == 'text/tokens':
          max_token_count = kwargs_subset['text/token_count'].max()
          x_subset.append(x_[subset_idx, :max_token_count])
          if 'text/token_duration' in kwargs_subset:
            kwargs_subset['text/token_duration'] = kwargs_subset['text/token_duration'][..., :max_token_count]
          
        else:
          x_subset.append(x_[subset_idx])
      else:
        x_subset.append(x_[subset_idx])        

    return x_subset, y_subset, kwargs_subset
  
  def get_distance_consistency_loss(self, x, y, time_steps, noise, consistency_loss_bsz=4, **kwargs):
    self.layer_idx = torch.randint(0, len(self.layers), size=(consistency_loss_bsz,))
    x_subset, y_subset, kwargs_subset = self.get_subset(x, y, consistency_loss_bsz, kwargs)
    with torch.no_grad():
      _, src_feats = self.G_src(x_subset, y_subset, time_steps, noise, **kwargs_subset)
      src_dist = torch.zeros(consistency_loss_bsz, consistency_loss_bsz-1).to(y.device)

      for sample1 in range(consistency_loss_bsz):
        count = 0
        for sample2 in range(consistency_loss_bsz):
          if sample1 != sample2:
            anchor_feat = src_feats[self.layers[self.layer_idx[sample1]]][sample1].reshape(-1).unsqueeze(0)
            compare_feat = src_feats[self.layers[self.layer_idx[sample1]]][sample2].reshape(-1).unsqueeze(0)
            src_dist[sample1, count] = self.sim(anchor_feat, compare_feat)
            count += 1

      src_dist = self.sfm(src_dist)

    _, tgt_feats = self.G_tgt(x_subset, y_subset, time_steps, noise, **kwargs_subset)
    tgt_dist = torch.zeros(consistency_loss_bsz, consistency_loss_bsz-1).to(y.device)

    for sample1 in range(consistency_loss_bsz):
      count = 0
      for sample2 in range(consistency_loss_bsz):
        if sample1 != sample2:
          anchor_feat = tgt_feats[self.layers[self.layer_idx[sample1]]][sample1].reshape(-1).unsqueeze(0)
          compare_feat = tgt_feats[self.layers[self.layer_idx[sample1]]][sample2].reshape(-1).unsqueeze(0)
          tgt_dist[sample1, count] = self.sim(anchor_feat, compare_feat)
          count += 1

    tgt_dist = self.sfm(tgt_dist)

    return self.kl_loss(torch.log(tgt_dist), src_dist) * self.kl_wt
    
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    if kwargs['sample_flag'] or kwargs['desc'] != 'train':
      out, internal_losses = self.G_ema(x, y, time_steps, noise, **kwargs)
    else:
      (out, internal_losses), feats_ = self.G_tgt(x, y, time_steps, noise, **kwargs)

      ## Cacluate Distance Consistency Loss
      internal_losses.append(self.get_distance_consistency_loss(x, y, time_steps, noise, consistency_loss_bsz=16, **kwargs))
    return out, internal_losses

