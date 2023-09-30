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

JointLateClusterSoftTransformerNoise_D = Speech2Gesture_D

'''
Same as JointLateClusterSoftTransformer12_G but an extra random input to model style.

Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
multimodal fusion is done both ways, Audio -> Text and Text -> Audio
'''

class JointLateClusterSoftTransformerNoise_G(nn.Module):
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256,
               in_noise_dim=100, out_noise_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    self.time_steps = time_steps
    self.in_noise_dim = in_noise_dim
    self.out_noise_dim = out_noise_dim

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        text_key=key
#    if text_key:
    self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])

    self.pos_encoder = PositionalEncoding(256, p)
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    ## decoder with noise and language concatenated
    self.decoder = nn.ModuleList([ConvNormRelu(in_channels + self.out_noise_dim, in_channels,
                                               type='1d', leaky=True, downsample=False,
                                               p=p, groups=self.num_clusters)])
    for i in range(3):
      self.decoder.append(ConvNormRelu(in_channels, in_channels,
                                       type='1d', leaky=True, downsample=False,
                                       p=p, groups=self.num_clusters))
    self.decoder = nn.Sequential(*self.decoder)

    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))


    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
    self.classify_loss = nn.CrossEntropyLoss()

    self.labels_cap_soft = None

    self.noise_dec = nn.Sequential(nn.Linear(self.in_noise_dim, time_steps*self.out_noise_dim),
                                   nn.BatchNorm1d(time_steps*self.out_noise_dim),
                                   nn.LeakyReLU(0.2))

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
    x = x[:-1]

    if False:
      x = self.pose_encoder(y, time_steps)
    else:
      mod_map = {}
      for i, modality in enumerate(kwargs['input_modalities']):
        if modality.split('/')[0] == "text":
          mod_map['text'] = i
          #.set_trace() #self.training FALSE
          for te in self.text_encoder:
            x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, 
                      pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

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

    x = self.unet(x)

    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft

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

    ## concatenate noise
    x = torch.cat([x, noise], dim=1)

    #repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)

    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)

    return x, internal_losses

  ## MineGAN functions
  def forward_enc(self, x, y, time_steps=None, noise=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    if False:
      x = self.pose_encoder(y, time_steps)
    else:
      mod_map = {}
      for i, modality in enumerate(kwargs['input_modalities']):
        if modality.split('/')[0] == "text":
          mod_map['text'] = i
          #.set_trace() #self.training FALSE
          for te in self.text_encoder:
            x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, 
                      pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

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
    return x, labels, internal_losses

  ## MineGAN functions
  def forward_dec(self, x, y, labels, noise, internal_losses, **kwargs):
    x = self.unet(x)

    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft

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

    ## concatenate noise
    x = torch.cat([x, noise], dim=1)

    #repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)

    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)

    return x, internal_losses


  ## DiffGAN functions
  def forward_DiffGAN(self, x, y, time_steps=None, noise=None,
                      internal_losses=[], LAYER_START=0, LAYER_END=1000, ## to get chunked forward pass
                      **kwargs):

    assert LAYER_START >= 0
    assert LAYER_END > 0
    assert LAYER_START < LAYER_END

    if LAYER_END == 0:
      return x, internal_losses
    
    if LAYER_START <= 0:
      ## language and audio encoders
      self.labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]
      
      self.mod_map = {}
      for i, modality in enumerate(kwargs['input_modalities']):
        if modality.split('/')[0] == "text":
          self.mod_map['text'] = i
          #.set_trace() #self.training FALSE
          for te in self.text_encoder:
            x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, 
                      pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

        if modality.split('/')[0] == 'audio':
          self.mod_map['audio'] = i
          if x[i].dim() == 3:
            x[i] = x[i].unsqueeze(dim=1)
          x[i] = self.audio_encoder(x[i], time_steps)

    if LAYER_END == 1:
      return x, internal_losses

    if LAYER_START <= 1:
      if len(x) >= 2:
        memory = x[self.mod_map['text']]
        tgt = x[self.mod_map['audio']]
        tgt = self.pos_encoder(tgt.permute(2, 0, 1)).permute(1, 2, 0)
        x_ = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, **kwargs)
        x = torch.cat([x_, x[self.mod_map['audio']]], dim=1)
        x = self.concat_encoder2(x)
      else:
        x = torch.cat(tuple(x),  dim=1)

    if LAYER_END == 2:
      return x, internal_losses

    if LAYER_START <= 2:
      x = self.unet(x)

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), self.labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

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

      ## concatenate noise
      x = torch.cat([x, noise], dim=1)

      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

    if LAYER_END == 3:
      return x, internal_losses

    if LAYER_START <=3:
      x = self.decoder[0](x)

    if LAYER_END == 4:
      return x, internal_losses

    if LAYER_START <=4:
      x = self.decoder[1](x)

    if LAYER_END == 5:
      return x, internal_losses

    if LAYER_START <=5:
      x = self.decoder[2](x)

    if LAYER_END == 6:
      return x, internal_losses

    if LAYER_START <=6:
      x = self.decoder[3](x)

    if LAYER_END == 7:
      return x, internal_losses

    if LAYER_START <=7:
      x = self.logits(x)
      x = self.index_select_outputs(x, self.labels_cap_soft)

    return x, internal_losses

  # ### enc 2
  #   if z_l_1 is None:
  #     x = self.decoder[3](x)
  #   else:
  #     x = self.decoder[3](z_l_1)
  #   return x, internal_losses

  # ### enc 3
  #   if z_l is None:
  #     x = self.logits(x)
  #   else:
  #     x = self.logits(z_l)
  #   x = self.index_select_outputs(x, self.labels_cap_soft)

  #   return x, internal_losses

  def enc1(self, x, y, time_steps=None, noise=None, **kwargs):
    return self.forward_DiffGAN(x, y, time_steps, noise,
                                internal_losses=[],
                                LAYER_START=0,
                                LAYER_END=self.fewshot_LAYER_START,
                                **kwargs)

  def enc2(self, x, y, internal_losses, z_l_1=None, time_steps=None, noise=None, **kwargs):
    if z_l_1 is not None:
      x = z_l_1
      
    return self.forward_DiffGAN(x, y, time_steps, noise,
                                internal_losses=internal_losses,
                                LAYER_START=self.fewshot_LAYER_START,
                                LAYER_END=self.fewshot_LAYER_END,
                                **kwargs)

  def enc3(self, x, y, internal_losses, z_l=None, time_steps=None, noise=None, **kwargs):
    if z_l is not None:
      x = z_l
      
    return self.forward_DiffGAN(x, y, time_steps, noise,
                                internal_losses=internal_losses,
                                LAYER_START=self.fewshot_LAYER_END,
                                LAYER_END=1000,
                                **kwargs)
  
  # def enc1(self, x, y, time_steps=None, noise=None, **kwargs):
  #   internal_losses = []
  #   labels = x[-1] ## remove the labels attached to the inputs
  #   x = x[:-1]

  #   if False:
  #     x = self.pose_encoder(y, time_steps)
  #   else:
  #     mod_map = {}
  #     for i, modality in enumerate(kwargs['input_modalities']):
  #       if modality.split('/')[0] == "text":
  #         mod_map['text'] = i
  #         #.set_trace() #self.training FALSE
  #         for te in self.text_encoder:
  #           x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =0, 
  #                     pos_encoder=self.pos_encoder, **kwargs) ## (B, channels, time)

  #       if modality.split('/')[0] == 'audio':
  #         mod_map['audio'] = i
  #         if x[i].dim() == 3:
  #           x[i] = x[i].unsqueeze(dim=1)
  #         x[i] = self.audio_encoder(x[i], time_steps)

  #     if len(x) >= 2:
  #       memory = x[mod_map['text']]
  #       tgt = x[mod_map['audio']]
  #       tgt = self.pos_encoder(tgt.permute(2, 0, 1)).permute(1, 2, 0)
  #       x_ = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, **kwargs)
  #       x = torch.cat([x_, x[mod_map['audio']]], dim=1)
  #       x = self.concat_encoder2(x)
  #     else:
  #       x = torch.cat(tuple(x),  dim=1)

  #   x = self.unet(x)

  #   ## Classify clusters using audio/text
  #   labels_score = self.classify_cluster(x).transpose(2, 1)
  #   internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

  #   #_, labels_cap = labels_score.max(dim=-1)
  #   labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
  #   self.labels_cap_soft = labels_cap_soft

  #   ## decode
  #   if noise is None: 
  #     if kwargs['sample_flag']:
  #       n_time_steps = y.shape[1] // self.time_steps 
  #       noise = torch.rand(n_time_steps, self.in_noise_dim)
  #     else:
  #       noise = torch.rand(y.shape[0], self.in_noise_dim)

  #   noise = noise.to(y.device)
  #   ## (B, self.out_noise_dim, self.time_steps)          
  #   noise = self.noise_dec(noise).view(-1, self.out_noise_dim, self.time_steps)
  #   if kwargs['sample_flag']:
  #     noise = noise.permute(1, 0, 2).reshape(noise.shape[1], -1).unsqueeze(0) ## (1, out_noise_dim, time_steps*n_time_steps)

  #   ## concatenate noise
  #   x = torch.cat([x, noise], dim=1)

  #   #repeat inputs before decoder
  #   x = torch.cat([x]*self.num_clusters, dim=1)

  #   for i in range(3):
  #     x = self.decoder[i](x)
    
  #   return x, internal_losses

  # def enc2(self, x, y, internal_losses, z_l_1=None, **kwargs):
  #   if z_l_1 is None:
  #     x = self.decoder[3](x)
  #   else:
  #     x = self.decoder[3](z_l_1)
  #   return x, internal_losses

  # def enc3(self, x, y, internal_losses, z_l=None, **kwargs):
  #   if z_l is None:
  #     x = self.logits(x)
  #   else:
  #     x = self.logits(z_l)
  #   x = self.index_select_outputs(x, self.labels_cap_soft)

  #   return x, internal_losses
