import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
from tqdm import tqdm

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

JointLateClusterSoftLatent_D = Speech2Gesture_D
JointLateClusterSoftLatent2_D = Speech2Gesture_D
JointLateClusterSoftLatent3_D = Speech2Gesture_D

class JointLateClusterSoftLatent_G(nn.Module):
  '''
  Learns a Latent Variable which decides the start position of a gesture
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
      if key in ['text/w2v', 'text/bert']:
        text_key=key
    if text_key:
      text_channels = kwargs['shape'][text_key][-1]
      self.text_encoder = TextEncoder1D(output_feats = time_steps,
                                        input_channels = text_channels,
                                        p=p)
    else:
      self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    
    self.latent_encoder = LatentEncoder(in_channels=in_channels*2, hidden_channels=in_channels, out_channels=2, p=p)
    
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

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None
    
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
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if True:
    pose_enc = self.pose_encoder(y, time_steps)
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

    pose_enc_t_1 = torch.cat([torch.zeros(pose_enc.shape[0], pose_enc.shape[1], 1).to(pose_enc.device).float(),
                              pose_enc[..., :-1]], dim=-1)

    ### causal encoder
    if self.training or True:
      ## estimate latent variable kappa
      kappa_score = self.latent_encoder(torch.cat([pose_enc_t_1, x], dim=1))
      kappa = torch.zeros_like(kappa_score)
      kappa[..., 1, 0] = 1
      kappa[..., 1:] = torch.softmax(kappa_score, dim=1)[..., 1:]

      #kappa = torch.softmax(kappa_score, dim=1)
      #kappa[..., 0, 0], kappa[..., 1, 0] = 0, 1 ## hardcode first time step to depend only on the audio
      self.kappa = kappa
      x = kappa[:, 0:1]*pose_enc + kappa[:, 1:2]*x
      #sm_loss = ((kappa[:, 1, 1:] - kappa[:, 1, :-1]) > 0.5).float().mean()
      sm_loss = ((kappa[..., 1:] - kappa[..., :-1]) !=0).float().mean()

      # if torch.rand(1).item() > self.thresh.step(self.training) and self.training: ## For joint training
      #   x = pose_enc
    # else:
    #   ## estimate latent variable kappa
    #   time_steps = x.shape[-1]
    #   for t in range(1, time_steps):
    #     x = torch.cat([pose_enc_t_1, x], dim=1)
    #     kappa_score = self.latent_encoder(x)
    #     kappa = torch.softmax(kappa_score[..., :t], dim=1)
    #     #kappa[..., 0, 0], kappa[..., 1, 0] = 0, 1 ## hardcode first time step to depend only on the audio
    #     x = kappa[:, 0:1]*pose_enc + kappa[:, 1:2]*x
    #     #sm_loss = ((kappa[:, 1, 1:] - kappa[:, 1, :-1]) > 0.5).float().mean()
    #   self.kappa = kappa
    #   sm_loss = ((kappa[..., 1:] - kappa[..., :-1]) !=0).float().mean()
    #   internal_losses.append(sm_loss)
    pdb.set_trace()
        
    x = self.unet(x)
    self.input_embedding = x
    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft
    
    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)
    internal_losses.append(sm_loss)
    return x, internal_losses

class JointLateClusterSoftLatent2_G(nn.Module):
  '''
  Train: single forward pass
  Inference: multiple forward passes
  Model starts with random pose sequences and then the model does a forward pass multiple times till convergence
  Learns a Latent Variable which decides the start position of a gesture
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
      if key in ['text/w2v', 'text/bert']:
        text_key=key
    if text_key:
      text_channels = kwargs['shape'][text_key][-1]
      self.text_encoder = TextEncoder1D(output_feats = time_steps,
                                        input_channels = text_channels,
                                        p=p)
    else:
      self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    
    self.latent_encoder = LatentEncoder(in_channels=in_channels*2, hidden_channels=in_channels, out_channels=2, p=p)
    
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

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None
    
  def index_select_outputs(self, x, labels):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], self.num_clusters, -1)
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x

  def forward_loop(self, x, y, time_steps, labels, **kwargs):
    internal_losses = []
    pose_enc = self.pose_encoder(y, time_steps)
    x = [x_.clone() for x_ in x]
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

    pose_enc_t_1 = torch.cat([torch.zeros(pose_enc.shape[0], pose_enc.shape[1], 1).to(pose_enc.device).float(),
                              pose_enc[..., :-1]], dim=-1)

    ## estimate latent variable kappa
    kappa_score = self.latent_encoder(torch.cat([pose_enc_t_1, x], dim=1))
    kappa = torch.zeros_like(kappa_score)
    kappa[..., 1, 0] = 1 ## hardcode first time step to depend only on the audio
    kappa[..., 1:] = torch.softmax(kappa_score, dim=1)[..., 1:]
    #kappa[..., 0, 0], kappa[..., 1, 0] = 0, 1 
    self.kappa = kappa
    x = kappa[:, 0:1]*pose_enc + kappa[:, 1:2]*x
    #sm_loss = ((kappa[:, 1, 1:] - kappa[:, 1, :-1]) > 0.5).float().mean()
    sm_loss = ((kappa[..., 1:] - kappa[..., :-1]) !=0).float().mean()
        
    x = self.unet(x)
    self.input_embedding = x
    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft
    
    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)

    internal_losses.append(sm_loss)

    stop = 0
    if kwargs['desc']!='train':
      diff = (torch.abs(y-x).mean())
      if diff < 0.1:
        stop = diff.item()

    return x, internal_losses, stop
  
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if True:
    if kwargs['desc'] == 'train':
      y, internal_losses, stop = self.forward_loop(x, y, time_steps, labels, **kwargs)
    else:
      y_inf = torch.rand_like(y)
      for t in tqdm(range(100), desc='markov chain'):
        y_inf, internal_losses, stop = self.forward_loop(x, y_inf, time_steps, labels, **kwargs)
        if stop:
          break
      y = y_inf

    return y, internal_losses

class JointLateClusterSoftLatent3_G(JointLateClusterSoftLatent2_G):
  '''
  Train: single forward pass
  Inference: 1 inference step for each time step.
  Model starts with random pose sequences and then the model does a forward pass multiple times till convergence
  Learns a Latent Variable which decides the start position of a gesture
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, **kwargs):
    super().__init__(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, num_clusters=num_clusters, cluster=cluster, **kwargs)
    self.classify_loss = nn.CrossEntropyLoss()
  def forward_loop(self, x, y, time_steps, labels, t=None, **kwargs):
    internal_losses = []
    pose_enc = self.pose_encoder(y, time_steps)
    x = [x_.clone() for x_ in x]
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

    pose_enc_t_1 = torch.cat([torch.zeros(pose_enc.shape[0], pose_enc.shape[1], 1).to(pose_enc.device).float(),
                              pose_enc[..., :-1]], dim=-1)

    ## estimate latent variable kappa
    kappa_score = self.latent_encoder(torch.cat([pose_enc_t_1, x], dim=1))
    kappa = torch.cat([torch.zeros_like(kappa_score[:, :1]), torch.ones_like(kappa_score[:, :1])], dim=1)
    #kappa[..., 0, 0] = 0 ## hardcode first time step to depend only on the audio
    if t is None:
      kappa[..., 1:] = torch.softmax(kappa_score, dim=1)[..., 1:]
    else:
      kappa[..., 1:t+1] = torch.softmax(kappa_score, dim=1)[..., 1:t+1]

    #kappa[..., 0, 0], kappa[..., 1, 0] = 0, 1 
    self.kappa = kappa
    x = kappa[:, 0:1]*pose_enc + kappa[:, 1:2]*x
    #sm_loss = ((kappa[:, 1, 1:] - kappa[:, 1, :-1]) > 0.5).float().mean()
    kappa_int = torch.argmax(kappa, dim=1)
    sm_loss = ((kappa_int[..., 1:] - kappa_int[..., :-1]) !=0).float().mean()
        
    x = self.unet(x)
    self.input_embedding = x
    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)

    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft
    
    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)

    internal_losses.append(sm_loss)

    stop = False
    if t is None:
      return x, internal_losses, stop
    else:
      return x[..., t:t+1, :], internal_losses, stop
  
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if True:
    if kwargs['desc'] == 'train':
      y, internal_losses, stop = self.forward_loop(x, y, time_steps, labels, **kwargs)
    else:
      y_inf = torch.rand_like(y)
      for t in tqdm(range(y.shape[1]), desc='time chain'):
        y_inf[..., t:t+1, :], internal_losses, stop = self.forward_loop(x, y_inf, time_steps, labels, t=t, **kwargs)
        # if (y - y_inf).abs().mean() > 10:
        #   pdb.set_trace()
      y = y_inf

    return y, internal_losses

