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

JointLateClusterSoftStyle_D = Speech2Gesture_D
JointLateClusterSoftStyle2_D = Speech2Gesture_D
JointLateClusterSoftStyle3_D = Speech2Gesture_D
JointLateClusterSoftStyle4_D = Speech2Gesture_D

class JointLateClusterSoftStyle_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

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
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=style_dim)
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

    self.style_emb = nn.Embedding(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
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

    #labels_style = (self.style_emb(kwargs['style']) + self.style_emb(1-kwargs['style']))/2
    labels_style = self.style_emb(kwargs['style'])
    x = torch.cat([x]*self.style_dim, dim=1)
    x = self.unet(x)
    x = self.index_select_outputs(x, labels_style, self.style_dim)
    x = x.transpose(2, 1)
    
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
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    return x, internal_losses

  
class JointLateClusterSoftStyle2_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    #self.style_dim = style_dim
    self.style_dim = len(self.style_dict)

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
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=self.style_dim)
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

    # self.style_emb = nn.Embedding(num_embeddings=len(self.style_dict),
    #                               embedding_dim=len(self.style_dict), weight_=torch.eye(len(self.style_dict)))


  def to_onehot(self, idxs):
    device = idxs.device
    idxs = idxs.unsqueeze(-1).long()
    x_shape = idxs.shape[:-1] + torch.Size([len(self.style_dict)])
    x = torch.zeros(x_shape).to(device)
    return x.scatter(-1, idxs, 1)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
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

    #labels_style = (self.style_emb(kwargs['style']) + self.style_emb(1-kwargs['style']))/2
    #labels_style = self.style_emb(kwargs['style'])

    labels_style = self.to_onehot(kwargs['style']).float()
    x = torch.cat([x]*self.style_dim, dim=1)
    x = self.unet(x)
    x = self.index_select_outputs(x, labels_style, self.style_dim)
    x = x.transpose(2, 1)
    
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
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    return x, internal_losses


class JointLateClusterSoftStyle3_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim
    #self.style_dim = len(self.style_dict)

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
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=1)

    ## Style
    self.style_emb = EmbLin(num_embeddings=len(self.style_dict),
                            embedding_dim=self.style_dim)
    self.style_dec = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.style_dim)
                                                 for i in range(2)]))
    self.style_dec_gr = Group([self.style_dec], groups=self.style_dim)

    ## Content
    decoder_list = nn.ModuleList()
    decoder_list.append(ConvNormRelu(self.style_dim+in_channels, in_channels,
                                     type='1d', leaky=True, downsample=False,
                                     p=p, groups=self.num_clusters))
    decoder_list += nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                type='1d', leaky=True, downsample=False,
                                                p=p, groups=self.num_clusters)
                                   for i in range(3)])
    self.decoder = nn.Sequential(*decoder_list)
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=1, input_channels=self.style_dim+in_channels)
    #self.classify_cluster_gr = Group([self.classify_cluster], groups=self.style_dim, dim=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None



  def to_onehot(self, idxs):
    device = idxs.device
    idxs = idxs.unsqueeze(-1).long()
    x_shape = idxs.shape[:-1] + torch.Size([len(self.style_dict)])
    x = torch.zeros(x_shape).to(device)
    return x.scatter(-1, idxs, 1)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
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

    #labels_style = (self.style_emb(kwargs['style']) + self.style_emb(1-kwargs['style']))/2
    #labels_style = self.style_emb(kwargs['style'])

    #labels_style = self.to_onehot(kwargs['style']).float()
    #x = torch.cat([x]*self.style_dim, dim=1)
    x = self.unet(x)
    #x = self.index_select_outputs(x, labels_style, self.style_dim)
    x = x.transpose(2, 1)

    ## Style
    if len(kwargs['style'].shape) == 2:
      mode = 'emb'
    elif len(kwargs['style'].shape) == 3:
      mode = 'lin' ## used while training for out-of-domain style embeddings
    labels_style = self.style_emb(kwargs['style'], mode=mode)
    if x.shape[1] != labels_style.shape[1]:
      labels_style = labels_style.view(x.shape[0], -1, labels_style.shape[-1])
    ## concatenate content and style
    x = torch.cat([x, labels_style], dim=-1).transpose(2, 1)
    
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
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    return x, internal_losses


class JointLateClusterSoftStyle4_G(nn.Module):
  '''
  gives id_in and id_out losses as well
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None,
               style_dict={}, style_dim=10,
               lambda_id=1, train_only=0,
               softmax=1, argmax=0,
               some_grad_flag=False, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim
    self.lambda_id = lambda_id
    self.train_only = train_only
    self.softmax = softmax
    self.argmax = argmax
    self.some_grad_flag = some_grad_flag
    #self.style_dim = len(self.style_dict)

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
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=1)

    ## Style
    self.pose_style_encoder = PoseStyleEncoder(input_channels=out_feats, p=p, num_speakers=len(self.style_dict))
    self.style_emb = EmbLin(num_embeddings=len(self.style_dict),
                            embedding_dim=self.style_dim)
    self.style_dec = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.style_dim)
                                                 for i in range(2)]))
    self.style_dec_gr = Group([self.style_dec], groups=self.style_dim)

    ## Content
    decoder_list = nn.ModuleList()
    decoder_list.append(ConvNormRelu(self.style_dim+in_channels, in_channels,
                                     type='1d', leaky=True, downsample=False,
                                     p=p, groups=self.num_clusters))
    decoder_list += nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                type='1d', leaky=True, downsample=False,
                                                p=p, groups=self.num_clusters)
                                   for i in range(3)])
    self.decoder = nn.Sequential(*decoder_list)
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=1, input_channels=self.style_dim+in_channels)
    #self.classify_cluster_gr = Group([self.classify_cluster], groups=self.style_dim, dim=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None



  def to_onehot(self, idxs):
    device = idxs.device
    idxs = idxs.unsqueeze(-1).long()
    x_shape = idxs.shape[:-1] + torch.Size([len(self.style_dict)])
    x = torch.zeros(x_shape).to(device)
    return x.scatter(-1, idxs, 1)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
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

    #labels_style = (self.style_emb(kwargs['style']) + self.style_emb(1-kwargs['style']))/2
    #labels_style = self.style_emb(kwargs['style'])

    #labels_style = self.to_onehot(kwargs['style']).float()
    #x = torch.cat([x]*self.style_dim, dim=1)
    x = self.unet(x)
    #x = self.index_select_outputs(x, labels_style, self.style_dim)
    x = x.transpose(2, 1)

    ## Pose Style
    pose_style_encoder_flag = not kwargs['sample_flag'] and (kwargs['description'] == 'train' or not self.train_only)

    if pose_style_encoder_flag: ## Training
      mode = 'lin'
      pose_style_score = self.pose_style_encoder(y)
      id_in_loss = torch.nn.functional.cross_entropy(pose_style_score, kwargs['style'][:, 0])
      pose_style_score = pose_style_score.unsqueeze(1).expand(pose_style_score.shape[0], x.shape[1], pose_style_score.shape[-1]) ## (B, T, num_speakers)
      if self.softmax:
        pose_style = torch.softmax(pose_style_score, dim=-1)
        if self.argmax:
          pose_style = torch.argmax(pose_style, dim=-1)
          mode = 'emb'
      else:
        pose_style = pose_style_score
    else:
      pose_style = kwargs['style']
      if len(kwargs['style'].shape) == 2:
        mode = 'emb'
      elif len(kwargs['style'].shape) == 3:
        mode = 'lin' ## used while training for out-of-domain style embeddings
      id_in_loss = torch.zeros(1)[0]
    labels_style = self.style_emb(pose_style, mode=mode)

    if x.shape[1] != labels_style.shape[1]:
      labels_style = labels_style.view(x.shape[0], -1, labels_style.shape[-1])
    ## concatenate content and style
    x = torch.cat([x, labels_style], dim=-1).transpose(2, 1)
    
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
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)        
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    if pose_style_encoder_flag:
      if self.some_grad_flag: ## the pose_style encoder is fixed for the generated outputs, hence only the generator receives the learning gradients.
        with some_grad(self.pose_style_encoder):
          pose_style_score_out = self.pose_style_encoder(x)
      else:
        pose_style_score_out = self.pose_style_encoder(x)
      id_out_loss = torch.nn.functional.cross_entropy(pose_style_score_out, kwargs['style'][:, 0])
    else:
      id_out_loss = torch.zeros(1)[0]

    internal_losses.append(id_in_loss*self.lambda_id)
    internal_losses.append(id_out_loss*self.lambda_id)
    return x, internal_losses


class JointLateClusterSoftStyle5_G(nn.Module):
  '''
  Modified for Continual Learning
  Both Language and Audio as input and rest is same as JointLateClusterSoftStyle4_G
  gives id_in and id_out losses as well
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None,
               style_dict={}, style_dim=10,
               lambda_id=1, train_only=0,
               softmax=1, argmax=0,
               some_grad_flag=False, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim
    self.lambda_id = lambda_id
    self.train_only = train_only
    self.softmax = softmax
    self.argmax = argmax
    self.some_grad_flag = some_grad_flag
    max_num_speakers = kwargs.get('max_num_speakers')
    self.max_num_speakers = max(max_num_speakers, len(self.style_dict)) if max_num_speakers is not None else len(self.style_dict)
    #self.style_dim = len(self.style_dict)

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        text_key=key
    if text_key:
      self.text_encoder = nn.ModuleList([BertEncoder(out_feats=in_channels)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

      
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=1)

    ## Style
    self.pose_style_encoder = PoseStyleEncoder(input_channels=out_feats, p=p, num_speakers=self.max_num_speakers)
    self.style_emb = EmbLin(num_embeddings=self.max_num_speakers,
                            embedding_dim=self.style_dim)

    ### --- unused params ----
    self.style_dec = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.style_dim)
                                                 for i in range(2)]))
    self.style_dec_gr = Group([self.style_dec], groups=self.style_dim)
    ### ----------------------

    ## Content
    decoder_list = nn.ModuleList()
    decoder_list.append(ConvNormRelu(self.style_dim+in_channels, in_channels,
                                     type='1d', leaky=True, downsample=False,
                                     p=p, groups=self.num_clusters))
    decoder_list += nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                type='1d', leaky=True, downsample=False,
                                                p=p, groups=self.num_clusters)
                                   for i in range(3)])
    self.decoder = nn.Sequential(*decoder_list)
    

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=1, input_channels=self.style_dim+in_channels)
    #self.classify_cluster_gr = Group([self.classify_cluster], groups=self.style_dim, dim=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None



  def to_onehot(self, idxs):
    device = idxs.device
    idxs = idxs.unsqueeze(-1).long()
    x_shape = idxs.shape[:-1] + torch.Size([self.max_num_speakers])
    x = torch.zeros(x_shape).to(device)
    return x.scatter(-1, idxs, 1)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x
  
  def forward(self, x, y, time_steps=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]
    kwargs['style'] = kwargs['style'].long()

    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.
    #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
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
    x = x.transpose(2, 1)

    ## Pose Style
    pose_style_encoder_flag = not kwargs['sample_flag'] and (kwargs['description'] == 'train' or not self.train_only)

    if pose_style_encoder_flag: ## Training
      mode = 'lin'
      pose_style_score = self.pose_style_encoder(y)
      id_in_loss = torch.nn.functional.cross_entropy(pose_style_score, kwargs['style'][:, 0])
      pose_style_score = pose_style_score.unsqueeze(1).expand(pose_style_score.shape[0], x.shape[1], pose_style_score.shape[-1]) ## (B, T, num_speakers)
      if self.softmax:
        pose_style = torch.softmax(pose_style_score, dim=-1)
        if self.argmax:
          pose_style = torch.argmax(pose_style, dim=-1)
          mode = 'emb'
      else:
        pose_style = pose_style_score
    else:
      pose_style = kwargs['style']
      if len(kwargs['style'].shape) == 2:
        mode = 'emb'
      elif len(kwargs['style'].shape) == 3:
        mode = 'lin' ## used while training for out-of-domain style embeddings
      id_in_loss = torch.zeros(1)[0]
    labels_style = self.style_emb(pose_style, mode=mode)

    if x.shape[1] != labels_style.shape[1]:
      labels_style = labels_style.view(x.shape[0], -1, labels_style.shape[-1])
    ## concatenate content and style
    x = torch.cat([x, labels_style], dim=-1).transpose(2, 1)
    
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
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)        
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    if pose_style_encoder_flag:
      if self.some_grad_flag: ## the pose_style encoder is fixed for the generated outputs, hence only the generator receives the learning gradients.
        with some_grad(self.pose_style_encoder):
          pose_style_score_out = self.pose_style_encoder(x)
      else:
        pose_style_score_out = self.pose_style_encoder(x)
      id_out_loss = torch.nn.functional.cross_entropy(pose_style_score_out, kwargs['style'][:, 0])
    else:
      id_out_loss = torch.zeros(1)[0]

    internal_losses.append(id_in_loss*self.lambda_id)
    internal_losses.append(id_out_loss*self.lambda_id)
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
      kwargs['style'] = kwargs['style'].long()
      
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
      x = x.transpose(2, 1)

      ## Pose Style
      pose_style_encoder_flag = not kwargs['sample_flag'] and (kwargs['description'] == 'train' or not self.train_only)

      if pose_style_encoder_flag: ## Training
        mode = 'lin'
        pose_style_score = self.pose_style_encoder(y)
        id_in_loss = torch.nn.functional.cross_entropy(pose_style_score, kwargs['style'][:, 0])
        pose_style_score = pose_style_score.unsqueeze(1).expand(pose_style_score.shape[0], x.shape[1], pose_style_score.shape[-1]) ## (B, T, num_speakers)
        if self.softmax:
          pose_style = torch.softmax(pose_style_score, dim=-1)
          if self.argmax:
            pose_style = torch.argmax(pose_style, dim=-1)
            mode = 'emb'
        else:
          pose_style = pose_style_score
      else:
        pose_style = kwargs['style']
        if len(kwargs['style'].shape) == 2:
          mode = 'emb'
        elif len(kwargs['style'].shape) == 3:
          mode = 'lin' ## used while training for out-of-domain style embeddings
        id_in_loss = torch.zeros(1)[0]
      labels_style = self.style_emb(pose_style, mode=mode)

      if x.shape[1] != labels_style.shape[1]:
        labels_style = labels_style.view(x.shape[0], -1, labels_style.shape[-1])
      ## concatenate content and style
      x = torch.cat([x, labels_style], dim=-1).transpose(2, 1)

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), self.labels.reshape(-1)))
      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## repeat inputs before decoder
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
      x = self.index_select_outputs(x, self.labels_cap_soft, self.num_clusters)

    return x, internal_losses


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


class JointLateClusterSoftStyleNoise5_G(nn.Module):
  '''
  Noise + Both Language and Audio as input and rest is same as JointLateClusterSoftStyle4_G

  gives id_in and id_out losses as well
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None,
               style_dict={}, style_dim=10,
               lambda_id=1, train_only=0,
               softmax=1, argmax=0,
               some_grad_flag=False,
               in_noise_dim=100, out_noise_dim=10, **kwargs):
    super().__init__()
    self.time_steps = time_steps
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim
    self.lambda_id = lambda_id
    self.train_only = train_only
    self.softmax = softmax
    self.argmax = argmax
    self.some_grad_flag = some_grad_flag
    self.in_noise_dim = in_noise_dim
    self.out_noise_dim = out_noise_dim
    #self.style_dim = len(self.style_dict)

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        text_key=key
    if text_key:
      self.text_encoder = nn.ModuleList([BertEncoder(out_feats=in_channels)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

      
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=1)

    ## Style
    self.pose_style_encoder = PoseStyleEncoder(input_channels=out_feats, p=p, num_speakers=len(self.style_dict))
    self.style_emb = EmbLin(num_embeddings=len(self.style_dict),
                            embedding_dim=self.style_dim)
    self.style_dec = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.style_dim)
                                                 for i in range(2)]))
    self.style_dec_gr = Group([self.style_dec], groups=self.style_dim)

    ## Content
    decoder_list = nn.ModuleList()
    decoder_list.append(ConvNormRelu(self.style_dim+in_channels+self.out_noise_dim, in_channels,
                                     type='1d', leaky=True, downsample=False,
                                     p=p, groups=self.num_clusters))
    decoder_list += nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                type='1d', leaky=True, downsample=False,
                                                p=p, groups=self.num_clusters)
                                   for i in range(3)])
    self.decoder = nn.Sequential(*decoder_list)
    

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=1, input_channels=self.style_dim+in_channels)
    #self.classify_cluster_gr = Group([self.classify_cluster], groups=self.style_dim, dim=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

    self.noise_dec = nn.Sequential(nn.Linear(self.in_noise_dim, time_steps*self.out_noise_dim),
                                   nn.BatchNorm1d(time_steps*self.out_noise_dim),
                                   nn.LeakyReLU(0.2))


  def to_onehot(self, idxs):
    device = idxs.device
    idxs = idxs.unsqueeze(-1).long()
    x_shape = idxs.shape[:-1] + torch.Size([len(self.style_dict)])
    x = torch.zeros(x_shape).to(device)
    return x.scatter(-1, idxs, 1)
    
  def index_select_outputs(self, x, labels, groups):
    '''
    x: (B, num_clusters*out_feats, T)
    labels: (B, T, num_clusters)
    '''
    x = x.transpose(2, 1)
    x = x.view(x.shape[0], x.shape[1], groups, -1)
    labels = labels.view(x.shape[0], x.shape[1], x.shape[2]) ## shape consistency while sampling
    x = (x * labels.unsqueeze(-1)).sum(dim=-2)
    return x
  
  def forward(self, x, y, time_steps=None, noise=None, **kwargs):
    internal_losses = []
    labels = x[-1] ## remove the labels attached to the inputs
    x = x[:-1]

    # Late Fusion with Joint
    ## Joint training intially helps train the classify_cluster model
    ## using pose as inputs, after a while when the generators have
    ## been pushed in the direction of learning the corresposing modes,
    ## we transition to speech and text as input.

    if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
    #if True:
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
    x = x.transpose(2, 1)

    ## Pose Style
    pose_style_encoder_flag = not kwargs['sample_flag'] and (kwargs['description'] == 'train' or not self.train_only)

    if pose_style_encoder_flag: ## Training
      mode = 'lin'
      pose_style_score = self.pose_style_encoder(y)
      id_in_loss = torch.nn.functional.cross_entropy(pose_style_score, kwargs['style'][:, 0])
      pose_style_score = pose_style_score.unsqueeze(1).expand(pose_style_score.shape[0], x.shape[1], pose_style_score.shape[-1]) ## (B, T, num_speakers)
      if self.softmax:
        pose_style = torch.softmax(pose_style_score, dim=-1)
        if self.argmax:
          pose_style = torch.argmax(pose_style, dim=-1)
          mode = 'emb'
      else:
        pose_style = pose_style_score
    else:
      pose_style = kwargs['style']
      if len(kwargs['style'].shape) == 2:
        mode = 'emb'
      elif len(kwargs['style'].shape) == 3:
        mode = 'lin' ## used while training for out-of-domain style embeddings
      id_in_loss = torch.zeros(1)[0]
    labels_style = self.style_emb(pose_style, mode=mode)

    if x.shape[1] != labels_style.shape[1]:
      labels_style = labels_style.view(x.shape[0], -1, labels_style.shape[-1])
    ## concatenate content and style
    x = torch.cat([x, labels_style], dim=-1).transpose(2, 1)
    
    ## Classify clusters using audio/text
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft

    ## Noise 
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

    x = torch.cat([x, noise], dim=1)
    
    ## repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)
      
    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft, self.num_clusters)        
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    if pose_style_encoder_flag:
      if self.some_grad_flag: ## the pose_style encoder is fixed for the generated outputs, hence only the generator receives the learning gradients.
        with some_grad(self.pose_style_encoder):
          pose_style_score_out = self.pose_style_encoder(x)
      else:
        pose_style_score_out = self.pose_style_encoder(x)
      id_out_loss = torch.nn.functional.cross_entropy(pose_style_score_out, kwargs['style'][:, 0])
    else:
      id_out_loss = torch.zeros(1)[0]

    internal_losses.append(id_in_loss*self.lambda_id)
    internal_losses.append(id_out_loss*self.lambda_id)
    return x, internal_losses
