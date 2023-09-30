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

JointLateClusterSoftTransformer_D = Speech2Gesture_D
JointLateClusterSoftTransformer2_D = Speech2Gesture_D
JointLateClusterSoftTransformer3_D = Speech2Gesture_D
JointLateClusterSoftTransformer4_D = Speech2Gesture_D
JointLateClusterSoftTransformer5_D = Speech2Gesture_D
JointLateClusterSoftTransformer6_D = Speech2Gesture_D
JointLateClusterSoftTransformer7_D = Speech2Gesture_D
JointLateClusterSoftTransformer8_D = Speech2Gesture_D
JointLateClusterSoftTransformer9_D = Speech2Gesture_D
JointLateClusterSoftTransformer10_D = Speech2Gesture_D
JointLateClusterSoftTransformer11_D = Speech2Gesture_D
JointLateClusterSoftTransformer12_D = Speech2Gesture_D
JointLateClusterSoftTransformer13_D = Speech2Gesture_D

class JointLateClusterSoftTransformer_G(nn.Module):
  '''
  decoder is unet, output of the text encoder is replicated based on the duration of each token

  Transformer as encoder for Language with a vanilla CNN as the decoder

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, S, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    #Encoder
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert']:
        text_key=key
    if text_key:
      text_channels = kwargs['shape'][text_key][-1]
      self.text_encoder = TransfomerEncoder(time_steps=time_steps, in_channels=text_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    else:
      self.text_encoder = TransfomerEncoder(time_steps=time_steps, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
    # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
    #                                         type='1d', leaky=True, downsample=False,
    #                                         p=p, groups=self.num_clusters)
    #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p, groups=self.num_clusters)
                                                 for i in range(4)]))

    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))



    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
    self.classify_loss = nn.CrossEntropyLoss()
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)

    self.thresh = Curriculum(0, 1, 1000)
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
      x = self.pose_encoder(y, time_steps)
    else:
      for i, modality in enumerate(kwargs['input_modalities']):
        if modality.split('/')[0] == "text":
          x[i] = self.text_encoder(x[i], y, output_repeat=1, **kwargs).transpose(1, 2) ## (B, channels, time)
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
    labels_score = self.classify_cluster(x).transpose(2, 1)
    internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

    #_, labels_cap = labels_score.max(dim=-1)
    labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
    self.labels_cap_soft = labels_cap_soft

    ## decode
    #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
    #x = x.transpose(-2, -1)
    #repeat inputs before decoder
    x = torch.cat([x]*self.num_clusters, dim=1)

    x = self.decoder(x)
    x = self.logits(x)
    x = self.index_select_outputs(x, labels_cap_soft)
    #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

    return x, internal_losses


class JointLateClusterSoftTransformer2_G(nn.Module):
   def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
     super().__init__()
     self.num_clusters = num_clusters
     self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
     #Encoder
     self.nhead = 8
     self.nhid = 3
     self.ninp = E

     text_key = None
     for key in kwargs['shape']:
       if key in ['text/w2v', 'text/bert']:
         text_key=key
     if text_key:
       text_channels = kwargs['shape'][text_key][-1]
       self.text_encoder = TransfomerEncoder2(time_steps=time_steps, in_channels=text_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
     else:
       self.text_encoder = TransfomerEncoder2(time_steps=time_steps, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

     self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
     self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

     self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
     # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
     #                                         type='1d', leaky=True, downsample=False,
     #                                         p=p, groups=self.num_clusters)
     #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
     self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                               type='1d', leaky=True, downsample=False,
                                                               p=p, groups=self.num_clusters)
                                                  for i in range(4)]))

     self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                      type='1d', leaky=True, downsample=False,
                                                                      p=p)]))



     self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
     self.classify_loss = nn.CrossEntropyLoss()
     self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
     self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                  downsample=False, p=p)

     self.thresh = Curriculum(0, 1, 1000)
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
       x = self.pose_encoder(y, time_steps)
     else:
       for i, modality in enumerate(kwargs['input_modalities']):
         if modality.split('/')[0] == "text":
           x[i] = self.text_encoder(x[i], y, output_repeat=1, **kwargs).transpose(1, 2) ## (B, channels, time)
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
     labels_score = self.classify_cluster(x).transpose(2, 1)
     internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

     #_, labels_cap = labels_score.max(dim=-1)
     labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
     self.labels_cap_soft = labels_cap_soft

     ## decode
     #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
     #x = x.transpose(-2, -1)
     #repeat inputs before decoder
     x = torch.cat([x]*self.num_clusters, dim=1)

     x = self.decoder(x)
     x = self.logits(x)
     x = self.index_select_outputs(x, labels_cap_soft)
     #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

     return x, internal_losses



class JointLateClusterSoftTransformer3_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
      #Encoder
      self.nhead = 8
      self.nhid = 3
      self.ninp = E

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert']:
          text_key=key
      if text_key:
        text_channels = kwargs['shape'][text_key][-1]
        self.text_encoder = TransfomerEncoder_WordPOS(time_steps=time_steps, in_channels=text_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
      else:
        self.text_encoder = TransfomerEncoder_WordPOS(time_steps=time_steps, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            x[i] = self.text_encoder(x[i], y, **kwargs).transpose(1, 2) ## (B, channels, time)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses


class JointLateClusterSoftTransformer4_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
      #Encoder
      self.nhead = 8
      self.nhid = 3
      self.ninp = E

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert']:
          text_key=key
      if text_key:
        text_channels = kwargs['shape'][text_key][-1]
        self.text_encoder = TransfomerEncoder_Multi(time_steps=time_steps, in_channels=text_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
      else:
        self.text_encoder = TransfomerEncoder_Multi(time_steps=time_steps, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            x[i] = self.text_encoder(x[i], y, **kwargs).transpose(1, 2) ## (B, channels, time)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

class JointLateClusterSoftTransformer5_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
      #Encoder
      self.nhead = 8
      self.nhid = 3
      self.ninp = E

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert']:
          text_key=key
      if text_key:
        text_channels = kwargs['shape'][text_key][-1]
        self.text_encoder = TransfomerEncoder(time_steps=time_steps, in_channels=text_channels, out_feats=out_feats, p=p, E= self.ninp, nhead = self.nhead, **kwargs)
        self.text_f_encoder = TransfomerEncoder_WordPOS(time_steps=time_steps, in_channels=self.ninp, out_feats=out_feats, p=p, E=self.ninp, nhead = self.nhead, **kwargs)
      else:
        self.text_encoder = TransfomerEncoder(time_steps=time_steps, out_feats=out_feats, p=p, E= self.ninp, nhead = self.nhead, **kwargs)
        self.text_f_encoder = TransfomerEncoder_WordPOS(time_steps=time_steps, in_channels=self.ninp, out_feats=out_feats, p=p, E=self.ninp, nhead = self.nhead, **kwargs)


      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            x[i] = self.text_encoder(x[i], y, input_repeat =0, output_repeat =1, **kwargs) ## (B, channels, time)
            x[i] = self.text_f_encoder(x[i], y, **kwargs).transpose(1, 2)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

class JointLateClusterSoftTransformer6_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0.5, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E),
        #                                    Transpose([-1, -2]),
        #                                    TextEncoder1D(output_feats = time_steps,
        #                                                  input_channels = E,
        #                                                  p=p)])

      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

      # Late Fusion with Joint
      ## Joint training intially helps train the classify_cluster model
      ## using pose as inputs, after a while when the generators have
      ## been pushed in the direction of learning the corresposing modes,
      ## we transition to speech and text as input.
      #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      if False:
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            for te in self.text_encoder:
              x[i] = te(x[i], y, input_repeat =0, output_repeat =1, **kwargs) ## (B, channels, time)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses


class JointLateClusterSoftTransformer7_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        #self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E),
                                           #Transpose([-1, -2]),
                                           ConvNormRelu(in_channels=E, out_channels=E, p=p)])

      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

      # Late Fusion with Joint
      ## Joint training intially helps train the classify_cluster model
      ## using pose as inputs, after a while when the generators have
      ## been pushed in the direction of learning the corresposing modes,
      ## we transition to speech and text as input.
      #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      if False:
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            for te in self.text_encoder:
              x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =1, **kwargs) ## (B, channels, time)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
Multi-Scale Bert Encoder Concatenated with Audio Features
'''
class JointLateClusterSoftTransformer8_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        #self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
                                           #Transpose([-1, -2]),
                                           ConvNormRelu(in_channels=E, out_channels=E, p=p)])

      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))



      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

      # Late Fusion with Joint
      ## Joint training intially helps train the classify_cluster model
      ## using pose as inputs, after a while when the generators have
      ## been pushed in the direction of learning the corresposing modes,
      ## we transition to speech and text as input.
      #if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      if False:
        x = self.pose_encoder(y, time_steps)
      else:
        for i, modality in enumerate(kwargs['input_modalities']):
          if modality.split('/')[0] == "text":
            #.set_trace() #self.training FALSE
            for te in self.text_encoder:
              x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =1, **kwargs) ## (B, channels, time)

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
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses


'''
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
'''
class JointLateClusterSoftTransformer9_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2)

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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
          x = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, **kwargs)
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
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is repeated, hence memory and tgt both are of the length 64
'''
class JointLateClusterSoftTransformer10_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2)

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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
              x[i] = te(x=x[i], y=y, input_repeat =0, output_repeat =1, 
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
          x = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=1, **kwargs)
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
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
src_mask is not provided, hence alignment is learnt automatically
'''
class JointLateClusterSoftTransformer11_G(JointLateClusterSoftTransformer9_G):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, num_clusters=num_clusters,
                       p=p, E=E, **kwargs)
    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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
          x = self.concat_encoder(tgt=tgt, memory=memory, y=y, input_repeat=0, src_mask=False, **kwargs)
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
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses


'''
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
multimodal fusion is done both ways, Audio -> Text and Text -> Audio
'''
class JointLateClusterSoftTransformer12_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
      self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))


      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # Q_t, K_a, V_a -> (B, N, C)
      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # 

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
nhid = 256, rest is the same as JointLateClusterSoftTransformer12_G
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
multimodal fusion is done both ways, Audio -> Text and Text -> Audio
'''
class JointLateClusterSoftTransformer13_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, nhid=256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                type='1d', leaky=True, downsample=False,
                                                                p=p, groups=self.num_clusters)
                                                   for i in range(4)]))

      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=nhid, nlayer=2) # Q_a, K_t, V_t -> (B, T, C)
      self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))


      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # Q_t, K_a, V_a -> (B, N, C)
      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # 

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
same as JointLateClusterSoftTransformer12_G except being conditioned on pose/startsC
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
multimodal fusion is done both ways, Audio -> Text and Text -> Audio
'''
class JointLateClusterSoftTransformer14_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)
      self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

      #self.logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
      # self.decoder = BatchGroup([ConvNormRelu(in_channels, in_channels,
      #                                         type='1d', leaky=True, downsample=False,
      #                                         p=p, groups=self.num_clusters)
      #                            for i in range(4)] + [self.logits], groups=self.num_clusters)
      #self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
      #                                                          type='1d', leaky=True, downsample=False,
      #                                                          p=p, groups=self.num_clusters)
      #                                             for i in range(4)]))
      self.decoder = PoseDecoder(input_channels=in_channels,
                                 style_dim=1, ## to get the pose/startsC at every decoder layer
                                 num_clusters=self.num_clusters,
                                 out_feats=out_feats)
      
      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
      self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))


      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # Q_t, K_a, V_a -> (B, N, C)
      #self.concat_encoder_audio = MultimodalTransformerFusion(out_feats=128, nhid=2) # 

      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()
      self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
      self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                   downsample=False, p=p)

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
      internal_losses = []
      labels = x[-1] ## remove the labels attached to the inputs
      x = x[:-1]

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

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))

      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)

      ## add pose/startsC
      startsC = kwargs['pose/startsC']
      if kwargs['sample_flag']:
        startsC = startsC.view(1,-1)
      x = torch.cat([x, startsC[:, None, :].float()], dim=1)
      
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      #x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses

'''
same as JointLateClusterSoftTransformer12_G except being conditioned on pose/startsC
Multi-Scale Bert with MultimodalFusion Transformer for concatenation
bert output is not repeated, hence it is a subword attention over the audio signals
multimodal fusion is done both ways, Audio -> Text and Text -> Audio
'''
class JointLateClusterSoftTransformer15_G(nn.Module):
    def __init__(self, time_steps=64, in_channels=256, out_feats=96, num_clusters=8, p = 0, E = 256, **kwargs):
      super().__init__()
      self.num_clusters = num_clusters
      self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

      text_key = None
      for key in kwargs['shape']:
        if key in ['text/w2v', 'text/bert', 'text/tokens']:
          text_key=key
      if text_key:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=E)])
        # self.text_encoder = nn.ModuleList([MultiScaleBertEncoder(out_feats=E),
        #                                    #Transpose([-1, -2]),
        #                                    ConvNormRelu(in_channels=E, out_channels=E, p=p)])
      self.pos_encoder = PositionalEncoding(256, p)
      self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p)

      self.decoder = PoseDecoder(input_channels=in_channels,
                                 style_dim=1, ## to get the pose/startsC at every decoder layer
                                 num_clusters=self.num_clusters,
                                 out_feats=out_feats)
      
      self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
      self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                       type='1d', leaky=True, downsample=False,
                                                                       p=p)]))


      self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
      self.classify_loss = nn.CrossEntropyLoss()

      self.thresh = Curriculum(0, 1, 1000)
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

    def forward(self, x, y, time_steps=None, **kwargs):
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

      ## Classify clusters using audio/text
      labels_score = self.classify_cluster(x).transpose(2, 1)
      internal_losses.append(self.classify_loss(labels_score.reshape(-1, labels_score.shape[-1]), labels.reshape(-1)))
      #_, labels_cap = labels_score.max(dim=-1)
      labels_cap_soft = torch.nn.functional.softmax(labels_score, dim=-1)
      self.labels_cap_soft = labels_cap_soft

      ## decode
      #[[x]] = self.decoder([[x]]*self.num_clusters, labels=[labels_cap_soft], transpose=False)
      #x = x.transpose(-2, -1)

      ## add pose/startsC
      startsC = kwargs['pose/startsC']
      if kwargs['sample_flag']:
        startsC = startsC.view(1,-1)
      x = torch.cat([x, startsC[:, None, :].float()], dim=1)
      
      #repeat inputs before decoder
      x = torch.cat([x]*self.num_clusters, dim=1)

      x = self.decoder(x)
      #x = self.logits(x)
      x = self.index_select_outputs(x, labels_cap_soft)
      #x = self.smoothen(x.transpose(2, 1)).transpose(2, 1)

      return x, internal_losses
