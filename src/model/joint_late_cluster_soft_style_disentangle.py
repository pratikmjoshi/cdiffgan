import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D, Speech2Gesture2_D, Speech2Gesture3_D
from .gan import GANClassify

import torch
import torch.nn as nn

JointLateClusterSoftStyleDisentangle_D = Speech2Gesture_D
JointLateClusterSoftStyleDisentangle2_D = Speech2Gesture_D
JointLateClusterSoftStyleDisentangle3_D = Speech2Gesture_D
JointLateClusterSoftStyleDisentangle4_D = Speech2Gesture_D
JointLateClusterSoftStyleDisentangle5_D = Speech2Gesture_D
JointLateClusterSoftStyleDisentangle6_D = Speech2Gesture2_D
JointLateClusterSoftStyleDisentangle7_D = Speech2Gesture3_D

class JointLateClusterSoftStyleDisentangle_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']
    # self.losses =  {'id_a':1, 'id_p':1, 
    #                 'style_a':1, 'style_p':1, 
    #                 'content_+':1, 'content_-':1, 
    #                 'rec_a':1, 'rec_p':1,
    #                 'lat_a':1, 'lat_p':1}

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
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
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.pose_encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.pose_encoder_gr = Group(self.pose_encoder, groups=2, dim=1)
    self.audio_encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.audio_encoder_gr = Group(self.audio_encoder, groups=2, dim=1)    

    ## Style ID
    self.pose_style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=1), nn.Softmax(dim=1))
    self.pose_style_id_gr = Group(self.pose_style_id, groups=2, dim=0)
    self.audio_style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=1), nn.Softmax(dim=1))
    self.audio_style_id_gr = Group(self.audio_style_id, groups=2, dim=0)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                 embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)
    
    ## Decoder for Style Embedding
    self.pose_unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=style_dim)
    self.pose_unet_gr = Group(self.pose_unet, groups=style_dim, dim=1)
    self.audio_unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=style_dim)
    self.audio_unet_gr = Group(self.audio_unet, groups=style_dim, dim=1)

    ## Classify cluster
    self.pose_classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
    self.audio_classify_cluster = ClusterClassify(num_clusters=self.num_clusters)
    self.pose_classify_cluster_gr = Group(self.pose_classify_cluster, groups=2, dim=0)
    self.audio_classify_cluster_gr = Group(self.audio_classify_cluster, groups=2, dim=0)
    self.softmax_cluster = nn.Softmax(dim=1)
    self.softmax_cluster_gr = Group(self.softmax_cluster, groups=4, dim=0)
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                   type='1d', leaky=True, downsample=False,
                                                                   p=p, groups=self.num_clusters)
                                                      for i in range(4)]))
    self.audio_decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                                    type='1d', leaky=True, downsample=False,
                                                                    p=p, groups=self.num_clusters)
                                                       for i in range(4)]))
    self.pose_logits = nn.Conv1d(in_channels*self.num_clusters, out_feats*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)
    self.audio_logits = nn.Conv1d(in_channels*self.num_clusters, self.audio_shape*self.num_clusters, kernel_size=1, stride=1, groups=self.num_clusters)

    self.pose_decoder_gr = Group([self.pose_decoder, self.pose_logits], groups=self.num_clusters, dim=1)
    self.audio_decoder_gr = Group([self.audio_decoder, self.audio_logits], groups=self.num_clusters, dim=1)

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
                                 downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.mse(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.mse(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]
    
  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]

    ## Pre-encoder
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    Pc, Ps = self.pose_encoder_gr([Ppre]*2)
    Pc_, Ps_ = self.pose_encoder_gr([Ppre_]*2)
    Ac, As = self.audio_encoder_gr([Apre]*2)
    Ac_, As_ = self.audio_encoder_gr([Apre_]*2)

    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    
    #if self.training:
    if torch.rand(1).item() > self.thresh.step(self.training) and self.training:
      Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])
      Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      
      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode='emb')
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode='emb')
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    PcAs_ = self.audio_unet_gr([Pc]*self.style_dim, transpose=False, labels=As_)
    PcAs = self.audio_unet_gr([Pc]*self.style_dim, transpose=False, labels=As)
    AcPs = self.pose_unet_gr([Ac]*self.style_dim, transpose=False, labels=Ps)
    AcPs_ = self.pose_unet_gr([Ac]*self.style_dim, transpose=False, labels=Ps_)

    score_PcAs_, score_PcAs = self.audio_classify_cluster_gr([PcAs_, PcAs], transpose=False)
    score_AcPs, score_AcPs_ = self.pose_classify_cluster_gr([AcPs, AcPs_], transpose=False)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly represent modes of the audio distribution
                                         [score_PcAs_, score_PcAs],
                                         [labels_, labels],
                                         self.cluster_loss))
    
    # internal_losses.append(self.get_loss('cluster_p',
    #                                      [score_AcPs, score_AcPs_],
    #                                      [labels, labels_],
    #                                      self.cluster_loss))
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss))

    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.softmax_cluster_gr([score_PcAs_, score_PcAs, score_AcPs, score_AcPs_], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs
    
    Acap1_ = self.audio_decoder_gr([PcAs_]*self.num_clusters, transpose=False, labels=labels_PcAs_)
    Acap1 = self.audio_decoder_gr([PcAs]*self.num_clusters, transpose=False, labels=labels_PcAs)
    Acap2 = self.audio_decoder_gr([AcPs]*self.num_clusters, transpose=False, labels=labels_AcPs)
    Acap2_ = self.audio_decoder_gr([AcPs_]*self.num_clusters, transpose=False, labels=labels_AcPs_)

    Pcap1_ = self.pose_decoder_gr([PcAs_]*self.num_clusters, transpose=False, labels=labels_PcAs_)
    Pcap1 = self.pose_decoder_gr([PcAs]*self.num_clusters, transpose=False, labels=labels_PcAs)
    Pcap2 = self.pose_decoder_gr([AcPs]*self.num_clusters, transpose=False, labels=labels_AcPs)
    Pcap2_ = self.pose_decoder_gr([AcPs_]*self.num_clusters, transpose=False, labels=labels_AcPs_)

    ## L_style
    internal_losses.append(self.get_loss('style_a',
                                         [Acap1_, Acap1, Acap2, Acap2_],
                                         [A_, A, A, A_],
                                         self.style_loss,
                                         models=self.audio_style_R_gr))
    internal_losses.append(self.get_loss('style_p',
                                         [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                         [P_, P, P, P_],
                                         self.style_loss,
                                         models=self.pose_style_R_gr))

    ## L_rec
    internal_losses.append(self.get_loss('rec_a',
                                         [Acap1, Acap2],
                                         [A, A],
                                         self.rec_loss))
    internal_losses.append(self.get_loss('rec_p',
                                         [Pcap1],
                                         [P],
                                         self.rec_loss))
    return Pcap2.transpose(-1, -2), internal_losses
    
  def forward1(self, x, y, time_steps=None, **kwargs):
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

class JointLateClusterSoftStyleDisentangle2_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
    # text_key = None
    # for key in kwargs['shape']:
    #   if key in ['text/w2v', 'text/bert']:
    #     text_key=key
    # if text_key:
    #   text_channels = kwargs['shape'][text_key][-1]
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps,
    #                                     input_channels = text_channels,
    #                                     p=p)
    # else:
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.encoder_gr = BatchGroup(self.encoder, groups=2)

    ## Encoder for Style and Content Embedding
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=2)
    self.unet_gr = BatchGroup(self.unet, groups=2)

    ## Style ID
    self.style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=2),
                                  nn.AvgPool1d(2), 
                                  Repeat(64, dim=-1)) ## 64= length of a sample
    self.style_id_gr = BatchGroup(self.style_id, groups=2)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                 embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)

    ## Classify cluster for Decoder
    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=2,
                                            input_channels=256+self.style_dim)
    self.classify_cluster_gr = BatchGroup(self.classify_cluster, groups=2)
    self.softmax = nn.Softmax(dim=1)
    self.softmax_gr = BatchGroup(self.softmax, groups=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                    num_clusters=self.num_clusters, out_feats=out_feats)
    self.pose_decoder_gr = BatchGroup(self.pose_decoder, groups=self.num_clusters)
    self.audio_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                     num_clusters=self.num_clusters, out_feats=self.audio_shape)
    self.audio_decoder_gr = BatchGroup(self.audio_decoder, groups=self.num_clusters)
  

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.mse(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.mse(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]

  def encode(self, P, P_, A, A_, style, style_, internal_losses=[]):
    ## Pre-encoder
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)
    
    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    #if self.training:
    if torch.rand(1).item() < self.thresh.step(self.training) and self.training:
      #Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])

      [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                                 [As, As_]],
                                                transpose=False)
      # Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      # As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      if len(style.shape) == 2:
        style = style.view(Ac.shape[0], Ac.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1])
        mode = 'emb'
      elif len(style.shape) == 3: ## used while training for out-of-domain style embeddings
        style = style.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        mode = 'lin'
        
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode=mode)
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode=mode)
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    Ps, Ps_, As, As_ = self.transpose_batch([Ps, Ps_, As, As_])
    
    return [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses

  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    self.device = A.device
    ## encoding style and content
    [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses = self.encode(P, P_,
                                                                          A, A_,
                                                                          style, style_,
                                                                          internal_losses)

    ## classify cluster
    PcAs_ = torch.cat([Pc, As_], dim=1)
    PcAs = torch.cat([Pc, As], dim=1)
    AcPs = torch.cat([Ac, Ps], dim=1)
    AcPs_ = torch.cat([Ac, Ps_], dim=1)

    [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[PcAs_, PcAs],
                                                                                       [AcPs, AcPs_]],
                                                                                      transpose=False)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly repres ent modes of the audio distribution
                                         [score_PcAs],
                                         [labels],
                                         self.cluster_loss))
    
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss))

    [[labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_]] = self.softmax_gr([[score_PcAs_, score_PcAs, score_AcPs, score_AcPs_]], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs

    if torch.rand(1).item() > 0.5 and self.training:
      [[Acap1_, Acap1, Acap2, Acap2_]] = self.audio_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                               transpose=False,
                                                               labels = [labels_PcAs_,
                                                                         labels_PcAs,
                                                                         labels_AcPs,
                                                                         labels_AcPs_])
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])

      ## L_style
      internal_losses.append(self.get_loss('style_a',
                                           [Acap1_, Acap1, Acap2, Acap2_],
                                           [A_, A, A, A_],
                                           self.style_loss,
                                           models=self.audio_style_R_gr))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

      internal_losses.append(self.get_loss('rec_a',
                                           [Acap1, Acap2],
                                           [A, A],
                                           self.rec_loss))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

    elif self.training:
      [[Pcap1_, Pcap1, Pcap2, Pcap2_]] = self.pose_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                              transpose=False,
                                                              labels = [labels_PcAs_,
                                                                        labels_PcAs,
                                                                        labels_AcPs,
                                                                        labels_AcPs_])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('style_p',
                                           [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                           [P_, P, P, P_],
                                           self.style_loss,
                                           models=self.pose_style_R_gr))

      ## L_rec
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('rec_p',
                                           [Pcap1],
                                           [P],
                                           self.rec_loss))
    else:
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      
    return Pcap2.transpose(-1, -2), internal_losses


class JointLateClusterSoftStyleDisentangle3_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
    # text_key = None
    # for key in kwargs['shape']:
    #   if key in ['text/w2v', 'text/bert']:
    #     text_key=key
    # if text_key:
    #   text_channels = kwargs['shape'][text_key][-1]
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps,
    #                                     input_channels = text_channels,
    #                                     p=p)
    # else:
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.encoder_gr = BatchGroup(self.encoder, groups=2)

    ## Encoder for Style and Content Embedding
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=2)
    self.unet_gr = BatchGroup(self.unet, groups=2)

    ## Style ID
    self.style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=2),
                                  nn.AvgPool1d(2), 
                                  Repeat(64, dim=-1)) ## 64= length of a sample
    self.style_id_gr = BatchGroup(self.style_id, groups=2)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                                  embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)

    ## Classify cluster for Decoder
    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=2,
                                            input_channels=256)
    self.classify_cluster_gr = BatchGroup(self.classify_cluster, groups=2)
    self.softmax = nn.Softmax(dim=1)
    self.softmax_gr = BatchGroup(self.softmax, groups=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                    num_clusters=self.num_clusters, out_feats=out_feats)
    self.pose_decoder_gr = BatchGroup(self.pose_decoder, groups=self.num_clusters)
    self.audio_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                     num_clusters=self.num_clusters, out_feats=self.audio_shape)
    self.audio_decoder_gr = BatchGroup(self.audio_decoder, groups=self.num_clusters)
  

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.l1(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]

  def get_style(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    self.device = A.device
    ## encoding style and content
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)

    [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                               [As, As_]],
                                                transpose=False)

    return Ps
    
  def encode(self, P, P_, A, A_, style, style_, internal_losses=[]):
    ## Pre-encoder
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)
    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    #if self.training:
    if torch.rand(1).item() < self.thresh.step(self.training) and self.training:
      #Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])

      [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                                 [As, As_]],
                                                transpose=False)
      # Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      # As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      if len(style.shape) == 2:
        style = style.view(Ac.shape[0], Ac.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1])
        mode = 'emb'
      elif len(style.shape) == 3: ## used while training for out-of-domain style embeddings
        style = style.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        mode = 'lin'
        
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode=mode)
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode=mode)
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    Ps, Ps_, As, As_ = self.transpose_batch([Ps, Ps_, As, As_])
    
    return [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses

  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    self.device = A.device
    ## encoding style and content
    [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses = self.encode(P, P_,
                                                                          A, A_,
                                                                          style, style_,
                                                                          internal_losses)

    ## classify cluster
    PcAs_ = torch.cat([Pc, As_], dim=1)
    PcAs = torch.cat([Pc, As], dim=1)
    AcPs = torch.cat([Ac, Ps], dim=1)
    AcPs_ = torch.cat([Ac, Ps_], dim=1)

    # [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[PcAs_, PcAs],
    #                                                                                    [AcPs, AcPs_]],
    #                                                                                   transpose=False)
    [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[Pc_, Pc],
                                                                                       [Ac, Ac_]],
                                                                                      transpose=False)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly repres ent modes of the audio distribution
                                         [score_PcAs],
                                         [labels],
                                         self.cluster_loss))
    
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss))

    [[labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_]] = self.softmax_gr([[score_PcAs_, score_PcAs, score_AcPs, score_AcPs_]], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs

    if torch.rand(1).item() > 0.5 and self.training:
      [[Acap1_, Acap1, Acap2, Acap2_]] = self.audio_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                               transpose=False,
                                                               labels = [labels_PcAs_,
                                                                         labels_PcAs,
                                                                         labels_AcPs,
                                                                         labels_AcPs_])
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])

      ## L_style
      internal_losses.append(self.get_loss('style_a',
                                           [Acap1_, Acap1, Acap2, Acap2_],
                                           [A_, A, A, A_],
                                           self.style_loss,
                                           models=self.audio_style_R_gr))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

      internal_losses.append(self.get_loss('rec_a',
                                           [Acap1, Acap2],
                                           [A, A],
                                           self.rec_loss))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

    elif self.training:
      [[Pcap1_, Pcap1, Pcap2, Pcap2_]] = self.pose_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                              transpose=False,
                                                              labels = [labels_PcAs_,
                                                                        labels_PcAs,
                                                                        labels_AcPs,
                                                                        labels_AcPs_])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('style_p',
                                           [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                           [P_, P, P, P_],
                                           self.style_loss,
                                           models=self.pose_style_R_gr))

      ## L_rec
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('rec_p',
                                           [Pcap1],
                                           [P],
                                           self.rec_loss))
    else:
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      
    return Pcap2.transpose(-1, -2), internal_losses


class JointLateClusterSoftStyleDisentangle4_G(nn.Module):
  '''
  GAN loss for cluster classification 

  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
    # text_key = None
    # for key in kwargs['shape']:
    #   if key in ['text/w2v', 'text/bert']:
    #     text_key=key
    # if text_key:
    #   text_channels = kwargs['shape'][text_key][-1]
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps,
    #                                     input_channels = text_channels,
    #                                     p=p)
    # else:
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.encoder_gr = BatchGroup(self.encoder, groups=2)

    ## Encoder for Style and Content Embedding
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=2)
    self.unet_gr = BatchGroup(self.unet, groups=2)

    ## Style ID
    self.style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=2),
                                  nn.AvgPool1d(2), 
                                  Repeat(64, dim=-1)) ## 64= length of a sample
    self.style_id_gr = BatchGroup(self.style_id, groups=2)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                                  embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)

    ## Classify cluster for Decoder
    self.pose_classify_cluster = ClusterClassifyGAN(num_clusters=self.num_clusters, groups=1,
                                                    input_channels=256+self.style_dim)
    self.D_classify_cluster = Speech2Gesture_D(self.num_clusters)
    self.pose_classify_cluster_gan = GANClassify(self.pose_classify_cluster, self.D_classify_cluster,
                                                 criterion='L1Loss')
    
    self.audio_classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=1,
                                                  input_channels=256+self.style_dim)
    self.audio_classify_cluster_gr = BatchGroup(self.audio_classify_cluster, groups=1)
    self.softmax = nn.Softmax(dim=1)
    self.softmax_gr = BatchGroup(self.softmax, groups=1)

    # self.cluster_classify_gr_gan = GANClassify(self.classify_cluster_gr,
    #                                            Speech2Gesture_D(self.num_clusters),
    #                                            criterion='CrossEntropyLoss')
    
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                    num_clusters=self.num_clusters, out_feats=out_feats)
    self.pose_decoder_gr = BatchGroup(self.pose_decoder, groups=self.num_clusters)
    self.audio_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                     num_clusters=self.num_clusters, out_feats=self.audio_shape)
    self.audio_decoder_gr = BatchGroup(self.audio_decoder, groups=self.num_clusters)
  

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.l1(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]

  def get_style(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    self.device = A.device
    ## encoding style and content
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)

    [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                               [As, As_]],
                                                transpose=False)

    return Ps
    
  def encode(self, P, P_, A, A_, style, style_, internal_losses=[]):
    ## Pre-encoder
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)
    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    #if self.training:
    if torch.rand(1).item() < self.thresh.step(self.training) and self.training:
      #Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])

      [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                                 [As, As_]],
                                                transpose=False)
      # Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      # As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      if len(style.shape) == 2:
        style = style.view(Ac.shape[0], Ac.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1])
        mode = 'emb'
      elif len(style.shape) == 3: ## used while training for out-of-domain style embeddings
        style = style.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        mode = 'lin'
        
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode=mode)
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode=mode)
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    Ps, Ps_, As, As_ = self.transpose_batch([Ps, Ps_, As, As_])
    
    return [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses

  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])
    self.device = A.device
    ## encoding style and content
    [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses = self.encode(P, P_,
                                                                          A, A_,
                                                                          style, style_,
                                                                          internal_losses)

    ## classify cluster
    PcAs_ = torch.cat([Pc, As_], dim=1)
    PcAs = torch.cat([Pc, As], dim=1)
    AcPs = torch.cat([Ac, Ps], dim=1)
    AcPs_ = torch.cat([Ac, Ps_], dim=1)

    # [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[PcAs_, PcAs],
    #                                                                                    [AcPs, AcPs_]],
    #                                                                                   transpose=False)
    # [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[Pc_, Pc],
    #                                                                                    [Ac, Ac_]],
    #                                                                                   transpose=False)
    [[score_PcAs_, score_PcAs]] = self.audio_classify_cluster_gr([[PcAs_, PcAs]],
                                                                 transpose=False)
    score_AcPs_, _ = self.pose_classify_cluster(AcPs_, None)
    score_AcPs_ = score_AcPs_.transpose(-2, -1)
    
    labels_onehot = torch.zeros(labels.shape + torch.Size([self.num_clusters])).to(labels.device).scatter(-1, labels.unsqueeze(-1), 1).float()
    score_AcPs, partial_i_loss = self.pose_classify_cluster_gan(AcPs, labels_onehot)
    score_AcPs = score_AcPs.transpose(-2, -1)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly repres ent modes of the audio distribution
                                         [score_PcAs],
                                         [labels],
                                         self.cluster_loss))
    
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss) + sum(partial_i_loss))
    
    [[labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_]] = self.softmax_gr([[score_PcAs_, score_PcAs, score_AcPs, score_AcPs_]], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs

    if torch.rand(1).item() > 0.5 and self.training:
      [[Acap1_, Acap1, Acap2, Acap2_]] = self.audio_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                               transpose=False,
                                                               labels = [labels_PcAs_,
                                                                         labels_PcAs,
                                                                         labels_AcPs,
                                                                         labels_AcPs_])
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])

      ## L_style
      internal_losses.append(self.get_loss('style_a',
                                           [Acap1_, Acap1, Acap2, Acap2_],
                                           [A_, A, A, A_],
                                           self.style_loss,
                                           models=self.audio_style_R_gr))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

      internal_losses.append(self.get_loss('rec_a',
                                           [Acap1, Acap2],
                                           [A, A],
                                           self.rec_loss))
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

    elif self.training:
      [[Pcap1_, Pcap1, Pcap2, Pcap2_]] = self.pose_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                              transpose=False,
                                                              labels = [labels_PcAs_,
                                                                        labels_PcAs,
                                                                        labels_AcPs,
                                                                        labels_AcPs_])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('style_p',
                                           [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                           [P_, P, P, P_],
                                           self.style_loss,
                                           models=self.pose_style_R_gr))

      ## L_rec
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('rec_p',
                                           [Pcap1],
                                           [P],
                                           self.rec_loss))
    else:
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      
    return Pcap2.transpose(-1, -2), internal_losses


class JointLateClusterSoftStyleDisentangle5_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
    # text_key = None
    # for key in kwargs['shape']:
    #   if key in ['text/w2v', 'text/bert']:
    #     text_key=key
    # if text_key:
    #   text_channels = kwargs['shape'][text_key][-1]
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps,
    #                                     input_channels = text_channels,
    #                                     p=p)
    # else:
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.encoder_gr = BatchGroup(self.encoder, groups=2)

    ## Encoder for Style and Content Embedding
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=2)
    self.unet_gr = BatchGroup(self.unet, groups=2)

    ## Style ID
    self.style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=2),
                                  nn.AvgPool1d(2), 
                                  Repeat(64, dim=-1)) ## 64= length of a sample
    self.style_id_gr = BatchGroup(self.style_id, groups=2)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                                  embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)

    ## Style Decoder
    self.pose_style_dec = StyleDecoder(256, self.style_dim, out_feats=256)
    self.pose_style_dec_gr = BatchGroup(self.pose_style_dec, groups=self.style_dim)
    self.audio_style_dec = StyleDecoder(256, self.style_dim, out_feats=256)
    self.audio_style_dec_gr = BatchGroup(self.audio_style_dec, groups=self.style_dim)
    
    ## Classify cluster for Decoder
    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=2,
                                            input_channels=256)
    self.classify_cluster_gr = BatchGroup(self.classify_cluster, groups=2)
    self.softmax = nn.Softmax(dim=1)
    self.softmax_gr = BatchGroup(self.softmax, groups=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                    num_clusters=self.num_clusters, out_feats=out_feats)
    self.pose_decoder_gr = BatchGroup(self.pose_decoder, groups=self.num_clusters)
    self.audio_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                     num_clusters=self.num_clusters, out_feats=self.audio_shape)
    self.audio_decoder_gr = BatchGroup(self.audio_decoder, groups=self.num_clusters)
  

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)

  def style_loss2(self, X, Y, **kwargs):
    with torch.no_grad():
      XY = self.pose_preencoder_gr(X+Y)
      [[X1_, X1, X2, X2_, Y1_, Y1, Y2, Y2_], _] = self.encoder_gr([XY, XY])
      #[Y_, _] = self.encoder_gr([Y, Y])
      # [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
      #                                       [Apre, Apre_]])

      ## Encoder Style and content
      inputs = [X1_, X1, X2, X2_, Y1_, Y1, Y2, Y2_]
      _, [X_cap, _] = self.unet_gr([inputs,
                                    inputs],
                                   return_bottleneck=True,
                                   transpose=False)

      style_losses = []
      for x, y in zip(X_cap[:4], X_cap[4:]):
        style_losses.append(self.l1(x, y))

    return sum(style_losses)
      #style_losses

      # [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
      #                                                                [A, A_]],
      #                                                               return_bottleneck=True,
      #                                                               transpose=False)
    
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.l1(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]
    
  def encode(self, P, P_, A, A_, style, style_, internal_losses=[]):
    ## Pre-encoder
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)
    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    #if self.training:
    if torch.rand(1).item() < self.thresh.step(self.training) and self.training:
      #Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])

      [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                                 [As, As_]],
                                                transpose=False)
      # Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      # As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      if len(style.shape) == 2:
        style = style.view(Ac.shape[0], Ac.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1])
        mode = 'emb'
      elif len(style.shape) == 3: ## used while training for out-of-domain style embeddings
        style = style.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        mode = 'lin'
        
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode=mode)
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode=mode)
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    Ps, Ps_, As, As_ = self.transpose_batch([Ps, Ps_, As, As_])
    
    return [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses

  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])

    self.device = A.device
    ## encoding style and content
    [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses = self.encode(P, P_,
                                                                          A, A_,
                                                                          style, style_,
                                                                          internal_losses)
    ## Encode content with style
    [[AcPs, AcPs_]] = self.pose_style_dec_gr([[Ac, Ac]] * self.style_dim,
                                             labels=[Ps, Ps_],
                                             transpose=False)
    [[PcAs, PcAs_]] = self.audio_style_dec_gr([[Pc, Pc]] * self.style_dim,
                                              labels=[As, As_],
                                              transpose=False)
    
    ## classify cluster
    # PcAs_ = torch.cat([Pc, As_], dim=1)
    # PcAs = torch.cat([Pc, As], dim=1)
    # AcPs = torch.cat([Ac, Ps], dim=1)
    # AcPs_ = torch.cat([Ac, Ps_], dim=1)

    [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[PcAs_, PcAs],
                                                                                       [AcPs, AcPs_]],
                                                                                      transpose=False)
    PcAs_ = torch.cat([PcAs_, As_], dim=1)
    PcAs = torch.cat([PcAs, As], dim=1)
    AcPs = torch.cat([AcPs, Ps], dim=1)
    AcPs_ = torch.cat([AcPs_, Ps_], dim=1)

    # [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[Pc_, Pc],
    #                                                                                    [Ac, Ac_]],
    #                                                                                   transpose=False)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly repres ent modes of the audio distribution
                                         [score_PcAs],
                                         [labels],
                                         self.cluster_loss))
    
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss))

    [[labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_]] = self.softmax_gr([[score_PcAs_, score_PcAs, score_AcPs, score_AcPs_]], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs

#    if torch.rand(1).item() > 0.5 and self.training:
      # [[Acap1_, Acap1, Acap2, Acap2_]] = self.audio_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
      #                                                          transpose=False,
      #                                                          labels = [labels_PcAs_,
      #                                                                    labels_PcAs,
      #                                                                    labels_AcPs,
      #                                                                    labels_AcPs_])
      # [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
      #                                  transpose=False,
      #                                  labels = [labels_AcPs])

      # ## L_style
      # internal_losses.append(self.get_loss('style_a',
      #                                      [Acap1_, Acap1, Acap2, Acap2_],
      #                                      [A_, A, A, A_],
      #                                      self.style_loss,
      #                                      models=self.audio_style_R_gr))
      # internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

      # internal_losses.append(self.get_loss('rec_a',
      #                                      [Acap1, Acap2],
      #                                      [A, A],
      #                                      self.rec_loss))
      # internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

    if self.training:
      [[Pcap1_, Pcap1, Pcap2, Pcap2_]] = self.pose_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                              transpose=False,
                                                              labels = [labels_PcAs_,
                                                                        labels_PcAs,
                                                                        labels_AcPs,
                                                                        labels_AcPs_])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      # internal_losses.append(self.get_loss('style_p',
      #                                      [Pcap1_, Pcap1, Pcap2, Pcap2_],
      #                                      [P_, P, P, P_],
      #                                      self.style_loss,
      #                                      models=self.pose_style_R_gr))

      internal_losses.append(self.get_loss('style_p',
                                           [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                           [P_, P, P, P_],
                                           self.style_loss2))

      ## L_rec
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(self.get_loss('rec_p',
                                           [Pcap1],
                                           [P],
                                           self.rec_loss))
    else:
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      
    return Pcap2.transpose(-1, -2), internal_losses

class JointLateClusterSoftStyleDisentangle6_G(nn.Module):
  '''
  Late Fusion with clustering in the input pose

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  input_shape pose: (N, time, pose_feats)

  output_shape pose: (N, time, pose_feats)
  output_shape audio: (N, time, frequency)

  kwargs['style_losses'] = {'id_a', 'id_p', 
                            'style_a', 'style_p', 
                            'content_a', 'content_p', 
                            'rec_a', 'rec_p',
                            'lat_a', 'lat_p'}
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, style_dict={}, style_dim=10, **kwargs):
    super().__init__()
    self.num_clusters = num_clusters
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.style_dict = style_dict
    self.style_dim = style_dim

    self.losses = kwargs['style_losses']

    audio_id = None
    for key in kwargs['shape']:
      if key.split('/')[0] == 'audio':
        audio_id = key
        break
    self.audio_shape = kwargs['shape'][audio_id][-1]
    
    # text_key = None
    # for key in kwargs['shape']:
    #   if key in ['text/w2v', 'text/bert']:
    #     text_key=key
    # if text_key:
    #   text_channels = kwargs['shape'][text_key][-1]
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps,
    #                                     input_channels = text_channels,
    #                                     p=p)
    # else:
    #   self.text_encoder = TextEncoder1D(output_feats = time_steps, p=p)
      
    ## shared weights for content and style
    self.pose_preencoder = PoseEncoder(output_feats = time_steps, input_channels=out_feats, p=p, groups=1)
    self.pose_preencoder_gr = Group(self.pose_preencoder, groups=2, dim=0)
    self.audio_preencoder = AudioEncoder(output_feats = time_steps, p=p, groups=1)
    self.audio_preencoder_gr = Group(self.audio_preencoder, groups=2, dim=0)

    ## different weights for content and style
    self.encoder = PoseEncoder(output_feats = time_steps, input_channels=256, p=p, groups=2)
    self.encoder_gr = BatchGroup(self.encoder, groups=2)

    ## Encoder for Style and Content Embedding
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p, groups=2)
    self.unet_gr = BatchGroup(self.unet, groups=2)

    ## Style ID
    self.style_id = nn.Sequential(ConvNormRelu(256, len(self.style_dict), p=p, groups=2),
                                  nn.AvgPool1d(2), 
                                  Repeat(64, dim=-1)) ## 64= length of a sample
    self.style_id_gr = BatchGroup(self.style_id, groups=2)
    
    ## Style Embedding
    self.pose_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                                  embedding_dim=style_dim)
    self.pose_style_emb_gr = Group(self.pose_style_emb, groups=2, dim=0)
    self.audio_style_emb = EmbLin(num_embeddings=len(self.style_dict),
                                  embedding_dim=style_dim)
    self.audio_style_emb_gr = Group(self.audio_style_emb, groups=2, dim=0)

    ## Style Decoder
    self.pose_style_dec = StyleDecoder(256, self.style_dim, out_feats=256)
    self.pose_style_dec_gr = BatchGroup(self.pose_style_dec, groups=self.style_dim)
    self.audio_style_dec = StyleDecoder(256, self.style_dim, out_feats=256)
    self.audio_style_dec_gr = BatchGroup(self.audio_style_dec, groups=self.style_dim)
    
    ## Classify cluster for Decoder
    self.classify_cluster = ClusterClassify(num_clusters=self.num_clusters, groups=2,
                                            input_channels=256)
    self.classify_cluster_gr = BatchGroup(self.classify_cluster, groups=2)
    self.softmax = nn.Softmax(dim=1)
    self.softmax_gr = BatchGroup(self.softmax, groups=1)
    
    self.classify_loss = nn.CrossEntropyLoss()
    
    ## Decoder for Mix-GAN
    self.pose_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                    num_clusters=self.num_clusters, out_feats=out_feats)
    self.pose_decoder_gr = BatchGroup(self.pose_decoder, groups=self.num_clusters)
    self.audio_decoder = PoseDecoder(in_channels, style_dim=self.style_dim,
                                     num_clusters=self.num_clusters, out_feats=self.audio_shape)
    self.audio_decoder_gr = BatchGroup(self.audio_decoder, groups=self.num_clusters)
  

    ## R function for L_style
    self.pose_style_R = nn.ModuleList([ConvNormRelu(out_feats, out_feats,
                                                    type='1d', leaky=True, downsample=False,
                                                    p=p, groups=1)
                                       for i in range(4)])
    self.audio_style_R = nn.ModuleList([ConvNormRelu(self.audio_shape, self.audio_shape,
                                                     type='1d', leaky=True, downsample=False,
                                                     p=p, groups=1)
                                        for i in range(4)])
    self.pose_style_R_gr = nn.ModuleList([Group(pose_style_R, groups=4, dim=0) for pose_style_R in self.pose_style_R])
    self.audio_style_R_gr = nn.ModuleList([Group(audio_style_R, groups=4, dim=0) for audio_style_R in self.audio_style_R])
    
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))
    
    self.eye = nn.Parameter(torch.eye(self.num_clusters, self.num_clusters), requires_grad=False)
    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.cluster = cluster

    self.thresh = Curriculum(0, 1, 1000)
    self.thresh2 = Curriculum(0, 1, 1000)
    self.labels_cap_soft = None

  def transpose_batch(self, X):
    return [X_.transpose(2,1) for X_ in X]

  def mse(self, x, y, reduction='mean'):
    return torch.nn.functional.mse_loss(x, y, reduction=reduction)
  
  def cce(self, x, y, reduction='mean'):
    return torch.nn.functional.cross_entropy(x, y, reduction=reduction)

  def l1(self, x, y, reduction='mean'):
    return torch.nn.functional.l1_loss(x, y, reduction=reduction)

  def style_loss2(self, X, Y, **kwargs):
    with torch.no_grad():
      XY = self.pose_preencoder_gr(X+Y)
      [[X1_, X1, X2, X2_, Y1_, Y1, Y2, Y2_], _] = self.encoder_gr([XY, XY])
      #[Y_, _] = self.encoder_gr([Y, Y])
      # [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
      #                                       [Apre, Apre_]])

      ## Encoder Style and content
      inputs = [X1_, X1, X2, X2_, Y1_, Y1, Y2, Y2_]
      _, [X_cap, _] = self.unet_gr([inputs,
                                    inputs],
                                   return_bottleneck=True,
                                   transpose=False)

      style_losses = []
      for x, y in zip(X_cap[:4], X_cap[4:]):
        style_losses.append(self.l1(x, y))

    return sum(style_losses)
      #style_losses

      # [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
      #                                                                [A, A_]],
      #                                                               return_bottleneck=True,
      #                                                               transpose=False)
    
  
  def style_loss(self, X, Y, models):
    gram_matrix = lambda x: torch.einsum('ijl, ikl -> ijk', x, x)/(x.shape[1]**2)
    gram_matrices_X, gram_matrices_Y = [], []
    for model in models:
      X = model(X, transpose=False)
      Y = model(Y, transpose=False)
      gram_matrices_X.append([gram_matrix(x) for x in X])
      gram_matrices_Y.append([gram_matrix(y) for y in Y])

    style_losses = []
    for X_list, Y_list in zip(gram_matrices_X, gram_matrices_Y):
      #style_losses_temp = []
      for x, y in zip(X_list, Y_list):
        style_losses.append(self.l1(x, y, 'sum'))
      #style_losses.append(style_losses_temp)
    return sum(style_losses)

  def id_loss(self, X, Y, **kwargs):
    losses = [self.cce(x.transpose(-1,-2).reshape(-1, x.shape[1]), y.view(-1)) for x,y in zip(X, Y)]
    return sum(losses)

  def cluster_loss(self, X, Y, **kwargs):
    return self.id_loss(X, Y, **kwargs)

  def content_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def rec_loss(self, X, Y, **kwargs):
    losses = [self.l1(x, y) for x,y in zip(X, Y)]
    return sum(losses)

  def lat_loss(self, X, Y, **kwargs):
    return self.content_loss(X, Y, **kwargs)

  def get_loss(self, key, X, Y, loss_fn, models=None):
    if key in self.losses:
      beta = self.losses[key]
      return beta * loss_fn(X, Y, models=models)
    else:
      return torch.FloatTensor([0]).to(X[0].device)[0]
    
  def encode(self, P, P_, A, A_, style, style_, internal_losses=[]):
    ## Pre-encoder
    Apre, Apre_ = A.unsqueeze(dim=1), A_.unsqueeze(dim=1)
    Ppre, Ppre_ = self.pose_preencoder_gr([P, P_])
    Apre, Apre_ = self.audio_preencoder_gr([Apre, Apre_])

    ## Encoder
    #P, P_= self.transpose_batch([P, P_])
    [[P, P_], [A, A_]] = self.encoder_gr([[Ppre, Ppre_],
                                          [Apre, Apre_]])

    ## Encoder Style and content
    [[Pc, Pc_], [Ac, Ac_]], [[Ps, Ps_], [As, As_]] = self.unet_gr([[P, P_],
                                                                   [A, A_]],
                                                                  return_bottleneck=True,
                                                                  transpose=False)
    internal_losses.append(self.get_loss('content_+',
                                         [Pc], [Ac],
                                         self.content_loss))
    internal_losses.append(self.get_loss('content_-',
                                         [Pc_], [Ac_],
                                         self.content_loss))    
    #if self.training:
    if torch.rand(1).item() < self.thresh.step(self.training) and self.training:
      #Ps, Ps_, As, As_= self.transpose_batch([Ps, Ps_, As, As_])

      [[Ps, Ps_], [As, As_]] = self.style_id_gr([[Ps, Ps_],
                                                 [As, As_]],
                                                transpose=False)
      # Ps, Ps_ = self.pose_style_id_gr([Ps, Ps_]) ## TODO, maybe audio and pose style could be the same matrix
      # As, As_ = self.audio_style_id_gr([As, As_])

      internal_losses.append(self.get_loss('id_a',
                                         [As, As_],
                                         [style, style_],
                                         self.id_loss))
      internal_losses.append(self.get_loss('id_p',
                                           [Ps, Ps_],
                                           [style, style_],
                                           self.id_loss))

      Ps, Ps_ = self.pose_style_emb_gr([Ps, Ps_]) 
      As, As_ = self.audio_style_emb_gr([As, As_])
    else:
      if len(style.shape) == 2:
        style = style.view(Ac.shape[0], Ac.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1])
        mode = 'emb'
      elif len(style.shape) == 3: ## used while training for out-of-domain style embeddings
        style = style.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        style_ = style_.view(Ac.shape[0], Ac.shape[-1], style.shape[-1])
        mode = 'lin'
        
      Ps, Ps_ = self.pose_style_emb_gr([style, style_], transpose=False, mode=mode)
      As, As_ = self.audio_style_emb_gr([style, style_], transpose=False, mode=mode)
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      #Ps_ = self.pose_style_emb(style_, mode='emb')
    
    Ps, Ps_, As, As_ = self.transpose_batch([Ps, Ps_, As, As_])
    
    return [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses

  def forward(self, a, p, time_steps=None, **kwargs):
    internal_losses = []
    labels = a[-1] ## remove the labels attached to the inputs
    a = a[:-1]

    audio_modality = [i for i, modality in enumerate(kwargs['input_modalities']) if modality.split('/')[0] == 'audio'][0]
    A = a[audio_modality]    
    P = p
    style = kwargs['style']
    
    if self.training:
      roll_value = get_roll_value(a, len(self.style_dict))
      a__ = roll(a, roll_value)
      P_ = roll(P, roll_value)
      style_ = roll(style, roll_value)
      labels_ = roll(labels, roll_value)
    else:
      a__ = a
      P_ = P
      style_ = style
      labels_ = labels

    A_ = a__[audio_modality]
    P, P_, A, A_= self.transpose_batch([P, P_, A, A_])

    self.device = A.device
    ## encoding style and content
    [Pc, Pc_, Ac, Ac_], [Ps, Ps_, As, As_], internal_losses = self.encode(P, P_,
                                                                          A, A_,
                                                                          style, style_,
                                                                          internal_losses)
    ## Encode content with style
    [[AcPs, AcPs_]] = self.pose_style_dec_gr([[Ac, Ac]] * self.style_dim,
                                             labels=[Ps, Ps_],
                                             transpose=False)
    [[PcAs, PcAs_]] = self.audio_style_dec_gr([[Pc, Pc]] * self.style_dim,
                                              labels=[As, As_],
                                              transpose=False)
    
    ## classify cluster
    # PcAs_ = torch.cat([Pc, As_], dim=1)
    # PcAs = torch.cat([Pc, As], dim=1)
    # AcPs = torch.cat([Ac, Ps], dim=1)
    # AcPs_ = torch.cat([Ac, Ps_], dim=1)

    [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[PcAs_, PcAs],
                                                                                       [AcPs, AcPs_]],
                                                                                      transpose=False)
    PcAs_ = torch.cat([PcAs_, As_], dim=1)
    PcAs = torch.cat([PcAs, As], dim=1)
    AcPs = torch.cat([AcPs, Ps], dim=1)
    AcPs_ = torch.cat([AcPs_, Ps_], dim=1)

    # [[score_PcAs_, score_PcAs], [score_AcPs, score_AcPs_]] = self.classify_cluster_gr([[Pc_, Pc],
    #                                                                                    [Ac, Ac_]],
    #                                                                                   transpose=False)
    
    internal_losses.append(self.get_loss('cluster_a',   ## TODO the labels do not particularly repres ent modes of the audio distribution
                                         [score_PcAs],
                                         [labels],
                                         self.cluster_loss))
    
    internal_losses.append(self.get_loss('cluster_p',
                                         [score_AcPs],
                                         [labels],
                                         self.cluster_loss))

    [[labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_]] = self.softmax_gr([[score_PcAs_, score_PcAs, score_AcPs, score_AcPs_]], transpose=False)
    labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_ = self.transpose_batch([labels_PcAs_, labels_PcAs, labels_AcPs, labels_AcPs_])
    self.labels_cap_soft = labels_AcPs

#    if torch.rand(1).item() > 0.5 and self.training:
      # [[Acap1_, Acap1, Acap2, Acap2_]] = self.audio_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
      #                                                          transpose=False,
      #                                                          labels = [labels_PcAs_,
      #                                                                    labels_PcAs,
      #                                                                    labels_AcPs,
      #                                                                    labels_AcPs_])
      # [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
      #                                  transpose=False,
      #                                  labels = [labels_AcPs])

      # ## L_style
      # internal_losses.append(self.get_loss('style_a',
      #                                      [Acap1_, Acap1, Acap2, Acap2_],
      #                                      [A_, A, A, A_],
      #                                      self.style_loss,
      #                                      models=self.audio_style_R_gr))
      # internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

      # internal_losses.append(self.get_loss('rec_a',
      #                                      [Acap1, Acap2],
      #                                      [A, A],
      #                                      self.rec_loss))
      # internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])

    if self.training:
      [[Pcap1_, Pcap1, Pcap2, Pcap2_]] = self.pose_decoder_gr([[PcAs_, PcAs, AcPs, AcPs_]]*self.num_clusters,
                                                              transpose=False,
                                                              labels = [labels_PcAs_,
                                                                        labels_PcAs,
                                                                        labels_AcPs,
                                                                        labels_AcPs_])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      # internal_losses.append(self.get_loss('style_p',
      #                                      [Pcap1_, Pcap1, Pcap2, Pcap2_],
      #                                      [P_, P, P, P_],
      #                                      self.style_loss,
      #                                      models=self.pose_style_R_gr))

      if torch.rand(1).item() <  self.thresh2.step(self.training):
        internal_losses.append(self.get_loss('style_p',
                                             [Pcap1_, Pcap1, Pcap2, Pcap2_],
                                             [P_, P, P, P_],
                                             self.style_loss2))

        ## L_rec
        internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
        internal_losses.append(self.get_loss('rec_p',
                                             [Pcap1],
                                             [P],
                                             self.rec_loss))
      else:
        internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
        internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
        internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
                
    else:
      [[Pcap2]] = self.pose_decoder_gr([[AcPs]]*self.num_clusters,
                                       transpose=False,
                                       labels = [labels_AcPs])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      internal_losses.append(torch.FloatTensor([0]).to(self.device)[0])
      
    return Pcap2.transpose(-1, -2), internal_losses


JointLateClusterSoftStyleDisentangle7_G = JointLateClusterSoftStyleDisentangle6_G
