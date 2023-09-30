import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

JointLateClusterVQ_D = Speech2Gesture_D
JointLateClusterVQ2_D = Speech2Gesture_D
JointLateClusterVQ3_D = Speech2Gesture_D

class JointLateClusterVQ_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, **kwargs):
    super().__init__()
#   vq_features = kwargs['cluster'].centers.shape[-1]
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
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]+
                                                [ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)for i in range(3)]))
    self.concat_encoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    #self.logits = nn.Conv1d(in_channels, out_feats, kernel_size=1, stride=1)
    # self.vq_precoder = ConvNormRelu(256, vq_features,
    #                                 type='1d', leaky=True, downsample=False,
    #                                 p=p)
    # self.vq = VQLayer(num_embeddings=kwargs['num_clusters'],
    #                   num_features=vq_features,
    #                   weight=kwargs['cluster'].centers)
    self.vq = VQLayer(num_embeddings=512,
                      num_features=256)

    # self.smoothen = ConvNormRelu(out_feats, out_feats, type='1d', leaky=True,
    #                              downsample=False, p=p)
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #pdb.set_trace()
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    #if random.random() > 0.5 and self.training:
    if True:
        x = self.pose_encoder(y, time_steps)
    else:
        #pdb.set_trace()
        for i, modality in enumerate(kwargs['input_modalities']):
            # UNSQUEEZE?
            #if x[i].dim() == 3:
            #    x[i] = x[i].unsqueeze(dim=1)
            if modality.split('/')[0] == "text":
                x[i] = self.text_encoder(x[i], time_steps)
            if modality.split('/')[0] == 'audio':
                if x[i].dim() == 3:
                    x[i] = x[i].unsqueeze(dim=1)
                x[i] = self.audio_encoder(x[i], time_steps)
        #pdb.set_trace()
        if len(x) >= 2:
            x = torch.cat(tuple(x[:-1]),  dim=1)
            x = self.concat_encoder(x)
        else:
            x = torch.cat(tuple(x[:-1]),  dim=1)

    #list no tuple
    x = self.unet(x)
    #pdb.set_trace()
    # = self.vq_precoder(x)
    x, dist = self.vq(x)
    x = self.decoder(x)
    #x = self.logits(x)
    #x, i_losses = self.vq(x)  ## VQLayer
    #internal_losses += i_losses
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses


class JointLateClusterVQ2_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_embeddings=512, num_features=256, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)
    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p) for i in range(3)]+
                                                [ConvNormRelu(in_channels, out_feats,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))

    self.vq = VQLayerSG(num_embeddings=num_embeddings,
                        num_features=num_features, **kwargs)
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    x = self.pose_encoder(y, time_steps)

    #list no tuple
    x = self.unet(x)
    x, idxs, i_loss = self.vq(x)
    internal_losses += i_loss
    x = self.decoder(x)
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses
  
  def encode(self, x, time_steps):
    x = self.pose_encoder(x, time_steps)
    x = self.unet(x)
    x, idxs, i_loss = self.vq(x)
    return x, idxs, i_loss

  def decode(self, logits):
    probs = F.softmax(logits, dim=-1)
    shape = probs.shape[:-1]
    probs = probs.view(-1, probs.shape[-1]) 
    idxs = probs.multinomial(1).view(shape) ## sampled idxs
    x = self.vq.emb(idxs).transpose(-2, -1) ## (B, C, T)
    x = self.decoder(x)
    x = self.smoothen(x)

    return x

  def decode_emb(self, emb):
    x = self.decoder(emb) ## (B, C, T)
    x = self.smoothen(x)

    return x

  def quantize(self, x):
    return self.vq(x)[0]
  
class JointLateClusterVQ3_G(nn.Module):
  '''
  VQ Bottleneck for each timescale 64, 32, 16, 8, 4
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_embeddings=512, max_depth=5, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.max_depth = max_depth
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet_enc = UNet1DEncoder(input_channels = in_channels, output_channels = in_channels,
                                  p=p, max_depth=max_depth)
    self.unet_dec = UNet1DDecoder(output_channels = in_channels, p=p, max_depth=max_depth)
    self.vqs = nn.ModuleList([VQLayerSG(num_embeddings=num_embeddings,
                                        num_features=256, **kwargs) for _ in range(self.max_depth)])

    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p) for i in range(3)]+
                                                [ConvNormRelu(in_channels, out_feats,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    x = self.pose_encoder(y, time_steps)

    x, residuals = self.unet_enc(x)
    i_loss = None
    residuals_quantized = []
    for depth in range(self.max_depth):
      res_, idxs_, i_loss_ = self.vqs[depth](residuals[depth])
      residuals_quantized.append(res_)
      if i_loss is None:
        i_loss = i_loss_
      else:
        for i in range(len(i_loss)):
          i_loss[i] += i_loss_[i]
    
    internal_losses += i_loss

    x = self.unet_dec(x, residuals_quantized)
    x = self.decoder(x)
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses
  
  def encode(self, x, time_steps):
    with torch.no_grad():
      x = self.pose_encoder(x, time_steps)
      x = self.unet(x)
      x, idxs, i_loss = self.vq(x)
    return x, idxs, i_loss

  def decode(self, logits):
    with torch.no_grad():
      probs = F.softmax(logits, dim=-1)
      shape = probs.shape[:-1]
      probs = probs.view(-1, probs.shape[-1]) 
      idxs = probs.multinomial(1).view(shape) ## sampled idxs
      x = self.vq.emb(idxs).transpose(-2, -1) ## (B, C, T)
      x = self.decoder(x)
      x = self.smoothen(x)

    return x
  
class JointLateClusterVQ4_G(nn.Module):
  '''
  VQ Bottleneck for each timescale 64, 32, 16, 8, 4, 2
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_embeddings=512, num_features=256, max_depth=5, residual_mask=None, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.max_depth = max_depth
    self.residual_mask = residual_mask
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet_enc = UNet1DEncoder(input_channels = in_channels, output_channels = in_channels,
                                  p=p, max_depth=max_depth)
    self.unet_dec = UNet1DDecoder(output_channels = in_channels, p=p, max_depth=max_depth)
    self.vqs = nn.ModuleList([VQLayerSG(num_embeddings=num_embeddings,
                                        num_features=num_features, **kwargs) 
                              if residual_mask[depth] else None
                              for depth in range(self.max_depth)])
    self.vqs.append(VQLayerSG(num_embeddings=num_embeddings, num_features=num_features, **kwargs))

    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p) for i in range(3)]+
                                                [ConvNormRelu(in_channels, out_feats,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    x = self.pose_encoder(y, time_steps)

    x, residuals = self.unet_enc(x)
    i_loss = None
    residuals_quantized = []
    for depth in range(self.max_depth):
      if self.residual_mask[depth]:
        res_, idxs_, i_loss_ = self.vqs[depth](residuals[depth])
        residuals_quantized.append(res_)
        if i_loss is None:
          i_loss = i_loss_
        else:
          for i in range(len(i_loss)):
            i_loss[i] += i_loss_[i]
      else:
        residuals_quantized.append(torch.zeros(1).to(y.device))
        
    ## The bottleneck is also quantized
    x_quantized, idxs, i_loss_ = self.vqs[self.max_depth](x)
    if i_loss is None:
      i_loss = i_loss_
    else:
      for i in range(len(i_loss)):
        i_loss[i] += i_loss_[i]
          
    internal_losses += i_loss

    x = self.unet_dec(x_quantized, residuals_quantized, residual_mask=self.residual_mask)
    x = self.decoder(x)
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses

  def encode(self, x, time_steps):
    x = self.pose_encoder(x, time_steps)

    x, residuals = self.unet_enc(x)
    i_loss = None
    idxs = []
    residuals_quantized = []
    for depth in range(self.max_depth):
      if self.residual_mask[depth]:
        res_, idxs_, i_loss_ = self.vqs[depth](residuals[depth])
        residuals_quantized.append(res_)
        idxs.append(idxs_)
        if i_loss is None:
          i_loss = i_loss_
        else:
          for i in range(len(i_loss)):
            i_loss[i] += i_loss_[i]
      else:
        residuals_quantized.append(torch.zeros(1).to(x.device))
        idxs.append(None)

    ## The bottleneck is also quantized
    x_quantized, idxs_, i_loss_ = self.vqs[self.max_depth](x)
    residuals_quantized.append(x_quantized)
    idxs.append(idxs_)
    if i_loss is None:
      i_loss = i_loss_
    else:
      for i in range(len(i_loss)):
        i_loss[i] += i_loss_[i]

    return residuals_quantized, idxs, i_loss

  def decode(self, logits):
    pass

  def decode_emb(self, emb):
    x = self.unet_dec(emb[-1], emb[:-1], residual_mask=self.residual_mask)
    x = self.decoder(x) ## (B, C, T)
    x = self.smoothen(x)

    return x

  def quantize(self, x):
    return [vq(x_)[0] if vq is not None else torch.zeros(1).to(x_.device)
            for vq, x_ in zip(self.vqs, x)]
  
  def decode_archive(self, logits):
    probs = F.softmax(logits, dim=-1)
    shape = probs.shape[:-1]
    probs = probs.view(-1, probs.shape[-1]) 
    idxs = probs.multinomial(1).view(shape) ## sampled idxs
    x = self.vq.emb(idxs).transpose(-2, -1) ## (B, C, T)
    x = self.decoder(x)
    x = self.smoothen(x)
    return x

class JointLateClusterVQ41_G(nn.Module):
  '''
  If audio -> pose, no vq_loss + gan 
  If pose -> pose, no gan training
  VQ Bottleneck + Joint training for each timescale 64, 32, 16, 8, 4, 2
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               num_embeddings=512, num_features=256, max_depth=5, residual_mask=None, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.num_features = num_features
    self.max_depth = max_depth
    self.residual_mask = residual_mask
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet_enc = UNet1DEncoder(input_channels = in_channels, output_channels = in_channels,
                                  p=p, max_depth=max_depth)
    self.unet_dec = UNet1DDecoder(output_channels = in_channels, p=p, max_depth=max_depth)
    self.vqs = nn.ModuleList([VQLayerSG(num_embeddings=num_embeddings,
                                        num_features=num_features, **kwargs) for _ in range(self.max_depth + 1)])

    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p) for i in range(3)]+
                                                [ConvNormRelu(in_channels, out_feats,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)

    ## Text audio encoder
    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=256)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, num_features,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    self.unet_enc_cross = UNet1DEncoder(input_channels = in_channels, output_channels = in_channels,
                                        p=p, max_depth=max_depth)

    self.thresh = Curriculum(0, 0.5, 100000)
    # self.n = 0
    # self.d = 0
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    threshold = self.thresh.step(not kwargs['sample_flag'] and kwargs['description'] == 'train')
    self.gan_flag = not (torch.rand(1)[0] > threshold and not kwargs['sample_flag'] and kwargs['description'] == 'train')
    if not self.gan_flag:
      # self.n += 1
      # self.d += 1
      x = self.pose_encoder(y, time_steps)
      x, residuals = self.unet_enc(x)
    else:
      # self.d += 1
      mod_map = {}
      for i, modality in enumerate(kwargs['input_modalities']):
        if modality.split('/')[0] == "text":
          mod_map['text'] = i
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
        x = self.concat_encoder2(x) ## B x C X T
      else:
        x = torch.cat(tuple(x),  dim=1)

      x, residuals = self.unet_enc_cross(x)
      
    i_loss = None
    residuals_quantized = []
    for depth in range(self.max_depth):
      res_, idxs_, i_loss_ = self.vqs[depth](residuals[depth])
      residuals_quantized.append(res_)
      if i_loss is None:
        i_loss = i_loss_
      else:
        for i in range(len(i_loss)):
          i_loss[i] += i_loss_[i]
          
    ## The bottleneck is also quantized
    x_quantized, idxs, i_loss_ = self.vqs[self.max_depth](x)
    for i in range(len(i_loss)):
      i_loss[i] += i_loss_[i]

    if self.gan_flag:
      i_loss[0].zero_() ## zero the vq loss for audio to pose generation
          
    internal_losses += i_loss
    #internal_losses.append(torch.FloatTensor([y.shape[0]*self.n/self.d])[0].to(x.device))
    x = self.unet_dec(x_quantized, residuals_quantized, residual_mask=self.residual_mask)
    x = self.decoder(x)
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses

  def encode(self, x, time_steps):
    x = self.pose_encoder(x, time_steps)

    x, residuals = self.unet_enc(x)
    i_loss = None
    idxs = []
    residuals_quantized = []
    for depth in range(self.max_depth):
      res_, idxs_, i_loss_ = self.vqs[depth](residuals[depth])
      residuals_quantized.append(res_)
      idxs.append(idxs_)
      if i_loss is None:
        i_loss = i_loss_
      else:
        for i in range(len(i_loss)):
          i_loss[i] += i_loss_[i]

    ## The bottleneck is also quantized
    x_quantized, idxs_, i_loss_ = self.vqs[self.max_depth](x)
    residuals_quantized.append(x_quantized)
    idxs.append(idxs_)
    for i in range(len(i_loss)):
      i_loss[i] += i_loss_[i]

    return residuals_quantized, idxs, i_loss

  def decode(self, logits):
    pass

  def decode_emb(self, emb):
    x = self.unet_dec(emb[-1], emb[:-1], residual_mask=self.residual_mask)
    x = self.decoder(x) ## (B, C, T)
    x = self.smoothen(x)

    return x

  def quantize(self, x):
    return [vq(x_)[0] for vq, x_ in zip(self.vqs, x)]
  
  def decode_archive(self, logits):
    probs = F.softmax(logits, dim=-1)
    shape = probs.shape[:-1]
    probs = probs.view(-1, probs.shape[-1]) 
    idxs = probs.multinomial(1).view(shape) ## sampled idxs
    x = self.vq.emb(idxs).transpose(-2, -1) ## (B, C, T)
    x = self.decoder(x)
    x = self.smoothen(x)
    return x

  
class JointLateClusterVQ5_G(nn.Module):
  '''
  VQ Bottleneck for each timescale 64, 32, 16, 8, 4, 2, the VQ layer is shared
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_embeddings=512, max_depth=5, residual_mask=None, **kwargs):
    super().__init__()
    self.num_embeddings = num_embeddings
    self.max_depth = max_depth
    self.residual_mask = residual_mask
    self.pose_encoder = PoseEncoder(output_feats = time_steps, p=p)
    self.unet_enc = UNet1DEncoder(input_channels = in_channels, output_channels = in_channels,
                                  p=p, max_depth=max_depth)
    self.unet_dec = UNet1DDecoder(output_channels = in_channels, p=p, max_depth=max_depth)
    self.vq = VQLayerSG(num_embeddings=num_embeddings,
                        num_features=256, **kwargs)

    self.decoder = nn.Sequential(*nn.ModuleList([ConvNormRelu(in_channels, in_channels,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p) for i in range(3)]+
                                                [ConvNormRelu(in_channels, out_feats,
                                                              type='1d', leaky=True, downsample=False,
                                                              p=p)]))
    
    self.smoothen = nn.Conv1d(out_feats, out_feats, kernel_size=1, stride=1)
    
  def forward(self, x, y, time_steps=None, **kwargs):
    #check kwargs, run thru seperate encoder in else conditional
    #x needs to be defined by pose or audio accordingly.
    internal_losses = []
    x = self.pose_encoder(y, time_steps)

    x, residuals = self.unet_enc(x)
    i_loss = None
    residuals_quantized = []
    for depth in range(self.max_depth):
      res_, idxs_, i_loss_ = self.vq(residuals[depth])
      residuals_quantized.append(res_)
      if i_loss is None:
        i_loss = i_loss_
      else:
        for i in range(len(i_loss)):
          i_loss[i] += i_loss_[i]

    ## The bottleneck is also quantized
    x_quantized, idxs, i_loss_ = self.vq(x)
    for i in range(len(i_loss)):
      i_loss[i] += i_loss_[i]
          
    internal_losses += i_loss

    x = self.unet_dec(x_quantized, residuals_quantized, residual_mask=self.residual_mask)
    x = self.decoder(x)
    x = self.smoothen(x)
    
    return x.transpose(-1, -2), internal_losses
  
  def encode(self, x, time_steps):
    with torch.no_grad():
      x = self.pose_encoder(x, time_steps)
      x = self.unet(x)
      x, idxs, i_loss = self.vq(x)
    return x, idxs, i_loss

  def decode(self, logits):
    with torch.no_grad():
      probs = F.softmax(logits, dim=-1)
      shape = probs.shape[:-1]
      probs = probs.view(-1, probs.shape[-1]) 
      idxs = probs.multinomial(1).view(shape) ## sampled idxs
      x = self.vq.emb(idxs).transpose(-2, -1) ## (B, C, T)
      x = self.decoder(x)
      x = self.smoothen(x)

    return x

