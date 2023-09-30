import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
from pycasper.torchUtils import non_deterministic
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

JointLateClusterVQPrior2_D = Speech2Gesture_D

def get_context_mask(time=64, kernel_size=5, device='cpu'):
  if kernel_size > 0:
    ones = torch.ones(time)
    mask = None
    if kernel_size % 2 == 0:
      start = -kernel_size/2 + 1 
      stop = kernel_size/2
    else:
      start = -(kernel_size - 1)/2
      stop = (kernel_size - 1)/2
    diags = list(np.linspace(start, stop, kernel_size).astype(np.int))
    for diag in diags:
      if mask is None:
        mask = torch.diag(ones[:ones.shape[0]-abs(diag)], diagonal=diag)
      else:
        mask += torch.diag(ones[:ones.shape[0]-abs(diag)], diagonal=diag)
    return (1-mask).bool().to(device)
  else:
    return torch.zeros(time, time).bool().to(device)  

def get_causal_pose_mask(sz, device):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).bool()
  return mask.to(device)

def combine_audio_pose_mask(m1, m2):
  assert m1.shape[0] == m2.shape[0], 'masks must be of the same size'
  mask = torch.zeros(m1.shape[0]+m2.shape[0], m1.shape[0]+m2.shape[0]).bool().to(m1.device)
  mask[:m1.shape[0], :m1.shape[0]] = m1
  mask[:m1.shape[0], m1.shape[0]:] = m2
  return mask

def LSE(x):
  c = x.max()
  return c + torch.log(torch.sum(torch.exp(x - c)))

class JointLateClusterVQPrior2_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_embeddings=512,
               pretrained_trainers=None, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=256)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, 256,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
      
    self.unet = UNet1D(input_channels = in_channels, output_channels = in_channels, p=p)

    self.vq = pretrained_trainers.model
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    self.post_encoder = nn.Sequential(ConvNormRelu(in_channels, num_embeddings,
                                                   type='1d', leaky=True, downsample=False,
                                                   p=p),
                                      ConvNormRelu(num_embeddings, num_embeddings,
                                                   type='1d', leaky=True, downsample=False,
                                                   p=p))


  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    sample_flag = False
    if kwargs['sample_flag']:
      x = [x_.view(-1, 64, x_.shape[-1]) if x_.shape[1] % 64 == 0 else x_.squeeze(0) for x_ in x ]
      y = y.view(-1, 64, y.shape[-1])
      kwargs['sample_flag'] = False
      sample_flag = True
      
    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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
      x = self.concat_encoder2(x)
    else:
      x = torch.cat(tuple(x),  dim=1)
    
    x = self.unet(x) ## B x C x T
    logits = self.post_encoder(x) ## B X num_embeddings X T
    logits_flattened = logits.transpose(-2, -1).reshape(-1, logits.shape[1]) ## (-1, num_embeddings)
    idxs = idxs.reshape(-1).detach()
    internal_losses.append(F.cross_entropy(logits_flattened, idxs.detach())) ## add cross_entropy loss

    ## decode to pose sequences
    x = self.vq.decode(logits.transpose(-1, -2)).detach()

    x = x.transpose(-1, -2)
    if sample_flag:
      x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses

class JointLateClusterVQPrior21_G(nn.Module):
  '''
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=256)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, num_embeddings,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    self.linear = nn.Linear(256, num_embeddings)
    
    self.classifier = nn.Transformer(d_model=num_embeddings, nhead=4, num_encoder_layers=2,
                                     num_decoder_layers=2, dim_feedforward=512, dropout=0.1,
                                     activation='relu')
    
    self.post_encoder = nn.Sequential(ConvNormRelu(in_channels, num_embeddings,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=1, stride=1,
                                                   p=p),
                                      ConvNormRelu(num_embeddings, num_embeddings,
                                                   type='1d', leaky=True, downsample=False,
                                                   kernel_size=1, stride=1,
                                                   p=p))


  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    sample_flag = False
    if kwargs['sample_flag']:
      x = [x_.view(-1, 64, x_.shape[-1]) if x_.shape[1] % 64 == 0 else x_.squeeze(0) for x_ in x ]
      y = y.view(-1, 64, y.shape[-1])
      kwargs['sample_flag'] = False
      sample_flag = True
      
    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    x_q_logits = self.linear(x_q.transpose(-2, -1)).transpose(-2, -1) ## x_q converted to logits for the prior
    if not sample_flag:
      tgt_mask = self.classifier.generate_square_subsequent_mask(x_q.shape[-1]).to(x.device)
      x = self.classifier(x.permute(2, 0, 1), x_q_logits.permute(2, 0, 1), tgt_mask=tgt_mask).permute(1, 2, 0)
    else:
      x_ = []
      memory = self.classifier.encoder(x.permute(2, 0, 1), mask=None, src_key_padding_mask=None)
      for b in range(x.shape[0]): ## each sample at a time
        # beam_width = 10
        # beam = BeamSearch(beam_width=beam_width,
        #                   beams=torch.ones(beam_width, 1).to(x.device),
        #                   beam_scores=torch.zeros(beam_width).to(x.device))
        tgt = x_q_logits[b:b+1, :, 0:1] ## First sample
        for t in range(x.shape[-1] - 1):
          tgt_mask = self.classifier.generate_square_subsequent_mask(tgt.shape[-1]).to(x.device)
          tgt_cap = self.classifier.decoder(tgt.permute(2, 0, 1), memory,
                                            tgt_mask=tgt_mask, memory_mask=None,
                                            tgt_key_padding_mask=None,
                                            memory_key_padding_mask=None).permute(1, 2, 0)
          tgt = torch.cat([tgt, tgt_cap[:, :, -1:].detach()], dim=-1)
          #beam(tgt) ## perform beam search
          #best_beam = beam.get_best_beam()
        x_.append(tgt)
      x = torch.cat(x_, dim=0)
    
    ## TODO output is one shifted. has to be done differently while sampling vs training
    ## TODO might have to remove post_encoder with causal convolution
    ## TODO the input at each step could be x_q from the embedding space.
    ## TODO positional encoding in encoder and decoder
    ## TODO also predict the first frame, instead of using that as the input
    ## TODO Beam Search
    ## TODO encode and decode for vqvae models
    ## TODO predict 5th pose instead of next pose
    
    
    #logits = self.post_encoder(x) ## B X num_embeddings X T
    logits = x
    logits_flattened = logits[..., 1:].transpose(-2, -1).reshape(-1, logits.shape[1]) ## (-1, num_embeddings)
    idxs = idxs[..., 1:].reshape(-1).detach()
    internal_losses.append(F.cross_entropy(logits_flattened, idxs.detach())) ## add cross_entropy loss

    ## decode to pose sequences
    x = self.vq.decode(logits.transpose(-1, -2)).detach()

    x = x.transpose(-1, -2)
    if sample_flag:
      x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses


class JointLateClusterVQPrior22_G(nn.Module):
  '''
  Only transformer encoder as the prior
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings

    text_key = None
    for key in kwargs['shape']:
      if key in ['text/w2v', 'text/bert', 'text/tokens']:
        self.text_encoder = nn.ModuleList([BertEncoder(out_feats=256)])
    self.pos_encoder = PositionalEncoding(256, p)
    self.concat_encoder = MultimodalTransformerFusion(out_feats=256, nhid=2) # Q_a, K_t, V_t -> (B, T, C)
    self.concat_encoder2 = nn.Sequential(*nn.ModuleList([ConvNormRelu(512, num_embeddings,
                                                                     type='1d', leaky=True, downsample=False,
                                                                     p=p)]))

    self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)

    #self.linear = nn.Linear(256, num_embeddings)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=num_embeddings, nhead=4,
                                               dim_feedforward=512, dropout=0.1, activation='relu')
    encoder_norm = nn.LayerNorm(num_embeddings)
    self.classifier = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                            num_layers=2,
                                            norm=encoder_norm)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    sample_flag = False
    if kwargs['sample_flag']:
      x = [x_.view(-1, 64, x_.shape[-1]) if x_.shape[1] % 64 == 0 else x_.squeeze(0) for x_ in x ]
      y = y.view(-1, 64, y.shape[-1])
      kwargs['sample_flag'] = False
      sample_flag = True
      
    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    #x_q_logits = self.linear(x_q.transpose(-2, -1)).transpose(-2, -1) ## x_q converted to logits for the prior
    x = self.classifier(x.permute(2, 0, 1)).permute(1, 2, 0)
    
    logits = x
    logits_flattened = logits[..., 1:].transpose(-2, -1).reshape(-1, logits.shape[1]) ## (-1, num_embeddings)
    idxs = idxs[..., 1:].reshape(-1).detach()
    internal_losses.append(F.cross_entropy(logits_flattened, idxs.detach())) ## add cross_entropy loss
    #internal_losses.append(torch.zeros(1)[0].to(x.device))

    ## decode pose sequences
    x = self.vq.decode(logits.transpose(-1, -2))#.detach() ## Use detach without gan

    x = x.transpose(-1, -2)
    if sample_flag:
      x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses


class JointLateClusterVQPrior23_G(nn.Module):
  '''
  Transformer encoder as prior. predict embeddings instead of embeddings ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features

    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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

    #self.linear = nn.Linear(256, num_embeddings)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=4,
                                               dim_feedforward=512, dropout=0.1, activation='relu')
    encoder_norm = nn.LayerNorm(num_features)
    self.classifier = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                            num_layers=2,
                                            norm=encoder_norm)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    # sample_flag = False
    # if kwargs['sample_flag']:
    #   x = [x_.view(-1, 64, x_.shape[-1]) if x_.shape[1] % 64 == 0 else x_.squeeze(0) for x_ in x ]
    #   y = y.view(-1, 64, y.shape[-1])
    #   kwargs['sample_flag'] = False
    #   sample_flag = True
      
    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    #x_q_logits = self.linear(x_q.transpose(-2, -1)).transpose(-2, -1) ## x_q converted to logits for the prior

    ## get context_mask
    mask = self.get_context_mask(time=x.shape[-1], device=x.device)
    x = self.classifier(x.permute(2, 0, 1), mask=mask).permute(1, 2, 0) ## B x C x T
    
    internal_losses.append(F.mse_loss(x, x_q.detach())) ## add mse loss between predicted and true embeddings
    
    ## decode embeddings instead of idxs
    x = self.vq.quantize(x) ## First Quantize x
    #x = x + (x_q - x).detach() ## gradient passing
    x = self.vq.decode_emb(x) ## Decode quantized x

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses

class JointLateClusterVQPrior24_G(nn.Module):
  '''
  Transformer encoder + previous poses as prior. predict embeddings instead of embeddings ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features

    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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

    #self.linear = nn.Linear(256, num_embeddings)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=4,
                                               dim_feedforward=512, dropout=0.1, activation='relu')
    encoder_norm = nn.LayerNorm(num_features)
    self.classifier = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                            num_layers=2,
                                            norm=encoder_norm)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    # sample_flag = False
    # if kwargs['sample_flag']:
    #   x = [x_.view(-1, 64, x_.shape[-1]) if x_.shape[1] % 64 == 0 else x_.squeeze(0) for x_ in x ]
    #   y = y.view(-1, 64, y.shape[-1])
    #   kwargs['sample_flag'] = False
    #   sample_flag = True
      
    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    #x_q_logits = self.linear(x_q.transpose(-2, -1)).transpose(-2, -1) ## x_q converted to logits for the prior

    mask1 = self.get_context_mask(time=x.shape[-1], device=x.device)
    mask2 = get_causal_pose_mask(sz=x.shape[-1], device=x.device)
    mask = combine_audio_pose_mask(mask1, mask2)

    audio_pose = torch.cat([x, x_q], dim=-1)
    time = x.shape[-1]

    if not kwargs['sample_flag']:
      x = self.classifier(audio_pose.permute(2, 0, 1), mask=mask).permute(1, 2, 0) ## B x C x T
      x = x[..., :time] ## remove the output of poses
    else:
      for t in range(time):
        x_insert = self.classifier(audio_pose[..., :time+t].permute(2, 0, 1), mask=mask[:time+t, :time+t]).permute(1, 2, 0)[..., t] ## B x C x 1
        audio_pose[..., time+t] = x_insert
      x = audio_pose[..., time:]
    
    internal_losses.append(F.mse_loss(x, x_q.detach())) ## add mse loss between predicted and true embeddings
    
    ## decode embeddings instead of idxs
    x = self.vq.quantize(x) ## First Quantize x
    #x = x + (x_q - x).detach() ## gradient passing
    x = self.vq.decode_emb(x) ## Decode quantized x

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses


class JointLateClusterVQPrior25_G(nn.Module):
  '''
  Transformer encoder as prior. predict embeddings instead of embeddings ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features

    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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

    
    # encoder_layer = nn.TransformerEncoderLayer(d_model=num_features, nhead=4,
    #                                            dim_feedforward=512, dropout=0.1, activation='relu')
    # encoder_norm = nn.LayerNorm(num_features)
    # self.classifier = nn.TransformerEncoder(encoder_layer=encoder_layer,
    #                                         num_layers=2,
    #                                         norm=encoder_norm)
    self.unet = UNet1D(num_features, num_features, p=p)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    x = self.unet(x)
    
    internal_losses.append(F.mse_loss(x, x_q.detach())) ## add mse loss between predicted and true embeddings
    
    ## decode embeddings instead of idxs
    x = self.vq.quantize(x) ## First Quantize x
    #x = x + (x_q - x).detach() ## gradient passing
    x = self.vq.decode_emb(x) ## Decode quantized x

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses


'''
Heirarchical VQ
'''
class JointLateClusterVQPrior41_G(nn.Module):
  '''
  Unet encoder as prior. predict embeddings instead of embeddings ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features
    self.residual_mask = self.vq.residual_mask

    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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
    self.unet_enc = UNet1DEncoder(num_features, num_features, p=p, max_depth=self.vq.unet_enc.max_depth)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    x, residuals = self.unet_enc(x)
    x = residuals + [x]

    i_loss = 0
    for depth, (x_, x_q_) in enumerate(zip(x, x_q)):
      if depth >= len(self.residual_mask):
        i_loss += F.mse_loss(x_, x_q_.detach())
      elif self.residual_mask[depth]:
        i_loss += F.mse_loss(x_, x_q_.detach())
    internal_losses.append(i_loss) ## add mse loss between predicted and true embeddings

    ## decode embeddings instead of idxs
    x = self.vq.quantize(x) ## First Quantize x
    #x = x + (x_q - x).detach() ## gradient passing
    x = self.vq.decode_emb(x) ## Decode quantized x

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses

class JointLateClusterVQPrior42_G(nn.Module):
  '''
  Transformer encoder as prior. predict embeddings instead of embeddings ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features
    self.num_chunks = self.vq.max_depth + 1
    
    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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

    self.linear = nn.Linear(num_features, num_features*self.num_chunks)
    encoder_layer = nn.TransformerEncoderLayer(d_model=num_features*self.num_chunks, nhead=4,
                                               dim_feedforward=512, dropout=0.1, activation='relu')
    encoder_norm = nn.LayerNorm(num_features*self.num_chunks)
    self.classifier = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                            num_layers=2,
                                            norm=encoder_norm)

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    ## get context_mask
    mask = self.get_context_mask(time=x.shape[-1], device=x.device)
    x = self.classifier(self.linear(x.permute(2, 0, 1)), mask=mask).permute(1, 2, 0) ## B x C x T
    x = list(torch.chunk(x, self.num_chunks, dim=1))
    x = [F.max_pool1d(x_, 2**i) for i, x_ in enumerate(x)]

    i_loss = 0
    for x_, x_q_ in zip(x, x_q):
      i_loss += F.mse_loss(x_, x_q_.detach())
    internal_losses.append(i_loss) ## add mse loss between predicted and true embeddings
    
    ## decode embeddings instead of idxs
    x = self.vq.quantize(x) ## First Quantize x
    #x = x + (x_q - x).detach() ## gradient passing
    x = self.vq.decode_emb(x) ## Decode quantized x

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses

class JointLateClusterVQPrior43_G(nn.Module):
  '''
  Unet encoder as prior. predict ids/logits
  input_shape:  (N, time, frequency)
  output_shape: (N, time, pose_feats)

  after encoding x always has the dim (256,64)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0,
               pretrained_trainers=None, prior_context=5, **kwargs):
    super().__init__()
    assert pretrained_trainers is not None, 'Pre-trained model needed for training the prior'
    self.vq = pretrained_trainers.model
    self.vq.requires_grad_(False)
    self.vq.eval()
    num_embeddings = self.vq.num_embeddings
    num_features = self.vq.num_features
    self.residual_mask = self.vq.residual_mask

    self.prior_context = prior_context
    self.get_context_mask = partial(get_context_mask, kernel_size=prior_context)
    
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
    self.unet_enc = UNet1DEncoder(num_features, num_features, p=p, max_depth=self.vq.unet_enc.max_depth)

    self.classify = nn.ModuleList([ConvNormRelu(in_channels=num_features,
                                                out_channels=num_embeddings,
                                                type='1d', leaky=True, downsample=False) 
                                   if self.residual_mask[depth] else None
                                   for depth in range(self.vq.unet_enc.max_depth)])
    self.classify.append(ConvNormRelu(in_channels=num_features,
                                      out_channels=num_embeddings,
                                      type='1d', leaky=True, downsample=False))

  def forward(self, x, y, time_steps=None, **kwargs):
    labels = x[-1] ## remove the cluster labels attached to the inputs
    x = x[:-1]
    internal_losses = []

    ## encoded without grad
    x_q, idxs, i_loss = self.vq.encode(y, time_steps) ## i_loss: [vq_loss, commit_loss]

    ## Text + Audio Prior
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

    x, residuals = self.unet_enc(x)
    x = residuals + [x]  

    ## Get logits for each granularity
    logits = []
    for x_, cl in zip(x, self.classify):
      if cl is None:
        logits.append(None)
      else:
        logits.append(cl(x_))

    ## Get cross_entropy loss for each granularity
    i_loss = 0
    for logit, idx in zip(logits, idxs):
      if logit is not None:
        with non_deterministic():
          i_loss += F.cross_entropy(logit, idx)
    internal_losses.append(i_loss)

    ## Soft Attention over all the embeddings
    embs = []
    for logit, vq in zip(logits, self.vq.vqs):
      if logit is not None:
        embs.append(logit.softmax(-2).permute(0, 2, 1).matmul(vq.emb.weight).permute(0, 2, 1))
      else:
        embs.append(torch.zeros(1).to(y.device))

    ## decode embeddings
    x = self.vq.decode_emb(embs)

    x = x.transpose(-1, -2)
    # if sample_flag:
    #   x = x.reshape(1, -1, x.shape[-1])
    return x, internal_losses
