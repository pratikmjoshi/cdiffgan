import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

TransformerLate_D = Speech2Gesture_D
TransformerLate2_D = Speech2Gesture_D
TransformerLate3_D = Speech2Gesture_D
TransformerLate4_D = Speech2Gesture_D
TransformerLate5_D = Speech2Gesture_D
TransformerLate6_D = Speech2Gesture_D
TransformerLate7_D = Speech2Gesture_D

class TransformerLate_G(nn.Module):
  '''
  Transformer as encoder for Language with a vanilla CNN as the decoder

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, dropout=0.5, p = 0, E = 256, **kwargs):
    super().__init__()
    #self.audio_encoder = AudioEncoder(output_feats = time_steps, p=p)
    #self.text_encoder = TextEncoderTransformer_d(output_feats = time_steps, p=p)

    #Linear layers
    self.plinear_enc = nn.Linear(out_feats, E)
    self.tlinear_enc = nn.Linear(in_channels, E)
    self.linear_decoder = nn.Linear(E, out_feats)
    self.decoder_emb = nn.Linear(E, out_feats)

    #Encoder
    self.nhead = 8
    self.nhid = 3
    self.ninp = E
    self.pos_encoder = PositionalEncoding(self.ninp, dropout)
    encoder_layers = nn.TransformerEncoderLayer(self.ninp, self.nhead, self.nhid)
    self.transformer_text_encoder = nn.TransformerEncoder(encoder_layers, self.nhid) # Norm
    self.src_mask = None #TODO

    #Decoder
    self.tgt_mask = None
    decoder_layers = nn.TransformerDecoderLayer(E, self.nhead, self.nhid)
    self.transformer_text_decoder = nn.TransformerDecoder(decoder_layers, self.nhid) # Norm\

    self.sos_idx = 0

  def _generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

  def _generate_source_key_padding_mask(self, token_count):
    max_count = token_count.max()
    mask = torch.ones(token_count.shape[0], max_count)
    for i in range(token_count.shape[0]):
      mask[i, :token_count[i]] = 0
    return mask.bool().to(token_count.device)

  def forward(self, x, y, time_steps=None, **kwargs):
    if time_steps is None:
      time_steps = y.shape[1]
    tgt_mask = self._generate_square_subsequent_mask(y.size(1)).to(y.device)
    self.tgt_mask = tgt_mask.float()
    #src_mask = self._generate_square_subsequent_mask(x[0].size(1)).to(y.device)
    #self.src_mask = src_mask.float() ## TODO this is incorrect as it depends on the length of each sentence
    src_key_padding_mask = self._generate_source_key_padding_mask(kwargs['token_count'])

    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = x[i].transpose(0,1)
        memory = self.tlinear_enc(memory)
        #memory = memory * math.sqrt(self.ninp)
        memory = self.pos_encoder(memory)
        memory = self.transformer_text_encoder(memory, src_key_padding_mask=src_key_padding_mask) #TODO: add mask

    if self.training:
      y = y.transpose(0,1)
      y = self.plinear_enc(y)
      output = self.transformer_text_decoder(y, memory, tgt_mask = self.tgt_mask)
      output = self.linear_decoder(output)
      output = output.transpose(0,1)
    else:
      batch_size = y.shape[0]
      output = torch.zeros(batch_size, time_steps, self.ninp).float().to(y.device)
      for t in range(1, time_steps):
        tgt_emb = output[:, :t].transpose(0,1)
        tgt_mask = torch.nn.Transformer().generate_square_subsequent_mask(t).transpose(0, 1).float().to(y.device)
        ## missing self-attention
        decoder_output = self.transformer_text_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        output_t =  decoder_output[-1, :, :]
        output[:, t] = output_t
      output = self.linear_decoder(output)

    internal_losses = []
    return output, internal_losses

class TransformerLate2_G(nn.Module):
  '''
  [Result]: gives a static output, propbably because the self attention has a lot of granularity
  Transformer as encoder for Language with a vanilla CNN as the decoder

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0.5, E = 256, **kwargs):
    super().__init__()
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    self.text_encoder = TransfomerEncoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    self.pose_decoder = TransfomerDecoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = self.text_encoder(x[i], y, repeat_text=0, **kwargs)

    output = self.pose_decoder(memory, y, time_steps, **kwargs)
    internal_losses = []
    return output, internal_losses

class TransformerLate3_G(nn.Module):
  '''
  [Result]: gives a static output, propbably because the self attention has a lot of granularity
  tgt=random while decoding

  Transformer as encoder for Language with a vanilla CNN as the decoder

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0, E = 256, **kwargs):
    super().__init__()
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    self.text_encoder = TransfomerEncoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    self.pose_decoder = TransfomerDecoderRand(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = self.text_encoder(x[i], y, repeat_text=0, **kwargs)

    output = self.pose_decoder(memory, y, time_steps, **kwargs)
    internal_losses = []
    return output, internal_losses

class TransformerLate4_G(nn.Module):
  '''

1. normal positional encoding with repeated embeddings (our original data format)
Write TransformerEncoder2 with forward pass modified that doesn't take kwargs as inputs using the same positional encoder

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0.5, E = 256, **kwargs):
    super().__init__()
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    self.text_encoder = TransfomerEncoder2(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    self.pose_decoder = TransfomerDecoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = self.text_encoder(x[i], y, repeat_text=0, **kwargs)

    output = self.pose_decoder(memory, y, time_steps, **kwargs)
    internal_losses = []
    return output, internal_losses


class TransformerLate5_G(nn.Module):
  '''

2. positional encoding of time frame for each word
write another positional encoder that takes in token/duration
to calculate the positional encoding of each time frame
and add it, with no positional encoding for each word in the sequence
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0.5, E = 256, **kwargs):
    super().__init__()
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    self.text_encoder = TransfomerEncoder_WordPOS(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    self.pose_decoder = TransfomerDecoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)



  def forward(self, x, y, time_steps=None, **kwargs):
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = self.text_encoder(x[i], y, repeat_text=1, **kwargs)
    output = self.pose_decoder(memory, y, time_steps, **kwargs)
    internal_losses = []
    return output, internal_losses

'''
HERE
'''

class TransformerLate6_G(nn.Module):
  '''

1. normal positional encoding with repeated embeddings + positional encoding of time frame for each word
include positional encodings of each word in the sequence
but add a token that marks the start/middle of the word for each time frame and
use a modification in the feed forward network that scales the dimension down back to E

  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=300, out_feats=96, p = 0.5, E = 256, **kwargs):
    super().__init__()
    self.nhead = 8
    self.nhid = 3
    self.ninp = E

    self.text_encoder = TransfomerEncoder2(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)
    self.pose_decoder = TransfomerDecoder(time_steps=time_steps, in_channels=in_channels, out_feats=out_feats, p=p, E=self.ninp, **kwargs)

  def forward(self, x, y, time_steps=None, **kwargs):
    for i, modality in enumerate(kwargs['input_modalities']):
      if modality.split('/')[0] == "text":
        memory = self.text_encoder(x[i], y, repeat_text=0, **kwargs)

    output = self.pose_decoder(memory, y, time_steps, **kwargs)
    internal_losses = []
    return output, internal_losses
