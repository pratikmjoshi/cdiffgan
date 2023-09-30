import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D

import torch
import torch.nn as nn

LateLM_D = Speech2Gesture_D

class LateLM_G(nn.Module):
  '''
  input_shape audio:  (N, time, frequency)
  input_shape text:  (N, time, embedding_size)
  output_shape: (N, time, pose_feats)
  '''
  def __init__(self, time_steps=64, in_channels=256, out_feats=104, p=0, num_clusters=8, cluster=None, mlm=True, clm=False, chunk_len=4, mask_percent=0.8, **kwargs):
    super().__init__()
    self.encoder = TransformerLM(out_feats=out_feats, ninp=in_channels, mlm=mlm, clm=clm, dropout=p)
    self.chunk_len = chunk_len
    self.mask_percent = mask_percent
    
  def forward(self, x, y, time_steps=None, **kwargs):
    '''
    y: (B, T, F)
    '''
    if kwargs['sample_flag']:
      T = kwargs['text/token_duration'].sum(-1)[0].long().item()
      y = y.view(-1, T, y.shape[-1])
      
    internal_losses = []
    #starting_buffer = 16
    #if kwargs['sample_flag'] == 0:
    y_cap, src_mask_loss = self.encoder(y.transpose(1, 0), mask_percent=self.mask_percent, chunk_len=self.chunk_len) ## (T, B, F) -> (T, B, F)
    y_cap = y_cap.transpose(1, 0)
    #else:
      # self.encoder.mlm = False
      # self.encoder.clm = True
      # y_cap = torch.zeros_like(y)
      # pdb.set_trace()
      # for t in range(starting_buffer, y.shape[1]): ## take starting k from the ground truth
      #   y_cap_, src_mask_loss = self.encoder(y.transpose(1, 0)[:t])
      #   y_cap[:, t:t+1] = y_cap_.transpose(1, 0)[:, t-1:t]

    if kwargs['sample_flag']:
      y_cap = y_cap.reshape(1, -1, y_cap.shape[-1])
        
    return y_cap, internal_losses, dict(src_mask_loss=src_mask_loss)
