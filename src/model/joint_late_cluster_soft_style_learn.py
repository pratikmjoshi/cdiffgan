import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb

from .layers import *
from .speech2gesture import Speech2Gesture_D
from .gan import GAN

import torch
import torch.nn as nn

JointLateClusterSoftStyleLearn_D = Speech2Gesture_D

class JointLateClusterSoftStyleLearn_G(nn.Module):
  def __init__(self, style_dict, pretrained_model, **kwargs):
    super().__init__()
    self.style_dict = style_dict
    self.pretrained_model = pretrained_model
    self.style_vector = nn.Parameter(torch.ones(len(style_dict))/len(style_dict))
    print(self.style_vector)
    
    ## Freeze the weights of the pre-trained model
    for param in self.pretrained_model.parameters(): 
      param.requires_grad = False

    if hasattr(self.pretrained_model, 'G'):
      model = self.pretrained_model.G
    else:
      model = self.pretrained_model
    if hasattr(model, 'pose_style_emb'):
      for param in model.pose_style_emb.parameters():
        param.requires_grad = True
    elif hasattr(model, 'style_emb'):
      for param in model.style_emb.parameters():
        param.requires_grad = True      

  def forward(self, a, p, **kwargs):
    #style_vector = torch.nn.functional.softmax(self.style_vector)
    style_emb = self.style_vector.unsqueeze(0).unsqueeze(0).repeat(p.shape[0], p.shape[1], 1)
    kwargs.update({'style':style_emb})
    self.pretrained_model.eval()
    # if isinstance(self.pretrained_model, GAN):
    #   get_style = self.pretrained_model.G.get_style
    # else:
    #   get_style = self.pretrained_model.get_style
    # pdb.set_trace()
    # style_emb = get_style(a, p, **kwargs)
    y_cap, internal_losses = self.pretrained_model(a, p, **kwargs)
    internal_losses = internal_losses[:3]
    
    return y_cap, internal_losses
