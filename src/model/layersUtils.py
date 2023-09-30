import torch
import torch.nn as nn
from copy import deepcopy

class FeatureExtractor(nn.Module):
  def __init__(self, model, layers):
    super().__init__()
    self.model = model
    self.layers = layers
    self._features = {layer: torch.empty(0) for layer in layers}

    for layer_id in layers:
      layer = dict([*self.model.named_modules()])[layer_id]
      layer.register_forward_hook(self.save_outputs_hook(layer_id))

  def save_outputs_hook(self, layer_id):
    def fn(_, __, output):
      self._features[layer_id] = output
    return fn

  def forward(self, *args, **kwargs):
    self._features = {layer: torch.empty(0) for layer in self.layers}
    out = self.model(*args, **kwargs)
    return out, self._features

def requires_grad(model, flag=True, target_layer=None):
  for name, param in model.named_parameters():
    if target_layer is None:  # every layer
      param.requires_grad = flag
    elif target_layer in name:  # target layer
      param.requires_grad = flag

'''Exponetial Moving Average for models'''
def accumulate(model1, model2, decay=0.999, modules=[]):
  par1 = dict(model1.named_parameters())
  par2 = dict(model2.named_parameters())
  
  for k in par1.keys():
    if k.split('.')[0] not in modules:
      par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)
