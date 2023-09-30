import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from model import *
from data import Data, ZNorm, Compose, RemoveJoints
from evaluation import PCK
from animation import save_animation
from parallel import parallel

from pycasper.name import Name
from pycasper.BookKeeper import *
from pycasper import torchUtils

from trainer import *

from tqdm import tqdm

'''
Class Heirarchy
Trainer Skeleton
  - TrainerBase 
     - Trainer 
     - TrainerGAN
     - TrainerTrans
'''

class TrainerTrans(TrainerBase):
  def __init__(self, args, args_subset, args_dict_update={}):
    super(TrainerTrans, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_model(self):
    modelKwargs = {}
    modelKwargs.update(self.args.modelKwargs)
    modelKwargs.update({'time_steps':self.data_shape[self.args.input_modalities[0]][0],
                        'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask)})

    return eval(self.args.model)(**modelKwargs)

  def running_loss_init(self):
    self.running_loss = [0]
    self.running_count = [1e-10]

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' {:.4f}'.format(*losses)
    else:
      return desc+' {:.4f}'.format(0)
      
  def zero_grad(self):
    self.model.zero_grad()
    self.optim.zero_grad()

  def forward_pass(self, desc, x1, x2, y): #x1=audio x2=text
    if desc == 'train' and self.model.training:
      y_cap, internal_losses = self.model(x1, x2)
    else:
      with torch.no_grad():
        y_cap, internal_losses = self.model(x1, x2)

    return y_cap, internal_losses

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    for i_loss in internal_losses:
      loss += i_loss

    self.running_loss[0] += loss.item() * x.shape[0]
    self.running_count[0] += x.shape[0]
 
    return loss

  def optimize(self, loss):
    loss.backward()
    self.optim.step()


  def get_processed_batch(self, batch):
    batch = self.pre(batch)
    
    audio = batch['audio/log_mel_512']
    text = batch['text/w2v']
    y_ = batch[self.output_modality]
    
    audio = audio.to(self.device)
    text = text.to(self.device)
    y = y_.to(self.device)
    
    ## remove the first joint
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    #y = torch.cat([y[..., 1:feats_shape], y[..., feats_shape+1:]], dim=-1)
    # y = y.view(y.shape[0], y.shape[1], 2, -1)
    # y, self.insert = torchUtils.remove_slices(y, mask=self.mask, dim=-1)
    # self.insert = self.insert.to('cpu')
    # y = y.view(y.shape[0], y.shape[1], -1)

    ## Remove the first joint
    y = self.transform(y)
    
    return audio, text, y_, y


  def train_loop(self, data, desc, epoch=0):
    ## init
    self.pck.reset()
    self.running_loss_init()

    if desc == 'train':
      self.model.train(True)
    else:
      self.model.eval()
      
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20)      
    for count, batch in enumerate(Tqdm):
      self.zero_grad()

      ## Transform batch before using in the model
      x1, x2, y_, y = self.get_processed_batch(batch) #x1=audio x2=text

      y_cap, internal_losses = self.forward_pass(desc, x1, x2, y)

      loss = self.calculate_loss(x1, y, y_cap, internal_losses)
      
      ## update tqdm
      losses = [l/c for l,c in zip(self.running_loss, self.running_count)]
      Tqdm.set_description(self.tqdm_desc(desc, losses))
      Tqdm.refresh()

      if desc == 'train':
        self.optimize(loss)

      ## Detach Variables to avoid memory leaks
      #x = x.detach()
      #y = y.detach()
      #loss = loss.detach()
      #y_cap = y_cap.detach()

      ## Evalutation
      y_cap = y_cap.to('cpu')
      self.calculate_pck(y_cap, y_)
      
      if count>=0 and self.args.debug: ## debugging by overfitting
        break

    return losses[0], self.pck.get_averages(desc)


  def sample_loop(self, data, desc):
    self.pck.reset()
    self.running_loss_init()
    self.model.eval()

    intervals = []
    start = []
    y_outs = []
    y_animates = []
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20)      
    for count, loader in enumerate(Tqdm):
      ### load ground truth
      Y = self.get_gt(loader.path2h5)
      
      loader = DataLoader(loader, self.batch_size, shuffle=False)
      Y_cap = []
      for batch in loader:
        with torch.no_grad():
          ## Transform batch before using in the model
          x1, x2, y_, y = self.get_processed_batch(batch) #x1=audio x2=text

          ## Forward pass
          y_cap, internal_losses = self.forward_pass(desc, x1, x2, y)

          loss = self.calculate_loss(x1, y, y_cap, internal_losses)

          ## Calculates PCK and reinserts data removed before training
          y_cap = y_cap.to('cpu')
          y_cap = self.calculate_pck(y_cap, y_) 
          Y_cap.append(y_cap)

          ## update tqdm
          losses = [l/c for l,c in zip(self.running_loss, self.running_count)]
          Tqdm.set_description(self.tqdm_desc(desc, losses))
          Tqdm.refresh()

      if Y_cap:
        intervals.append(batch['meta']['interval_id'][0])
        start.append(torch.Tensor([0]).to(torch.float))
        y_outs.append(torch.cat(Y_cap, dim=0))
        y_animates.append([torch.cat(Y_cap, dim=0), Y])

      if count >= 0 and self.args.debug: ## debugging by overfitting
        break

    filenames = [(Path(self.dir_name)/'keypoints'/'{}/{}/{}.h5'.format(desc,
                                                                       self.data.getSpeaker(interval),
                                                                       interval)).as_posix()
                 for interval in intervals]
    keys = [self.output_modality] * len(intervals)

    ## Save Keypoints
    parallel(self.data.modality_classes[self.output_modality].append, # fn
             -1, # n_jobs
             filenames, keys, y_outs) # fn_args

    ## Render Animations
    if self.args.render:
      sample_idxs = np.random.randint(0, len(y_animates), size=(self.args.render,))
      sample_from_list = lambda x, idxs: [x[idx] for idx in idxs]
      y_animates = sample_from_list(y_animates, sample_idxs)
      intervals = sample_from_list(intervals, sample_idxs)
      start = sample_from_list(start, sample_idxs)      
      save_animation(y_animates, intervals, self.dir_name, desc, self.data, start)

    return losses[0], self.pck.get_averages(desc)


