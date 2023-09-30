import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import collections
import math

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
from model.modules import *

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
        -TrainerTrans_SB
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


class TrainerTrans_SB(TrainerTrans):
  def __init__(self, args, args_subset, args_dict_update={}):
    super(TrainerTrans_SB, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    
    self.history_length = self.batch_size * 40
    self.beta = 2  #larger beta is more selective in sampling
    self.sampling_min = 0
    self.historical_losses = UnboundedHistogram(self.history_length)
    self.selected_x1 = []
    self.selected_x2 = []
    self.selected_y = []
  
  def update_history(self, losses):
    for loss in losses:
        self.historical_losses.append(loss)
    print("history length{0}##".format(len(self.historical_losses.history)))

  def calculate_probability(self, loss):
    percentile = self.historical_losses.percentile_of_score(loss)
    return math.pow(percentile / 100., self.beta)

  def select(self, select_probability):
    draw = np.random.uniform(0, 1)
    return draw < select_probability

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    print("calculate_loss {0}".format(loss))
    #for i_loss in internal_losses:
    #    loss += i_loss

    self.running_loss[0] += loss.item() * x.shape[0]
    self.running_count[0] += x.shape[0]

    return loss

  def create_selected_sample(self):
    x1 = torch.stack(self.selected_x1[:self.batch_size])
    x2 = torch.stack(self.selected_x2[:self.batch_size])
    y = torch.stack(self.selected_y[:self.batch_size])
    print("create_selected_sample x1:{0} x2:{1} y:{2}".format(x1.shape, x2.shape, y.shape)) 
    return x1, x2, y

    
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
        #print("{0}  {1} th batch".format(desc, count))
        ## Transform batch before using in the model
        x1, x2, y_, y = self.get_processed_batch(batch) #x1=audio x2=text
        #print("data shape x1:{0} x2:{1} y:{2}".format(x1.shape, x2.shape, y.shape))
        y_cap, internal_losses = self.forward_pass(desc, x1, x2, y)
        loss = self.calculate_loss(x1, y, y_cap, internal_losses)
        
        if desc == 'train':
          losses_em = []
          for y_em, y_cap_em in zip(y, y_cap):
            em_loss = self.criterion(y_cap_em, y_em)
            em_loss = em_loss.detach()
            #print("em_loss {0}".format(em_loss))
            losses_em.append(em_loss)
          self.update_history(losses_em)
          for x1_em, x2_em, y_em, em_loss in zip(x1, x2, y, losses_em):
            prob = self.calculate_probability(em_loss)
            print("em_loss:{0} em_loss prob:{1}".format(em_loss, prob))
            if self.select(prob):
                self.selected_x1.append(x1_em)
                self.selected_x2.append(x2_em)
                self.selected_y.append(y_em)
                print("selected_x1:{0}, selected_x2:{1}, selected_y:{2}".format(len(self.selected_x1), len(self.selected_x2), len(self.selected_y)))
          losses_em = []
                
        ## update tqdm
        losses = [l/c for l,c in zip(self.running_loss, self.running_count)]
        Tqdm.set_description(self.tqdm_desc(desc, losses))
        Tqdm.refresh()
        
        if desc == 'train':
            
            #if enough pool
            print("before judge x1:{0} x2:{1} y:{2} batch:{3}".format(len(self.selected_x1), len(self.selected_x2), len(self.selected_y), self.batch_size))
            if len(self.selected_x1) >= self.batch_size:
                print("Sampling backpropper!!!!!!!")
                x1_s, x2_s, y_s = self.create_selected_sample()
                y_s_cap, internal_losses_s = self.model(x1_s, x2_s)
                sb_loss = self.criterion(y_s_cap, y_s)
                print("sb_loss {0}".format(sb_loss))
                self.optimize(sb_loss)
                #self.optimize(loss)
                self.selected_x1= self.selected_x1[self.batch_size:]
                self.selected_x2= self.selected_x2[self.batch_size:]
                self.selected_y= self.selected_y[self.batch_size:]
                print("updated lelected list x1:{0}, x2:{1}, y:{2}".format(len(self.selected_x1), len(self.selected_x2), len(self.selected_y)))
                x1_s = x1_s.detach()
                x2_s = x2_s.detach()
                y_s = y_s.detach()
                y_s_cap = y_s_cap.detach()
                sb_loss =sb_loss.detach()

                
        ## Detach Variables to avoid memory leaks
        x1 = x1.detach()
        x2 = x2.detach()
        y = y.detach()
        loss = loss.detach()
        y_cap = y_cap.detach()
                
        ## Evalutation
        y_cap = y_cap.to('cpu')
        self.calculate_pck(y_cap, y_)
                
        if count>=0 and self.args.debug: ## debugging by overfitting
            break

    return losses[0], self.pck.get_averages(desc)


class UnboundedHistogram:
    def __init__(self, max_history):
        self.max_history =  max_history
        self.history = collections.deque(maxlen=self.max_history)
    
    def append(self, value):
        self.history.append(value)
    
    def get_count(self, it, score):
        count = 0
        for i in it:
            if i < score:
                count += 1
        return count

    def percentile_of_score(self, score):
        num_lower_scores = self.get_count(self.history, score)
        return num_lower_scores * 100. / len(self.history)

