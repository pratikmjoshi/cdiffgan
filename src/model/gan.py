import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from .layersUtils import *

import math

from pycasper.torchUtils import LambdaScheduler
from tqdm import tqdm

import pdb
  
class GAN(nn.Module):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True,
               tgt_train_ratio = 0.25,
               **kwargs):
    super(GAN, self).__init__()
    self.G = G
    self.D = D
    self.D_prob = dg_iter_ratio/(dg_iter_ratio + 1) ## discriminator generator iteration ratio
    self.lambda_D = lambda_D
    self.lambda_gan = lambda_gan
    self.lambda_scheduler = LambdaScheduler([self.lambda_D, self.lambda_gan],
                                            kind='incremental', max_interval=300,

                                            max_lambda=2)
    self.G_flag = True
    self.lr = lr

#    self.G_optim = eval('torch.optim.' + optim)(self.G.parameters(), lr=self.lr)
#    self.D_optim = eval('torch.optim.' + optim)(self.D.parameters(), lr=self.lr)

    self.criterion = eval('torch.nn.' + criterion)(reduction='none')

    self.joint = joint
    self.input_modalities = kwargs['input_modalities']
    self.update_D_prob_flag = update_D_prob_flag
    self.no_grad = no_grad
    self.tgt_train_ratio = tgt_train_ratio

    ## variable for g_path_regularize
    self.mean_path_length = 0 

  def get_velocity(self, x, x_audio):
    x_v = x[..., 1:, :] - x[..., :-1, :]
    if self.joint:
      return torch.cat([torch.cat([torch.zeros_like(x[..., 0:1, :]), x_v], dim=-2), torch.cat(x_audio[:len(self.input_modalities)], dim=-1)], dim=-1)
    else:
      return torch.cat([torch.zeros_like(x[..., 0:1, :]), x_v], dim=-2)

  def get_velocity_archive(self, x, x_audio):
    x_v = x[..., 1:] - x[..., :-1]
    return torch.cat([torch.zeros_like(x[..., 0:1]), x_v], dim=-1)

  def get_real_gt(self, x):
    return torch.ones_like(x)
  
  def get_fake_gt(self, x):
    return torch.zeros_like(x)

  def sample_wise_weight_mean(self, loss, W):
    W = W.view([W.shape[0]] + [1]*(len(loss.shape) - 1))
    loss = (W.expand_as(loss) * loss).mean()
    return loss
  
  def get_gan_loss(self, y_cap, y, W):
    loss = self.criterion(y_cap, y)
    return self.sample_wise_weight_mean(loss, W)

  def get_loss(self, y_cap, y, W):
    loss = self.criterion(y_cap, y)
    return self.sample_wise_weight_mean(loss, W)

  def estimate_weights(self, x_audio, y_pose, **kwargs):
    return torch.ones(y_pose.shape[0]).to(y_pose.device), None

  def estimate_weights_loss(self, W):
    return W

  def update_D_prob(self, W):
    pass

  def compute_gradient_penalty(self, **kwargs):
    return None

  def g_path_regularize(self, fake_img, latents, decay=0.01):
    return 0

  def get_idx_subset(self, bs, num_fewshot_samples, **kwargs):
    fewshot_sampler = kwargs.get('fewshot_sampler')
    if kwargs.get('replay') == 'pseudo-align-fs-src':
      max_style = len(fewshot_sampler.class_count_sub)
      bs_r = max(bs//max_style, 1)
      bs_l = bs - bs_r 
      low_l = 0
      high_l = fewshot_sampler.num_samples_per_class * (max_style - 1)
      low_r = fewshot_sampler.num_samples_per_class * (max_style - 1)
      high_r = fewshot_sampler.num_samples
      kwargs['replay-pivot'] = bs_l
      return torch.cat([torch.randint(low=low_l, high=high_l, size=(bs_l, )), torch.randint(low=low_r, high=high_r, size=(bs_r, ))]), kwargs
      
    return torch.randint(high=num_fewshot_samples, size=(bs, )), kwargs
  
  def get_fewshot_data(self, y_pose, x_audio, **kwargs):
    if kwargs.get('fewshot_data') is not None:
      bs = y_pose.shape[0]
      num_fewshot_samples = kwargs['fewshot_data']['fewshot_y'].shape[0]
      #idx_subset = torch.randint(high=num_fewshot_samples, size=(bs,))
      idx_subset, kwargs = self.get_idx_subset(bs, num_fewshot_samples, **kwargs)
      if torch.rand(1).item()<self.tgt_train_ratio: ## Use fewshot (Target) data 1 in 4 times by default
        common_kwargs = set(kwargs['fewshot_data']['fewshot_batch'].keys()).intersection(kwargs.keys())
        for key in common_kwargs:
          fewshot_batch_subset = kwargs['fewshot_data']['fewshot_batch'][key][idx_subset]
          if isinstance(kwargs[key], torch.Tensor):
            kwargs[key] = fewshot_batch_subset.to(kwargs[key].device)
          else:
            kwargs[key] = fewshot_batch_subset
        x = []
        for i, x_ in enumerate(kwargs['fewshot_data']['fewshot_x']):
          if i < len(self.input_modalities):
            if self.input_modalities[i] == 'text/tokens': ## hardocoded for text/tokens
              max_token_count = kwargs['text/token_count'].max()
              if 'text/token_duration' in kwargs:
                kwargs['text/token_duration'] = kwargs['text/token_duration'][..., :max_token_count]
              x.append(x_[idx_subset, :max_token_count].to(y_pose.device))
            else:
              x.append(x_[idx_subset].to(y_pose.device))
          else:
            x.append(x_[idx_subset].to(y_pose.device))
        kwargs['fewshot_flag'] = 'fewshot'

        return kwargs['fewshot_data']['fewshot_y'][idx_subset].to(y_pose.device), x, 1, kwargs
      else:
        kwargs['fewshot_flag'] = 'src_tgt'
        return kwargs['fewshot_data']['fewshot_y'][idx_subset].to(y_pose.device), x_audio, 0, kwargs
    else: ## usual case when there is no fewshot data
      return y_pose, x_audio, 1, kwargs


  '''
  Use kwargs['replay'] to decide on the pseduo output model
  - pseudo-fs-src        - random-fs
  - pseduo-align-fs-src  - random-fs
  - pseudo-fs-tgt        - no-replay-fs
  - pseudo-align-fs-tgt  - no-replay-fs

  Pass a copy of the model which also generates internal layer outputs
  '''

  def get_max_style(self, kwargs):
    if not hasattr(self, 'max_style'):
      self.max_style = int(kwargs['fewshot_data']['fewshot_batch']['style'].max().item())
    return self.max_style
  
  def get_pseudo_outputs(self, y_pose, x_audio, **kwargs):
    if kwargs.get('replay') == 'pseudo-fs-src' or kwargs.get('replay') == 'pseudo-align-fs-src':
      max_style = self.get_max_style(kwargs)
      with torch.no_grad():
        y_pose_, internal_losses, *args = self.G.G_src(x_audio, y_pose, **kwargs)
        y_pose = (kwargs['style'][:, 0]<max_style).int()[:, None, None]*y_pose_ + (kwargs['style'][:, 0]==max_style).int()[:, None, None]*y_pose
    elif kwargs.get('replay') == 'pseudo-fs-tgt':
      max_style = self.get_max_style(kwargs)
      with torch.no_grad():
        ## add older styles to the kwargs['style'] which originally contains only tgt styles
        kwargs['style'] = torch.randint(max_style + 1, size=(kwargs['style'].shape[0],), device=kwargs['style'].device)[:, None].expand(kwargs['style'].shape)
        y_pose_, internal_losses, *args = self.G.G_src(x_audio, y_pose, **kwargs)
        y_pose = (kwargs['style'][:, 0]<max_style).int()[:, None, None]*y_pose_ + (kwargs['style'][:, 0]==max_style).int()[:, None, None]*y_pose
    elif kwargs.get('replay') == 'pseudo-align-fs-tgt':
      max_style = self.get_max_style(kwargs)
      with torch.no_grad():
        bs = y_pose.shape[0]
        bs_r = bs // (max_style + 1)
        bs_l = bs - bs_r

        ## add older styles to the kwargs['style'] which originally contains only tgt styles        
        kwargs['style'] = torch.cat([torch.randint(max_style, size=(bs_l,)), torch.ones(bs_r)*max_style]).to(kwargs['style'].device)[:, None].expand(kwargs['style'].shape)
        kwargs['replay-pivot'] = bs_l
        y_pose_, internal_losses, *args = self.G.G_src(x_audio, y_pose, **kwargs)
        y_pose = (kwargs['style'][:, 0]<max_style).int()[:, None, None]*y_pose_ + (kwargs['style'][:, 0]==max_style).int()[:, None, None]*y_pose
        
    return y_pose, x_audio, kwargs
  
  def forward(self, x_audio, y_pose, **kwargs):
    internal_losses = []

    ## in a fewshot setting real_pose is taken from the fewshot_data
    ## if rec_flag == 1, x_audio is from fewshot_data (i.e. target) else it is from source data
    ## if rec_flag == 1, calculate the reconstruction loss as the x_audio and y_pose are parallel

    y_pose, x_audio, rec_flag, kwargs = self.get_fewshot_data(y_pose, x_audio, **kwargs)
    y_pose, x_audio, kwargs = self.get_pseudo_outputs(y_pose, x_audio, **kwargs)

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)

    ## if the generator has a gan_flag, it can be used to control if a gan is being trained or not
    gan_flag = self.G.gan_flag if hasattr(self.G, 'gan_flag') else True
    if self.training and gan_flag:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':False})
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose

        ## convert pose to velocity
        real_pose_v = self.get_velocity(real_pose, x_audio)
        fake_pose_v = self.get_velocity(fake_pose, x_audio)

        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones_like(W_loss))

        ## add gradient_penalty
        gp = self.compute_gradient_penalty(D=self.D,
                                           real_samples=real_pose_v.data,
                                           fake_samples=fake_pose_v.data)
        if gp is not None:
          real_D_loss += gp/2 * self.lambda_gp
          fake_D_loss += gp/2 * self.lambda_gp
        
        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        self.G_flag = False
      else:
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':True})
        
        fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
        args = args[0] if len(args)>0 else {}
        ## convert pose to velocity
        fake_pose_v = self.get_velocity(fake_pose, x_audio)
        if self.no_grad:
          with torch.no_grad():
            fake_pose_score, _ = self.D(fake_pose_v)
        else:
          fake_pose_score, _ = self.D(fake_pose_v)

        G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)

        pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag

        internal_losses.append(pose_loss)
        internal_losses.append(G_gan_loss)
        internal_losses += partial_i_loss
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args

class GANAligned(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)

  def forward(self, x_audio, y_pose, **kwargs):
    internal_losses = []

    ## in a fewshot setting real_pose is taken from the fewshot_data
    ## if rec_flag == 1, x_audio is from fewshot_data (i.e. target) else it is from source data
    ## if rec_flag == 1, calculate the reconstruction loss as the x_audio and y_pose are parallel

    y_pose, x_audio, rec_flag, kwargs = self.get_fewshot_data(y_pose, x_audio, **kwargs)
    y_pose, x_audio, kwargs = self.get_pseudo_outputs(y_pose, x_audio, **kwargs)
    pivot = kwargs.get('replay-pivot')
    # text/token_count, text/token_duration, style

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)

    ## if the generator has a gan_flag, it can be used to control if a gan is being trained or not
    gan_flag = self.G.gan_flag if hasattr(self.G, 'gan_flag') else True
    if self.training and gan_flag:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':False})
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose
        
        ## convert pose to velocity
        if pivot is not None: ## for replay alignment previous experiences do not impact the discriminator
          real_pose_v = self.get_velocity(real_pose[pivot:], [x_[pivot:] for x_ in x_audio])
          fake_pose_v = self.get_velocity(fake_pose[pivot:], [x_[pivot:] for x_ in x_audio])
        else:
          real_pose_v = self.get_velocity(real_pose, x_audio)
          fake_pose_v = self.get_velocity(fake_pose, x_audio)

        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          #fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones(fake_pose_score.shape[0]).to(fake_pose_score.device))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones(real_pose_score.shape[0]).to(real_pose_score.device))

        ## add gradient_penalty
        gp = self.compute_gradient_penalty(D=self.D,
                                           real_samples=real_pose_v.data,
                                           fake_samples=fake_pose_v.data)
        if gp is not None:
          real_D_loss += gp/2 * self.lambda_gp
          fake_D_loss += gp/2 * self.lambda_gp
        
        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        self.G_flag = False
      else:
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':True})
        
        fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
        args = args[0] if len(args)>0 else {}

        ## convert pose to velocity
        if pivot is not None: ## for replay alignment previous experiences do not impact the discriminator
          fake_pose_v = self.get_velocity(fake_pose[pivot:], [x_[pivot:] for x_ in x_audio])
        else:
          fake_pose_v = self.get_velocity(fake_pose, x_audio)

        if self.no_grad:
          with torch.no_grad():
            fake_pose_score, _ = self.D(fake_pose_v)
        else:
          fake_pose_score, _ = self.D(fake_pose_v)

        #G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)
        G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), torch.ones(fake_pose_score.shape[0]).to(fake_pose_score.device))

        #pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag
        pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones(fake_pose.shape[0]).to(fake_pose.device)) * rec_flag

        internal_losses.append(pose_loss)
        internal_losses.append(G_gan_loss)
        internal_losses += partial_i_loss
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args

  
class GANWeighted(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint, **kwargs)
    self.gan_criterion = torch.nn.CrossEntropyLoss(reduction='none')
  
  def get_real_gt(self, x):
    return torch.ones(x.shape[0], x.shape[1]).long().to(x.device)

  def get_fake_gt(self, x):
    return torch.zeros(x.shape[0], x.shape[1]).long().to(x.device)

  def get_gan_loss(self, y_cap, y, W):
    orig_shape = y.shape
    y_cap = y_cap.reshape(-1, y_cap.shape[-1])
    y = y.reshape(-1)
    loss = self.gan_criterion(y_cap, y).view(*orig_shape)
    return self.sample_wise_weight_mean(loss, W)

  def estimate_weights(self, x_audio, y_pose, **kwargs):
    with torch.no_grad():
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      
      ## convert pose to velocity
      fake_pose_v = self.get_velocity(fake_pose, x_audio)
      fake_pose_score, _ = self.D(fake_pose_v)
      #fake_pose_score = fake_pose_score.reshape(-1, fake_pose_score.shape[-1])
      
      outputs = torch.nn.functional.softmax(fake_pose_score, dim=-1).mean(-2)
      w = ((outputs[:, 1]/outputs[:, 0]))

      #w = ((outputs[:, 0]/outputs[:, 1]) / gamma)    
      w = 1/w
      if torch.isnan(w).any(): ## if there is some anomaly default to ones
        w = torch.ones_like(w)
      if torch.isinf(w).any():
        w = torch.ones_like(w)
      max_weight = 10
      mask = w > max_weight 
      w[mask] = max_weight

    return w, outputs

  def estimate_weights_loss(self, W):
    return torch.ones_like(W)

  def update_D_prob(self, W):
    W_ = min(max(W.mean().item(), 0.1), 10)
    W_ = math.log(W_)/math.log(10)
    W_ = (W_ + 1)/2
    W_ = 1 - W_
    W_ = max(min(W_, 0.9), 0.1) ## clip values between 0.1 and 0.9
    self.D_prob = 1-W_ ## if the samples are not able to fool the discriminator, imrpove the generator in their favour
    
    
class GANClassify(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', **kwargs):
    super().__init__(G, D, dg_iter_ratio=dg_iter_ratio,
                     lambda_D=lambda_D, lambda_gan=lambda_gan, lr=lr,
                     criterion=criterion, optim=optim, **kwargs)

  def get_velocity(self, x):
    return x


class WGAN_GP(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True,
               lambda_gp=10, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)
    self.lambda_gp = lambda_gp


  def get_real_gt(self, x):
    return -1
  
  def get_fake_gt(self, x):
    return 1

  def get_gan_loss(self, y_cap, y, W):
    return F.softplus(y_cap).mean()

  def compute_gradient_penalty(self, D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.shape[0], 1, 1).to(real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    #fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(real_samples.device)
    fake = torch.ones_like(d_interpolates)
    fake.requires_grad_(True)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
      outputs=d_interpolates,
      inputs=interpolates,
      grad_outputs=fake,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

  def g_path_regularize(self, fake_img, latents, decay=0.01):
    fake_img = fake_img.transpose(-2, -1)
    noise = torch.randn_like(fake_img) / math.sqrt(
      fake_img.shape[2]
    )
    grad, = autograd.grad(
      outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
                (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty

class WGAN_GP_NoRecon(WGAN_GP):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True,
               lambda_gp=10, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)
    self.lambda_gp = lambda_gp

  def get_loss(self, y_cap, y, W):
    return torch.zeros(1)[0].to(y_cap.device)

  def g_path_regularize(self, fake_img, latents, decay=0.01):
    return 0


class DiffGAN(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)
    self.fewshot_style = 1 if kwargs.get('fewshot_style') is None else kwargs.get('fewshot_style')
    self.fewshot_timing = 1 if kwargs.get('fewshot_timing') is None else kwargs.get('fewshot_timing')
    self.fewshot_rec = 1 if kwargs.get('fewshot_timing') is None else kwargs.get('fewshot_rec')

    ## Start (Layer l) and end layers (Layer m) in the paper
    self.G.G.fewshot_LAYER_START = 6 if kwargs.get('fewshot_LAYER_START') is None else kwargs.get('fewshot_LAYER_START')
    self.G.G.fewshot_LAYER_END = 7 if kwargs.get('fewshot_LAYER_END') is None else kwargs.get('fewshot_LAYER_END')

    self.fewshot_k = 10 if kwargs.get('fewshot_k') is None else kwargs.get('fewshot_k')

  def forward(self, x_audio, y_pose, **kwargs):
    internal_losses = []
    ## in a fewshot setting real_pose is taken from the fewshot_data
    ## if rec_flag == 1, x_audio is from fewshot_data (i.e. target) else it is from source data
    ## if rec_flag == 1, calculate the reconstruction loss as the x_audio and y_pose are parallel

    y_pose, x_audio, rec_flag, kwargs = self.get_fewshot_data(y_pose, x_audio, **kwargs)
    y_pose, x_audio, kwargs = self.get_pseudo_outputs(y_pose, x_audio, **kwargs)

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)

    ## if the generator has a gan_flag, it can be used to control if a gan is being trained or not
    gan_flag = self.G.gan_flag if hasattr(self.G, 'gan_flag') else True
    if self.training and gan_flag:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':False})
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose

        ## convert pose to velocity
        real_pose_v = self.get_velocity(real_pose, x_audio)
        fake_pose_v = self.get_velocity(fake_pose, x_audio)

        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones_like(W_loss))

        ## add gradient_penalty
        gp = self.compute_gradient_penalty(D=self.D,
                                           real_samples=real_pose_v.data,
                                           fake_samples=fake_pose_v.data)
        if gp is not None:
          real_D_loss += gp/2 * self.lambda_gp
          fake_D_loss += gp/2 * self.lambda_gp
        
        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        self.G_flag = False
      else:
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':True})
        i_loss_grounding = []
        if rec_flag and self.fewshot_timing:
          ## Learn (1) Crossmodal Grounding Shift
          
          ## Deactive Decoder Layer
          self.G.deactivate_decoder('big')

          ## Step 1: Extract the output of the z_l layer
          z_l, z_l_1 = self.G.estimate_z_l(x_audio, y_pose, **kwargs)

          z_l_param = torch.nn.Parameter(z_l.clone())

          ## Step 2: Update z_l based on gan and pose losses
          num_iters = 100
          #optim_z_l = torch.optim.AdamW(params=(z_l_param, ), lr=1e-3)
          optim_z_l = torch.optim.SGD(params=(z_l_param, ), lr=1000)
          requires_grad(self.D, False)
          # pbar = tqdm(range(num_iters), desc='loss: {:.3f}'.format(0))
          # for _ in pbar:
          for _ in range(num_iters):
            fake_pose, partial_i_loss, *args = self.G.forward_z_l(x_audio, y_pose,
                                                                  z_l=z_l_param, internal_losses=[],
                                                                  **kwargs)
            args = args[0] if len(args)>0 else {}
            ## convert pose to velocity
            fake_pose_v = self.get_velocity(fake_pose, x_audio)
            if self.no_grad:
              with torch.no_grad():
                fake_pose_score, _ = self.D(fake_pose_v)
            else:
              fake_pose_score, _ = self.D(fake_pose_v)

            # G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score,
            #                                                  self.get_real_gt(fake_pose_score),
            #                                                  1/W_loss)

            G_gan_loss = torch.zeros(1)[0].to(fake_pose.device)
            pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag

            loss = (pose_loss + G_gan_loss)
            loss.backward()
            optim_z_l.step()
            optim_z_l.zero_grad()
            # pbar.set_description('loss: {:.3f}'.format(pose_loss.item()))
            # pbar.refresh()
          requires_grad(self.D, True)

          ## Step 3: Update Layer L using mask
          mask_ = (z_l - z_l_param.data).abs().mean(1).unsqueeze(1).detach()
          mask = torch.zeros_like(mask_)
          val, idxs = mask_.topk(self.fewshot_k)
          mask.scatter_(-1, idxs, 1)

          self.G.activate_decoder('big')
          fake_pose, z_l_cap, partial_i_loss, *args = self.G.forward_z_l_1(x_audio, y_pose,
                                                                           z_l_1.detach(),
                                                                           internal_losses=[],
                                                                           **kwargs)
          args = args[0] if len(args)>0 else {}
          lambda_diff = 1 ## TODO Hardcoded to 1
          i_loss_grounding.append((torch.nn.functional.mse_loss(z_l_param.data, z_l_cap,
                                                              reduction='none') * mask).mean() * lambda_diff)
          i_loss_grounding.append(loss.detach())

        ## Learn the (2) Output Domain Shift
        self.G.activate_decoder('big')

        if self.fewshot_style:
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
          ## convert pose to velocity
          fake_pose_v = self.get_velocity(fake_pose, x_audio)
          if self.no_grad:
            with torch.no_grad():
              fake_pose_score, _ = self.D(fake_pose_v)
          else:
            fake_pose_score, _ = self.D(fake_pose_v)

          G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)
          if not self.fewshot_rec: ## TODO it is connected to fewshot_style. If fewshot_style is false, reconstruction is ignored.
            rec_flag = 0 ## ablation to remove the reconstruction loss
          pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag


          internal_losses.append(pose_loss)
          internal_losses.append(G_gan_loss)
        else:
          with torch.no_grad():
            fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
          internal_losses.append(torch.zeros(1)[0].to(y_pose.device))
          internal_losses.append(torch.zeros(1)[0].to(y_pose.device))

        partial_i_loss += i_loss_grounding
        internal_losses += partial_i_loss
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args
  

class DiffGANAligned(DiffGAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)

  def forward(self, x_audio, y_pose, **kwargs):
    internal_losses = []
    ## in a fewshot setting real_pose is taken from the fewshot_data
    ## if rec_flag == 1, x_audio is from fewshot_data (i.e. target) else it is from source data
    ## if rec_flag == 1, calculate the reconstruction loss as the x_audio and y_pose are parallel

    y_pose, x_audio, rec_flag, kwargs = self.get_fewshot_data(y_pose, x_audio, **kwargs)
    y_pose, x_audio, kwargs = self.get_pseudo_outputs(y_pose, x_audio, **kwargs)
    pivot = kwargs.get('replay-pivot')

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)

    ## if the generator has a gan_flag, it can be used to control if a gan is being trained or not
    gan_flag = self.G.gan_flag if hasattr(self.G, 'gan_flag') else True
    if self.training and gan_flag:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':False})
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose

        ## convert pose to velocity
        if pivot is not None: ## for replay alignment previous experiences do not impact the discriminator
          real_pose_v = self.get_velocity(real_pose[pivot:], [x_[pivot:] for x_ in x_audio])
          fake_pose_v = self.get_velocity(fake_pose[pivot:], [x_[pivot:] for x_ in x_audio])
        else:
          real_pose_v = self.get_velocity(real_pose, x_audio)
          fake_pose_v = self.get_velocity(fake_pose, x_audio)
  
        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          # fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones(fake_pose_score.shape[0]).to(fake_pose_score.device))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        # real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones_like(W_loss))
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones(real_pose_score.shape[0]).to(real_pose_score.device))


        ## add gradient_penalty
        gp = self.compute_gradient_penalty(D=self.D,
                                           real_samples=real_pose_v.data,
                                           fake_samples=fake_pose_v.data)
        if gp is not None:
          real_D_loss += gp/2 * self.lambda_gp
          fake_D_loss += gp/2 * self.lambda_gp
        
        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        self.G_flag = False
      else:
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':True})
        i_loss_grounding = []
        if rec_flag and self.fewshot_timing:
          ## Learn (1) Crossmodal Grounding Shift
          
          ## Deactive Decoder Layer
          self.G.deactivate_decoder('big')

          ## Step 1: Extract the output of the z_l layer
          z_l, z_l_1 = self.G.estimate_z_l(x_audio, y_pose, **kwargs)

          z_l_param = torch.nn.Parameter(z_l.clone())

          ## Step 2: Update z_l based on gan and pose losses
          num_iters = 100
          #optim_z_l = torch.optim.AdamW(params=(z_l_param, ), lr=1e-3)
          optim_z_l = torch.optim.SGD(params=(z_l_param, ), lr=1000)
          requires_grad(self.D, False)
          # pbar = tqdm(range(num_iters), desc='loss: {:.3f}'.format(0))
          # for _ in pbar:
          for _ in range(num_iters):
            fake_pose, partial_i_loss, *args = self.G.forward_z_l(x_audio, y_pose,
                                                                  z_l=z_l_param, internal_losses=[],
                                                                  **kwargs)
            args = args[0] if len(args)>0 else {}
            ## convert pose to velocity
            fake_pose_v = self.get_velocity(fake_pose, x_audio)
            if self.no_grad:
              with torch.no_grad():
                fake_pose_score, _ = self.D(fake_pose_v)
            else:
              fake_pose_score, _ = self.D(fake_pose_v)

            # G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score,
            #                                                  self.get_real_gt(fake_pose_score),
            #                                                  1/W_loss)

            G_gan_loss = torch.zeros(1)[0].to(fake_pose.device)
            pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag

            loss = (pose_loss + G_gan_loss)
            loss.backward()
            optim_z_l.step()
            optim_z_l.zero_grad()
            # pbar.set_description('loss: {:.3f}'.format(pose_loss.item()))
            # pbar.refresh()
          requires_grad(self.D, True)

          ## Step 3: Update Layer L using mask
          mask_ = (z_l - z_l_param.data).abs().mean(1).unsqueeze(1).detach()
          mask = torch.zeros_like(mask_)
          val, idxs = mask_.topk(self.fewshot_k)
          mask.scatter_(-1, idxs, 1)

          self.G.activate_decoder('big')
          fake_pose, z_l_cap, partial_i_loss, *args = self.G.forward_z_l_1(x_audio, y_pose,
                                                                           z_l_1.detach(),
                                                                           internal_losses=[],
                                                                           **kwargs)
          args = args[0] if len(args)>0 else {}
          lambda_diff = 1 ## TODO Hardcoded to 1
          i_loss_grounding.append((torch.nn.functional.mse_loss(z_l_param.data, z_l_cap,
                                                              reduction='none') * mask).mean() * lambda_diff)
          i_loss_grounding.append(loss.detach())

        ## Learn the (2) Output Domain Shift
        self.G.activate_decoder('big')

        if self.fewshot_style:
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}

          ## convert pose to velocity
          if pivot is not None: ## for replay alignment previous experiences do not impact the discriminator
            fake_pose_v = self.get_velocity(fake_pose[pivot:], [x_[pivot:] for x_ in x_audio])
          else:
            fake_pose_v = self.get_velocity(fake_pose, x_audio)

          if self.no_grad:
            with torch.no_grad():
              fake_pose_score, _ = self.D(fake_pose_v)
          else:
            fake_pose_score, _ = self.D(fake_pose_v)

          # G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)
          G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), torch.ones(fake_pose_score.shape[0]).to(fake_pose_score.device))

          if not self.fewshot_rec: ## TODO it is connected to fewshot_style. If fewshot_style is false, reconstruction is ignored.
            rec_flag = 0 ## ablation to remove the reconstruction loss

          # pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag
          pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones(fake_pose.shape[0]).to(fake_pose.device)) * rec_flag

          internal_losses.append(pose_loss)
          internal_losses.append(G_gan_loss)
        else:
          with torch.no_grad():
            fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
          internal_losses.append(torch.zeros(1)[0].to(y_pose.device))
          internal_losses.append(torch.zeros(1)[0].to(y_pose.device))

        partial_i_loss += i_loss_grounding
        internal_losses += partial_i_loss
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args

  
class EWC():
  def __init__(self, G, D, data, parent, input_modalities):
    self.G = deepcopy(G)
    self.D = deepcopy(D)
    self.data = data
    self.parent = parent
    self.input_modalities = input_modalities
        
    ## Store the old parameters
    self._means = {'G':{}, 'D':{}}
    for n, p in self.G.named_parameters():
      self._means['G'][n] = p.data
    for n, p in self.D.named_parameters():
      self._means['D'][n] = p.data

    self._fisher = {'G':{}, 'D':{}}
    for n, p in self.G.named_parameters():
      self._fisher['G'][n] = torch.zeros_like(p.data)
    for n, p in self.D.named_parameters():
      self._fisher['D'][n] = torch.zeros_like(p.data)

    self.estimate_fisher()
    
  def estimate_fisher(self):
    count = 0
    
    ## make it a one sized batch
    
    for batch in tqdm(self.data, desc='estimating fisher'):
      self.G.zero_grad()
      self.D.zero_grad()

      x, y_, y = self.parent.get_processed_batch(batch)
      new_kwargs = {'input_modalities':self.input_modalities,
                    'sample_flag': False,
                    'text/token_duration':batch['text/token_duration'].cuda(),
                    'text/token_count':batch['text/token_count'].cuda(),
                    'description':'train'}
      
      y_cap, internal_losses = self.G(x, y, **new_kwargs)
      logits, _ = self.D(self.get_velocity(y_cap))
      loss = torch.nn.functional.l1_loss(logits, torch.ones_like(logits))
      loss.backward()
      
      count += y.shape[0]
      for n, p in self.G.named_parameters():
        if p.grad is not None:
          self._fisher['G'][n] += p.grad.data ** 2
      for n, p in self.D.named_parameters():
        if p.grad is not None:
          self._fisher['D'][n] += p.grad.data ** 2

    for key in ['G', 'D']:
      for n, p in self._fisher[key].items():
        self._fisher[key][n] = p / count
    
    
  def penalty(self, model, kind):
    loss = 0
    for n, p in model.named_parameters():
      loss += (self._fisher[kind][n] * (p - self._means[kind][n]) ** 2).sum()
    return loss

  def get_velocity(self, x):
    x_v = x[..., 1:, :] - x[..., :-1, :]
    return torch.cat([torch.zeros_like(x[..., 0:1, :]), x_v], dim=-2)

class EWCsample():
  def __init__(self, *args, **kwargs):
    pass

  def penalty(self, *args, **kwargs):
    return 0 ## no penalty while sampling

  
class ewcGAN(GAN):
  def __init__(self, G, D, dg_iter_ratio=1,
               lambda_D=1, lambda_gan=1, lr=0.0001,
               criterion='MSELoss', optim='Adam', joint=False,
               update_D_prob_flag=True, no_grad=True, **kwargs):
    super().__init__(G=G, D=D, dg_iter_ratio=dg_iter_ratio, lambda_D=lambda_D, lambda_gan=lambda_gan,
                     lr=lr, criterion=criterion, optim=optim, joint=joint,
                     update_D_prob_flag=update_D_prob_flag, no_grad=no_grad, **kwargs)

    self.ewc = None
    self.data_train = kwargs['data_train']
    self.parent = kwargs['parent']
    self.ewc_lambda = 1 if kwargs.get('ewc_lambda') is None else kwargs.get('ewc_lambda')

  def forward(self, x_audio, y_pose, **kwargs):
    ## initialize ewc
    if self.ewc is None and not kwargs['sample_flag']:
      self.ewc = EWC(self.G, self.D, self.data_train, self.parent, kwargs['input_modalities'])
    elif self.ewc is None and kwargs['sample_flag']:
      self.ewc = EWCsample()

    internal_losses = []

    ## in a fewshot setting real_pose is taken from the fewshot_data
    ## if rec_flag == 1, x_audio is from fewshot_data (i.e. target) else it is from source data
    ## if rec_flag == 1, calculate the reconstruction loss as the x_audio and y_pose are parallel

    y_pose, x_audio, rec_flag, kwargs = self.get_fewshot_data(y_pose, x_audio, **kwargs)

    ## get confidence values
    if 'confidence' in kwargs:
      confidence = kwargs['confidence']
    else:
      confidence = 1

    W, outputs = self.estimate_weights(x_audio, y_pose, **kwargs)
    W_loss = self.estimate_weights_loss(W)
    if self.update_D_prob_flag:
      self.update_D_prob(W)

    ## if the generator has a gan_flag, it can be used to control if a gan is being trained or not
    gan_flag = self.G.gan_flag if hasattr(self.G, 'gan_flag') else True
    if self.training and gan_flag:
    #if True:
      ## update lambdas
      self.lambda_D, self.lambda_gan = self.lambda_scheduler.step()

      if torch.rand(1).item() < self.D_prob: ## run discriminator
        self.G.eval() ## generator must be in eval mode to activate eval mode of bn and dropout
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':False})
        with torch.no_grad():
          fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
          args = args[0] if len(args)>0 else {}
        self.G.train(self.training) ## take back generator to it's parent's mode
        real_pose = y_pose

        ## convert pose to velocity
        real_pose_v = self.get_velocity(real_pose, x_audio)
        fake_pose_v = self.get_velocity(fake_pose, x_audio)

        self.fake_flag = True
        #if torch.rand(1).item() < 0.5:
        if True:
          fake_pose_score, _ = self.D(fake_pose_v.detach())
          fake_D_loss = self.lambda_D * self.get_gan_loss(fake_pose_score, self.get_fake_gt(fake_pose_score), torch.ones_like(1/W_loss))
          self.fake_flag = True
        else:
          fake_D_loss = torch.zeros(1)[0].to(fake_pose_v.device)
          self.fake_flag = False
        real_pose_score, _ = self.D(real_pose_v)
        real_D_loss = self.get_gan_loss(real_pose_score, self.get_real_gt(real_pose_score), torch.ones_like(W_loss))

        ## add gradient_penalty
        gp = self.compute_gradient_penalty(D=self.D,
                                           real_samples=real_pose_v.data,
                                           fake_samples=fake_pose_v.data)
        if gp is not None:
          real_D_loss += gp/2 * self.lambda_gp
          fake_D_loss += gp/2 * self.lambda_gp
        
        internal_losses.append(real_D_loss)
        internal_losses.append(fake_D_loss)
        internal_losses += partial_i_loss
        internal_losses += [self.ewc.penalty(self.D, 'D') * self.ewc_lambda] ## add the ewc penalty to the discriminator
        self.G_flag = False
      else:
        ## Pass the information to Generator about the training cycle
        kwargs.update({'G_flag':True})
        
        fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
        args = args[0] if len(args)>0 else {}
        ## convert pose to velocity
        fake_pose_v = self.get_velocity(fake_pose, x_audio)
        if self.no_grad:
          with torch.no_grad():
            fake_pose_score, _ = self.D(fake_pose_v)
        else:
          fake_pose_score, _ = self.D(fake_pose_v)

        G_gan_loss = self.lambda_gan * self.get_gan_loss(fake_pose_score, self.get_real_gt(fake_pose_score), 1/W_loss)

        pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, 1/W_loss) * rec_flag

        internal_losses.append(pose_loss)
        internal_losses.append(G_gan_loss)
        internal_losses += partial_i_loss
        internal_losses += [self.ewc.penalty(self.G, 'G') * self.ewc_lambda] ## add the ewc penalty to the generator
        self.G_flag = True
    else:
      fake_pose, partial_i_loss, *args = self.G(x_audio, y_pose, **kwargs)
      args = args[0] if len(args)>0 else {}
      pose_loss = self.get_loss(fake_pose*confidence, y_pose*confidence, torch.ones_like(W_loss))

      internal_losses.append(pose_loss)
      internal_losses.append(torch.tensor(0))
      internal_losses += partial_i_loss
      self.G_flag = True

    args.update(dict(W=W))
    return fake_pose, internal_losses, args
    
