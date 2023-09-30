import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AdamW

from model import *
from layersUtils import requires_grad, accumulate

from data import *
#Data, ZNorm, Compose, RemoveJoints, KMeans, POSCluster, Relative2Parent, collate_fn_pad, PoseStarts, RandomTranslate
#from evaluation import PCK, L1, VelL1, Diversity, Expressiveness, F1
from argsUtils import get_args_perm
import evaluation
from animation import save_animation
from parallel import parallel

from pycasper.name import Name
from pycasper.BookKeeper import *
from pycasper import torchUtils

import wandb

import trainer_chooser
from argparse import Namespace
from pathlib import Path

from tqdm import tqdm
import json
from functools import partial
import itertools
import pickle as pkl
from collections import Counter
import pdb

FLOAT = torch.float # torch.float | torch.double

## Imports for get_pretrained_trainer
from argparse import Namespace
from argsUtils import get_args_perm
from pycasper.BookKeeper import BookKeeper
from pathlib import Path
import copy
import trainer_chooser

def get_pretrained_trainer(path2weights, cuda):
  args_new = Namespace(load=path2weights, cuda=cuda, save_dir=Path(path2weights).parent.as_posix(), pretrained_model=1)
  args, args_perm = get_args_perm()
  args.__dict__.update(args_perm[0])

  ## Load the correct args from the weight file to choose the correct Trainer
  args.__dict__.update(args_new.__dict__)
  book = BookKeeper(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1, 'sample_all_styles':0, 'mix':0, 'optim_separate':None, 'path2data':args.path2data})
  
  ## Choose the correct Trainer
  Trainer = trainer_chooser.trainer_chooser(book.args)

  ## Load the Trainer
  trainer = Trainer(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1, 'path2data':args.path2data})
  return trainer

def get_pretrained_trainers(pathdict, cuda):
  if isinstance(pathdict, str):
    return get_pretrained_trainer(pathdict, cuda)
  elif isinstance(pathdict, list):
    return [get_pretrained_trainer(path, cuda) for path in pathdict]
  elif isinstance(pathdict, dict):
    return {key:get_pretrained_trainer(pathdict[key], cuda) for key in pathdict}
  else:
    assert False, 'pathdict must be a string, list or dict kind'
'''
Class Heirarchy
Trainer Skeleton
  - TrainerBase
     - Trainer
     - TrainerGAN
'''

class TrainerBase():
  def __init__(self, args, args_subset, args_dict_update={}):
    self.book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                           tensorboard=args.tb)
    self.args = self.book.args
    
    ## Training parameters
    self.path2data = self.args.path2data
    self.path2outdata = self.args.path2outdata
    self.speaker = self.args.speaker
    self.modalities = self.args.modalities
    if self.args.input_modalities is None: ## infer inputs and outputs from self.modalities
      self.input_modalities = self.modalities[1:]
    else:
      self.input_modalities = self.args.input_modalities
    if self.args.output_modalities is None:
      self.output_modalities = self.modalities[:1]
    else:
      self.output_modalities = self.args.output_modalities

    self.mask = self.args.mask
    self.mask = list(np.concatenate([np.r_[i] if isinstance(i, int) else np.r_[eval(i)] for i in self.mask])) ## convert ranges to list of numbers
    self.split = self.args.split
    self.batch_size = self.args.batch_size
    self.shuffle = True if self.args.shuffle else False
    self.time = self.args.time
    self.fs_new = self.args.fs_new if isinstance(self.args.fs_new, list) else [self.args.fs_new] * len(self.modalities)
    self.window_hop = self.args.window_hop
    self.num_epochs = self.args.num_epochs
    self.num_clusters = args.num_clusters
    self.feats = self.args.feats
    self.num_training_sample = self.args.num_training_sample
    self.style_losses = self.args.style_losses
    self.style_iters = self.args.style_iters
    self.sample_all_styles = self.args.sample_all_styles
    self.repeat_text = self.args.repeat_text
    self.num_workers = self.args.num_workers
    
    self.relative2parent = self.args.relative2parent
    self.quantile_sample = self.args.quantile_sample
    self.quantile_num_training_sample = self.args.quantile_num_training_sample
    self.heuristic_pose_starts = self.args.heuristic_pose_starts

    self.metrics = self.args.metrics
    self.load_data = self.args.load_data
    self.pretrained_model = self.args.pretrained_model
    self.modelKwargs = {}

    ## parameter to use pad_collate for the dataloaders
    self.text_in_modalities = False
    for modality in self.modalities:
      if 'text' in modality:
        self.text_in_modalities = True

    ## Device
    self.device = torch.device('cuda:{}'.format(self.args.cuda)) if self.args.cuda>=0 else torch.device('cpu')

    ## Get Data
    self.data, self.data_train, self.data_dev, self.data_test = self.get_data()

    ## Get style
    self.style_dict = self.data.style_dict
    self.style_dim = self.args.style_dim

    ## Data shape
    self.data_shape = self.data.shape

    # define input and output modalities TODO hadcoded
    self.output_modality = self.output_modalities[0]

    ## Parents
    self.parents = self.data.modality_classes[self.output_modality].parents

    ## Get cluster Transform for Cluster based models
    # if self.num_clusters is not None or self.args.pos:
    #   self.cluster = self.get_cluster()
    #   if self.args.pos:
    #     self.num_clusters = len(self.cluster.tagset)

    if self.num_clusters is not None:
      self.cluster = self.get_cluster()
    if self.args.pos:
      self.cluster_pos = self.get_pos_cluster()
      self.num_clusters_pos = len(self.cluster_pos.tagset)

    if args.preprocess_only:
      print('Data Preprocessing done')
      exit(1)

    ## ZNorm + other transforms
    self.pre = self.get_pre()

    ## Remove Joints / Reinsert Joints from data
    self.transform = self.get_transforms()

    ## transform the confidence matrix
    self.transform_confidence = self.get_transforms()
    self.confidence_loss = Confidence(beta=1, epsilon=0.5)

    ## label histogram
    if self.num_clusters is not None:
      if self.args.replay is not None and self.args.fewshot_seed is not None:
        self.num_styles = len(self.fewshot_data.speaker) if self.speaker[0] != 'all' else len(self.data.speakers)
      else:
        self.num_styles = len(self.speaker) if self.speaker[0] != 'all' else len(self.data.speakers)
      if self.sample_all_styles: ## if all styles are being sampled, create the permutation of the kwargs_names
        kwargs_names = ['{}_{}'.format(sp1, sp2) for sp2 in self.speaker for sp1 in self.speaker if sp1 != sp2]
      else:
        kwargs_names = ['style']
        kwargs_names.append('same')
      self.labels_hist = {kwargs_name:{desc:{i:torch.zeros(self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in kwargs_names}
      self.labels_hist_tensor = {kwargs_name:{desc:{i:torch.zeros(1, self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in kwargs_names}

      #self.labels_hist = {kwargs_name:{desc:{i:torch.zeros(self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in ['same', 'style']}
      #self.labels_hist_tensor = {kwargs_name:{desc:{i:torch.zeros(1, self.num_clusters) for i in range(self.num_styles)} for desc in ['test', 'train', 'dev']} for kwargs_name in ['same', 'style']}
    if args.mix and args.load:
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=['mix'])
    elif self.args.sample_all_styles != 0 and args.load and self.args.replay is not None:
      sample_styles = ['same'] + ['_'.join(list(perm)) for perm in itertools.permutations(self.fewshot_data.speaker, 2)]
      self.Stack = partial(evaluation.Stack, n=len(self.fewshot_data.speaker), speakers=self.fewshot_data.speaker, sample_styles=sample_styles)    
    elif self.args.sample_all_styles != 0 and args.load:
      sample_styles = ['same'] + ['_'.join(list(perm)) for perm in itertools.permutations(self.speaker, 2)]
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=sample_styles)
    elif self.args.load and self.args.replay is not None and self.args.fewshot_seed is not None:
      self.Stack = partial(evaluation.Stack, n=len(self.fewshot_data.speaker), speakers=self.fewshot_data.speaker, sample_styles=['same', 'style'])
    elif self.args.load:
      self.Stack = partial(evaluation.Stack, n=len(self.data.speaker), speakers=self.data.speaker, sample_styles=['same', 'style'])
    else:
      self.Stack = partial(evaluation.Stack, n=0, speakers=[], sample_styles=['same'])

    ## Metrics
    self.metrics_init()

    ## Counter for reweighting
    self.weight_counter = Counter()

    ## Create Model
    self.update_modelKwargs()
    self.model = self.get_model()
    self.model.to(device=self.device, dtype=FLOAT)
    self.update_model()
    #device_ids = list(range(torch.cuda.device_count()))
    #self.model = nn.DataParallel(self.model, device_ids=device_ids)

    ## Load model
    if self.args.load:
      print('Loading Model')
      self.book._load_model(self.model, map_location=self.device)

    if not self.pretrained_model:
      self.book._copy_best_model(self.model)
    print('Model Created')
    
    ## Loss Function
    self.criterion = self.get_criterion()

    ## Optimizers
    self.G_optim, self.D_optim = self.get_optims()

    ## Scheduler
    self.schedulers = self.get_scheduler()

    ## Wandb
    self.wandb_init = partial(wandb.init,
                              dir=self.args.wandb_dir,
                              config=self.args.__dict__,
                              name=self.book.name.name,
                              project=self.args.wandb_project,
                              group='-'.join(self.speaker),
                              settings=wandb.Settings(start_method='fork'))
    if self.args.wandb:
      os.environ["WANDB_MODE"] = "dryrun" ## set to offline mode as nodes may not have internet access
      ## use wandb sync from the root node to sync
      # os.environ["WANDB_SILENT"] = "true" ## Silence the wandb logger


  def get_data(self):
    ## Load data iterables
    data = Data(self.path2data, self.speaker, self.modalities, self.fs_new,
                time=self.time, split=self.split, batch_size=self.batch_size,
                shuffle=self.shuffle, num_workers=self.num_workers,
                window_hop=self.window_hop, style_iters=self.style_iters,
                num_training_sample=self.num_training_sample,
                load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                quantile_num_training_sample=self.quantile_num_training_sample,
                weighted=self.args.weighted, filler=self.args.filler,
                num_training_iters=self.args.num_training_iters, audio_average=self.args.audio_average,
                fewshot_seed=self.args.fewshot_seed, k_shot=self.args.k_shot, p_avg=self.args.p_avg,
                cache=self.args.cache, replay=self.args.replay)

    data_train = data.train
    data_dev = data.dev
    data_test = data.test

    print('Data Loaded')

    return data, data_train, data_dev, data_test

  def get_criterion(self):
    return eval('torch.nn.' + self.args.loss)(**self.args.lossKwargs)

  def get_pck(self):
    return self.Stack(evaluation.PCK(num_joints=int(self.data_shape[self.output_modality][-1]/2)))

  def get_l1(self):
    return self.Stack(evaluation.L1())

  def get_VelL1(self):
    return self.Stack(evaluation.VelL1())

  def get_Diversity(self):
    mean = self.pre.transforms[-1].variable_dict[self.output_modality][0]
    remove_joints = RemoveJoints(self.mask)
    mean = remove_joints(mean).squeeze(0)
    return self.Stack(evaluation.Diversity(mean))

  def get_Expressiveness(self):
    mean = self.pre.transforms[-1].variable_dict[self.output_modality][0]
    remove_joints = RemoveJoints(self.mask)
    mean = remove_joints(mean).squeeze(0)
    return self.Stack(evaluation.Expressiveness(mean))

  def get_F1(self):
    cluster = KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=8, mask=self.mask, feats=self.feats)
    return self.Stack(evaluation.F1(num_clusters=8)), cluster

  def get_IS(self):
    speakers_rev = {sp:i for i,sp in enumerate(self.data.speakers)}
    if 'all' in self.speaker:
      speaker = self.data.speakers
    else:
      speaker = self.speaker

    weight = torch.Tensor([speakers_rev[sp.split('|')[0]] for sp in speaker]).float().unsqueeze(-1)
    return self.Stack(evaluation.InceptionScoreStyle(len(self.data.speakers), weight))

  def get_FID(self):
    return self.Stack(evaluation.FID())

  def get_W1(self):
    return self.Stack(evaluation.W1())

  def get_optims(self):
    if self.args.gan !=0:
      model_params = list(self.model.G.parameters())
    else:
      model_params = list(self.model.parameters())

    if self.args.optim_separate is not None: ## TODO  harcoded to work with text_encoder
      if self.args.gan != 0:
        bert_params = self.model.G.text_encoder.parameters()
      else:
        bert_params = self.model.text_encoder.parameters()
      bert_params = list(bert_params)
      G_optim = eval('torch.optim.' + self.args.optim)([{'params': bert_params,
                                                         'lr':self.args.optim_separate},
                                                        {'params': list(set(model_params) \
                                                                        - set(bert_params))}],
                                                       lr=self.args.lr, **self.args.optimKwargs)
    else:
      G_optim = eval('torch.optim.' + self.args.optim)(model_params, lr=self.args.lr, **self.args.optimKwargs)

    if self.args.gan != 0:
      D_optim = eval('torch.optim.' + self.args.optim)(self.model.D.parameters(), lr=self.args.lr, **self.args.optimKwargs)
    else:
      D_optim = None

    return G_optim, D_optim
    #return AdamW(self.model.parameters(), lr=self.args.lr, **self.args.optimKwargs)

  def get_scheduler(self):
    schedulers = []
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
      """ Create a schedule with a learning rate that decreases linearly after
      linearly increasing during a warmup period.
      """

      def lr_lambda(current_step):
        if current_step < num_warmup_steps:
          return float(current_step) / float(max(1, num_warmup_steps))
        return max(
          0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
          )

      return lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
    if self.args.scheduler == 'linear_decay':
      warmup_steps = self.args.scheduler_warmup_steps
      schedulers.append(get_linear_schedule_with_warmup(self.G_optim, warmup_steps, len(self.data.train)*self.num_epochs))
      if self.D_optim is not None:
        schedulers.append(get_linear_schedule_with_warmup(self.D_optim, warmup_steps, len(self.data.train)*self.num_epochs))
    else:
      schedulers.append(lr_scheduler.ExponentialLR(self.G_optim, gamma=self.args.gamma))
      if self.D_optim is not None:
        schedulers.append(lr_scheduler.ExponentialLR(self.D_optim, gamma=self.args.gamma))
    return schedulers

  def get_pre(self):
    transforms = []
    if self.heuristic_pose_starts:
      transforms.append(PoseStarts(self.modalities, key=self.speaker, data=self.data_train))

    if self.relative2parent:
      transforms.append(Relative2Parent())
      pre_op = Compose(transforms) ## if the mean variance is being calculated for the first time, it uses the pre_op on each batch before calculating mean var
    else:
      pre_op = None

    ## remove text/tokens
    hidden_modalities = ['text/tokens', 'text/filler', 'audio/silence']
    modalities = [mod for mod in self.modalities if mod not in hidden_modalities]

    transforms.append(ZNorm(modalities, key=self.speaker, data=self.data_train, relative2parent=self.relative2parent, pre=pre_op))
    return Compose(transforms)

  def get_transforms(self):
    return Compose([RemoveJoints(self.mask, self.parents)])

  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_gt(self, path2h5):
    Y, h5 = self.data.load(path2h5, self.output_modality)
    feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    Y = Y[()].reshape(-1, 2, feats_shape)
    Y[..., 0] = 0
    h5.close()
    return Y

  def get_confidence_loss(self, batch, y, y_cap):
    key = 'pose/confidence'
    if key in batch:
      confidence = self.transform_confidence(batch[key].to(self.device))
    else:
      return 0

    confidence = confidence.view(*y.shape)
    confidence_loss = self.confidence_loss(y, y_cap, confidence).mean()

    return confidence_loss

  def _update_labels(self, desc, style, kwargs_name):
    if self.num_clusters is not None:
      if self.args.gan and self.args.replay is not None:
        model = self.model.G.G
      elif self.args.gan:
        model = self.model.G
      else:
        model = self.model

      if kwargs_name is None:
        kwargs_name = 'same'
      ## update only if labels_cap_soft is an attribute in the model
      try:
        if hasattr(model, 'labels_cap_soft'):
          if desc == 'test':
            self.labels_hist_tensor[kwargs_name][desc][style] = torch.cat([self.labels_hist_tensor[kwargs_name][desc][style], model.labels_cap_soft.squeeze(0).detach().cpu().float()], dim=0)
          label = torch.argmax(model.labels_cap_soft.squeeze(0), dim=-1)
          label = label.detach().cpu()
          emb = torch.nn.Embedding(num_embeddings=self.num_clusters,
                                   embedding_dim=self.num_clusters,
                                   _weight=torch.eye(self.num_clusters))
          self.labels_hist[kwargs_name][desc][style] += emb(label).sum(dim=0)
      except:
        pass

  def _save_labels(self):
    if self.num_clusters is not None:
      if self.args.replay is not None:
        speakers = self.data.speakers if self.speaker[0] == 'all' else self.fewshot_data.speaker
      else:
        speakers = self.data.speakers if self.speaker[0] == 'all' else self.speaker
      labels_hist = {kwargs_name:{desc:{speakers[i]:self.labels_hist[kwargs_name][desc][i].numpy().tolist() for i in self.labels_hist[kwargs_name][desc]} for desc in ['test', 'train', 'dev']} for kwargs_name in self.labels_hist}
      labels_hist_tensor = {kwargs_name:{desc:{speakers[i]:self.labels_hist_tensor[kwargs_name][desc][i].numpy() for i in self.labels_hist_tensor[kwargs_name][desc]} for desc in ['test', 'train', 'dev']} for kwargs_name in self.labels_hist_tensor}

      hist_filename = self.book.name('histogram', 'json', self.book.save_dir)
      json.dump(labels_hist, open(hist_filename, 'w'))
      tensor_filename = self.book.name('style', 'pkl', self.book.save_dir)
      pkl.dump(labels_hist_tensor, open(tensor_filename, 'wb'))

  def metrics_init(self): ## metric objects
    self.pck = self.get_pck()
    self.l1 = self.get_l1()
    self.vel_l1 = self.get_VelL1()
    self.diversity = self.get_Diversity()
    self.expressiveness = self.get_Expressiveness()
    self.f1, self.f1_cluster = self.get_F1()
    if not self.pretrained_model: ## if this is a pretrained model do not get self.IS to avoid a loop
      try:
        self.IS = self.get_IS()
      except:
        pass
      
    self.fid = self.get_FID()
    self.w1 = self.get_W1()

    self.metrics_objects = [self.pck, self.l1, self.vel_l1, self.diversity, self.expressiveness, self.f1, self.fid, self.w1]

    if hasattr(self, 'IS'):
      self.metrics_objects.append(self.IS)

  def metrics_reset(self, **kwargs):
    for obj in self.metrics_objects:
      obj.reset(**kwargs)

  @property
  def metric_order(self): ## order of metrics to show while training
    if self.metrics:
      metric_order = ['pck', 'spatialNorm',
                      'diversity', 'diversity_gt', 'F1',
                      'FID', 'W1_vel', 'W1_acc',
                      'style_IS'
                      # 'style_IS_subset',
                      # 'style_F1', 'style_subset_F1',
                      # 'style_cce'
      ]
    else:
      metric_order = []
    return metric_order

  def get_metrics(self, desc):
    metrics = {}
    metrics_split = {}

    for metric in self.metrics_objects:
      avgs = metric.get_averages(desc)
      if isinstance(avgs, tuple):
        metrics.update(avgs[0])
        if not metrics_split:
          metrics_split = {kwargs_name:{speaker:{} for speaker in avgs[1][kwargs_name]} for kwargs_name in avgs[1]}

        for kwargs_name in avgs[1]:
          for speaker in avgs[1][kwargs_name]:
            metrics_split[kwargs_name][speaker].update(avgs[1][kwargs_name][speaker])
      else:
        metrics.update(avgs)
    return metrics, metrics_split

  def _save_metrics(self, metrics, filename='metrics'):
    metrics_filename = self.book.name(filename, 'json', self.book.save_dir)
    json.dump(metrics, open(metrics_filename, 'w'))

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = {}
    keys = ['text/token_count', 'text/token_duration', 'audio/silence', 'text/filler', 'pose/starts', 'pose/startsC']
    for key in keys:
      if key in batch:
        kwargs[key] = batch[key].to(self.device)

    ## add speaker name
    kwargs.update({'speaker':self.speaker})

    ## add current epoch
    kwargs.update(kwargs_subset)

    return kwargs

  def update_kwargs(self, kwargs, y=None):
    '''
    Update kwargs for sample_loop
    '''
    yield kwargs, None

  def start_exp(self):
    self.book._start_log()

  def finish_exp(self):
    self.book._stop_log()

  def _is_warmup(self, epoch, min_epoch):
    return False

  def mem_usage(self):
    out = subprocess.check_output(['nvidia-smi'])
    used = int(out.decode('utf-8').split('\n')[8].split('|')[2].strip().split('/')[0].strip()[:-3])
    total = int(out.decode('utf-8').split('\n')[8].split('|')[2].strip().split('/')[1].strip()[:-3])
    return used, total, (float(used)/total) * 100

  def detach(self, *args):
    for var in args:
      if isinstance(var, list):
        for va in var:
          del va
      elif isinstance(var, torch.Tensor):
        del var

    # for p in self.model.parameters():
    #   if p.grad is not None:
    #     del p.grad
    #torch.cuda.empty_cache()
    #used, total, percent = self.mem_usage()
    #tqdm.write('{}/{}: {}%'.format(used, total, percent))

  def train(self, exp_num):
    ## Wandb
    if self.args.wandb:
      self.wandb_init(job_type='train')
    for epoch in tqdm(range(self.num_epochs), ncols=20):
      train_loss, train_metrics, train_metrics_split = self.train_loop(self.data_train, 'train', epoch, num_iters=self.args.num_iters)
      dev_loss, dev_metrics, dev_metrics_split = self.train_loop(self.data_dev, 'dev', num_iters=self.args.num_iters)
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.data_test, 'test', num_iters=self.args.num_iters)
      if self.args.scheduler not in ['linear_decay']: ## update lr after each iteration if training bert
        self.schedulers_step() ## Change the Learning Rate

      ## update the weights for data_train
      if self.args.weighted:
        ## Normalize weights
        max_W = 10
        min_W = 0.1
        W_ = self.data_train.sampler.weights
        W_ = (W_ - W_.mean())/W_.std() + 1
        W_ = torch.min(torch.ones(1)[0].float()*max_W,
                       torch.max(torch.zeros(1)[0].float() + min_W, W_))
        if torch.isnan(W_).any():
          W_ = torch.ones_like(W_) ## reinit to ones if Weights suffer a large variation
        self.data_train.sampler.weights = W_

        W = self.data_train.sampler.weights
        D_prob = self.model.D_prob if hasattr(self.model, 'D_prob') else 0
        tqdm.write('W: {}/{}/{}/{}/{}'.format(W.mean(), W.std(), W.min(), W.max(), D_prob))
        most_common = str(self.weight_counter.most_common()[:5])
        least_common = str(self.weight_counter.most_common()[-5:])
        tqdm.write('samples: {} -- {}'.format(most_common, least_common))

      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()

      ## update wandb
      if self.args.wandb:
        wandb.log(train_metrics, commit=False)
        wandb.log(dev_metrics, commit=False)
        wandb.log(test_metrics, commit=True)

      ## update tensorboard
      if self.args.tb:
        self.book.update_tb({'scalar':[[f'{self.args.cpk}/train', train_loss, epoch],
                                       [f'{self.args.cpk}/dev', dev_loss, epoch],
                                        [f'{self.args.cpk}/test', test_loss, epoch],
                                       [f'{self.args.cpk}/pck_train',
                                        train_metrics['train_pck'], epoch],
                                       [f'{self.args.cpk}/pck_dev',
                                        dev_metrics['dev_pck'], epoch],
                                       [f'{self.args.cpk}/pck_test',
                                        test_metrics['test_pck'],
                                        epoch],
                                       [f'{self.args.cpk}/train_spatialNorm',
                                        train_metrics['train_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/dev_spatialNorm',
                                        dev_metrics['dev_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/test_spatialNorm',
                                        test_metrics['test_spatialNorm'], epoch]
        ]})
                             #'histogram':[[f'{self.args.cpk}/'+name,
                             #param.clone().cpu().detach().numpy(), epoch]
                             # for name, param in model.named_parameters()]})

      ## print results
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

#      warmup = self._is_warmup(epoch, np.ceil(len(self.data_train)/self.batch_size))
      if self.book.stop_training(self.model, epoch):
        if self.args.wandb:
          wandb.finish()
        break

    if self.args.num_iters > 0:
      #get the best model
      self.book._load_model(self.model, map_location=self.device)
      #calculate test loss for the complete dataset
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.data_test, 'test', 0)
      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()
      print('Final Results')
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])


  def train_loop(self, data, desc, epoch=0, num_iters=0, **train_loop_kwargs):
    ## init
    self.metrics_reset(description=desc)
    self.running_loss_init()

    if desc == 'train':
      self.model.train(True)
    else:
      self.model.eval()

    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    bar_format = '{desc}:' +'{n_fmt}/{total_fmt}[{elapsed}<{remaining}]'
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    for count, batch in enumerate(Tqdm):
      self.zero_grad()

      ## update weight counter
      if desc == 'train':
        self.weight_counter.update(batch['idx'].numpy())

      ## Transform batch before using in the model
      x, y_, y = self.get_processed_batch(batch)

      ## get kwargs like style
      kwargs = self.get_kwargs(batch, epoch=epoch, sample_flag=0, description=desc, **train_loop_kwargs)

      ## add noise to output to improve robustness of the model
      noise = torch.randn_like(y) * self.args.noise if self.args.noise > 0 else 0

      y_cap, internal_losses, args = self.forward_pass(desc, x, y+noise, **kwargs)

      args = args[0] if len(args)>0 else {} ## dictionary of args returned by model

      ## check if there are weights in *args
      if args.get('W') is not None and desc=='train' and self.args.weighted > 0:
        W = args['W']
        W_min = 0.1
        self.data_train.sampler.weights[batch['idx']] = torch.max(torch.zeros(1)[0].float() + W_min, W.cpu()) ## clip the weights to positive values

      ## Get mask to calculate the loss function
      src_mask_loss = args.get('src_mask_loss')
      src_mask_loss = src_mask_loss.unsqueeze(-1) if src_mask_loss is not None else torch.ones_like(y[:, :, 0:1])

      ## get confidence values and
      ## calculate confidence loss
      confidence_loss = self.get_confidence_loss(batch, y, y_cap)

      loss = self.calculate_loss(x, (y+noise)*src_mask_loss, y_cap*src_mask_loss, internal_losses)

      ## update tqdm
      losses = [l/c for l,c in zip(self.running_loss, self.running_count)] + [confidence_loss]
      Tqdm.set_description(self.tqdm_desc(desc, losses))
      Tqdm.refresh()

      if np.isnan(losses[0]):
        pdb.set_trace()
      if desc == 'train':
        self.optimize(loss + confidence_loss)

      ## Detach Variables to avoid memory leaks
      #x = x.detach()
      #y = y.detach()
      #loss = loss.detach()
      #y_cap = y_cap.detach()

      ## Evalutation
      y_cap = y_cap.to('cpu')
      src_mask_loss = src_mask_loss.to('cpu')
      with torch.no_grad():
        self.calculate_metrics(y_cap*src_mask_loss, y_*src_mask_loss, 'same', **{**kwargs, **args})
      self.detach(x, y, loss, y_cap, internal_losses)

      if count>=self.args.debug and self.args.debug: ## debugging by overfitting
        break

      ## if self.args.num_iters > 0, break training
      if count >= num_iters and num_iters > 0 and desc != 'train':
        Tqdm.close()
        break

    metrics = {}
    if self.metrics:
      metrics, metrics_split = self.get_metrics(desc)
    else:
      metrics, metrics_split = {}, {}

    return losses[0], metrics, metrics_split
    #return sum(losses), metrics

  def weight_estimate_loop(self, data, desc, epoch=0, num_iters=0):
    self.model.eval()

    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    bar_format = '{desc}:' +'{n_fmt}/{total_fmt}[{elapsed}<{remaining}]'
    Tqdm = tqdm(data, desc='update weights: '+self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    W = []
    for count, batch in enumerate(Tqdm):
      ## Transform batch before using in the model
      x, y_, y = self.get_processed_batch(batch)

      ## get kwargs like style
      kwargs = self.get_kwargs(batch, epoch=0, sample_flag=0, description=desc)

      w = self.forward_pass_weight(desc, x, y, **kwargs)
      W.append(w)

      if count>=self.args.debug and self.args.debug: ## debugging by overfitting
        break

      ## if self.args.num_iters > 0, break training
      if count >= num_iters and num_iters > 0:
        break

    ## update the weights for data sampler
    W = torch.cat(W)
    return W

  def sample(self, exp_num):
    ## Wandb
    if self.args.wandb:
      if self.sample_all_styles:
        self.wandb_init(job_type='sample_all_styles')
      else:
        self.wandb_init(job_type='sample')

    ## Create Output Directory
    self.dir_name = self.book.name.dir(self.args.save_dir)

    ## Load best Model
    self.book._load_model(self.model, map_location=self.device)

    train_loss, train_metrics, train_metrics_split = self.sample_loop(self.data_train.dataset.datasets, 'train')
    dev_loss, dev_metrics, dev_metrics_split = self.sample_loop(self.data_dev.dataset.datasets, 'dev')
    test_loss, test_metrics, test_metrics_split = self.sample_loop(self.data_test.dataset.datasets, 'test')    

    if self.sample_all_styles == 0: ## if all styles are sampled, then the results change, hence we don't update it in this case
      ## Save labels histogram
      self._save_labels()

      ## Save sample time metrics
      self._save_metrics(test_metrics_split, 'metrics')
      self._save_metrics(test_metrics, 'cummMetrics')

    print('Sampled- Train:{:.4f}/{:.4f}, '.format(train_loss, train_metrics['train_pck']) + \
          'Dev:{:.4f}/{:.4f}, '.format(dev_loss, dev_metrics['dev_pck']) + \
          'Test:{:.4f}/{:.4f}'.format(test_loss, test_metrics['test_pck']))

    ## print results
    self.book.print_res(epoch=0,
                        key_order=['train', 'dev', 'test'],
                        metric_order=self.metric_order,
                        exp=exp_num,
                        lr=0)

    ## Wandb
    if self.args.wandb:
      if len(self.speaker) > 1:
        wandb.log(test_metrics_split, commit=False)
      wandb.log(test_metrics, commit=True)
      wandb.finish()

  def sample_loop(self, data, desc, DATA=None, **sample_loop_kwargs):
    if DATA is None:
      DATA = self.data
    self.metrics_reset(description=desc)
    self.running_loss_init()
    self.model.eval()

    intervals = []
    start = []
    y_outs = []
    y_animates = []
    filenames = []
    keys = []

    ## collate function
    #if not self.repeat_text:
    if self.text_in_modalities:
      pad_keys = ['text/w2v', 'text/bert', 'text/token_duration', 'text/tokens']
      collate_fn = partial(collate_fn_pad, pad_key=pad_keys, dim=0)
    else:
      collate_fn = None

    len_data = len(data)
    bar_format = '{percentage:3.0f}%|' + '|' + ':{desc}'
    bar_format = '{percentage:3.0f}%[{elapsed}<{remaining}]' + ':{desc}'
    Tqdm = tqdm(data, desc=self.tqdm_desc(desc), leave=False, ncols=20, bar_format=bar_format)
    for count, loader in enumerate(Tqdm):
      ### load ground truth
      Y = self.get_gt(loader.path2h5)

      if len(loader) > 0:
        loader = DataLoader(loader, len(loader), shuffle=False, collate_fn=collate_fn)
        Y_cap = []

        for batch in loader:
          with torch.no_grad():
            ## Transform batch before using in the model
            x, y_, y = self.get_processed_batch(batch)
            kwargs = self.get_kwargs(batch, epoch=0, sample_flag=1, description=desc)

        batch_size = y.shape[0]
        X_ = [x_.view(1, -1, x_.shape[-1]) for x_ in x[:len(self.input_modalities)]]
        for x_ in x[len(self.input_modalities):]: ## hardcoded for auxillary labels
          X_.append(x_.view(1, -1))
        #if len(x) > len(self.input_modalities):
        #  X_.append(x[-1].view(1, -1))

        y = y.view(1, -1, y.shape[-1])
        x = X_

          ## based on kwargs_batch_size, repeat x, and y
          #y = torch.cat([y]*kwargs_batch_size, dim=0)
          #x = [torch.cat([x_]*kwargs_batch_size, dim=0) for x_ in x]
        for kwargs, kwargs_name in self.update_kwargs(kwargs, y_): ## update kwargs like style
          with torch.no_grad():
            ## Forward pass
            y_cap, internal_losses, args = self.forward_pass(desc, x, y, **kwargs)
            args = args[0] if len(args) > 0 else {}

            ## update labels histogram ## only update when the speaker is sampled with it's style
            self._update_labels(desc=desc, style=int(batch['style'][0, 0].item()), kwargs_name=kwargs_name)

            ## get confidence loss
            confidence_loss = self.get_confidence_loss(batch, y, y_cap)

            loss = self.calculate_loss(x, y, y_cap, internal_losses)

            ## Calculates PCK and reinserts data removed before training
            y_cap = y_cap.to('cpu')
            with torch.no_grad():
              y_cap = y_cap.view(batch_size, -1, y_cap.shape[-1])
              y_cap = self.calculate_metrics(y_cap, y_, kwargs_name, **{**kwargs, **args, **sample_loop_kwargs})
            Y_cap.append(y_cap)

            ## update tqdm
            losses = [l/c for l,c in zip(self.running_loss, self.running_count)] + [confidence_loss]
            Tqdm.set_description(self.tqdm_desc(desc, losses))
            Tqdm.refresh()

            self.detach(x, y, y_cap, loss, internal_losses)

          if Y_cap:
            intervals.append(batch['meta']['interval_id'][0])
            start.append(torch.Tensor([0]).to(torch.float))
            y_outs.append(torch.cat(Y_cap, dim=0))
            y_animates.append([torch.cat(Y_cap, dim=0), Y])

            dir_name = 'keypoints' if kwargs_name is None else 'keypoints_{}'.format(kwargs_name)
            filenames.append((Path(self.dir_name)/dir_name/'{}/{}/{}.h5'.format(desc,
                                                                                DATA.getSpeaker(intervals[-1]),
                                                                                intervals[-1])).as_posix())
            keys.append(self.output_modality)
            Y_cap = []
            #keys += [self.output_modality] * len(intervals)

      ## Save Keypoints
      if (count + 1) % 100 == 0 or count == len_data - 1: ## save files every 100 batches to prevent memory errors
        parallel(DATA.modality_classes[self.output_modality].append, # fn
                 -1, # n_jobs
                 filenames, keys, y_outs) # fn_args
        intervals = []
        start = []
        y_outs = []
        y_animates = []
        filenames = []
        keys = []

    if self.metrics:
      metrics, metrics_split = self.get_metrics(desc)
    else:
      metrics, metrics_split = {}, {}

    return losses[0], metrics, metrics_split

  def get_processed_batch(self, batch):
    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    y = self.transform(y)

    return x, y_, y

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    if kwargs_name is None:
      kwargs_name = 'same'
    elif kwargs_name in ['1', '2', '3', '4']:
      kwargs_name = 'same'
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    if 'style' in kwargs:
      idx = int(kwargs['style'].view(-1)[0].detach().cpu().item())
      style_vector = kwargs['style'].detach().cpu()
    else:
      idx = 0
      style_vector = torch.zeros(y_cap.shape[0], y_cap.shape[1]).long()

    try:
      self.IS(y_cap, style_vector, self.mask, idx=idx, kwargs_name=kwargs_name)
    except:
      pass

    ## Re-insert Joints
    y_cap = self.transform(y_cap, inv=True, batch_gt=y_)

    ## calculate L1
    self.l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.vel_l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.fid(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## undo normalization
    y_cap = self.pre({self.output_modality:y_cap}, inv=True)[self.output_modality]
    y_cap = y_cap.view(y_cap.shape[0], y_cap.shape[1], 2, -1) ## (B, T, 2, feats)
    y_ = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
    y_ = y_.view(y_.shape[0], y_.shape[1], 2, -1) ## (B, T, 2, feats)

    ## calculate wasserstein_distance-1 for avg velocity and accelaration
    self.w1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## Hardcode root as (0,0) for eternity for y and gt
    y_cap = y_cap.view(-1, 2, y_cap.shape[-1]) ## (BxT, 2, feats)
    y_cap[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity
    y_cap_out = y_cap

    y_gt = y_.view(-1, 2, y_cap.shape[-1])
    y_gt[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity

    ## calculate and add pck to the average meter
    self.pck(y_cap, y_gt, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## calculate STEEr, SEA and MoCA-{self.num_clusters} scores
    y_cap = self.transform(y_cap.view(1, y_cap.shape[0], -1), save_insert=False)
    y_gt = self.transform(y_gt.view(1, y_gt.shape[0], -1), save_insert=False)
    self.diversity(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.expressiveness(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.f1(self.f1_cluster(y_cap), self.f1_cluster(y_gt), idx=idx, kwargs_name=kwargs_name)
    return y_cap_out

  def get_model(self):
    raise NotImplementedError

  def update_modelKwargs(self):
    raise NotImplementedError

  def update_model(self):
    pass

  # def debug_model(self, model):
  #   try:
  #     model()
  #   except RuntimeError as e:
  #     if 'out of memory' in str(e):
  #       print('| WARNING: ran out of memory, retrying batch',sys.stdout)
  #       sys.stdout.flush()
  #       for p in model.parameters():
  #         if p.grad is not None:
  #           del p.grad  # free some memory
  #       torch.cuda.empty_cache()
  #       y= model()
  #     else:
  #       raise e

  def running_loss_init(self):
    raise NotImplementedError

  def tqdm_desc(self):
    raise NotImplementedError

  def zero_grad(self):
    raise NotImplementedError

  def forward_pass(self):
    raise NotImplementedError

  def calculate_loss(self):
    raise NotImplementedError

  def optimize(self, loss):
    if self.args.scheduler in ['linear_decay']:
      self.schedulers_step()

  def schedulers_step(self):
    for sched in self.schedulers:
      sched.step()

class Trainer(TrainerBase):
  '''
  Single modality Trainer with early fusion
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super(Trainer, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_model(self):
    return eval(self.args.model)(**self.modelKwargs)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape})

  def running_loss_init(self):
    self.running_loss = [0]
    self.running_count = [1e-10]

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' {:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' {:.4f} H:{:.4f}'.format(0, 0)

  def zero_grad(self):
    self.model.zero_grad()
    self.G_optim.zero_grad()
    if self.D_optim is not None:
      self.D_optim.zero_grad()

  def forward_pass(self, desc, x, y, **kwargs):
    x = torch.cat(x, dim=-1) ## Early Fusion
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y)

    return y_cap, internal_losses, args

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    for i_loss in internal_losses:
      loss += i_loss

    self.running_loss[0] += loss.item() * y_cap.shape[0]
    self.running_count[0] += y_cap.shape[0]

    return loss

  def optimize(self, loss):
    loss.backward()
    self.G_optim.step()
    super().optimize(loss)

class TrainerLate(Trainer):
  '''
  the inputs are not concatenated, passed as a list to the model
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super(TrainerLate, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)

    return y_cap, internal_losses, args

TrainerJointLate = TrainerLate
TrainerJoint = Trainer


class TrainerLateGest(TrainerLate):
  '''
  the inputs are not concatenated, passed as a list to the model
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_criterion(self):
    criterion = eval('torch.nn.' + self.args.loss)(**self.args.lossKwargs)
    def crit(y, y_cap):
        y_vel = y[:,1:,:] - y[:,:-1,:]
        y_cap_vel =  y_cap[:,1:,:] - y_cap[:,:-1,:]
        return criterion(y, y_cap) + 0.6 * criterion(y_vel, y_cap_vel)
    return crit

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, **kwargs)

    return y_cap, internal_losses, args



class TrainerGAN(TrainerBase):
  def __init__(self, args, args_subset, args_dict_update):
    super(TrainerGAN, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_model(self):
    ## Generator
    G = eval(self.args.model)(**self.modelKwargs)

    ## Discriminator
    if self.args.discriminator is None: ## infer the name of the discriminator
      D_modelname = '_'.join(self.args.model.split('_')[:-1] + ['D'])
    else:
      D_modelname = self.args.discriminator

    ## GAN Wrapper
    D_modelKwargs = {}
    if self.args.weighted:
      GANWrapper = GANWeighted
      D_modelKwargs.update({'out_shape':2})
    if self.args.gan == 'wgan-gp':
      GANWrapper = WGAN_GP
    else:
      GANWrapper = GAN

    ### add input_shape for self.args.joint
    input_shape = 0
    if self.args.joint:
      for mod in self.input_modalities:
        input_shape += self.data_shape[mod][-1]

    D_modelKwargs.update({'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask) + input_shape})
    if 'p' in self.modelKwargs: ## get the dropout parameter in the discrimiator as well
      D_modelKwargs.update({'p':self.args.modelKwargs['p']})

    try:
      D = eval(D_modelname)(**D_modelKwargs)
    except:
      print('{} not defined, hence defaulting to Speech2Gesture_D'.format(D_modelname))
      D = eval('Speech2Gesture_D')(**D_modelKwargs)

    ## GAN
    model = GANWrapper(G, D, lr=self.args.lr, criterion=self.args.loss, optim=self.args.optim,
                       dg_iter_ratio=self.args.dg_iter_ratio, lambda_gan=self.args.lambda_gan,
                       lambda_D=self.args.lambda_D, joint=self.args.joint, input_modalities=self.input_modalities,
                       update_D_prob_flag=self.args.update_D_prob_flag, no_grad=self.args.no_grad)
    return model

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape})

  def running_loss_init(self):
    self.running_loss = [0]*4
    self.running_count = [1e-10]*4

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} H:{:.4f}'.format(0, 0, 0, 0, 0)

  def zero_grad(self):
    self.model.zero_grad()
    self.G_optim.zero_grad()
    self.D_optim.zero_grad()

  def forward_pass(self, desc, x, y, **kwargs):
    x = torch.cat(x, dim=-1) ## Early Fusion
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, **kwargs)
    return y_cap, internal_losses, args

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
          self.running_count[i+2] += y_cap.shape[0]

      loss += i_loss
    return loss

  def get_norm(self, model):
    params = []
    for param in model.parameters():
      params.append(param.grad.view(-1))
    return torch.norm(torch.cat(params))

  def optimize(self, loss):
    loss.backward()
    if self.model.G_flag: ## TODO
      torch.nn.utils.clip_grad_norm_(self.model.G.parameters(), 1) ## TODO
      self.G_optim.step() ## TODO
    else:
      torch.nn.utils.clip_grad_norm_(self.model.D.parameters(), 1) ## TODO
      self.D_optim.step() ## TODO
    super().optimize(loss)

class TrainerLateGAN(TrainerGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def forward_pass_weight(self, desc, x, y, **kwargs):
    w = self.model.estimate_weights(x, y, input_modalities=self.input_modalities, **kwargs)
    return w

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)

    return y_cap, internal_losses, args

class TrainerLateTransformerGAN(TrainerGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_processed_batch(self, batch):
    batch = self.pre(batch)

    text_modalities = self.input_modalities
    #text_modalities.append('text/token_count')

    x = [batch[mod] for mod in text_modalities]
    y_ = batch[self.output_modality]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    y = self.transform(y)

    return x, y_, y

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, input_modalities=self.input_modalities, desc=desc, **kwargs)

    return y_cap, internal_losses, args

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = super().get_kwargs(batch, **kwargs_subset)
    return kwargs

class TrainerNoiseOnly(Trainer):
  '''
  Trainer with Noise as input
  '''
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_processed_batch(self, batch):
    batch = self.pre(batch)
    x = [torch.randn_like(batch[mod]) for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y


class TrainerLateCluster(TrainerLate):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.transform_cluster = self.get_transforms()

  def running_loss_init(self):
    self.running_loss = [0]*3
    self.running_count = [1e-10]*3

  def tqdm_desc(self, desc, losses=[]):
    if losses:
      return desc+' pose:{:.4f} label:{:.4f} H:{:.4f}'.format(*losses)
    else:
      return desc+' pose:{:.4f} label:{:.4f} H:{:.4f}'.format(0, 0, 0)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = self.criterion(y_cap, y)
    self.running_loss[0] += loss.item() * y_cap.shape[0]
    self.running_count[0] += y_cap.shape[0]
    for i, i_loss in enumerate(internal_losses):
      self.running_loss[i+1] += i_loss.item() * y_cap.shape[0]
      self.running_count[i+1] += y_cap.shape[0]
      loss += i_loss

    return loss

  # def update_modelKwargs(self):
  #   modelKwargs = {}
  #   modelKwargs.update(self.args.modelKwargs)
  #   modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
  #                       'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
  #                       'num_clusters':self.num_clusters,
  #                       'cluster':self.cluster,
  #                       'shape':self.data_shape})

  # def get_model(self):

  #   return eval(self.args.model)(**modelKwargs)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})


  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_processed_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)
    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    ## Append cluster labels
    x.append(labels)

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y

TrainerJointLateCluster = TrainerLateCluster

class TrainerLateClusterGAN(TrainerLateGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super(TrainerGAN, self).__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    self.transform_cluster = self.get_transforms()

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'cntr', 'H', 'mem%']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    if not losses:
      losses = [0]* len(self.running_loss)
      if self.args.mem_usage:
        losses[-1] = self.mem_usage()[-1]
    else:
      if self.args.mem_usage:
        losses[-2] = self.mem_usage()[-1]
    return desc + loss_str.format(*losses)

  # def running_loss_init(self):
  #   self.running_loss = [0]*5
  #   self.running_count = [1e-10]*5

  # def tqdm_desc(self, desc, losses=[]):
  #   if losses:
  #     return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'.format(*losses)
  #   else:
  #     return desc+' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'.format(0, 0, 0, 0, 0, 0)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0

    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          if not self.model.fake_flag and i == 1:
            pass
          else:
            self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
            self.running_count[i+2] += y_cap.shape[0]
      else:
        self.running_loss[i+2] = i_loss.item() * y_cap.shape[0]
        self.running_count[i+2] += y_cap.shape[0]
      loss += i_loss
    return loss

  def get_cluster(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=self.num_clusters, mask=self.mask, feats=self.feats)

  def get_processed_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)
    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    ## Append cluster labels
    x.append(labels)

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y

TrainerJointLateClusterGAN = TrainerLateClusterGAN

class TrainerNN(Trainer):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.audio, self.pose = self.get_train_data()


  def get_model(self):
    modelKwargs = {}
    modelKwargs.update(self.args.modelKwargs)
    modelKwargs.update({'shape':self.data_shape})
    return eval(self.args.model)(**modelKwargs)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'shape':self.data_shape})

  def get_train_data(self):
    audio, pose = [], []
    for batch in self.data_train:
      x, y_, y = self.get_processed_batch(batch)
      audio.append(x[0].mean(dim=1))
      pose.append(y)
    return torch.cat(audio, dim=0), torch.cat(pose, dim=0)

  def forward_pass(self, desc, x, y, **kwargs):
    if desc == 'train' and self.model.training:
      y_cap, internal_losses, *args = self.model(x, y, audio=self.audio, pose=self.pose)
    else:
      with torch.no_grad():
        y_cap, internal_losses, *args = self.model(x, y, audio=self.audio, pose=self.pose)

    return y_cap, internal_losses, args

  def optimize(self, loss):
    pass

TrainerRand = TrainerNN
TrainerMean = TrainerNN

class TrainerStyleClassifier(Trainer):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'shape':self.data_shape,
                             'style_dict':self.style_dict})

  def get_processed_batch(self, batch):
    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch['style'].long()[:,0]

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    x = [self.transform(x_) for x_ in x]

    return x, y_, y

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    return y_cap


class TrainerLateClusterStyleGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'style_dict':self.style_dict,
                             'style_dim':self.style_dim})

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = super().get_kwargs(batch, **kwargs_subset)
    ## Style Vector
    kwargs.update({'style':batch['style'].long().to(self.device)})
    return kwargs

  def update_kwargs(self, kwargs, y=None):
    if self.sample_all_styles:
      style_id = kwargs['style'].view(-1)[0].cpu().item()
      kwargs_list = [kwargs.copy()]
      kwargs_names = [None]
      for style_shift in range(1, self.num_styles):
        kwargs_temp = kwargs.copy()
        kwargs_temp['style'] = (kwargs_temp['style'] + style_shift) % self.num_styles
        kwargs_list.append(kwargs_temp)
        style_shift_id = (style_id + style_shift) % self.num_styles
        kwargs_names.append('{}_{}'.format(self.speaker[style_id], self.speaker[style_shift_id]))
        #kwargs_names = [None, 'style']
    else:
      kwargs_list = [kwargs.copy()]
      kwargs['style'] = (kwargs['style'] + 1) % self.num_styles
      kwargs_list.append(kwargs)
      kwargs_names = [None, 'style']

    for kwargs_, kwargs_name in zip(kwargs_list, kwargs_names):
      yield kwargs_, kwargs_name

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'label',
            'id_in', 'id_out',
            'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

  # def update_kwargs(self, kwargs):
  #   kwargs_list = [kwargs.copy()]
  #   kwargs['style'] = (kwargs['style'] + 1) % self.num_styles
  #   kwargs_list.append(kwargs)
  #   kwargs_names = [None, 'style']
  #   for kwargs, kwargs_name in zip(kwargs_list, kwargs_names):
  #     yield kwargs, kwargs_name

TrainerJointLateClusterStyleGAN = TrainerLateClusterStyleGAN

class TrainerLateClusterStyleGANMix(TrainerLateClusterStyleGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_kwargs(self, kwargs, y=None):
    kwargs_list = []
    kwargs['style'] = torch.ones(kwargs['style'].shape[0], kwargs['style'].shape[1], len(self.style_dict)).to(kwargs['style'].device).float()/len(self.style_dict)
    kwargs_list.append(kwargs)
    kwargs_names = ['mix']
    for kwargs, kwargs_name in zip(kwargs_list, kwargs_names):
      yield kwargs, kwargs_name

TrainerJointLateClusterStyleGANMix = TrainerLateClusterStyleGANMix

class TrainerLateClusterStyleDisentangleGAN(TrainerLateClusterStyleGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'style_dict':self.style_dict,
                             'style_dim':self.style_dim,
                             'style_losses':self.style_losses})


  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'con_+', 'con_-',
            'id_a', 'id_p',
            'c_a', 'c_p',
            'st_a', 'st_p',
            'rec_a', 'rec_p',
            'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      if i < 2:
        if self.model.G_flag: ## TODO
          self.running_loss[i] += i_loss.item() * y_cap.shape[0]
          self.running_count[i] += y_cap.shape[0]
        else:
          self.running_loss[i+2] += i_loss.item() * y_cap.shape[0]
          self.running_count[i+2] += y_cap.shape[0]
      else:
        self.running_loss[i+2] = i_loss.item() * y_cap.shape[0]
        self.running_count[i+2] += y_cap.shape[0]
      loss += i_loss
    return loss

TrainerJointLateClusterStyleDisentangleGAN = TrainerLateClusterStyleDisentangleGAN

class TrainerLateClusterStyleLearn(TrainerLateCluster):
  def __init__(self, args, args_subset, args_dict_update):
    self.pretrained_trainer = self.get_pretrained_model(args.pretrained_model_weights)
    args_dict_update.update({'num_clusters':self.pretrained_trainer.num_clusters})
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]

  def get_pretrained_model(self, path2weights):
    args_new = Namespace(load=path2weights, cuda=-1, save_dir=Path(path2weights).parent.as_posix(), pretrained_model=1)
    args, args_perm = get_args_perm()
    args.__dict__.update(args_perm[0])
    args.__dict__.update(args_new.__dict__)

    book = BookKeeper(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1})
    Trainer = trainer_chooser.trainer_chooser(book.args)
    trainer = Trainer(args, [], args_dict_update = {'load_data':0, 'pretrained_model':1})
    return trainer

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.pretrained_trainer = self.get_pretrained_model(self.args.pretrained_model_weights)
    self.modelKwargs.update({'style_dict':self.pretrained_trainer.style_dict,
                             'pretrained_model':self.pretrained_trainer.model})

  @property
  def loss_kinds(self):
    return ['pose', 'label', 'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

  def calculate_loss(self, x, y, y_cap, internal_losses):
    #loss = self.criterion(y_cap, y)
    #self.running_loss[0] += loss.item() * y_cap.shape[0]
    #self.running_count[0] += y_cap.shape[0]
    loss = 0
    for i, i_loss in enumerate(internal_losses):
      self.running_loss[i] += i_loss.item() * y_cap.shape[0]
      self.running_count[i] += y_cap.shape[0]
      loss += i_loss

    return loss

  def get_kwargs(self, batch, **kwargs_subset):
    kwargs = super().get_kwargs(batch, **kwargs_subset)

    ## Style Vector
    kwargs.update({'style':batch['style'].long().to(self.device)})
    return kwargs

TrainerJointLateClusterStyleLearn = TrainerLateClusterStyleLearn

class TrainerLateClusterPOSGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    #self.num_clusters = len(self.cluster.tagset) ## update the number of clusters
  def get_pos_cluster(self):
    return POSCluster()

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'num_clusters_pos':self.num_clusters_pos})

  def get_processed_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))
    self.cluster_pos.update(batch)
    labels_pos = self.cluster_pos(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    ## Append cluster labels
    x.append(labels)
    x.append(labels_pos)

    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the masked joints
    y = self.transform(y)

    return x, y_, y

TrainerJointLateClusterPOSGAN = TrainerLateClusterPOSGAN


class TrainerLateClusterContrastiveGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    self.transforms_contrastive.transforms[0].znorm = self.pre.transforms[-1]

  def get_contrastive_transforms(self):
    skel = self.data.modality_classes[self.output_modality]
    if 'translate_limits' in self.modelKwargs:
      max = self.modelKwargs['translate_limits']
    else:
      max = [50, 50]

    transforms = [RandomTranslate(max=max, mask=self.mask, skel=skel,
                                  znorm=None, output_modality=self.output_modality)]
    return Compose(transforms)

  def update_modelKwargs(self):
    super().update_modelKwargs()
    self.transforms_contrastive = self.get_contrastive_transforms()
    self.modelKwargs.update({'transforms':self.transforms_contrastive})
    self.modelKwargs.update({'num_batches':len(self.data_train)})

  # @property
  # def loss_kinds(self):
  #   return ['pose', 'G_gan',
  #           'real_D', 'fake_D',
  #           'cntr', 'H', 'mem%']

  # def running_loss_init(self):
  #   self.running_loss = [0]*len(self.loss_kinds)
  #   self.running_count = [1e-10]*len(self.loss_kinds)

  # def tqdm_desc(self, desc, losses=[]):
  #   loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
  #   #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
  #   if not losses:
  #     losses = [0]* len(self.running_loss)
  #     losses[-1] = self.mem_usage()[-1]
  #   else:
  #     losses[-2] = self.mem_usage()[-1]
  #   return desc + loss_str.format(*losses)

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    y_cap_out = super().calculate_metrics(y_cap, y_, kwargs_name, **kwargs)
    if 'z' in kwargs:
      z = kwargs['z'].cpu().transpose(-2,-1).numpy()
      if kwargs['description'] == 'train':
        self.iou.metric.cluster_z.update(z) ## train Kmeans
      p = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
      p = self.transform_cluster(p)
      self.iou(z, p) ## calculate_metrics

      z = torch.from_numpy(z).float().reshape(-1, z.shape[-1])
      label = self.cluster_p(p).view(-1)
      if True:#kwargs['sample_flag']:
        if kwargs['description'] == 'train':
          self.knn.metric.update(z, label)
          for kwargs_name in self.knn.metrics:
            for sp in self.knn.metrics[kwargs_name]:
              sp.feature_bank = self.knn.metric.feature_bank
              sp.feature_labels = self.knn.metric.feature_labels
          #self.knn(z, label)
        else:
          self.knn(z, label)

    return y_cap_out

  def get_cluster_p(self):
    return KMeans(variable_list=[self.output_modality], key=self.speaker, data=self.data_train, num_clusters=8, mask=self.mask, feats=['pose', 'velocity', 'speed'])
  
  def metrics_init(self):
    super().metrics_init()
    ## Get metric_init and add it to metrics_object
    self.cluster_p = self.get_cluster_p()

    ## Cluster IOU
    self.iou = self.Stack(evaluation.IOU(num_clusters=8, cluster_p=self.cluster_p))
    for kwargs_name in self.iou.metrics:
      for sp in self.iou.metrics[kwargs_name]:
        sp.cluster_z = self.iou.metric.cluster_z
    self.metrics_objects.append(self.iou)

    ## KNN Monitor
    self.knn = self.Stack(evaluation.KNNMonitor(num_clusters=8))
    self.metrics_objects.append(self.knn)

  @property
  def metric_order(self):
    metric_order = super().metric_order + ['cluster_IOU', 'knn_monitor']
    return metric_order

  def update_model(self):
    self.transform_cluster = self.get_transforms()
    ## Run one forward pass to initialize the MLPs for patchwise loss
    with torch.no_grad():
      ## Transform batch before using in the model
      for batch in self.data_train:
        break
      x, y_, y = self.get_processed_batch(batch)

      ## get kwargs like style
      kwargs = self.get_kwargs(batch, epoch=0, sample_flag=0, description='train')

      ## add noise to output to improve robustness of the model
      noise = torch.randn_like(y) * self.args.noise if self.args.noise > 0 else 0

      y_cap, internal_losses, args = self.forward_pass('train', x, y+noise, **kwargs)
    

TrainerJointLateClusterContrastiveGAN = TrainerLateClusterContrastiveGAN


class TrainerLateClusterContrastiveDTWGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]
    self.running_count = [1e-10]
    self.transforms_contrastive.transforms[0].znorm = self.pre.transforms[-1]

  def get_contrastive_transforms(self):
    skel = self.data.modality_classes[self.output_modality]
    if 'translate_limits' in self.modelKwargs:
      max = self.modelKwargs['translate_limits']
    else:
      max = [50, 50]

    transforms = [RandomTranslate(max=max, mask=self.mask, skel=skel,
                                  znorm=None, output_modality=self.output_modality)]
    return Compose(transforms)

  def update_modelKwargs(self):
    super().update_modelKwargs()
    self.transforms_contrastive = self.get_contrastive_transforms()
    self.modelKwargs.update({'transforms':self.transforms_contrastive})

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'cntr', 'H']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    #loss_str = ' pose:{:.4f} G_gan:{:.4f} real_D:{:.4f} fake_D:{:.4f} label:{:.4f} H:{:.4f}'
    if not losses:
      losses = [0]* len(self.running_loss)

    return desc + loss_str.format(*losses)

TrainerJointLateClusterContrastiveDTWGAN = TrainerLateClusterContrastiveDTWGAN


class TrainerLateClusterVQ(TrainerLateCluster):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2

  @property
  def loss_kinds(self):
    return ['pose', 'vq',
            'commiit', 'mem%']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    if not losses:
      losses = [0]* len(self.running_loss)
      if self.args.mem_usage:
        losses[-1] = self.mem_usage()[-1]
    else:
      if self.args.mem_usage:
        losses[-2] = self.mem_usage()[-1]
    return desc + loss_str.format(*losses)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})


TrainerJointLateClusterVQ = TrainerLateClusterVQ

class TrainerLateClusterVQGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'vq', 'commit',
            'H', 'mem%']
    
  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    if not losses:
      losses = [0]* len(self.running_loss)
      if self.args.mem_usage:
        losses[-1] = self.mem_usage()[-1]
    else:
      if self.args.mem_usage:
        losses[-2] = self.mem_usage()[-1]
    return desc + loss_str.format(*losses)

  def update_modelKwargs(self):
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape})


TrainerJointLateClusterVQGAN = TrainerLateClusterVQGAN

class TrainerLateClusterVQPrior(TrainerLateCluster):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2

  @property
  def loss_kinds(self):
    return ['pose', 'cce', 'mem%']

  def running_loss_init(self):
    self.running_loss = [0]*len(self.loss_kinds)
    self.running_count = [1e-10]*len(self.loss_kinds)

  def tqdm_desc(self, desc, losses=[]):
    loss_str = ''.join([' {}'.format(l) + ':{:.3f}' for l in self.loss_kinds])
    if not losses:
      losses = [0]* len(self.running_loss)
      if self.args.mem_usage:
        losses[-1] = self.mem_usage()[-1]
    else:
      if self.args.mem_usage:
        losses[-2] = self.mem_usage()[-1]
    return desc + loss_str.format(*losses)

  def update_modelKwargs(self):
    ## load pretrained VQ model
    pretrained_trainers = get_pretrained_trainers(self.args.pretrained_model_weights, self.args.cuda)
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'pretrained_trainers':pretrained_trainers})

TrainerJointLateClusterVQPrior = TrainerLateClusterVQPrior

class TrainerLateClusterVQPriorGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'cce', 'H', 'mem%']

  def update_modelKwargs(self):
    ## load pretrained VQ model
    pretrained_trainers = get_pretrained_trainers(self.args.pretrained_model_weights, self.args.cuda)
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'pretrained_trainers':pretrained_trainers})

TrainerJointLateClusterVQPriorGAN = TrainerLateClusterVQPriorGAN


class TrainerLateClusterNoiseGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2


  def lerp(self, n1, n2, t):
    noises = []
    for t_ in t:
      noises.append(n1*(1-t_) + n2*t_)
    return noises
          
  def update_kwargs(self, kwargs, y=None):
    n_time_steps = y.shape[0]
    in_noise_dim = self.model.G.in_noise_dim
    kwargs_list = []
    kwargs_names = []
    torch.manual_seed(11212)
    n1 = torch.rand(n_time_steps, in_noise_dim)
    n2 = torch.rand(n_time_steps, in_noise_dim)
    noises = self.lerp(n1, n2, np.linspace(0, 1, 5))
    for i, noise in enumerate(noises):
      kwargs['noise'] = noise
      kwargs_list.append(kwargs)
      if i == 0:
        kwargs_names.append(None)
      else:
        kwargs_names.append('{}'.format(i))

    for kwargs_, kwargs_name in zip(kwargs_list, kwargs_names):
      yield kwargs_, kwargs_name

TrainerJointLateClusterNoiseGAN = TrainerLateClusterNoiseGAN

class TrainerMineGAN(TrainerLateClusterGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1
    self.fewshot_mu = None
    self.fewshot_std = None

  # @property
  # def loss_kinds(self):
  #   return ['pose', 'G_gan',
  #           'real_D', 'fake_D',
  #           'cce', 'H', 'mem%']

  def update_modelKwargs(self):
    ## Load Pretrained model(s)
    try:
      pretrained_trainers = get_pretrained_trainers(self.args.pretrained_model_weights, self.args.cuda)
    except:
      pretrained_trainers = None
      warnings.warn('pretrained_weights not found. Skipping!!')
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'pretrained_trainers':pretrained_trainers})

  '''Load Discriminator from the pretrained_model'''
  def get_model(self):
    ## Generator
    G = eval(self.args.model)(**self.modelKwargs)

    ## Discriminator
    if self.args.discriminator is None: ## infer the name of the discriminator
      D_modelname = '_'.join(self.args.model.split('_')[:-1] + ['D'])
    else:
      D_modelname = self.args.discriminator

    ## GAN Wrapper
    D_modelKwargs = {}
    if self.args.weighted:
      GANWrapper = GANWeighted
      D_modelKwargs.update({'out_shape':2})
    elif self.args.gan == 'wgan_gp':
      GANWrapper = WGAN_GP
      print('wgan_gp GAN')
    elif self.args.gan == 'wgan_gp_norecon':
      GANWrapper = WGAN_GP_NoRecon
      print('wgan_gp_norecon GAN')
    elif self.args.gan == 'diffgan':
      GANWrapper = DiffGAN
      print('DiffGAN')
    elif self.args.gan == 'ewcgan':
      GANWrapper = ewcGAN
      print('ewcGAN')
    else:
      GANWrapper = GAN
      print('GAN')

    ### add input_shape for self.args.joint
    input_shape = 0
    if self.args.joint:
      for mod in self.input_modalities:
        input_shape += self.data_shape[mod][-1]

    ## Load Discriminator from the pretrained_model
    D_modelKwargs.update({'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask) + input_shape})
    if 'p' in self.modelKwargs: ## get the dropout parameter in the discrimiator as well
      D_modelKwargs.update({'p':self.args.modelKwargs['p']})
    try:
      if self.modelKwargs.get('get_pretrained_D') is not None:
        D = self.modelKwargs['pretrained_trainers'].model.D
      else:
        print('Using randomly initialized Discriminator')
        D = eval('Speech2Gesture_D')(**D_modelKwargs)
    except:
      print('{} not defined, hence defaulting to Speech2Gesture_D'.format(D_modelname))
      D = eval('Speech2Gesture_D')(**D_modelKwargs)
      
    ## GAN
    model = GANWrapper(G, D, lr=self.args.lr, criterion=self.args.loss, optim=self.args.optim,
                       dg_iter_ratio=self.args.dg_iter_ratio, lambda_gan=self.args.lambda_gan,
                       lambda_D=self.args.lambda_D, joint=self.args.joint,
                       input_modalities=self.input_modalities,
                       update_D_prob_flag=self.args.update_D_prob_flag, no_grad=self.args.no_grad,
                       **{**self.args.modelKwargs, **{'data_train':self.data_train, 'parent':self}})
    return model

  def get_data(self):
    assert self.args.fewshot_seed is not None, 'fewshot_seed should be an integer for fewshot learning experiments'
    assert isinstance(self.speaker[0], list), 'speaker should be a list of lists for a fewshot learning dataset'

    ## Load data iterables
    if self.speaker[0]:
      data = Data(self.path2data, self.speaker[0], self.modalities, self.fs_new,
                  time=self.time, split=self.split, batch_size=self.batch_size,
                  shuffle=self.shuffle, num_workers=self.num_workers,
                  window_hop=self.window_hop, style_iters=self.style_iters,
                  num_training_sample=self.num_training_sample,
                  load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                  repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                  quantile_num_training_sample=self.quantile_num_training_sample,
                  weighted=self.args.weighted, filler=self.args.filler,
                  num_training_iters=self.args.num_training_iters,
                  audio_average=self.args.audio_average,
                  fewshot_seed=None,
                  k_shot=self.args.k_shot, p_avg=self.args.p_avg,
                  cache=self.args.cache)

    else:
      assert 0, 'Fewshot train not available. Exiting!!'

    if self.speaker[1]:
      fewshot_data = Data(self.path2data, self.speaker[1], self.modalities, self.fs_new,
                          time=self.time, split=self.split, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          window_hop=self.window_hop, style_iters=self.style_iters,
                          num_training_sample=self.num_training_sample,
                          load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                          repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                          quantile_num_training_sample=self.quantile_num_training_sample,
                          weighted=self.args.weighted, filler=self.args.filler,
                          num_training_iters=None,
                          audio_average=self.args.audio_average,
                          fewshot_seed=self.args.fewshot_seed,
                          k_shot=self.args.k_shot, p_avg=self.args.p_avg,
                          cache=self.args.cache)

    else:
      assert False, 'Fewshot Test not available. Exiting!!'



    data_train = data.train
    data_dev = data.dev
    data_test = data.test
    self.fewshot_data = fewshot_data
    
    print('Data Loaded')

    ## assigning first speaker to self.speaker
    self.speaker_all = self.speaker.copy()
    self.speaker = self.speaker[0]
    
    return data, data_train, data_dev, data_test

  def get_train_data_for_paradigm(self, data_src, data_tgt, call_freq=[1,1], paradigm=None):
    if paradigm is None or paradigm == 'no_src_tgt' or paradigm == 'src_only':
      if paradigm == 'no_src_tgt' or paradigm is None:
        self.model.tgt_train_ratio = 1 ## only get samples from fewshotdata
      return data_src
    elif paradigm == 'tgt_only':
      return data_tgt
    elif paradigm == 'src_tgt':
      return ConcatIter([data_src, data_tgt],
                        call_freq=call_freq, num_iters=self.args.num_training_iters)
    
  def train(self, exp_num):
    ## Wandb
    if self.args.wandb:
      self.wandb_init(job_type='train')

    fewshot_data = self.update_fewshot_data(self.fewshot_data.train)
    self.fewshot_data_samples = fewshot_data
    #data_train = self.data.train ## TODO can change depending on type of training
    data_train = self.get_train_data_for_paradigm(self.data.train, self.fewshot_data.train,
                                                  paradigm=self.args.paradigm)
    for epoch in tqdm(range(self.num_epochs), ncols=20):
      if epoch == self.num_epochs/2 and self.modelKwargs.get('step2') is not None:
        self.step = 2
        ## Load best Model
        self.book._load_model(self.model, map_location=self.device)
        tqdm.write('Entering Step 2')
        
      train_loss, train_metrics, train_metrics_split = self.train_loop(data_train,
                                                                       'train', epoch,
                                                                       num_iters=self.args.num_iters,
                                                                       fewshot_data=fewshot_data,
                                                                       step=self.step)
      dev_loss, dev_metrics, dev_metrics_split = self.train_loop(self.fewshot_data.dev,
                                                                 'dev', epoch,
                                                                 num_iters=self.args.num_iters,
                                                                 fewshot_data=None)
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.fewshot_data.test,
                                                                    'test', epoch,
                                                                    num_iters=self.args.num_iters,
                                                                    fewshot_data=None)
      if self.args.scheduler not in ['linear_decay']: ## update lr after each iteration if training bert
        self.schedulers_step() ## Change the Learning Rate

      ## update the weights for data_train
      if self.args.weighted:
        ## Normalize weights
        max_W = 10
        min_W = 0.1
        W_ = self.data_train.sampler.weights
        W_ = (W_ - W_.mean())/W_.std() + 1
        W_ = torch.min(torch.ones(1)[0].float()*max_W,
                       torch.max(torch.zeros(1)[0].float() + min_W, W_))
        if torch.isnan(W_).any():
          W_ = torch.ones_like(W_) ## reinit to ones if Weights suffer a large variation
        self.data_train.sampler.weights = W_

        W = self.data_train.sampler.weights
        D_prob = self.model.D_prob if hasattr(self.model, 'D_prob') else 0
        tqdm.write('W: {}/{}/{}/{}/{}'.format(W.mean(), W.std(), W.min(), W.max(), D_prob))
        most_common = str(self.weight_counter.most_common()[:5])
        least_common = str(self.weight_counter.most_common()[-5:])
        tqdm.write('samples: {} -- {}'.format(most_common, least_common))

      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()

      ## update wandb
      if self.args.wandb:
        wandb.log(train_metrics, commit=False)
        wandb.log(dev_metrics, commit=False)
        wandb.log(test_metrics, commit=True)

      ## update tensorboard
      if self.args.tb:
        self.book.update_tb({'scalar':[[f'{self.args.cpk}/train', train_loss, epoch],
                                       [f'{self.args.cpk}/dev', dev_loss, epoch],
                                        [f'{self.args.cpk}/test', test_loss, epoch],
                                       [f'{self.args.cpk}/pck_train',
                                        train_metrics['train_pck'], epoch],
                                       [f'{self.args.cpk}/pck_dev',
                                        dev_metrics['dev_pck'], epoch],
                                       [f'{self.args.cpk}/pck_test',
                                        test_metrics['test_pck'],
                                        epoch],
                                       [f'{self.args.cpk}/train_spatialNorm',
                                        train_metrics['train_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/dev_spatialNorm',
                                        dev_metrics['dev_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/test_spatialNorm',
                                        test_metrics['test_spatialNorm'], epoch]
        ]})
                             #'histogram':[[f'{self.args.cpk}/'+name,
                             #param.clone().cpu().detach().numpy(), epoch]
                             # for name, param in model.named_parameters()]})

      ## print results
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

#      warmup = self._is_warmup(epoch, np.ceil(len(self.data_train)/self.batch_size))
      if self.book.stop_training(self.model, epoch):
        if self.args.wandb:
          wandb.finish()
        break

    if self.args.num_iters > 0:
      #get the best model
      self.book._load_model(self.model, map_location=self.device)
      #calculate test loss for the complete dataset
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.fewshot_data.test,
                                                                    'test', 0, fewshot_batch=None)
      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()
      print('Final Results')
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

  def get_processed_fewshot_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    x.append(labels)
    
    x = [x_.to(self.device) for x_ in x]
    y = y_.to(self.device)

    ## Remove the first joint
    y = self.transform_fewshot(y)

    return x, y_, y

      
  def update_fewshot_data(self, data):
    self.transform_fewshot = self.get_transforms()
    ## iter_frequency, num_samples
    for batch in data:
      continue

    x, y_, y = self.get_processed_fewshot_batch(batch)

    keys = ['text/token_count', 'text/token_duration', 'audio/silence', 'text/filler', 'pose/starts', 'pose/startsC']
    for key in keys:
      if key in batch:
        batch[key] = batch[key].to(self.device)

    ## Give fewshotdata access to all samplesx
    train_dataLoader_kwargs = self.fewshot_data.dataLoader_kwargs.copy()
    train_dataLoader_kwargs['shuffle'] = False
    self.fewshot_data.train = DataLoader(data.dataset,
                                         sampler=torch.utils.data.RandomSampler(data.dataset,
                                                                                replacement=True,
                                                                                num_samples=self.args.num_training_iters*self.args.batch_size),
                                         **train_dataLoader_kwargs)
    #data.sampler.seed = None
    #data.sampler._num_samples = self.args.num_training_iters * self.args.batch_size
    
    #concatData = ConcatIter([data, fewshot_data], [4, 1])
    ## X_s, X_t^{all}, (X_t, Y_t)  
    return {'fewshot_batch':batch, 'fewshot_x':x, 'fewshot_y_':y_, 'fewshot_y':y} #, concatData
  
  def sample(self, exp_num):
    ## Wandb
    if self.args.wandb:
      if self.sample_all_styles:
        self.wandb_init(job_type='sample_all_styles')
      else:
        self.wandb_init(job_type='sample')

    ## get fewshot data
    fewshot_data = self.update_fewshot_data(self.fewshot_data.train)
    self.fewshot_data_samples = fewshot_data
        
    ## Create Output Directory
    self.dir_name = self.book.name.dir(self.args.save_dir)

    ## Load best Model
    self.book._load_model(self.model, map_location=self.device)

    train_loss, train_metrics, train_metrics_split = self.sample_loop(self.fewshot_data.train.dataset.datasets, 'train', DATA=self.fewshot_data, fewshot_data=fewshot_data)
    dev_loss, dev_metrics, dev_metrics_split = self.sample_loop(self.fewshot_data.dev.dataset.datasets, 'dev', DATA=self.fewshot_data, fewshot_data=fewshot_data)
    test_loss, test_metrics, test_metrics_split = self.sample_loop(self.fewshot_data.test.dataset.datasets, 'test', DATA=self.fewshot_data, fewshot_data=fewshot_data)    

    if self.sample_all_styles == 0: ## if all styles are sampled, then the results change, hence we don't update it in this case
      ## Save labels histogram
      self._save_labels()

      ## Save sample time metrics
      self._save_metrics(test_metrics_split, 'metrics')
      self._save_metrics(test_metrics, 'cummMetrics')

    print('Sampled- Train:{:.4f}/{:.4f}, '.format(train_loss, train_metrics['train_pck']) + \
          'Dev:{:.4f}/{:.4f}, '.format(dev_loss, dev_metrics['dev_pck']) + \
          'Test:{:.4f}/{:.4f}'.format(test_loss, test_metrics['test_pck']))

    ## print results
    self.book.print_res(epoch=0,
                        key_order=['train', 'dev', 'test'],
                        metric_order=self.metric_order,
                        exp=exp_num,
                        lr=0)

    ## Wandb
    if self.args.wandb:
      if len(self.speaker) > 1:
        wandb.log(test_metrics_split, commit=False)
      wandb.log(test_metrics, commit=True)
      wandb.finish()

  def optimize(self, loss):
    super().optimize(loss)
    if self.step == 2 and self.model.G_flag:
      accumulate(self.model.G.G_ema, self.model.G.G, decay=0.999)

  def znorm_inv_fewshot(self, y_cap):
    fewshot_data = self.fewshot_data_samples['fewshot_batch'][self.output_modality]

    if self.fewshot_mu is None:
      fewshot_data  = fewshot_data.view(-1, fewshot_data.shape[-1])
      self.fewshot_mu, self.fewshot_std = fewshot_data.mean(0)[None, None, :], fewshot_data.std(0)[None, None, :]
    fewshot_y_cap = (y_cap * self.fewshot_std) + self.fewshot_mu
    fewshot_y_cap = fewshot_y_cap.view(fewshot_y_cap.shape[0] * fewshot_y_cap.shape[1], 2, -1)
    fewshot_y_cap[..., 0] = 0

    return fewshot_y_cap
    
  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    if kwargs_name is None:
      kwargs_name = 'same'
    elif kwargs_name in ['1', '2', '3', '4']:
      kwargs_name = 'same'
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    if 'style' in kwargs:
      idx = int(kwargs['style'].view(-1)[0].detach().cpu().item())
      style_vector = kwargs['style'].detach().cpu()
    else:
      idx = 0
      style_vector = torch.zeros(y_cap.shape[0], y_cap.shape[1]).long()

    try:
      self.IS(y_cap, style_vector, self.mask, idx=idx, kwargs_name=kwargs_name)
    except:
      pass

    ## Re-insert Joints
    y_cap = self.transform(y_cap, inv=True, batch_gt=y_)

    ## calculate L1
    self.l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.vel_l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.fid(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## undo normalization using fewshot data statistics
    fewshot_y_cap = self.znorm_inv_fewshot(y_cap)
    if torch.isnan(fewshot_y_cap).any() or torch.isinf(fewshot_y_cap).any():
      pdb.set_trace()
      
    ## undo normalization
    y_cap = self.pre({self.output_modality:y_cap}, inv=True)[self.output_modality]
    y_cap = y_cap.view(y_cap.shape[0], y_cap.shape[1], 2, -1) ## (B, T, 2, feats)
    y_ = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
    y_ = y_.view(y_.shape[0], y_.shape[1], 2, -1) ## (B, T, 2, feats)

    ## Complete undo normalization using fewshot statistics
    fewshot_y_cap[..., self.mask] = y_cap.view(-1, y_cap.shape[-2], y_cap.shape[-1])[..., self.mask]   
    
    ## calculate wasserstein_distance-1 for avg velocity and accelaration
    self.w1(fewshot_y_cap.view(y_.shape[0], y_.shape[1], 2, -1), y_,
            self.mask, idx=idx, kwargs_name=kwargs_name)

    ## Hardcode root as (0,0) for eternity for y and gt
    y_cap = y_cap.view(-1, 2, y_cap.shape[-1]) ## (BxT, 2, feats)
    y_cap[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity
    y_cap_out = y_cap

    y_gt = y_.view(-1, 2, y_cap.shape[-1])
    y_gt[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity

    ## calculate and add pck to the average meter
    self.pck(fewshot_y_cap, y_gt, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## calculate STEEr, SEA and MoCA-{self.num_clusters} scores
    y_cap = self.transform(y_cap.view(1, y_cap.shape[0], -1), save_insert=False)
    y_gt = self.transform(y_gt.view(1, y_gt.shape[0], -1), save_insert=False)
    self.diversity(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.expressiveness(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.f1(self.f1_cluster(y_cap), self.f1_cluster(y_gt), idx=idx, kwargs_name=kwargs_name)

    return fewshot_y_cap
    #return y_cap_out


class TrainerConsistentGAN(TrainerMineGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1

  def optimize(self, loss):
    super().optimize(loss)
    accumulate(self.model.G.G_ema, self.model.G.G, decay=0.999)

class TrainerAdaptiveGAN(TrainerMineGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    if kwargs_name is None:
      kwargs_name = 'same'
    elif kwargs_name in ['1', '2', '3', '4']:
      kwargs_name = 'same'
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    if 'style' in kwargs:
      idx = int(kwargs['style'].view(-1)[0].detach().cpu().item())
      style_vector = kwargs['style'].detach().cpu()
    else:
      idx = 0
      style_vector = torch.zeros(y_cap.shape[0], y_cap.shape[1]).long()

    try:
      self.IS(y_cap, style_vector, self.mask, idx=idx, kwargs_name=kwargs_name)
    except:
      pass

    ## Re-insert Joints
    y_cap = self.transform(y_cap, inv=True, batch_gt=y_)

    ## calculate L1
    self.l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.vel_l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.fid(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## undo normalization
    y_cap = self.pre({self.output_modality:y_cap}, inv=True)[self.output_modality]
    y_cap = y_cap.view(y_cap.shape[0], y_cap.shape[1], 2, -1) ## (B, T, 2, feats)
    y_ = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
    y_ = y_.view(y_.shape[0], y_.shape[1], 2, -1) ## (B, T, 2, feats)

    ## calculate wasserstein_distance-1 for avg velocity and accelaration
    self.w1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## Hardcode root as (0,0) for eternity for y and gt
    y_cap = y_cap.view(-1, 2, y_cap.shape[-1]) ## (BxT, 2, feats)
    y_cap[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity
    y_cap_out = y_cap

    y_gt = y_.view(-1, 2, y_cap.shape[-1])
    y_gt[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity

    ## calculate and add pck to the average meter
    self.pck(y_cap, y_gt, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## calculate STEEr, SEA and MoCA-{self.num_clusters} scores
    y_cap = self.transform(y_cap.view(1, y_cap.shape[0], -1), save_insert=False)
    y_gt = self.transform(y_gt.view(1, y_gt.shape[0], -1), save_insert=False)
    self.diversity(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.expressiveness(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.f1(self.f1_cluster(y_cap), self.f1_cluster(y_gt), idx=idx, kwargs_name=kwargs_name)
    return y_cap_out
  

class TrainerDiffGAN(TrainerAdaptiveGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1

TrainerTransferringGAN = TrainerAdaptiveGAN
TrainerEWCGAN = TrainerAdaptiveGAN


'''
------------------
Mix-Stage trainers
------------------
'''

class TrainerStyleAdaptiveGAN(TrainerLateClusterStyleGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1
    self.fewshot_mu = None
    self.fewshot_std = None

  @property
  def loss_kinds(self):
    return ['pose', 'G_gan',
            'real_D', 'fake_D',
            'ground',
            'label',
            'id_in', 'id_out',
            'H']  

  def update_modelKwargs(self):
    ## Load Pretrained model(s)
    #try:
    pretrained_trainers = get_pretrained_trainers(self.args.pretrained_model_weights, self.args.cuda)
    
    # except:
    #   pretrained_trainers = None
    #   warnings.warn('pretrained_weights not found. Skipping!!')
    self.modelKwargs.update(self.args.modelKwargs)
    self.modelKwargs.update({'time_steps':self.data_shape[self.input_modalities[0]][0],
                             'out_feats':self.data_shape[self.output_modality][-1]-2*len(self.mask),
                             'num_clusters':self.num_clusters,
                             'cluster':self.cluster,
                             'shape':self.data_shape,
                             'style_dict':self.style_dict,
                             'style_dim':self.style_dim,
                             'pretrained_trainers':pretrained_trainers})

  '''Load Discriminator from the pretrained_model'''
  def get_model(self):
    ## Generator
    G = eval(self.args.model)(**self.modelKwargs)

    ## Discriminator
    if self.args.discriminator is None: ## infer the name of the discriminator
      D_modelname = '_'.join(self.args.model.split('_')[:-1] + ['D'])
    else:
      D_modelname = self.args.discriminator

    ## GAN Wrapper
    D_modelKwargs = {}
    if self.args.weighted:
      GANWrapper = GANWeighted
      D_modelKwargs.update({'out_shape':2})
    elif self.args.gan == 'wgan_gp':
      GANWrapper = WGAN_GP
      print('wgan_gp GAN')
    elif self.args.gan == 'wgan_gp_norecon':
      GANWrapper = WGAN_GP_NoRecon
      print('wgan_gp_norecon GAN')
    elif self.args.gan == 'diffgan':
      GANWrapper = DiffGAN
      print('DiffGAN')
    elif self.args.gan == 'ewcgan':
      GANWrapper = ewcGAN
      print('ewcGAN')
    elif self.args.gan == 'align_gan':
      GANWrapper = GANAligned
      print('GANAligned')
    elif self.args.gan == 'align_diffgan':
      GANWrapper = DiffGANAligned
      print('DiffGANAligned')      
    else:
      GANWrapper = GAN
      print('GAN')

    ### add input_shape for self.args.joint
    input_shape = 0
    if self.args.joint:
      for mod in self.input_modalities:
        input_shape += self.data_shape[mod][-1]

    ## Load Discriminator from the pretrained_model
    D_modelKwargs.update({'in_channels':self.data_shape[self.output_modality][-1]-2*len(self.mask) + input_shape})
    if 'p' in self.modelKwargs: ## get the dropout parameter in the discrimiator as well
      D_modelKwargs.update({'p':self.args.modelKwargs['p']})
    try:
      if self.modelKwargs.get('get_pretrained_D') is not None:
        D = self.modelKwargs['pretrained_trainers'].model.D
      else:
        print('Using randomly initialized Discriminator')
        D = eval('Speech2Gesture_D')(**D_modelKwargs)
    except:
      print('{} not defined, hence defaulting to Speech2Gesture_D'.format(D_modelname))
      D = eval('Speech2Gesture_D')(**D_modelKwargs)
      
    ## GAN
    model = GANWrapper(G, D, lr=self.args.lr, criterion=self.args.loss, optim=self.args.optim,
                       dg_iter_ratio=self.args.dg_iter_ratio, lambda_gan=self.args.lambda_gan,
                       lambda_D=self.args.lambda_D, joint=self.args.joint,
                       input_modalities=self.input_modalities,
                       update_D_prob_flag=self.args.update_D_prob_flag, no_grad=self.args.no_grad,
                       **{**self.args.modelKwargs, **{'data_train':self.data_train, 'parent':self}})
    return model

  def get_data(self):
    assert self.args.fewshot_seed is not None, 'fewshot_seed should be an integer for fewshot learning experiments'
    assert isinstance(self.speaker[0], list), 'speaker should be a list of lists for a fewshot learning dataset'

    ## Load data iterables
    if self.speaker[0]:
      data = Data(self.path2data, self.speaker[0], self.modalities, self.fs_new,
                  time=self.time, split=self.split, batch_size=self.batch_size,
                  shuffle=self.shuffle, num_workers=self.num_workers,
                  window_hop=self.window_hop, style_iters=self.style_iters,
                  num_training_sample=self.num_training_sample,
                  load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                  repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                  quantile_num_training_sample=self.quantile_num_training_sample,
                  weighted=self.args.weighted, filler=self.args.filler,
                  num_training_iters=self.args.num_training_iters,
                  audio_average=self.args.audio_average,
                  fewshot_seed=None, ## force fewshot seed to None for the source data
                  k_shot=self.args.k_shot, p_avg=self.args.p_avg,
                  cache=self.args.cache, replay=None)

    else:
      assert 0, 'Fewshot train not available. Exiting!!'

    if self.speaker[1]:
      fewshot_data = Data(self.path2data, self.speaker[1], self.modalities, self.fs_new,
                          time=self.time, split=self.split, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          window_hop=self.window_hop, style_iters=self.style_iters,
                          num_training_sample=self.num_training_sample,
                          load_data=self.load_data, sample_all_styles=self.sample_all_styles,
                          repeat_text=self.repeat_text, quantile_sample=self.quantile_sample,
                          quantile_num_training_sample=self.quantile_num_training_sample,
                          weighted=self.args.weighted, filler=self.args.filler,
                          num_training_iters=None, ## force it to be non for fewshot dataloader
                          audio_average=self.args.audio_average,
                          fewshot_seed=self.args.fewshot_seed,
                          k_shot=self.args.k_shot, p_avg=self.args.p_avg,
                          cache=self.args.cache, replay=self.args.replay)

    else:
      assert False, 'Fewshot Test not available. Exiting!!'

    data_train = data.train
    data_dev = data.dev
    data_test = data.test
    self.fewshot_data = fewshot_data
    
    print('Data Loaded')

    ## assigning first speaker to self.speaker
    self.speaker_all = self.speaker.copy()
    self.speaker = self.speaker[0]
    
    return data, data_train, data_dev, data_test

  def get_train_data_for_paradigm(self, data_src, data_tgt, call_freq=[1,1], paradigm=None):
    if paradigm is None or paradigm == 'no_src_tgt' or paradigm == 'src_only':
      if paradigm == 'no_src_tgt' or paradigm is None:
        self.model.tgt_train_ratio = 1 ## only get samples from fewshotdata
      return data_src
    elif paradigm == 'tgt_only':
      return data_tgt
    elif paradigm == 'src_tgt':
      return ConcatIter([data_src, data_tgt],
                        call_freq=call_freq, num_iters=self.args.num_training_iters)
    
  def train(self, exp_num):
    ## Wandb
    if self.args.wandb:
      self.wandb_init(job_type='train')

    fewshot_data = self.update_fewshot_data(self.fewshot_data.train)
    self.fewshot_data_samples = fewshot_data
    #data_train = self.data.train ## TODO can change depending on type of training
    data_train = self.get_train_data_for_paradigm(self.data.train, self.fewshot_data.train,
                                                  paradigm=self.args.paradigm)
    for epoch in tqdm(range(self.num_epochs), ncols=20):
      if epoch == self.num_epochs/2 and self.modelKwargs.get('step2') is not None:
        self.step = 2
        ## Load best Model
        self.book._load_model(self.model, map_location=self.device)
        tqdm.write('Entering Step 2')

      ## data_train: contains src/train data for different paradigms
      ## fewshot_data: contains fewshot parallel data for just the target
      train_loss, train_metrics, train_metrics_split = self.train_loop(data_train,
                                                                       'train', epoch,
                                                                       num_iters=self.args.num_iters,
                                                                       fewshot_data=fewshot_data,
                                                                       step=self.step,
                                                                       replay=self.args.replay,
                                                                       fewshot_sampler=self.fewshot_data.train_sampler)
      dev_loss, dev_metrics, dev_metrics_split = self.train_loop(self.fewshot_data.dev,
                                                                 'dev', epoch,
                                                                 num_iters=self.args.num_iters,
                                                                 fewshot_data=None)
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.fewshot_data.test,
                                                                    'test', epoch,
                                                                    num_iters=self.args.num_iters,
                                                                    fewshot_data=None)
      if self.args.scheduler not in ['linear_decay']: ## update lr after each iteration if training bert
        self.schedulers_step() ## Change the Learning Rate

      ## update the weights for data_train
      if self.args.weighted:
        ## Normalize weights
        max_W = 10
        min_W = 0.1
        W_ = self.data_train.sampler.weights
        W_ = (W_ - W_.mean())/W_.std() + 1
        W_ = torch.min(torch.ones(1)[0].float()*max_W,
                       torch.max(torch.zeros(1)[0].float() + min_W, W_))
        if torch.isnan(W_).any():
          W_ = torch.ones_like(W_) ## reinit to ones if Weights suffer a large variation
        self.data_train.sampler.weights = W_

        W = self.data_train.sampler.weights
        D_prob = self.model.D_prob if hasattr(self.model, 'D_prob') else 0
        tqdm.write('W: {}/{}/{}/{}/{}'.format(W.mean(), W.std(), W.min(), W.max(), D_prob))
        most_common = str(self.weight_counter.most_common()[:5])
        least_common = str(self.weight_counter.most_common()[-5:])
        tqdm.write('samples: {} -- {}'.format(most_common, least_common))

      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()

      ## update wandb
      if self.args.wandb:
        wandb.log(train_metrics, commit=False)
        wandb.log(dev_metrics, commit=False)
        wandb.log(test_metrics, commit=True)

      ## update tensorboard
      if self.args.tb:
        self.book.update_tb({'scalar':[[f'{self.args.cpk}/train', train_loss, epoch],
                                       [f'{self.args.cpk}/dev', dev_loss, epoch],
                                        [f'{self.args.cpk}/test', test_loss, epoch],
                                       [f'{self.args.cpk}/pck_train',
                                        train_metrics['train_pck'], epoch],
                                       [f'{self.args.cpk}/pck_dev',
                                        dev_metrics['dev_pck'], epoch],
                                       [f'{self.args.cpk}/pck_test',
                                        test_metrics['test_pck'],
                                        epoch],
                                       [f'{self.args.cpk}/train_spatialNorm',
                                        train_metrics['train_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/dev_spatialNorm',
                                        dev_metrics['dev_spatialNorm'], epoch],
                                       [f'{self.args.cpk}/test_spatialNorm',
                                        test_metrics['test_spatialNorm'], epoch]
        ]})
                             #'histogram':[[f'{self.args.cpk}/'+name,
                             #param.clone().cpu().detach().numpy(), epoch]
                             # for name, param in model.named_parameters()]})

      ## print results
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

#      warmup = self._is_warmup(epoch, np.ceil(len(self.data_train)/self.batch_size))
      if self.book.stop_training(self.model, epoch):
        if self.args.wandb:
          wandb.finish()
        break

    if self.args.num_iters > 0:
      #get the best model
      self.book._load_model(self.model, map_location=self.device)
      #calculate test loss for the complete dataset
      test_loss, test_metrics, test_metrics_split = self.train_loop(self.fewshot_data.test,
                                                                    'test', 0, fewshot_batch=None)
      ## save results
      self.book.update_res({'train':train_loss,
                            'dev':dev_loss,
                            'test':test_loss})
      ## add metrics
      self.book.update_res(train_metrics)
      self.book.update_res(dev_metrics)
      self.book.update_res(test_metrics)

      self.book._save_res()
      print('Final Results')
      self.book.print_res(epoch,
                          key_order=['train', 'dev', 'test'],
                          metric_order=self.metric_order,
                          exp=exp_num,
                          lr=self.schedulers[0].get_last_lr())#self.G_optim.state_dict()['param_groups'][0]['lr'])

  def get_processed_fewshot_batch(self, batch):
    ## Get cluster Labels
    self.cluster.update(batch)
    labels = self.cluster(self.transform_cluster(batch[self.output_modality]))

    batch = self.pre(batch)

    x = [batch[mod] for mod in self.input_modalities]
    y_ = batch[self.output_modality]

    x.append(labels)
    
    # x = [x_.to(self.device) for x_ in x]
    # y = y_.to(self.device)

    ## Remove the first joint
    y = self.transform_fewshot(y_)

    return x, y_, y

      
  def update_fewshot_data(self, data):
    self.transform_fewshot = self.get_transforms()
    ## iter_frequency, num_samples
    for batch in data:
      continue

    x, y_, y = self.get_processed_fewshot_batch(batch)

    # keys = ['text/token_count', 'text/token_duration', 'audio/silence', 'text/filler', 'pose/starts', 'pose/startsC']
    # for key in keys:
    #   if key in batch:
    #     batch[key] = batch[key].to(self.device)

    ## Give fewshotdata access to all samplesx
    train_dataLoader_kwargs = self.fewshot_data.dataLoader_kwargs.copy()
    train_dataLoader_kwargs['shuffle'] = False
    self.fewshot_data.train = DataLoader(data.dataset,
                                         sampler=torch.utils.data.RandomSampler(data.dataset,
                                                                                replacement=True,
                                                                                num_samples=self.args.num_training_iters*self.args.batch_size),
                                         **train_dataLoader_kwargs)
    #data.sampler.seed = None
    #data.sampler._num_samples = self.args.num_training_iters * self.args.batch_size
    
    #concatData = ConcatIter([data, fewshot_data], [4, 1])
    ## X_s, X_t^{all}, (X_t, Y_t)  
    return {'fewshot_batch':batch, 'fewshot_x':x, 'fewshot_y_':y_, 'fewshot_y':y} #, concatData
  
  def sample(self, exp_num):
    ## Wandb
    if self.args.wandb:
      if self.sample_all_styles:
        self.wandb_init(job_type='sample_all_styles')
      else:
        self.wandb_init(job_type='sample')

    ## get fewshot data
    fewshot_data = self.update_fewshot_data(self.fewshot_data.train)
    self.fewshot_data_samples = fewshot_data
        
    ## Create Output Directory
    self.dir_name = self.book.name.dir(self.args.save_dir)

    ## Load best Model
    self.book._load_model(self.model, map_location=self.device)

    train_loss, train_metrics, train_metrics_split = self.sample_loop(self.fewshot_data.train.dataset.datasets, 'train', DATA=self.fewshot_data, fewshot_data=fewshot_data)
    dev_loss, dev_metrics, dev_metrics_split = self.sample_loop(self.fewshot_data.dev.dataset.datasets, 'dev', DATA=self.fewshot_data, fewshot_data=fewshot_data)
    test_loss, test_metrics, test_metrics_split = self.sample_loop(self.fewshot_data.test.dataset.datasets, 'test', DATA=self.fewshot_data, fewshot_data=fewshot_data)    

    if self.sample_all_styles == 0: ## if all styles are sampled, then the results change, hence we don't update it in this case
      ## Save labels histogram
      self._save_labels()

      ## Save sample time metrics
      self._save_metrics(test_metrics_split, 'metrics')
      self._save_metrics(test_metrics, 'cummMetrics')

    print('Sampled- Train:{:.4f}/{:.4f}, '.format(train_loss, train_metrics['train_pck']) + \
          'Dev:{:.4f}/{:.4f}, '.format(dev_loss, dev_metrics['dev_pck']) + \
          'Test:{:.4f}/{:.4f}'.format(test_loss, test_metrics['test_pck']))

    ## print results
    self.book.print_res(epoch=0,
                        key_order=['train', 'dev', 'test'],
                        metric_order=self.metric_order,
                        exp=exp_num,
                        lr=0)

    ## Wandb
    if self.args.wandb:
      if len(self.speaker) > 1:
        wandb.log(test_metrics_split, commit=False)
      wandb.log(test_metrics, commit=True)
      wandb.finish()

  def optimize(self, loss):
    super().optimize(loss)
    if self.step == 2 and self.model.G_flag:
      accumulate(self.model.G.G_ema, self.model.G.G, decay=0.999)

  def znorm_inv_fewshot(self, y_cap):
    fewshot_data = self.fewshot_data_samples['fewshot_batch'][self.output_modality]

    if self.fewshot_mu is None:
      fewshot_data  = fewshot_data.view(-1, fewshot_data.shape[-1])
      self.fewshot_mu, self.fewshot_std = fewshot_data.mean(0)[None, None, :], fewshot_data.std(0)[None, None, :]
    fewshot_y_cap = (y_cap * self.fewshot_std) + self.fewshot_mu
    fewshot_y_cap = fewshot_y_cap.view(fewshot_y_cap.shape[0] * fewshot_y_cap.shape[1], 2, -1)
    fewshot_y_cap[..., 0] = 0

    return fewshot_y_cap

  def calculate_metrics(self, y_cap, y_, kwargs_name, **kwargs):
    if kwargs_name is None:
      kwargs_name = 'same'
    elif kwargs_name in ['1', '2', '3', '4']:
      kwargs_name = 'same'
    #feats_shape = int(self.data_shape[self.output_modality][-1]/2)
    if 'style' in kwargs:
      idx = int(kwargs['style'].view(-1)[0].detach().cpu().item())
      style_vector = kwargs['style'].detach().cpu()
    else:
      idx = 0
      style_vector = torch.zeros(y_cap.shape[0], y_cap.shape[1]).long()

    try:
      self.IS(y_cap, style_vector, self.mask, idx=idx, kwargs_name=kwargs_name)
    except:
      pass

    ## Re-insert Joints
    y_cap = self.transform(y_cap, inv=True, batch_gt=y_)

    ## calculate L1
    self.l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.vel_l1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)
    self.fid(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## undo normalization
    y_cap = self.pre({self.output_modality:y_cap}, inv=True)[self.output_modality]
    y_cap = y_cap.view(y_cap.shape[0], y_cap.shape[1], 2, -1) ## (B, T, 2, feats)
    y_ = self.pre({self.output_modality:y_}, inv=True)[self.output_modality]
    y_ = y_.view(y_.shape[0], y_.shape[1], 2, -1) ## (B, T, 2, feats)

    ## calculate wasserstein_distance-1 for avg velocity and accelaration
    self.w1(y_cap, y_, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## Hardcode root as (0,0) for eternity for y and gt
    y_cap = y_cap.view(-1, 2, y_cap.shape[-1]) ## (BxT, 2, feats)
    y_cap[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity
    y_cap_out = y_cap

    y_gt = y_.view(-1, 2, y_cap.shape[-1])
    y_gt[..., 0] = 0 ## Hardcode to have root as (0,0) for eternity

    ## calculate and add pck to the average meter
    self.pck(y_cap, y_gt, self.mask, idx=idx, kwargs_name=kwargs_name)

    ## calculate STEEr, SEA and MoCA-{self.num_clusters} scores
    y_cap = self.transform(y_cap.view(1, y_cap.shape[0], -1), save_insert=False)
    y_gt = self.transform(y_gt.view(1, y_gt.shape[0], -1), save_insert=False)
    self.diversity(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.expressiveness(y_cap.squeeze(0), y_gt.squeeze(0), idx=idx, kwargs_name=kwargs_name)
    self.f1(self.f1_cluster(y_cap), self.f1_cluster(y_gt), idx=idx, kwargs_name=kwargs_name)
    return y_cap_out

class TrainerStyleConsistentGAN(TrainerStyleAdaptiveGAN):
  def __init__(self, args, args_subset, args_dict_update={}):
    super().__init__(args, args_subset, args_dict_update)
    self.running_loss = [0]*2
    self.running_count = [1e-10]*2
    self.step = 1

  def optimize(self, loss):
    super().optimize(loss)
    accumulate(self.model.G.G_ema, self.model.G.G, decay=0.999)

TrainerStyleDiffGAN = TrainerStyleAdaptiveGAN
