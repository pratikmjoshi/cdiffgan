import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import pdb
import copy

import torch
import torch.nn as nn
from transformers import BertModel
import logging
import random
from .layers import *


from soft_dtw_cuda import SoftDTW, SoftDTW_DBS
#from fastdtw_mod import fastdtw



class MoCo_GlobalDTW(MoCo):
  def __init__(self, enc, in_channels=128, time_steps=64, DTW = True, K=512, m=0.99, T=0.1, bn_splits=1, margin = 0, symmetric=False, **kwargs):
    super().__init__(enc, in_channels, time_steps, K, m, T, bn_splits, symmetric, **kwargs)
    self.softdtw_DBS = SoftDTW_DBS(use_cuda = True)
    self.softdtw = SoftDTW(use_cuda = True)
    self.clusterdict = dict()
    self.margin = margin
    self.select_idx = None
    self.DTW_true = DTW
    self.kwargs = kwargs
    self.running_mean_list = []
    self.running_mean_counts = 0
    self.running_mean = None
    self.running_sd = None
    #self.SoftDTW = SoftDTW(use_cuda=True, gamma=0.1)
    self.global_dict = {}
    self.cluster_count = 0

    #online-kmeans inits
    self.cluster_centers = None
    self.w_star = None
    self.r = 1
    self.q = 0
    self.f = None

  def contrastive_loss(self, im_q, im_k):
    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

    # compute logits
    # Einstein sum is more intuitive
    # positive logits: Nx1xT
    l_pos = torch.einsum('nct,nct->nt', [q, k]).unsqueeze(1)
    # negative logits: NxKxT
    l_neg = torch.einsum('nct,ckt->nkt', [q, self.queue.clone().detach()])
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= self.T

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda()

    loss = nn.CrossEntropyLoss().cuda()(logits, labels)


    return loss, q, k

  def cluster_DTW(self, q, k, labels, iter, top_k = 5, bootstrap = True, **kwargs): #anchor: 1 X T X E

    with torch.no_grad():
        if self.select_idx == None:
            sample_indices = torch.nonzero(labels)
            select_idx = torch.randperm(len(sample_indices))[:1]
            anchor = q[select_idx]
        else:
            select_idx = self.select_idx[0]
            anchor = q[select_idx]

        labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
        sample_indices = torch.nonzero(labels) #update sample_indices to be used later
        remain_size = sample_indices.shape[0]



        if self.DTW_true:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation

            sim_scores = self.softdtw(anchor,q[sample_indices[:,0]].clone()) # calculate sim scores via dtw
            print("Iter: {}    Mean:".format(iter), torch.mean(sim_scores))
            #pdb.set_trace()
            # for i in range(sample_indices.shape[0]):
            #     try:
            #         distance, path = fastdtw(anchor.squeeze(), q[sample_indices[i,0]].clone())
            #     except Exception as e:
            #         print(e)
            #         pdb.set_trace()
        if self.DTW_true == False:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores = torch.einsum('nct,nct->n', [anchor, q[sample_indices[:,0]].clone()])
        scores_size = sim_scores.shape[0]

        if (top_k + 1) >= scores_size:
            best_values, top_indices = torch.topk(sim_scores, scores_size)

            self.clusterdict[iter] = sample_indices[top_indices]
            self.clusterdict[iter] = dict()

            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx])
            self.clusterdict[iter]["vals"] = best_values.cuda()
            labels[sample_indices[top_indices]] = 0

        else:
            best_values, top_indices = torch.topk(sim_scores, top_k) # TODO: calculate mean of sim scores, kind of wasting medium values of DTW
            worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
            self.clusterdict[iter] = dict()
            #try pdb post mortem
            self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx]) #last elem is anchor seq.
            self.clusterdict[iter]["vals"] = best_values.cuda()
            self.select_idx = sample_indices[bot_indices]
            labels[sample_indices[top_indices]] = 0
            labels[sample_indices[bot_indices]] = 0

    return labels



  def bootstrap_cluster_DTW(self, q, k, labels, iter, top_k = 5, bootstrap = True, **kwargs): #anchor: 1 X T X E
    with torch.no_grad():
        if self.select_idx == None:
            sample_indices = torch.nonzero(labels)
            select_idx = torch.randperm(len(sample_indices))[:1]
            anchor = q[select_idx]
        else:
            select_idx = self.select_idx[0]
            anchor = q[select_idx]

        labels[select_idx] = 0 #selected anchor is not a part of sampleable indices for DTW
        sample_indices = torch.nonzero(labels) #update sample_indices to be used later
        remain_size = sample_indices.shape[0]



        if self.DTW_true:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation

            sim_scores = self.softdtw(anchor,q[sample_indices[:,0]].clone()) # calculate sim scores via dtw

        if self.DTW_true == False:
            anchor = anchor.repeat(remain_size,1,1) #repeat anchor for DTW calculation
            sim_scores = torch.einsum('nct,nct->n', [anchor, q[sample_indices[:,0]].clone()])

        scores_size = sim_scores.shape[0]

        #V1
        if kwargs['epoch'] == 0:
          if iter == 0:
              mean = torch.mean(self.softdtw(anchor,q[sample_indices[:,0]].clone()))
              self.running_mean_list.append(mean)
              self.running_mean_counts += 1
          return torch.zeros_like(labels)
        #   # implement running mean

        if kwargs['epoch'] == 1 and self.running_mean == None:
          self.running_mean = torch.stack(self.running_mean_list)
          self.running_sd = torch.std(self.running_mean)
          self.running_mean = torch.mean(self.running_mean)
          self.thresh = self.running_mean
          print("check")

        if iter == 0: #exponential moving average
          mean = self.running_mean
          N = self.running_mean_counts
          K = (2/(N+1))
          self.running_mean = (mean * K) + (self.running_mean * (1-K))
          self.thresh = self.running_mean
          self.running_mean_counts += 1
          print("check")

        
        if kwargs['epoch'] >= 1:

          top_indices = (sim_scores > self.thresh).nonzero()
          
          if top_indices.shape[0] > 0:
              # self.clusterdict[iter]["idx"] = torch.cat([sample_indices[top_indices].squeeze(1), select_idx]) #last elem is anchor seq.
              # self.clusterdict[iter]["vals"] = torch.stack(q[top_indices], q[select_idx])
              try:
                self.clusterdict[self.cluster_count]= torch.cat((q[select_idx].unsqueeze(0), q[top_indices])).squeeze()
                self.cluster_count += 1
              except Exception:
                pdb.post_mortem()
          else: #버려 씨발
              # self.clusterdict[iter]["idx"] = torch.tensor(select_idx)
              # self.clusterdict[iter]["vals"] =  q[select_idx]
              self.clusterdict[self.cluster_count] =  q[select_idx]
          
          worst_values, bot_indices = torch.topk(sim_scores, 1, largest=False)
          self.select_idx = sample_indices[bot_indices]
          labels[sample_indices[top_indices]] = 0
          labels[sample_indices[bot_indices]] = 0
          
    return labels


  def global_lifted_embedding_loss(self, im_q, im_k, y, training, global_bool, self_sup = False, **kwargs):
    # https://arxiv.org/pdf/1703.07737.pdf
    '''
    #losses for TopK clustering, the im_k is not used at all, only q is used
    '''

    # compute query features
    q = self.encoder_q(im_q)  # queries: NxCxT
    q = nn.functional.normalize(q, dim=1)  # already normalized

    # compute key features
    with torch.no_grad():  # no gradient to keys
      # shuffle for making use of BN
      im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

      k = self.encoder_k(im_k_)  # keys: NxCxT
      k = nn.functional.normalize(k, dim=1)  # already normalized

      # undo shuffle
      k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

    if training:
        batch_size = y.shape[0]
        not_all_assigned = True
        labels = torch.ones(batch_size)
        iter = 0
        self.select_idx = None
        self.cluster_count = 0
        while not_all_assigned:
            
            if not self_sup:
            # labels = self.cluster_DTW(q.transpose(1,2),  k.transpose(1,2), labels, iter, top_k = 5)
              labels = self.bootstrap_cluster_DTW(y.transpose(1,2).clone().detach(),  y.transpose(1,2).clone().detach(), labels, iter, top_k = batch_size//8, **kwargs)
            if self_sup:
              labels = self.bootstrap_cluster_DTW(q.transpose(1,2).clone().detach(),  q.transpose(1,2).clone().detach(), labels, iter, top_k = batch_size//8, **kwargs)
            
            iter += 1

            if torch.sum(labels) == 0:
                not_all_assigned = False


        loss = torch.zeros(1).cuda()

        if kwargs['epoch'] == 0:
          loss, q, k = self.contrastive_loss(im_q, im_k)
          return loss.squeeze(), q, k
          

        if bool(self.global_dict) == False:
            self.global_dict = self.clusterdict.copy()

        else:
          centroids = [cluster_vals[random.randint(0 ,cluster_vals.shape[0]-1)] for cluster_vals in self.global_dict.values()] #do random sampling better
          #centroids = [cluster_vals[0] for cluster_vals in self.global_dict.values()]
          for clusters,v in self.clusterdict.items():
              sim_scores_global = self.softdtw(v[0].unsqueeze(0),torch.stack(centroids))
              top_idx = int(torch.argmax(sim_scores_global))

              if sim_scores_global[top_idx] > self.thresh:
                try:
                  #get only 10 seqs as cluster
                  t = torch.cat([self.global_dict[top_idx], v]) #last elem is anchor seq
                  # idx = torch.randperm(t.shape[0])
                  # t = t[idx].view(t.size()) 
                  self.global_dict[top_idx] = t[-10:]

                except Exception:
                  pdb.post_mortem()
              else:
                  self.global_dict[len(self.global_dict)]= v


        global_vals = self.global_dict.values()
        global_tensor_vals = torch.cat(list(self.global_dict.values()))
        print("GLOBALDICT:" , len(self.global_dict), "\n")
        pos_labels = torch.zeros(global_tensor_vals.shape[0], dtype=torch.bool)


        for cluster, v in self.global_dict.items():
          
          pos_labels[v.shape[0]] = True
          neg_labels = ~(pos_labels.clone())
  
          other_cluster_vals = [v for k,v in self.global_dict.items() if cluster != k]

          l_pos = torch.einsum('nct,nct->nt', [v, v.detach()]).unsqueeze(1)
          # negative logits: NxKxT
          l_neg = torch.einsum('nct, kct->nkt', [v,torch.cat(other_cluster_vals)])

          # logits: Nx(1+K)
          logits = torch.cat([l_pos, l_neg], dim=1)
          #stability
          logits = logits - torch.max(logits)

          # apply temperature
          logits /= self.T

          # labels: positive key indicators
          labels = torch.zeros(logits.shape[0], logits.shape[-1], dtype=torch.long).cuda() # TODO: mean???

          loss += nn.CrossEntropyLoss().cuda()(logits, labels)


        loss = loss/iter
        self.clusterdict = dict()
    
    if not training:
      loss, q, k = self.contrastive_loss(im_q, im_k)
      #self.global_dict.clear()
      return loss.squeeze(), q, k

    print(kwargs['desc'])

    if kwargs['desc'] != "train":
      pdb.set_trace()
      self.global_dict.clear()

    return loss.squeeze(), q, k


 

  def forward(self, im1, im2, y, global_var, self_sup, **kwargs):
    """
    Input:
        im_q: a batch of query images
        im_k: a batch of key images
    Output:
        loss
    """

    # update the key encoder
    with torch.no_grad():  # no gradient to keys
      self._momentum_update_key_encoder()

    # compute loss

    if global_var and not self_sup:
        if self.symmetric:  # asymmetric loss
          loss_12, q1, k2 = self.global_lifted_embedding_loss(im1, im2, y, self.training, global_var, **kwargs)
          loss_21, q2, k1 = self.global_lifted_embedding_loss(im2, im1, y, self.training, global_var, **kwargs)
          loss = loss_12 + loss_21
          k = torch.cat([k1, k2], dim=0)
          q = q1
        else:  # asymmetric loss
          loss, q, k = self.global_lifted_embedding_loss(im1, im2, y, self.training, global_var, **kwargs)

    if global_var and self_sup:
      if self.symmetric:  # asymmetric loss
        loss_12, q1, k2 = self.global_lifted_embedding_loss(im1, im2, y, self.training, global_var, self_sup = True, **kwargs)
        loss_21, q2, k1 = self.global_lifted_embedding_loss(im2, im1, y, self.training, global_var, self_sup = True, **kwargs)
        loss = loss_12 + loss_21
        k = torch.cat([k1, k2], dim=0)
        q = q1
      else:  # asymmetric loss
        loss, q, k = self.global_lifted_embedding_loss(im1, im2, y, self.training, global_var, self_sup = True, **kwargs)


    if not global_var:
        if self.symmetric:  # asymmetric loss
          loss_12, q1, k2 = self.lifted_embedding_loss(im1, im2, y, self.training, global_var,**kwargs)
          loss_21, q2, k1 = self.lifted_embedding_loss(im2, im1, y, self.training, global_var, **kwargs)
          loss = loss_12 + loss_21
          k = torch.cat([k1, k2], dim=0)
          q = q1
        else:  # asymmetric loss
          loss, q, k = self.lifted_embedding_loss(im1, im2, y, self.training, **kwargs)

    self._dequeue_and_enqueue(k)


    return q, [loss]
