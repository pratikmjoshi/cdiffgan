#!/usr/bin/env python
# coding: utf-8

# In[129]:


import argparse


# In[77]:


import pdb
from animation.animation import frames
from data import Data
import numpy as np


# In[132]:


parser = argparse.ArgumentParser()
parser.add_argument('-speaker', type=str, default='lec_cosmic',
                   help='speaker')
args = parser.parse_args()


# In[136]:


data = Data('../dataset/groot/data', [args.speaker], modalities=['pose/data'], fs_new=[15, 15, 15])


# In[137]:


batches = []
batches_vel = []
for batch in data.train:
  ys = batch['pose/data']
  ys = ys.view(ys.shape[0], ys.shape[1], 2, 52).numpy()
  ys[..., 0] = 0
  ys_v = ys[:, 1:] - ys[:, :-1]
  ys_v = np.concatenate([ys_v, np.zeros_like(ys[:, 0:1])], axis=1)
  ys = ys.reshape(-1, 2, 52)
  ys_v = ys_v.reshape(-1, 2, 52)
  ys = ys.reshape(-1, 104)
  ys_v = ys_v.reshape(-1, 104)
  batches.append(ys)
  batches_vel.append(ys_v)
batches = np.concatenate(batches)
batches_vel = np.concatenate(batches_vel)
batches_both = np.concatenate([batches, batches_vel], axis=-1)


# In[138]:


interval_id = batch['meta']['interval_id'][0]


# ## Clustering

# In[139]:


import sklearn.cluster


# In[140]:


# In[265]:


def cluster(batches, n_clusters=4, bs=None):
  model = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)
  model.fit(batches)
#   subset = batches[::int(batches.shape[0]/200)]
#   bs_subset = bs[::int(bs.shape[0]/200)]
  classes_total = model.predict(batches)
  bb = []
  cc = []
  for clus in range(n_clusters):
    idx = np.where(classes_total == clus)[0]
    idx_subset = np.random.choice(idx, size=(8,), replace=False)
    bb.append(bs[idx_subset])
    cc.append(classes_total[idx_subset])
  return np.concatenate(bb), np.concatenate(cc), classes_total, model


# In[266]:


n_clusters = 8


# In[267]:


subset, classes, classes_total, model= cluster(batches, n_clusters, bs=batches)


# In[268]:


subset_v, classes_v, classes_total_v, model = cluster(batches_vel, n_clusters, bs=batches)


# In[269]:


subset_both, classes_both, classes_total_both, model = cluster(batches_both, n_clusters, bs=batches)


# In[270]:


print('Models Learnt')


# In[271]:


import matplotlib.pyplot as plt


# In[272]:


plt.hist(classes_total, bins=range(9))


# In[273]:


plt.hist(classes_total_v, bins=range(9))


# In[274]:


plt.hist(classes_total_both, bins=range(9))


# ## Plot

# In[275]:


import os
import PIL
import torch


# In[276]:


import torchvision
import torchvision.transforms.functional as TF


# In[277]:


parents  = data.modality_classes['pose/data'].parents


# In[278]:


path2imgs = frames([subset.reshape(-1, 2, 52)], classes, interval_id, parents, 'frames_test', data, None)


# In[279]:


path2imgs_v = frames([subset_v.reshape(-1, 2, 52)], classes_v, interval_id, parents, 'frames_test', data, 'vel')


# In[280]:


path2imgs_both = frames([subset_both.reshape(-1, 2, 52)], classes_both, interval_id, parents, 'frames_test', data, 'both')


means = model.cluster_centers_[:, :104].reshape(-1, 2, 52)
path2imgs_mean = frames([means], [0]*len(means), batch['meta']['interval_id'][0], parents, 'frames_test/', data, 'mean')
# ### make a grid of images

# In[281]:


listdir = lambda x: sorted(list(set(os.listdir(x)) - {'.ipynb_checkpoints'}))


# In[282]:


def make_grid(path2imgs):
  image_list = []
  for dirs in listdir(path2imgs):
    path2imgdir = path2imgs/dirs
    if os.path.isdir(path2imgdir):
      image_list.append([])
      for file in listdir(path2imgs/dirs):
        img = PIL.Image.open((path2imgs/dirs/file).as_posix())
        image_list[-1].append(img)

  min_len = min([len(imgs) for imgs in image_list])
  batch = []
  for imgs in image_list:
    imgs_ = imgs[:min_len]
    imgs_ = [TF.to_tensor(img)[:-1, 33:-33, 112:-112].unsqueeze(0) for img in imgs_]
    batch.append(imgs_)

  batch = [torchvision.utils.make_grid(torch.cat(b), nrow=len(b)).unsqueeze(0) for b in batch]
  torchvision.utils.save_image(torch.cat(batch), path2imgs/'grid.png', nrow=1)


# In[283]:


make_grid(path2imgs)


# In[284]:


make_grid(path2imgs_v)


# In[285]:


make_grid(path2imgs_both)

make_grid(path2imgs_mean)
# In[ ]:




