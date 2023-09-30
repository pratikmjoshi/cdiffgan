#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data import DataSample, RemoveJoints, Data
from argparse import Namespace
import json
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm

import seaborn as sns
from scipy.stats import beta, norm
import scipy.stats as stats
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pdb
mod_map = {'diffgan_pseudo-align-fs-src':'C-DiffGAN',
          'mixstage_pseudo-align-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{diffgan}$',
          'diffgan_no-replay-fs':r'DiffGAN',
          'diffgan_random-fs':'Random Replay',
          'diffgan_pseudo-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{ccf}$',
          'diffgan_pseudo-align-fs-tgt':'C-DiffGAN w/o Source Inputs',
          'mixstage_baseline':r'MixStAGe',
          'gt':'Ground Truth'}
color_map = {'diffgan_pseudo-align-fs-src':'tab:green',
          'mixstage_pseudo-align-fs-src':'tab:orange',
          'diffgan_no-replay-fs':'tab:red',
          'diffgan_random-fs':'tab:purple',
          'diffgan_pseudo-fs-src':'tab:pink',
          'diffgan_pseudo-align-fs-tgt':'tab:gray',
          'mixstage_baseline':'tab:brown',
          'gt':'tab:cyan'}


# In[2]:


from pycasper.results import walkthroughMetrics, walkthroughResults


# In[3]:


def get_xlim(pdf):
  mask = (RemoveJoints< 1e-4).astype(np.int)
  pdf = pdf+mask*2
  return np.argmin(pdf)

def vel(x):
  x = (((x[:, 1:] - x[:, :-1])**2).sum(dim=-2)**0.5).mean(-1).mean(-1)
  return x


# In[4]:


def get_datasample(view, sp_idx):
  view = Path(view)
  args_name = view.parent / (view.name + '_args.args')
  args = Namespace(**json.load(open(args_name, 'r')))
  speaker = args.speaker
  view = view.as_posix()
  window_hop = args.window_hop
  modalities = args.modalities[:1]
  mask = args.mask
  print(args.note)
  return DataSample('../dataset/groot/data/', speaker[1][sp_idx], modalities=modalities, window_hop=window_hop, view=view), speaker, modalities, window_hop, mask, args.note


# In[5]:


@torch.no_grad()
def get_samples(d, input_modality, mask):
  samples = []
  interval_ids = []
  idxs = []
  transform = RemoveJoints(mask)
  for batch in d:
    sample = batch[input_modality]
    sample = transform(sample).view(sample.shape[0], sample.shape[1], 2, -1)
    samples.append(vel(sample))
    interval_ids += batch['meta']['interval_id']
    idxs += list(batch['meta']['idx'].numpy())
    
  samples = torch.cat(samples, dim=0)
  samples_df = pd.DataFrame({'value':samples.numpy(), 'interval_id':interval_ids, 'idx':idxs})
  return samples_df


# In[6]:


df = walkthroughMetrics(['save7/cl/'], args_subset=['note', 'paradigm', 'speaker', 'fewshot_seed', 'k_shot', 'save_dir'], res_subset=['test_pck', 'test_L1', 'test_F1', 'test_FID', 'test_W1_vel', 'test_W1_acc'], depth=6)


# In[7]:


def get_speaker_1(x):
  sp = x[0]
  if isinstance(sp, list):
    return sp[0]
  else:
    return sp

def get_speaker_2(x):
  if isinstance(x[0], list):
    return x[1]
  else:
    return x


# In[8]:


df['sp1'] = df['speaker'].apply(get_speaker_1)
df['sp2'] = df['speaker'].apply(get_speaker_2)
df['num_sp'] = df['sp2'].apply(lambda x:len(x))


# In[9]:


def plot_views(views, prefix, k_shot, fewshot_seed, sp_idx, sp, num_sp):
  samples_list = []
  notes = []
  for view in views:
    data_model, speaker, modalities, window_hop, mask, note = get_datasample(view, sp_idx)
    notes.append(note)
    samples_model = get_samples(data_model.test, modalities[0], mask)
    samples_list.append(samples_model)

  data = Data('../dataset/groot/data', speaker[1][sp_idx], modalities=modalities, window_hop=window_hop)
  samples = get_samples(data.test, modalities[0], mask)
  notes.append('Ground Truth')

  ## plt
  patches = []
  support = []
  alphas = np.arange(0.8, 0.2, -0.6/len(samples_list))
  for s, alpha, note in zip(samples_list, alphas, notes):
    ax = sns.distplot(s['value'], kde=True,  kde_kws={'shade':True}, hist=False, color=color_map[note])
    model = mod_map[note] if note in mod_map else note
    patches.append(mpatches.Patch(color=color_map[note], label=model))
    support.append([s['value'].min(), s['value'].max()])  
  
  sns.distplot(samples['value'], kde=True,  kde_kws={'shade':True}, hist=False, color=color_map['gt'])
  patches.append(mpatches.Patch(color=color_map['gt'], label='Ground Truth'))
  support.append([samples['value'].min(), samples['value'].max()])

  ## add a line to denote the support
  y0, y1 = ax.get_ylim()
  y1 = 0.2 ## hardcoded TDDO
  diff = y1/5#0.005
  ax.set_ylim([y0, y1+len(notes)*diff])
  
  color_list = [color_map[m] for m in notes[:-1]] + [color_map['gt']]
  for count, (note, c, sup) in enumerate(zip(notes[1:] + notes[:1], color_list[1:] + color_list[:1], support[1:] + support[:1])):
    height = y1 + count*diff
    plt.plot(sup, [height, height], c=c)
    plt.plot([sup[0]]*2, [height-diff/2, height+diff/2], c=c)
    plt.plot([sup[1]]*2, [height-diff/2, height+diff/2], c=c)

  ## plot the legend
  plt.legend(handles=patches, loc = 'upper right')
  plt.title(speaker[1][sp_idx])
  plt.xlabel('Average Absolute Velocity')
  plt.ylabel('Probability')


  ## Save figure
  dir_path = 'results/figs/vel_graphs/{}'.format(prefix)
  os.makedirs(dir_path, exist_ok=True)
  plt.savefig(os.path.join(dir_path, '{}_{}_{}_{}_{}.pdf'.format(prefix, sp, num_sp, k_shot, fewshot_seed)))
  plt.savefig(os.path.join(dir_path, '{}_{}_{}_{}_{}.png'.format(prefix, sp, num_sp, k_shot, fewshot_seed)))  
  plt.close()

# ### Distribution for each speaker at the final stage

# In[11]:


# model_order = ['diffgan_no-replay-fs', 'diffgan_random-fs', 'diffgan_pseudo-align-fs-src']
# speakers = ['oliver', 'maher', 'chemistry', 'ytch_prof', 'lec_evol']
# k_shot = 140
# fewshot_seed = 1
# prefix = 'final'


# # In[12]:


# for sp_idx, sp in enumerate(speakers):
#   views = []
#   for model in model_order:
#     row = df[df.note.isin([model]) & df.num_sp.isin([5]) & df.k_shot.isin([k_shot]) & df.fewshot_seed.isin([fewshot_seed])].iloc[0]
#     views.append(Path(row['save_dir'])/row['name'])
#   plot_views(views, prefix, k_shot, fewshot_seed, sp_idx, sp, 5)


# In[ ]:



model_order = ['diffgan_no-replay-fs', 'diffgan_random-fs', 'diffgan_pseudo-align-fs-src']
model_order = ['diffgan_no-replay-fs', 'diffgan_pseudo-align-fs-src']
sp = 'maher'
sp_idx = 1
k_shot = 28
fewshot_seed = 1
prefix = 'overExp'

speakers = ['oliver', 'maher', 'chemistry', 'ytch_prof', 'lec_evol']

for sp_idx, sp in enumerate(speakers):
  for num_sp in range(2, 6):
    views = []
    for model in model_order:
      row = df[df.note.isin([model]) & df.num_sp.isin([num_sp]) & df.k_shot.isin([k_shot]) & df.fewshot_seed.isin([fewshot_seed]) & df.sp1.isin(['oliver'])].iloc[0]
      views.append(Path(row['save_dir'])/row['name'])
    plot_views(views, prefix, k_shot, fewshot_seed, sp_idx, sp, num_sp)
