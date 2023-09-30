## speaker
## models
## velocity vs accelaration

import pandas as pd
from data import Data, DataSample

from pathlib import Path
from argparse import Namespace
import json
import pdb
import os

import argparse

from data import Data
from data import Skeleton2D
from data import ZNorm, RemoveJoints
import torch
import numpy as np
from scipy.stats import beta, norm
import scipy.stats as stats
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from metrics_loader import get_raw_metrics, remap_df, note_map
from ast import literal_eval

def get_datasample(view):
  view = Path(view)
  args_name = view.parent / (view.name + '_args.args')
  args = Namespace(**json.load(open(args_name, 'r')))
  speaker = args.speaker
  view = view.as_posix()
  window_hop = args.window_hop
  modalities = args.modalities[:1]
  mask = args.mask
  print(args.note)
  return DataSample('../dataset/groot/data/', speaker, modalities=modalities, window_hop=window_hop, view=view), speaker, modalities, window_hop, mask, args.note

def get_xlim(pdf):
  mask = (pdf < 1e-4).astype(np.int)
  pdf = pdf+mask*2
  return np.argmin(pdf)

def vel(x):
  x = (((x[:, 1:] - x[:, :-1])**2).sum(dim=-2)**0.5).mean(-1).mean(-1)
  return x

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-speaker', type=str, help='speaker')
  parser.add_argument('-notes_list', type=str, help='choice of notes_list', default='main')
  args = parser.parse_args()

  #df = get_raw_metrics(['save3/emnlp4/'])
  df = pd.read_csv('emnlp_results/results.csv')
  df['speaker'] = df.speaker.apply(lambda x:literal_eval(x))

  df1, df1_std, df2, missing_speakers = remap_df(df, note_map, speakers=[args.speaker], metrics=['test_L1'])

  if args.notes_list == 'main':
    notes_list = ['gesticulator2', 'multiscaleBert_lfiw3', 'mmsbert', 'mmsbert_lfiw_no_update3']
  elif args.notes_list == 'rebalance':
    notes_list = ['mmsbert_rebalance', 'mmsbert_quantile0.9', 'mmsbert_lfiw_no_update']
    
  df2 = df2[df2.note.isin(notes_list)]
  views = df2.apply(lambda x:os.path.join(x['save_dir'], x['name']), axis=1).reset_index().set_index('note').loc[notes_list][0].values

  samples_list = []
  notes = []
  for view in views:
    data_model, speaker, modalities, window_hop, mask, note = get_datasample(view)
    notes.append(note)
    samples_model = get_samples(data_model.test, modalities[0], mask)
    samples_list.append(samples_model)

  data = Data('../dataset/groot/data', speaker, modalities=modalities, window_hop=window_hop)
  samples = get_samples(data.test, modalities[0], mask)
  notes.append('Ground Truth')

  ## plt
  patches = []
  support = []
  sns.distplot(samples['value'], kde=True,  kde_kws={'shade':True}, hist=False, color='tab:cyan')
  patches.append(mpatches.Patch(color='tab:cyan', label='Ground Truth'))
  support.append([samples['value'].min(), samples['value'].max()])
  
  alphas = np.arange(0.8, 0.2, -0.6/len(samples_list))
  colour_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:red']
  for s, alpha, note, c in zip(samples_list, alphas, notes, colour_list):
    ax = sns.distplot(s['value'], kde=True,  kde_kws={'shade':True}, hist=False, color=c)
    model = note_map[note] if note in note_map else note
    patches.append(mpatches.Patch(color=c, label=model))
    support.append([s['value'].min(), s['value'].max()])

  ## add a line to denote the support
  y0, y1 = ax.get_ylim()
  diff = y1/5#0.005
  ax.set_ylim([y0, y1+len(notes)*diff])
  for count, (note, c, sup) in enumerate(zip(notes[1:] + notes[:1], colour_list[:len(notes) - 1] + ['tab:cyan'], support[1:] + support[:1])):
    height = y1 + count*diff
    plt.plot(sup, [height, height], c=c)
    plt.plot([sup[0]]*2, [height-diff/2, height+diff/2], c=c)
    plt.plot([sup[1]]*2, [height-diff/2, height+diff/2], c=c)

  ## plot the legend
  plt.legend(handles=patches)
  plt.title(speaker[0])
  plt.xlabel('Average Absolute Velocity')
  plt.ylabel('Probability')

  ## Save figure
  dir_path = 'emnlp_results/vel_graphs/{}/'.format(args.notes_list)
  os.makedirs(dir_path, exist_ok=True)
  plt.savefig(os.path.join(dir_path, '{}.pdf'.format(speaker[0])))
