import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from data import Skeleton2D, HDF5
import os
plt.rcParams.update({'font.size': 40, 'axes.linewidth':2, 'text.usetex':True})
import torch
import numpy as np
import pdb
from tqdm import tqdm

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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pdb
from cl_modelmap import mod_map, color_map


parents = Skeleton2D('../dataset/groot/data/').parents



from pycasper.results import walkthroughMetrics, walkthroughResults


# In[22]:


df = walkthroughMetrics(['save7/cl/'], args_subset=['note', 'paradigm', 'speaker', 'fewshot_seed', 'k_shot', 'save_dir'], res_subset=['test_pck', 'test_L1', 'test_F1', 'test_FID', 'test_W1_vel', 'test_W1_acc'], depth=6)


# In[23]:


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


# In[24]:


df['sp1'] = df['speaker'].apply(get_speaker_1)
df['sp2'] = df['speaker'].apply(get_speaker_2)
df['num_sp'] = df['sp2'].apply(lambda x:len(x))


# In[1]:


def find_model_paths(path, model, paradigm, k_shot, src_sp, tgt_sp, fewshot_seeds=[1,2,3]):
  ret_paths = []
  if model != 'gt':
    path = Path(path)/'models'
    if model == 'scratch':
      src_sp = 'lec_cosmic'
    for seed in fewshot_seeds:
      basepath = Path(path)/model/'{}_{}'.format(model, paradigm)/'{}'.format(k_shot)/'{}_{}'.format(src_sp, tgt_sp)/'{}'.format(seed)
      for file in os.listdir(basepath):
        if os.path.isdir(basepath/file) and file.split('_')[-1] != 'tb':
          break
      ret_paths.append(basepath/file/'keypoints'/'test'/tgt_sp)
  else:
    path = Path(path)/'pretrained/aisle_noise'
    basepath = Path(path)/tgt_sp
    for file in os.listdir(basepath):
      if os.path.isdir(basepath/file) and file.split('_')[-1] != 'tb':
        break
    ret_paths.append(basepath/file/'keypoints'/'test'/tgt_sp)
    
  return ret_paths


# In[2]:


def find_model_paths_grid(modelpath, models, paradigm, k_shot, src_sp, tgt_sps):
  paths = []
  src_sp_path = [find_model_paths(modelpath, 'gt', paradigm, k_shot, src_sp, src_sp)] + [[]]*(len(models) -1)
  paths.append(src_sp_path)
  for tgt_sp in tgt_sps:
    paths_ = []
    for model in models:
      paths_.append(find_model_paths(modelpath, model, paradigm, k_shot, src_sp, tgt_sp))
    paths.append(paths_)
  return paths


def get_colour(joint):
  if joint in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    return 'k'
  elif joint in range(10, 31):
    return 'r'
  else:
    return 'b'


def plot_grid(ys, parents, filename, speakers, modelmap):
  def escape_underscore(sp):
    sp_new = ''
    for char in sp:
      if char == '_':
        sp_new += '\\'
      sp_new += char
    return sp_new  
  R = len(ys)
  C = len(ys[0])
  square_size = 5
  fig = plt.figure(figsize=(square_size*C, square_size*R))
  gs1 = gridspec.GridSpec(int(R), int(C))
  axs = [plt.subplot(gs1[i]) for i in range(C*R)]
  gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes. 
  for row, y_sp in enumerate(y_list_grid):
    for col, y_model in enumerate(y_sp):
      if isinstance(y_model, list):
        axs[row*C + col].axis('off')

  for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.set_aspect(aspect=1)
        
  for row, y_sp in tqdm(enumerate(ys)):
    for col, y_model in enumerate(y_sp):
      if not isinstance(y_model, list):
        for y in y_model:
          for joint, parent in enumerate(parents):
            if joint != 0:
              axs[col + row*C].plot([y[0, parent], y[0, joint]],
                                    [-y[1, parent], -y[1, joint]], 
                                    linestyle='-', marker='.', color=get_colour(joint), alpha=0.01)
              if col == 0 and row < R - 1:
                axs[col + row*C].set_ylabel('{}'.format(row + 1))
              elif col == 0 and row == R - 1:
                axs[col + row*C].set_ylabel('GT')
              if row == R - 1:
                axs[col + row*C].set_xlabel('{}'.format(escape_underscore(speakers[col])))


#   plt.figtext(0.1, 0.43,r'$\underline{\hspace{%dpt}\textbf{Target Speakers}\hspace{%dpt}}$' % (350, 350), va="center", ha="center", size=40, rotation=90)
#   plt.figtext(0.1, 0.81,r'\underline{\textbf{Source Speaker}}', va="center", ha="center", size=40, rotation=90)
#   plt.figtext(0.56,0.78,r'$\underline{\hspace{%dpt}\textbf{Models}\hspace{%dpt}}$' %(840, 840), va="center", ha="center", size=40)
#   plt.figtext(0.1, 0.05, '1', va="center", ha="center", size=40)

#   for row, y_sp in enumerate(y_list_grid):
#     for col, y_model in enumerate(y_sp):
#       if col == 0:
#         if row == 0:
#           axs[col + row*C].set_ylabel('{}'.format(escape_underscore(src_sp)), size=40)
#         else:
#           axs[col + row*C].set_ylabel('{}'.format(escape_underscore(tgt_sps[row-1])), size=40)
#       if row == 0 and not isinstance(y_model, list):
#         axs[col + row*C].set_title('{}'.format(modelmap[models[col]]), size=40)
#       if row == 1 and not isinstance(y_model, list) and isinstance(y_list_grid[row-1][col], list):
#         axs[col + row*C].set_title('{}'.format(modelmap[models[col]]), size=40)    

  
  os.makedirs(Path(filename).parent, exist_ok=True)
  for suffix in ['.png', '.pdf']:
    plt.savefig(filename.as_posix() + suffix, bbox_inches='tight')
  plt.close()


# In[185]:


def find_model_paths_grid_cl(df, model, k_shot, speakers, sp1, fewshot_seed):
  df_ = pd.concat([df[df.note.isin(['mixstage_t1']) & df.sp1.isin([sp1])], df[df.note.isin([model]) & df.sp1.isin([sp1]) & df.k_shot.isin([k_shot]) & df.fewshot_seed.isin([fewshot_seed])].sort_values('num_sp')])
  paths_grid = []
  #for i, (_, row) in enumerate(df_.iterrows()):
  for i in range(len(speakers)):
    try:
      row = df_[df_.num_sp == i + 1].iloc[0]
    except:
      continue
    paths = []
    for j, sp in enumerate(speakers):
      if j > i:
        paths.append([])
      else:
        paths.append([Path(row['save_dir'])/Path(row['name'])/'keypoints'/'test'/sp])
    paths_grid.append(paths)
  paths = []
  for sp in speakers:
    paths.append([Path('../dataset/groot/data/processed/{}'.format(sp))])
  paths_grid.append(paths)
  return paths_grid


# In[172]:


def plot_heatmaps_grid(df, model, modelmap, k_shot, speakers, sp1, fewshot_seed=1, count=600, sample_size=100):
  paths_grid = find_model_paths_grid_cl(df, model, k_shot, speakers, sp1, fewshot_seed)
  y_list_grid = []
  for paths_sp in tqdm(paths_grid):
    y_list_sp = []
    for paths_model in paths_sp:
      if paths_model != []:
        y_list_model = []
        np.random.seed(0)
        pathlist = []
        for i, path in enumerate(paths_model):
          pathlist += [path/f for f in os.listdir(path)]
        pathlist = sorted(pathlist)
        pathlist_idx = np.random.randint(0, len(pathlist), size=(sample_size, ))
        for idx in pathlist_idx:
          f = pathlist[idx]
          try:
            data, h5= HDF5.load(f, 'pose/normalize')
            data = data[()]
            h5.close()

            if len(data.shape) == 1:
              continue ## ignore spurious data
            y_list_model.append(data)
          except:
            continue
        try:
          y = np.concatenate(y_list_model, axis=0)
        except:
          pdb.set_trace()
        if len(y.shape) == 2:
          y = y.reshape(y.shape[0], 2, -1)
        y[..., 0] = 0 ## set root joint to zero
        #y = y.reshape(1, *y.shape)
        y_list_sp.append(y[np.random.permutation(y.shape[0])[:count]])
      else:
        y_list_sp.append([])
    y_list_grid.append(y_list_sp)

  return y_list_grid

# In[164]:


modelpath = "save7/cl/"
sp1 = 'oliver'
k_shots = [28, 140]
fewshot_seed = 1

speakers = ['oliver', 'maher', 'chemistry', 'ytch_prof', 'lec_evol']
models = ['diffgan_no-replay-fs', 'diffgan_random-fs', 'diffgan_pseudo-align-fs-src', 'diffgan_pseudo-fs-src', 'diffgan_pseudo-align-fs-tgt', 'diffgan_pseudo-fs-tgt']
models = ['diffgan-sp2_pseudo-align-fs-src','diffgan-sp3_pseudo-align-fs-src'] 
#models = ['diffgan_pseudo-align-fs-src']
#model = models[0]
for model in models:
  for k_shot in k_shots:
    y_list_grid= plot_heatmaps_grid(df, model, mod_map, k_shot, speakers, sp1, fewshot_seed=1, count=600)
    path2file = Path('results/figs/histograms/{}_{}_{}'.format(model, k_shot, sp1))
    plot_grid(y_list_grid, parents, path2file, speakers, mod_map)
