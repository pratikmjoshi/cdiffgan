import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
import pandas as pd
import time
import shutil
import pdb

from argsUtils import *
from data import *

import torch
import numpy as np

def copy_files(speaker, transform, path2data):
  path2processed = Path(path2data)/'processed'/speaker
  path2transformed = Path(path2data)/'processed'/'{}|{}'.format(speaker, transform)
  os.makedirs(path2transformed, exist_ok=True)

  for filename in tqdm(os.listdir(path2processed), desc=transform):
    filename_transformed = filename.split('.')[0] + '|{}.h5'.format(transform)
    shutil.copy(path2processed/filename, path2transformed/filename_transformed)

  return path2transformed
#    return False

#    warnings.warn('{} already exists. Delete it before running the script again'.format(path2processed.as_posix()))
#    return True

def mirror(data, **kwargs):
  data = data.reshape(data.shape[0], 2, -1)
  data[:, 0, 1:] = -data[:, 0, 1:]
  data = data.reshape(data.shape[0], -1)
  return data

def leftarm(data, **kwargs):
  joints = [4, 5, 6, 9] + list(range(10, 31))
  mean = kwargs['mean']
  std = kwargs['std']/100
  data = data.reshape(data.shape[0], 2, -1)
  torch.manual_seed(11212)
  data[:, :, joints] = mean[:, :, joints] + torch.randn(data.shape[0], 1, 1) * std[:, :, joints]
  data = data.reshape(data.shape[0], -1)
  return data
    
def rightarm(data, **kwargs):
  joints = [1, 2, 3, 8] + list(range(31, 52))
  mean = kwargs['mean']
  data = data.reshape(data.shape[0], 2, -1)
  data[:, :, joints] = mean[:, :, joints]
  data = data.reshape(data.shape[0], -1)
  return data

def botharm(data, **kwargs):
  rjoints = [1, 2, 3, 8] + list(range(31, 52))
  ljoints = [4, 5, 6, 9] + list(range(10, 31))
  data = data.reshape(data.shape[0], 2, -1)
  data[:, 0, rjoints] = -data[:, 0, ljoints]
  data[:, 1, rjoints] = data[:, 1, ljoints]
  data = data.reshape(data.shape[0], -1)
  return data

def loop(args, exp_num):
  ## speakers and transforms
  ## transform - mirror image, stop one hand
  ## add cmu_intervals_df_transformed to dataUtils
  ## render audio also needs to know the speakers

  speaker = args.speaker[0]
  path2data = args.path2data
  transforms = args.transforms

  modalities = ['pose/normalize'] #['pose/data', 'pose/normalize']
  pre = ZNorm(modalities, key=speaker, data=None)

  df = pd.read_csv((Path(path2data)/'cmu_intervals_df.csv').as_posix())
  df.loc[:, 'interval_id'] = df['interval_id'].apply(str)
      
  path2dftransforms = Path(path2data)/'cmu_intervals_df_transforms.csv'
  if path2dftransforms.exists():
    df_transforms = pd.read_csv(path2dftransforms.as_posix())
    df_transforms.loc[:, 'interval_id'] = df_transforms['interval_id'].apply(str)
  else:
    df_transforms = pd.DataFrame(columns=df.columns)
    
  for transform in transforms:
    ## Copy files
    #path2transformed = copy_files(speaker, transform, path2data)
    path2transformed = Path('/projects/dataset_processed/groot/data/processed/oliver|leftarm/')
    ## Copy relevant speakers of the dataframe
    df_temp = df[df['speaker'] == speaker]

    ## modify the speaker name and interval_ids
    df_temp.loc[:, 'speaker'] = '{}|{}'.format(speaker, transform)
    df_temp.loc[:, 'interval_id'] = df_temp['interval_id'].apply(lambda x: '{}|{}'.format(x, transform))

    ## save df_transforms
    df_transforms = df_transforms[df_transforms['speaker'] != '{}|{}'.format(speaker, transform)]
    df_transforms = df_transforms.append(df_temp, ignore_index=True)
    df_transforms.to_csv(path2dftransforms.as_posix(), index=False)

    ## transform the copied data
    transform_fn = eval(transform)
    for filename in tqdm(os.listdir(path2transformed), desc=transform+':update hdf5'):
      for key in modalities:
        try:
          data, h5 = HDF5.load(path2transformed/filename, key)
          data = data[()]
          h5.close()

          ## get mean
          mean = pre.variable_dict[key][0].reshape(1, 2, -1)
          std = pre.variable_dict[key][1].reshape(1, 2, -1)
          data = transform_fn(data, mean=mean, std=std)
          HDF5.append(path2transformed/filename, key, data)
        except:
          continue
    
if __name__ == '__main__':
  argparseNloop(loop)
