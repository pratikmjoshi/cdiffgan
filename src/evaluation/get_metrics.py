import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from pathlib import Path
import time
import pdb

from argsUtils import *

import torch
from sklearn.metrics import confusion_matrix
import numpy as np
import json

from data import HDF5
from data import KMeans, RemoveJoints, ZNorm
from evaluation import F1, Diversity, Expressiveness

def get_metrics(row, num_clusters=2, feats=['velocity'], mask=None):
  speaker = row['speaker']
  if mask is None:
    mask = row['mask']
  remove_joints = RemoveJoints(mask=mask)
  f1_dict = {num_cls:F1(num_clusters=num_cls) for num_cls in num_clusters}
  try:
    kmeans_dict = {num_cls:KMeans(['pose/data'], key=speaker, num_clusters=num_cls, mask=mask, feats=feats, verbose=False) for num_cls in num_clusters}
  except:
    pdb.set_trace()
    kmeans_dict = {num_cls:KMeans(['pose/data'], key=json.dumps(speaker), num_clusters=num_cls, mask=mask, feats=feats, verbose=False) for num_cls in num_clusters}
  znorm = ZNorm(['pose/data'], key=speaker, verbose=False)
  mean = znorm.variable_dict['pose/data'][0]
  mean = remove_joints(mean).squeeze(0)
  diversity = Diversity(mean)
  expressiveness = Expressiveness(mean)
  
  path2predictions = Path(row['save_dir'])/row['name']/'keypoints/test/{}'.format(speaker[0])
  path2gt = '../dataset/groot/data/processed/{}'.format(speaker[0])
  
  for i, file in enumerate(os.listdir(path2predictions)):
    ## load data and close h5 files
    predictions, h5 = HDF5.load(Path(path2predictions)/file, 'pose/data')
    predictions = predictions[()]
    h5.close()
    labels, h5 = HDF5.load(Path(path2gt)/file, 'pose/data')
    labels = labels[()]
    h5.close()

    ## Reshape prediction files
    predictions = torch.from_numpy(predictions.reshape(predictions.shape[0], -1))

    ## Trim gt files
    labels = torch.from_numpy(labels[:predictions.shape[0]])

    ## Remove Joints
    predictions = predictions.view(1, *predictions.shape)
    labels = labels.view(1, *labels.shape)
    predictions = remove_joints(predictions)
    labels = remove_joints(labels)

    ## Calculate Diversity
    diversity(predictions.squeeze(0), labels.squeeze(0))

    ## Calculate Expressiveness
    expressiveness(predictions.squeeze(0), labels.squeeze(0))
    
    ## calculate f1
    for num_cls in num_clusters:
      ## Get labels
      predictions1 = kmeans_dict[num_cls](predictions)
      labels1 = kmeans_dict[num_cls](labels)
      f1_dict[num_cls](predictions1, labels1)
    
  metrics = {}
  for num_cls in num_clusters:
    metrics.update(f1_dict[num_cls].get_averages('test_{}'.format(num_cls)))
  metrics.update(diversity.get_averages('test'))
  metrics.update(expressiveness.get_averages('test'))
  return metrics
