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

from data import HDF5, ZNorm
from data import KMeans, RemoveJoints
from evaluation import Diversity

def get_diversity(row):
  speaker = row['speaker']
  mask = row['mask']
  #kmeans = KMeans(['pose/data'], key=speaker, num_clusters=num_clusters, mask=mask, feats=feats)
  remove_joints = RemoveJoints(mask=mask)
  znorm = ZNorm(['pose/data'], key=speaker, verbose=False)
  mean = znorm.variable_dict['pose/data'][0]
  mean = remove_joints(mean).squeeze(0)
  diversity = Diversity(mean)
  path2predictions = Path(row['save_dir'])/row['name']/'keypoints/test/{}'.format(speaker[0])
  path2gt = '../dataset/groot/data/processed/{}'.format(speaker[0])
  for file in os.listdir(path2predictions):
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
    predictions = remove_joints(predictions).squeeze(0)
    labels = remove_joints(labels).squeeze(0)

    ## calculate diversity
    diversity(predictions, labels)
  return diversity.get_averages('test')
