import json
import itertools
from pathlib import Path
import os
import warnings

def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x

def get_pretrained_model_weights(base_path, speakers):
  weight_paths = []
  count_weights = 1
  for speaker in speakers:
    path = base_path/'_'.join(speaker)
    weight_path = None
    count = 0
    for f in os.listdir(path):
      if f.split('_')[-1] == 'weights.p':
        weight_path = (path/f).as_posix()
        print('{: 2d}: {} ----: {}'.format(count_weights, speaker, weight_path))
        print('----')
        count+=1
        count_weights += 1
    if count > 1:
      warnings.warn('more than one weight file found for speaker: {}'.format(speaker))
    weight_paths.append(weight_path)
  return weight_paths

from data import Modality

speakers = Modality().speakers
speakers = [[sp] for sp in speakers]

base_path = Path('save6/fewshot/vqvae_pretraining/vqvae_timescale5/')
pretrained_model_weights = get_pretrained_model_weights(base_path, speakers)

#speakers = [['oliver']]
modalities = [['pose/normalize', 'text/tokens', 'audio/log_mel_400']]
repeat_text = [0]
model = 'JointLateClusterVQPrior43_G'
note = 'vqvae_timescale5_prior2_gan'

window_hop = 5
batch_size = 32
num_epochs = 100
stop_thresh = 3
overfit = 0
gan = 1
early_stopping = 1
dev_key = 'dev_L1'

min_epochs = 50

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

fs_new = 15

save_dirs = ['save6/fewshot/vqvae_prior/{}/{}'.format(note, sp[0]) for sp in speakers]

optim = 'AdamW'
lr = 0.00001

wandb = 1
wandb_project = 'fewshot'
wandb_dir = 'save6/fewshot/wandb'

speakers = [convert2str(speaker) for speaker in speakers]
modalities = [convert2str(modality) for modality in modalities]
fs_new = convert2str(fs_new)
feats = convert2str(feats)
pretrained_model_weights = [json.dumps(w) for w in pretrained_model_weights]

conv_config={
  'cpk':[note],
  'tb':[1],
  'exp':[1],
  ('speaker', 'save_dir', 'pretrained_model_weights'):zip(speakers, save_dirs, pretrained_model_weights),
  'model':[model],
  'fs_new':[fs_new],
  'modalities': modalities,
  'repeat_text': repeat_text,
  'gan':[gan],
  'num_clusters':[num_clusters],
  'feats':[feats],
  'loss':['L1Loss'],
  'window_hop':[window_hop],
  'render':[0],
  'batch_size':[batch_size],
  'num_epochs':[num_epochs],
  'stop_thresh':[stop_thresh],
  'overfit':[overfit],
  'early_stopping':[early_stopping],
  'dev_key':[dev_key],
  'note':[note],
  'min_epochs': [min_epochs],
  'optim':[optim],
  'lr':[lr],
  'wandb':[wandb],
  'wandb_project':[wandb_project],
  'wandb_dir':[wandb_dir]
}

