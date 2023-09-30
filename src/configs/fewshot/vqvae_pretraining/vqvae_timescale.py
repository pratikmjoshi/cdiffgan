import json
import itertools
def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x

from data import Modality

speakers = Modality().speakers
speakers = [[sp] for sp in speakers]

#speakers = [['oliver']]
modalities = [['pose/normalize', 'audio/log_mel_400']]
model = 'JointLateClusterVQ3_G'
note = 'vqvae_timescale'

window_hop = 5
batch_size = 32
num_epochs = 100
stop_thresh = 3
overfit = 0
gan = 0
early_stopping = 1
dev_key = 'dev_L1'

min_epochs = 50

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

fs_new = 15

save_dirs = ['save6/fewshot/vqvae_pretraining/{}/{}'.format(note, sp[0]) for sp in speakers]

optim = 'AdamW'
lr = 0.0005

wandb = 1
wandb_project = 'fewshot'
wandb_dir = 'save6/fewshot/wandb'

speakers = [convert2str(speaker) for speaker in speakers]
modalities = [convert2str(modality) for modality in modalities]
fs_new = convert2str(fs_new)
feats = convert2str(feats)

conv_config={
  'cpk':[note],
  'tb':[1],
  'exp':[1],
  ('speaker', 'save_dir'):zip(speakers, save_dirs),
  'model':[model],
  'fs_new':[fs_new],
  'modalities': modalities,
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
