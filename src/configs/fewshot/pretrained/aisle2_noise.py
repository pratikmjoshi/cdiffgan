import json
import itertools
def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x

from data import Modality

speakers = Modality().speakers
speakers = [[sp] for sp in speakers]

modalities = [['pose/normalize', 'text/tokens', 'audio/log_mel_400']]
model = 'JointLateClusterSoftTransformerNoise_G'
note = 'aisle2_noise'

window_hop = 5
batch_size = 32
num_epochs = 100
stop_thresh = 3
overfit = 0
gan = 1
early_stopping = 1
dev_key = 'dev_spatialNorm'
dg_iter_ratio = [1]
repeat_text = 0

num_iters = 100
min_epochs = 50
update_D_prob_flag = 0
no_grad = 0

num_training_iters = 400

fs_new = 15

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

save_dirs = ['save6/fewshot/pretrained/{}/{}'.format(note, sp[0]) for sp in speakers]

optim = 'AdamW'
lr = 0.0001
optim_separate = 0.00003

speakers = [convert2str(speaker) for speaker in speakers]
modalities = [convert2str(modality) for modality in modalities]
feats = convert2str(feats)
fs_new = convert2str(fs_new)

conv_config={
  'cpk':[note],
  'tb':[1],
  'exp':[1],
  ('speaker', 'save_dir'):zip(speakers, save_dirs),
  'num_cluster': [num_clusters],
  'model':[model],
  'fs_new':[fs_new],
  'modalities': modalities,
  'gan':[gan],
  'loss':['L1Loss'],
  'window_hop':[window_hop],
  'render':[0],
  'batch_size':[batch_size],
  'num_epochs':[num_epochs],
  'stop_thresh':[stop_thresh],
  'overfit':[overfit],
  'early_stopping':[early_stopping],
  'dev_key':[dev_key],
  'feats':[feats],
  'note':[note],
  'dg_iter_ratio':dg_iter_ratio,
  'repeat_text':[repeat_text],
  'num_iters' : [num_iters],
  'min_epochs': [min_epochs],
  'update_D_prob_flag': [update_D_prob_flag],
  'num_training_iters': [num_training_iters],
  'optim':[optim],
  'lr':[lr],
  'optim_separate':[optim_separate],
  'no_grad':[no_grad]
}
