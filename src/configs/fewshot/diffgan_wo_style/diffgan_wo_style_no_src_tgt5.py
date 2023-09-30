import json
import itertools
from pathlib import Path
import os
import pdb

def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x

def get_pretrained_model_weights(base_path, speakers):
  weight_paths = []
  for speaker in speakers:
    path = base_path/'_'.join(speaker)
    weight_path = None
    for f in os.listdir(path):
      if f.split('_')[-1] == 'weights.p':
        weight_path = (path/f).as_posix()
        break
    weight_paths.append(json.dumps(weight_path))
  return weight_paths


## Speakers combinations
src_speakers = ['maher', 'oliver']
tgt_speakers = ['sp0001', 'sp0003', 'sp0006']

speakers = []
for src_sp in src_speakers:
  for tgt_sp in tgt_speakers:
    if src_sp != tgt_sp:
      speakers.append([[src_sp], [tgt_sp]])

speakers_rev = {sp:i for i, sp in enumerate(src_speakers)}

## pretrained_models
base_path = Path('save6/fewshot/pretrained/aisle_noise/')
pretrained_model_weights = get_pretrained_model_weights(base_path, [[sp] for sp in src_speakers])

## Fewshot Seeds
fewshot_seeds = [1, 2, 3]
seed = 11212

## Kshots
k_shots = [28, 140]

## Paradigm
paradigm = 'no_src_tgt'

## Model
modalities = [['pose/normalize', 'text/tokens', 'audio/log_mel_400']]
model = 'DiffGAN2_G'
modelKwargs = [{'tgt_train_ratio':1, 'get_pretrained_D':1, 'fewshot_style':0, 'fewshot_timing':1}]
note = 'diffgan_wo_style_{}'.format(paradigm)

## Choice of GAN
gan = 'diffgan'

window_hop = 5
batch_size = 32
num_epochs = 20
stop_thresh = 3
overfit = 1
early_stopping = 1
dev_key = 'dev_FID'
dg_iter_ratio = [1]
repeat_text = 0

num_iters = 200
cache = 1
#min_epochs = 50
update_D_prob_flag = 0
no_grad = 0

num_training_iters = 200

fs_new = 15

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

#save_dirs = ['save6/fewshot/models/{}/{}/{}_{}/{}'.format(note.split('_')[0], note, sp[0][0], sp[1][0], seed) for sp in speakers for seed in fewshot_seeds]

kshot_sp_seeds_save_dir = []
for k_shot in k_shots:
  for sp in speakers:
    for sd in fewshot_seeds:
      try:
        kshot_sp_seeds_save_dir.append((pretrained_model_weights[speakers_rev[sp[0][0]]], k_shot, convert2str(sp), sd, 'save6/fewshot/models/{}/{}/{}/{}_{}/{}'.format(note.split('_')[0], note, k_shot, sp[0][0], sp[1][0], sd)))
      except:
        pdb.set_trace()
optim = 'AdamW'
lr = 0.0001
optim_separate = 0.00003

speakers = [convert2str(speaker) for speaker in speakers]
modalities = [convert2str(modality) for modality in modalities]
modelKwargs = [convert2str(mk) for mk in modelKwargs]
feats = convert2str(feats)
fs_new = convert2str(fs_new)

conv_config={
  'cpk':[note],
  'tb':[1],
  'exp':[1],
  'seed':[seed],
  #('speaker', 'save_dir'):zip(speakers, save_dirs),
  ('pretrained_model_weights', 'k_shot', 'speaker', 'fewshot_seed', 'save_dir'):kshot_sp_seeds_save_dir,
  'paradigm':[paradigm],
  'num_cluster': [num_clusters],
  'model':[model],
  'modelKwargs':modelKwargs,
  'cache':[cache],
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
  #'min_epochs': [min_epochs],
  'update_D_prob_flag': [update_D_prob_flag],
  'num_training_iters': [num_training_iters],
  'optim':[optim],
  'lr':[lr],
#  'optim_separate':[optim_separate],
  'no_grad':[no_grad]
}
