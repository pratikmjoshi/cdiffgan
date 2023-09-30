import json
def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x
  
speakers = [['maher', 'lec_cosmic'],
            ['maher', 'lec_cosmic', 'ytch_prof', 'oliver'],
            ['maher', 'lec_cosmic', 'ytch_prof', 'oliver', 'ellen', 'noah', 'lec_evol', 'corden'],
            ['corden', 'lec_cosmic'],
            ['corden', 'lec_cosmic', 'ytch_prof', 'oliver'],
            ['corden', 'lec_cosmic', 'ytch_prof', 'oliver', 'ellen', 'noah', 'lec_evol', 'maher']]


modalities = ['pose/normalize', 'text/tokens', 'audio/log_mel_400']
model = 'JointLateClusterSoftStyleNoise5_G'
modelKwargs = [{'lambda_id':0.1, 'argmax':1, 'some_grad_flag':1, 'train_only':1}]
note = 'mixstage_language_noise'

save_dir = ['save6/fewshot/pretrained/{}/{}_{}'.format(note, sp[0], len(sp)) for sp in speakers]

window_hop = 5
batch_size = 8
num_epochs = 40
stop_thresh = 3
overfit = 0
gan = 1
early_stopping = 0
dev_key = 'dev_spatialNorm'
style_iters = 3000
num_iters = 3000

repeat_text = 0

optim = 'AdamW'
lr = 0.0001
optim_separate = 0.00003


input_modalities = modalities[1:]
output_modalities = modalities[:1]
fs_new = [15] * len(modalities)

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

speakers = [convert2str(speaker) for speaker in speakers]
modelKwargs = [convert2str(mk) for mk in modelKwargs]
modalities = convert2str(modalities)
input_modalities = convert2str(input_modalities)
output_modalities = convert2str(output_modalities)
fs_new = convert2str(fs_new)
feats = convert2str(feats)

conv_config={
  'cpk':[model],
  'tb':[1],
  'exp':[1],
  'model':[model],
  ('save_dir', 'speaker'):zip(save_dir, speakers),
  'modelKwargs':modelKwargs,
  'note':[note],
  'modalities':[modalities],
  'fs_new':[fs_new],
  'input_modalities':[input_modalities],
  'output_modalities':[output_modalities],
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
  'num_clusters':[num_clusters],
  'feats':[feats],
  'style_iters':[style_iters],
  'num_iters':[num_iters],
  'no_grad':[0],
  'repeat_text':[repeat_text],
  'optim':[optim],
  'lr':[lr],
  'optim_separate':[optim_separate]
}

