import json
def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x
  
# speakers = [['corden', 'lec_cosmic'],
#             ['corden', 'lec_cosmic', 'ytch_prof', 'oliver'],
#             ['corden', 'lec_cosmic', 'ytch_prof', 'oliver', 'ellen', 'noah', 'lec_evol', 'maher']]
speakers = [['oliver', 'maher'],
            ['oliver', 'maher', 'chemistry'],
            ['oliver', 'maher', 'chemistry', 'ytch_prof'],
            ['oliver', 'maher', 'chemistry', 'ytch_prof', 'lec_evol']]
modalities = ['pose/normalize', 'audio/log_mel_400', 'text/tokens']
model = 'JointLateClusterSoftStyle5_G'
modelKwargs = [{'lambda_id':0.1, 'argmax':1, 'some_grad_flag':1, 'train_only':1, 'max_num_speakers':25}]
note = ['mixstage_baseline']
save_dir = [(n, convert2str(sp), 'save7/cl/baseline/{}/{}'.format(n, '_'.join(sp))) for sp in speakers for n in note]

window_hop = 5
batch_size = 16
num_epochs = 20
stop_thresh = 3
overfit = 0
gan = 1
early_stopping = 0
dev_key = 'dev_spatialNorm'
style_iters = 3000
num_iters = 3000
repeat_text = 0

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
  'modelKwargs':modelKwargs,
  'model':[model],
  ('note', 'speaker', 'save_dir'):save_dir,
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
  'repeat_text':[repeat_text]
}

