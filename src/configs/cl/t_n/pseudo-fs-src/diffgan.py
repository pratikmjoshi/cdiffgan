import json
import pdb
def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x
  
SP = [['oliver'], ['maher'], ['chemistry'], ['ytch_prof'], ['lec_evol']]
speakers = [SP[0]]
for sp in SP[1:]:
  speakers.append(speakers[-1] + sp)
src_sp = speakers[0]
tgt_sps = speakers[1:]

modalities = ['pose/normalize', 'audio/log_mel_400', 'text/tokens']
model = 'StyleDiffGAN4_G'
modelKwargs = [{'tgt_train_ratio':1, 'get_pretrained_D':1}]

replay = 'pseudo-fs-src' ## replay type
k_shots = [28, 140] ## num fewshot
fewshot_seeds = [1, 2, 3] ## fewshot_seed
paradigm = 'no_src_tgt' #
cache = 1 #
gan = 'diffgan' #

note = ['diffgan_{}'.format(replay)] ## model

## pretrained_weights
base_path = 'save7/cl/t1/mixstage_t1/{}/"$(ls save7/cl/t1/mixstage_t1/{} | grep weights | tail -1)"'.format(src_sp[0], src_sp[0])

save_dir = []
for fewshot_seed in fewshot_seeds:
  for k_shot in k_shots:
    for n in note:
      for idx, sp in enumerate(tgt_sps):
        if idx == 0:
          pretrained_model_weights = base_path
        else:
          pretrained_model_weights = '{}/"$(ls {} | grep weights | tail -1)"'.format(save_path, save_path)

        save_path = 'save7/cl/t{}/{}/{}/{}/{}/{}'.format(idx+2, replay, n, k_shot, '-'.join(sp),fewshot_seed)
        save_dir.append((fewshot_seed,
                         k_shot,
                         n,
                         convert2str([src_sp]+[sp]),
                         save_path,
                         pretrained_model_weights))
        


window_hop = 5
batch_size = 32
num_epochs = 20
stop_thresh = 3
overfit = 1
dg_iter_ratio = 1 #
early_stopping = 1
dev_key = 'dev_FID'
num_iters = 200 #
num_training_iters = 200 #
update_D_prob_flag = 0 #
no_grad = 0 #
repeat_text = 0

input_modalities = modalities[1:]
output_modalities = modalities[:1]
fs_new = [15] * len(modalities)

num_clusters = 8
feats = ['pose', 'velocity', 'speed']

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
  ('fewshot_seed', 'k_shot', 'note', 'speaker', 'save_dir', 'pretrained_model_weights'):save_dir,
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
  'num_iters':[num_iters],
  'num_training_iters':[num_training_iters],
  'no_grad':[no_grad],
  'repeat_text':[repeat_text],
  'dg_iter_ratio':[dg_iter_ratio],
  'update_D_prob_flag':[update_D_prob_flag],
  'cache':[cache],
  'replay':[replay],
  'paradigm':[paradigm]
}

