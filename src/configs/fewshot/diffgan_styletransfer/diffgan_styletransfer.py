import json
import itertools
from pathlib import Path
import os
import pdb

def convert2str(x):
  x = json.dumps(x)
  x = "'" + x + "'"
  return x

def get_pretrained_model_weights(base_path):
  weight_paths = []
  speakers = []
  for paradigm in os.listdir(base_path):
    for k_shot in os.listdir(base_path/paradigm):
      for sp in os.listdir(base_path/paradigm/k_shot):
        for fewshot_seed in os.listdir(base_path/paradigm/k_shot/sp):
          for f in os.listdir(base_path/paradigm/k_shot/sp/fewshot_seed):
            if f.split('_')[-1] == 'weights.p':
              weight_path = (base_path/paradigm/k_shot/sp/fewshot_seed/f).as_posix()
              weight_paths.append(json.dumps(weight_path))
              speakers.append([[sp.split('_')[0]]]*2)
              break

  return weight_paths, speakers 

base_path = Path('save6/fewshot/styletransfer/diffgan_styletransfer')
loads, speakers = get_pretrained_model_weights(base_path)

fewshot_seed = 1
k_shot = 140
p_avg = 1
cache = 1

speakers = [convert2str(sp) for sp in speakers]

conv_config={
'fewshot_seed':[fewshot_seed],
'k_shot':[k_shot],
'p_avg':[p_avg],
'cache':[cache],
('load', 'speaker'):zip(loads, speakers),
}
