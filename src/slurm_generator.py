from argsUtils import argparseNloop
import itertools
import pdb
from pathlib import Path
import os

def load_config(conv_config):
  import importlib.util as U
  spec = U.spec_from_file_location('config_loader', conv_config)
  config_loader = U.module_from_spec(spec)
  spec.loader.exec_module(config_loader)
  return config_loader.conv_config

def loop(args, exp_num):
  config = args.config
  script = args.script
  assert config, "config can't be None"
  assert script, "script can't be None"

  prequel = args.prequel if args.prequel is not None else 'source activate torch\\n'
  sequel = args.sequel if args.sequel is not None else ''
  config = load_config(config)
  config_keys = config.keys()
  #config_keys = sorted(config)
  config_perm = [dict(zip(config_keys, prod)) for prod in itertools.product(*(config[names] for names in config_keys))]

  command = ''
  for i, perm in enumerate(config_perm):
    command += prequel
    command += 'python {} '.format(script)
    cmd_list = []
    for key in perm:
      if isinstance(key, tuple):
        for key_tup, value in zip(key, perm[key]):
          cmd_list.append('-{} {}'.format(key_tup, value))
      else:
        cmd_list.append('-{} {}'.format(key, perm[key]))
      
    command += ' '.join(cmd_list)
    command += sequel
    command += '\\n'

  jobs = Path('jobs')/Path(args.config).relative_to('configs')
  os.makedirs(jobs.parent, exist_ok=True)
  with open(jobs, 'w') as f:
    f.write('\n'.join(command.split('\\n')))

  print('{} jobs created'.format((len(command.split('\\n'))-1)/2))
if __name__ == '__main__':
  argparseNloop(loop)
