''' Usage:
python render_after_training.py -render 100 -view log_file
'''

from slurmpy import Slurm
import os
from pathlib import Path

from argsUtils import argparseNloop
import datetime

import time
import pdb

TMPL = """\
#!/bin/bash

#SBATCH -e logs/{name}.%J.err
#SBATCH -o logs/{name}.%J.out
#SBATCH -J {name}

{header}

__script__"""


def render_samples(args, filename):
  filename = '_'.join(filename.split('_')[:-1] + ['weights.p'])
  render = Slurm('render',
                 slurm_kwargs={'partition':'cpu_low', 'time':'3-00:00', 'n':args.cpu, 'mem':args.mem},
                 tmpl=TMPL)
  python_cmd = ['source activate torch',
                'python render.py -load \"{}\" -render {} -render_text {} -render_transparent {}'.format(
                  filename,
                  args.render,
                  args.render_text,
                  args.render_transparent)]

  render.run('\n'.join(python_cmd))

def render_new(args, filename):
  filename = '_'.join(filename.split('_')[:-1] + ['weights.p'])
  render = Slurm('render_new', slurm_kwargs={'partition':'cpu_long', 'time':'10-00:00', 'n':38, 'mem':16000})
  python_cmd = ['source activate torch',
                'python render_newSentence.py -load {} -view {} -path2data {} -dataset {} -feats_kind {}'.format(
                  filename,
                  args.view,
                  args.path2data,
                  args.dataset,
                  args.feats_kind)]
  print(python_cmd)
  render.run('\n'.join(python_cmd))


def add_line(filename):
  with open(filename, 'a') as f:
    f.write('Rendering job Submitted\n')

def get_lines(filename):
  with open(filename, 'r') as f:
    lines = f.readlines()
  return lines

def find_completed_exps(args, suffix='log.log'):
  filenames = []
  for tup in os.walk(args.view):
    for files in tup[2]:
      if files.split('_')[-1] == suffix:
        try:
          filename = (Path(tup[0])/files).as_posix()
          lines = get_lines(filename)
          if len(lines) == 2:
            filenames.append(files)
            ## run rendering jobs
            render_samples(args, filename)
            print('Jobs submitted for {} at {}'.format(filename, datetime.datetime.now()))
            ## add a third line
            add_line(filename)
        except:
          pdb.set_trace()

def loop(args, exp_num):
  while(1):
    time.sleep(5)
    find_completed_exps(args)
    
if __name__ == '__main__':
  argparseNloop(loop)
