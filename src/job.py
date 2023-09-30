'''
Usage:
python job.py -p <partition> -mem <memory> -c <number of cores> -t <time> -log <log directory> -lines <number of lines per job> -job <path2jobfile> -mail <0 or 1>

Example:
python job.py -p cpu_low -mem 1000 -c 4 -t 1-00:00 -log .log -lines 2 -job jobs/job.sh

where a jobs/job.sh could look like:

source activate torch
python hello.py --args <args>
source activate torch
python hello.py --args <args>
source activate torch
python hello.py --args <args>
...

'''

from argparse import ArgumentParser
import pdb
from pathlib import Path
import subprocess
import os

def run_bash(bashCommand):
  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
  output, error = process.communicate()
  return output

def run_unit_test(output):
  ## runs a successtest after previous program completes successfully. 
  dependency = int(output.decode("utf-8").strip().split(' ')[-1])
  src = 'jobs/successtest/successtest.py.sbatch'
  dest = 'jobs/successtest/{}.py.sbatch'.format(dependency)
  run_bash('cp {} {}'.format(src, dest))
  output_successtest = run_bash('sbatch --dependency=afterok:{} {}'.format(dependency, dest))
  print(output_successtest.decode("utf-8").strip() + ' (Success Test)')

import sys

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

parser = ArgumentParser()

## Dataset Parameters
parser.add_argument('-p', type=str, default='gpu_low',
                    help='partition')
parser.add_argument('-gpu', type=int, default=0,
                    help='number of gpus')
parser.add_argument('-mem', type=str, default='32000',
                    help='memory')
parser.add_argument('-N', type=str, default='1',
                    help='num_nodes')
parser.add_argument('-c', type=str, default='1',
                    help='num_cores')
parser.add_argument('-t', type=str, default='3-00:00',
                    help='time')
parser.add_argument('-log', type=str, default='.log',
                    help='log directory')
parser.add_argument('-lines', type=str, default='2',
                    help='lines per task')
parser.add_argument('-job', type=str, default='jobs/job.sh',
                    help='job file')
parser.add_argument('-mail', type=str, default=0,
                    help='ALL, BEGIN, END, FAIL or 0')
parser.add_argument('-dependency', type=int, default=None,
                    help='dependency')
parser.add_argument('-successtest', type=int, default=0,
                    help='run unit test or not')
parser.add_argument('-ids', nargs='+', type=int, default=None,
                    help='list of ids to use')

args, unknown = parser.parse_known_args()
print(args)

sbatch_str = '''#!/bin/bash
#
'''
if args.gpu:
  sbatch_str += '''#SBATCH --gres=gpu:{}
'''.format(args.gpu)

if args.dependency:
  sbatch_str += '''#SBATCH --dependency=afterok:{}
'''.format(args.dependency)
  
## log path
job_path = Path(args.job)
log_path = Path(args.log)/job_path.relative_to('jobs')
log_name = log_path.stem
log_path = log_path.parent
os.makedirs(log_path, exist_ok=True)

## find the number of lines in the job file
total_lines = run_bash('wc -l {}'.format(args.job))
total_lines = int(total_lines.decode("utf-8").split(' ')[0])

if args.ids is not None:
  start_list = [(idx-1)*int(args.lines) + 1 for idx in args.ids]
  end_list = [idx + int(args.lines)-1 for idx in start_list]
  start = 1
  end = len(start_list)
else:
  start_list = [idx for idx in range(1, total_lines, int(args.lines))]
  end_list = [idx + int(args.lines)-1 for idx in start_list]
  start = 1
  end = int(total_lines/int(args.lines))

def get_list_str(start_list):
  start_list_str = '(' + '{} '*(len(start_list) - 1) + '{})'
  start_list_str = start_list_str.format(*start_list)
  return start_list_str

start_list_str = get_list_str(start_list)
end_list_str = get_list_str(end_list)
print(start_list, end_list)

## Exclude list: compute-1-33,compute-1-17,compute-0-37,compute-1-5,compute-1-29,compute-2-9
sbatch_str += '''#SBATCH -p {}    # partition
#SBATCH --mem {}   # memory pool for all cores
#SBATCH -N {}
#SBATCH -c {}
#SBATCH -t {}    # time (D-HH:MM)
#SBATCH -o {}/{}_%A_%a.out        # STDOUT. %j specifies JOB_ID.
#SBATCH -e {}/{}_%A_%a.err        # STDERR. See the first link for more options.
#SBATCH -x compute-2-9,compute-1-17,compute-1-37

LINES={}
ID=$SLURM_ARRAY_TASK_ID
ID=`expr $ID - 1`
START_LIST={}
END_LIST={}
START="${{START_LIST[ID]}}"
END="${{END_LIST[ID]}}"
for i in `seq $START $END`
do
    command=$(sed -n "$i p" {})
    eval $command
done
'''.format(args.p, args.mem, args.N, args.c, args.t, log_path, log_name, log_path, log_name, args.lines, start_list_str, end_list_str, args.job)


with open(args.job + '.sbatch', 'w') as f:
  f.write(sbatch_str)

bashCommand = Path('./') / Path(args.job + '.sbatch')
bashCommand = bashCommand.as_posix()

## convert sbatch file to executable
_ = run_bash('chmod u+x {}'.format(bashCommand))

## find username
username = run_bash('whoami')
username = username.decode('utf-8').strip()
user_email = '{}+slurm@andrew.cmu.edu'.format(username)

answer = query_yes_no('Are you sure you want to submit a job array of size {}?'.format(end), default='no')

## run the sbatch file
if answer:
  sbatch_args = '--array={}-{}'.format(start, end)
  if args.mail:
    sbatch_args = '{} --mail-type={} --mail-user={}'.format(sbatch_args, args.mail, user_email)
  output = run_bash('sbatch {} {}'.format(sbatch_args, bashCommand))
  print(output.decode("utf-8").strip())

  ## unit test
  if args.successtest:
    run_unit_test(output)
else:
  print('Jobs NOT submitted')
  sys.exit(0)
