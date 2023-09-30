import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-load', type=str)
parser.add_argument('-a', nargs='+', type=str,
                    help='list of arguments')

args = parser.parse_args()

filename = args.load
filename = '_'.join(filename.split('_')[:-1] + ['args.args'])
arg_file = json.load(open(filename, 'r'))
print('ARGS')
print('----')
for arg in args.a:
  if arg in arg_file:
    print('{}:{}'.format(arg, arg_file.get(arg)))
  else:
    print('{}:{}'.format(arg, 'not found'))
  
