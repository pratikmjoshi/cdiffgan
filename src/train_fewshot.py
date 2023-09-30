from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import gc
import pdb
import torch
import copy

def loop(args, exp_num):
  args_copy = copy.deepcopy(args)
  for i, num_training_sample in enumerate([1, 10, 100, 1000, 1000000]):
    args = copy.deepcopy(args_copy)
    args.__dict__.update({'num_training_sample':num_training_sample,
                          'note':args.note+'_{}_{}'.format(args.speaker[0], num_training_sample)})
    if i > 0:
      args.__dict__.update({'load_data':0})
    args_subset = ['exp', 'cpk', 'speaker', 'model']
    args_dict_update = {}

    ## Choose Trainer
    Trainer = trainer_chooser(args)

    ## Train
    trainer = Trainer(args, args_subset, args_dict_update)
    trainer.start_exp()  ## Start Log
    trainer.book._set_seed()

    trainer.train(exp_num)  ## Train
    print('Loading the best model and running the sample loop')
    args.__dict__.update({'load':trainer.book.name(trainer.book.weights_ext[0],
                                                   trainer.book.weights_ext[1],
                                                   trainer.args.save_dir)})
    args_dict_update = {'render':args.render, 'window_hop':0, 'load_data':0}
    #trainer = None ## remove Trainer Object to avoid memory errors
  #  del trainer
  #  gc.collect()    

    ## Sample
    trainer_sample = Trainer(args, args_subset, args_dict_update)
    trainer_sample.data = trainer.data
    trainer_sample.data_train = trainer.data_train
    trainer_sample.data_dev = trainer.data_dev
    trainer_sample.data_test = trainer.data_test
    #trainer.book.args.__dict__.update({'window_hop':0})
    trainer_sample.sample(exp_num)

    ## Finish
    trainer_sample.finish_exp()

    ## Print Experiment No.
    print(args.exp)
  
if __name__ == '__main__':
  argparseNloop(loop)
