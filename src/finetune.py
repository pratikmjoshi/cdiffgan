from argsUtils import argparseNloop
from trainer_chooser import trainer_chooser
import pdb
from pycasper.BookKeeper import *
from pycasper.argsUtils import *

def loop(args, exp_num):
  args_subset = ['exp', 'cpk', 'speaker', 'model']
  args_dict_update = {'render':args.render, 'finetune_quantile_sample':args.finetune_quantile_sample}
  args_dict_update.update(get_args_update_dict(args)) ## update all the input args

  ## Load Args
  book = BookKeeper(args, args_subset, args_dict_update=args_dict_update,
                    tensorboard=args.tb)
  args = book.args

  ## choose trainer
  Trainer = trainer_chooser(args)
  pdb.set_trace()
  ## Init Trainer
  trainer = Trainer(args, args_subset, args_dict_update)

  trainer.book._set_seed()

  ## FINE TUNE over quantile
  ## --------------------------------
  if args.finetune_quantile_sample is not None:
    ## Load best model
    try:
      trainer.book._load_model(trainer.model)
    except:
      pass
    
    ## update train_sampler
    trainer.data.quantile_sample = args.finetune_quantile_sample
    trainer.data.train_sampler = trainer.data.get_train_sampler(trainer.data.dataset_train,
                                                               trainer.data.train_intervals_dict)
    ## update dataloader
    trainer.data.update_dataloaders(trainer.data.time, trainer.data.window_hop)
    trainer.data_train = trainer.data.train
    trainer.data_dev = trainer.data.dev
    trainer.data_test = trainer.data.test
    
    ## update args, trainer.args.weighted, trainer.args.epochs
    trainer.args.__dict__.update({'weighted':0, 'num_epochs':20})
    trainer.num_epochs = 20

    ## update bookkeeper to start the training afresh
    trainer.book.best_dev_score = np.inf * trainer.book.dev_sign
    trainer.book.stop_count = 0

    ## Reset optims and learning rates
    trainer.G_optim, trainer.D_optim = trainer.get_optims()
    trainer.schedulers = trainer.get_scheduler()

    ## train again
    trainer.train(exp_num)

  ## SAMPLE
  ## -----------------------------
  
  ## Sample Prep.
  del trainer
  gc.collect()    

  print('Loading the best model and running the sample loop')
  args_dict_update = {'render':args.render, 'window_hop':0, 'sample_all_styles':0}
  
  ## Sample
  trainer = Trainer(args, args_subset, args_dict_update)
  trainer.sample(exp_num)
  
  ## Finish
  trainer.finish_exp()

  ## Print Experiment No.
  print('\nExperiment Number: {}'.format(args.exp))
      
if __name__ == '__main__':
  argparseNloop(loop)
