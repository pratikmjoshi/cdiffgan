import model.trainer

import pdb
def trainer_chooser(args):
  trainer_name = 'model.trainer.Trainer'
  if args.trainer is not None:
    trainer_name += args.trainer
  else:
    if args.noise_only:
      trainer_name += 'NoiseOnly'
    if 'Joint' in args.model:
      trainer_name += 'Joint'
    if 'Late' in args.model:
      trainer_name += 'Late'
    if 'Gest' in args.model:
      trainer_name += 'Gest'
    # if 'Transformer' in args.model:
    #   trainer_name += 'Transformer'
    if 'Cluster' in args.model:
      trainer_name += 'Cluster'
    if 'VQ' in args.model:
      trainer_name += 'VQ'
    if 'Prior' in args.model:
      trainer_name += 'Prior'
    if 'Style' in args.model:
      trainer_name += 'Style'
    if 'Disentangle' in args.model:
      trainer_name += 'Disentangle'
    if 'Learn' in args.model:
      trainer_name += 'Learn'
    if args.pos:
      trainer_name += 'POS'
    if 'Contrastive' in args.model:
      trainer_name += 'Contrastive'
    if 'DTW' in args.model:
      trainer_name += 'DTW'
    # if 'Noise' in args.model:
    #   trainer_name += 'Noise'
    if 'Mine' in args.model:
      trainer_name += 'Mine'
    if 'Consistent' in args.model:
      trainer_name += 'Consistent'
    if 'Adaptive' in args.model:
      trainer_name += 'Adaptive'
    if 'Diff' in args.model:
      trainer_name += 'Diff'
    if 'Transferring' in args.model:
      trainer_name += 'Transferring'
    if 'ewc' in args.model:
      trainer_name += 'EWC'
    if args.gan:
      trainer_name += 'GAN'
    # if args.sample_all_styles:
    #   trainer_name += 'Sample'
    if args.mix:
      trainer_name += 'Mix'
    if 'NN' in args.model:
      trainer_name += 'NN'
    if 'Rand' in args.model:
      trainer_name += 'Rand'
    if 'Mean' in args.model:
      trainer_name += 'Mean'
    if 'Classifier' in args.model:
      trainer_name += 'Classifier'

  try:
    eval(trainer_name)
  except:
    raise '{} trainer not defined'.format(trainer_name)
  print('{} selected'.format(trainer_name))
  return eval(trainer_name)
