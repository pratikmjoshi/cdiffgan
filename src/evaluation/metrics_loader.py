from pycasper.results import walkthroughMetrics, walkthroughResults

def get_raw_metrics(paths, args_subset=None, res_subset=None, val_key=None, metrics=True):
  if args_subset is None:
    args_subset=['exp', 'modalities', 'scheduler', 'lr', 'model', 'optim', 'speaker', 'note', 'save_dir']
  if res_subset is None:
    res_subset=['test_L1', 'test_pck', 'test_F1', 'test_spatialNorm', 'test_FID', 'test_W1_vel', 'test_W1_acc', 'test_diversity', 'test_diversity_gt']
  
  if metrics:
    df = walkthroughMetrics(paths, args_subset=args_subset, res_subset=res_subset)
    df['S'] = df.speaker.apply(lambda x:x[0])
    return df
  else:
    df, df_all = walkthroughResults(paths, args_subset=args_subset, res_subset=res_subset, val_key=val_key)
    df_all['S'] = df_all.speaker.apply(lambda x:x[0])
    df['S'] = df.speaker.apply(lambda x:x[0])
    return df, df_all


note_map = {'mmsbert':'MMS-Bert_archive',
            'mmsbert_quantile0.9':'MMS-Bert w/ 90th percentile',
            'mmsbert_rebalance':'MMS-Bert w/ Manual Rebalancing',
            'multiscaleBert_lfiw':'MMS-bert w/o Multihead fusion', 
            'mmsbert_lfiw':'MMS-Bert w/o AIST',
            'mmsbert_text_lfiw':'MMS-Bert w/o Audio',
            'joint_audio_cluster_soft_pvs8_gan':'MMS-Bert w/o Text',
            'joint_late_cluster_soft_pvs8_gan_bert':'MMS-Bert w/o Multiscale',
            'unet_audio_gan':'S2G',
            'gesticulator2':'Gesticulator',
            'mmsbert_lfiw_no_update':'MMS-Bert'}

metrics = ['test_pck', 'test_F1', 'test_FID', 'test_W1_vel', 'test_W1_acc']
def remap_df(df, note_map, note_order=None, speakers=None, metrics=None):
  if note_order is None:
    note_order  = note_map.keys()
  if speakers is None:
    speakers = df.speaker.apply(lambda x:x[0]).unique()
  
  ## convert all metrics to numerals
  if metrics:
    df[metrics] = df[metrics].fillna(-1)
  
  ## Rename Models
  df['model'] = df.note.apply(lambda x:note_map[x] if x in note_map else x)
  models_in_df = df.model.unique()

  ## get speaker from speaker list
  df['S'] = df.speaker.apply(lambda x:x[0])
  
  ## keep the required speakers
  df = df[df.S.isin(speakers)]

  ## only take the speakers that have all the models
  df = df.groupby(['S']).filter(lambda x:x.shape[0] >= len(note_order))
  missing_speakers = set(speakers) - set(df.S.unique())
  
  ## keep the best experiments
  df = df.groupby(['S', 'note']).apply(lambda x:x[x['test_L1'] == x['test_L1'].min()])
  for col in df.columns: ## convert each column to numerical values if possible
    df[col] = pd.to_numeric(df[col], errors='ignore')
        
  ## Group results by model
  df_ = df.groupby('model').mean()
  df__ = df.groupby('model').std()
  if metrics is not None:
    df_ = df_[metrics]
    df__ = df__[metrics]
    
  ## set the model order
  if note_order is not None:
    model_order_ = [note_map[note]  if note in note_map else note for note in note_order] ## find the model order
    
    ## remove unavailable models
    model_order = []
    for model in model_order_:
      if model in models_in_df:
        model_order.append(model)
    
    df_ = df_.loc[model_order]
    df__ = df__.loc[model_order]
  
  return df_, df__, df, missing_speakers
