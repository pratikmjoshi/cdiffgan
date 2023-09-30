source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/1 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/1 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/1/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/1 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof-lec_evol/1 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/1/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/1 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/1 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/1 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/1/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/1 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 1 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof-lec_evol/1 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/1/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/1 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/2 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/2 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/2/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/2 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof-lec_evol/2 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/2/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/2 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/2 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/2 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/2/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/2 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 2 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof-lec_evol/2 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/2/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/2 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/3 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/3 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/3/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry/3 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 28 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof-lec_evol/3 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/3/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/28/oliver-maher-chemistry-ytch_prof/3 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry"]]' -save_dir save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/3 -pretrained_model_weights save7/cl/baseline/mixstage_baseline/oliver_maher/"$(ls save7/cl/baseline/mixstage_baseline/oliver_maher | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof"]]' -save_dir save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/3 -pretrained_model_weights save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/3/"$(ls save7/cl/t2/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry/3 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
source activate torch
python train.py -cpk StyleDiffGAN4_G -tb 1 -exp 1 -modelKwargs '{"tgt_train_ratio": 1, "get_pretrained_D": 1}' -model StyleDiffGAN4_G -fewshot_seed 3 -k_shot 140 -note diffgan-sp2_pseudo-align-fs-src -speaker '[["oliver", "maher"], ["oliver", "maher", "chemistry", "ytch_prof", "lec_evol"]]' -save_dir save7/cl/t4/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof-lec_evol/3 -pretrained_model_weights save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/3/"$(ls save7/cl/t3/pseudo-align-fs-src/diffgan-sp2_pseudo-align-fs-src/140/oliver-maher-chemistry-ytch_prof/3 | grep weights | tail -1)" -modalities '["pose/normalize", "audio/log_mel_400", "text/tokens"]' -fs_new '[15, 15, 15]' -input_modalities '["audio/log_mel_400", "text/tokens"]' -output_modalities '["pose/normalize"]' -gan align_diffgan -loss L1Loss -window_hop 5 -render 0 -batch_size 32 -num_epochs 20 -stop_thresh 3 -overfit 1 -early_stopping 1 -dev_key dev_FID -num_clusters 8 -feats '["pose", "velocity", "speed"]' -num_iters 200 -num_training_iters 200 -no_grad 0 -repeat_text 0 -dg_iter_ratio 1 -update_D_prob_flag 0 -cache 1 -replay pseudo-align-fs-src -paradigm no_src_tgt || exit
