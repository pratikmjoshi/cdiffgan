mod_map = {'diffgan_pseudo-align-fs-src':'C-DiffGAN (Ours)',
          'mixstage_pseudo-align-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{diffgan}$',
          'diffgan_no-replay-fs':r'DiffGAN ~\cite{ahuja2022low}',
          'diffgan_random-fs':'Buffer Replay',
           'mixstage_pseudo-fs-tgt': r'C-DiffGAN w/o $\mathcal{L}_{ccf}$',
          'diffgan_pseudo-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{ccf} archive$',
          'diffgan_pseudo-align-fs-tgt':'MeRGAN-RA ~\cite{wu2018memory}',
          'diffgan_pseudo-fs-tgt':'MeRGAN-JTR ~\cite{wu2018memory}',
          'mixstage_baseline':r'MixStAGe ~\cite{ahuja2020style}',
           'mixstage-fs-140-preloaded':r'MixStAGe low resource 2',
           'mixstage-fs-140-nopreloaded': r'MixSTAGe low ~\cite{ahuja2020style}',
           'diffgan_pseudo-align-fs-src-no-ldiff':r'C-DiffGAN w/o $\mathcal{L}_{diff}$',
           'diffgan_pseudo-align-fs-src-no-lshift':r'C-DiffGAN w/o $\mathcal{L}_{shift}$',
          'gt':'Ground Truth'}
mod_map_fig = {'diffgan_pseudo-align-fs-src':'C-DiffGAN (Ours)',
          'mixstage_pseudo-align-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{diffgan}$',
          'diffgan_no-replay-fs':r'DiffGAN',
          'diffgan_random-fs':'Buffer Replay',
          'diffgan_pseudo-fs-src':r'C-DiffGAN w/o $\mathcal{L}_{ccf}$',
          'diffgan_pseudo-align-fs-tgt':'MeRGAN-RA',
          'diffgan_pseudo-fs-tgt':'MeRGAN-JTR',
          'mixstage_baseline':r'MixStAGe',
          'gt':'Ground Truth'}          
color_map = {'diffgan_pseudo-align-fs-src':'tab:green',
          'mixstage_pseudo-align-fs-src':'tab:orange',
          'diffgan_no-replay-fs':'tab:red',
          'diffgan_random-fs':'tab:purple',
          'diffgan_pseudo-fs-src':'tab:pink',
          'diffgan_pseudo-align-fs-tgt':'tab:gray',
          'mixstage_baseline':'tab:brown',
          'diffgan_pseudo-fs-tgt':'tab:olive',
          'gt':'tab:cyan'}
round_map = {('Accuracy', 'test_pck'):2, 
            ('Accuracy', 'test_FID'):1, 
            ('Forgetting', 'test_pck'):2, 
            ('Forgetting', 'test_FID'):1}