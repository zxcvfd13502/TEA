
__all__ = ['configs_arxiv_coral', 'configs_arxiv_irm', 'configs_arxiv_erm',
           'configs_arxiv_erm_lisa', 'configs_arxiv_erm_mixup', 'configs_arxiv_agem',
           'configs_arxiv_ewc', 'configs_arxiv_ft', 'configs_arxiv_si',
            'configs_arxiv_drain', 'configs_arxiv_evos']

configs_arxiv_erm =        {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/erm'}

configs_arxiv_irm =        {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/irm'}

configs_arxiv_coral =      {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/coral'}

configs_arxiv_erm_mixup  = {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/erm_mixup'}

configs_arxiv_erm_lisa =   {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'lisa': True, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/erm_lisa'}

configs_arxiv_gi =         {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'time_dim': 8, 'time_append_dim': 32, 'gi_finetune_bs': 64, 'gi_finetune_epochs': 50, 'gi_start_to_finetune': None, 'method': 'gi', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/gi'}

configs_arxiv_ft =         {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'K': 1, 'method': 'ft', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/ft'}

configs_arxiv_ewc =        {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/ewc'}

configs_arxiv_si =         {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 1e-6, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/si'}

configs_arxiv_agem =       {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 1e-6, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/agem'}

configs_arxiv_drain =      {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, "hidden_dim": 64, "latent_dim": 64, "num_rnn_layers": 10, "num_layer_to_replace": 1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/drain'}

configs_arxiv_evos =       {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'dim_head': 8, 'num_head': 128, 'scale': 3, 'truncate': 1.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 2048, 'warm_max_iters': 100000, 'warm_multiply': None, 'method': 'evos', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/evos'}

configs_arxiv_tsi =         {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 4, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'tsi', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/tsi', 'split_epochs': 1, 'split_lr': 2e-6}

configs_arxiv_stft =   {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'K': 1 , 'si_c': 0.1, 'epsilon': 0.001, 'method': 'stft', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/stft'}

configs_arxiv_swad =        {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 5, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'method': 'swad', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/swad', 'swad_method': "LossValley", 'n_converge':3, 'n_tolerance': 6, 'tolerance_ratio': 0.3, 'swad_eval_iter': 100,'ref_bn_model_path': '/projectnb/ivc-ml/amliu/tdg/EvoS/ckpts/decrease_all/tsi_arxiv_2/64_1_3e-06/base/step_26677.pth'}

configs_arxiv_diwa =         {'dataset': 'arxiv', 'device': 0, 'random_seed': 1, 'epochs': 4, 'lr': 2e-5, 'eval_fix': True, 'init_timestamp': 2007, 'split_time': 2015, 'method': 'diwa', 'data_dir': '/projectnb/ivc-ml/amliu/tdg/EvoS/dataset/arxiv', 'log_dir': './checkpoints/arxiv/diwa', 'split_epochs': 10, 'diwa_ratio': 0.1}