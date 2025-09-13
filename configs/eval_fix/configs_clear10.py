
__all__ = ['configs_clear10_coral', 'configs_clear10_irm', 'configs_clear10_erm',
           'configs_clear10_erm_lisa', 'configs_clear10_erm_mixup', 'configs_clear10_agem',
           'configs_clear10_ewc', 'configs_clear10_ft', 'configs_clear10_si',
           'configs_clear10_simclr', 'configs_clear10_swav', "configs_clear10_drain", 'configs_clear10_gi',
           "configs_clear10_lssae", 'configs_clear10_cdot', 'configs_clear10_cida', 'configs_clear10_evos', 'configs_clear10_swad', 'configs_clear10_sep']

configs_clear10_erm =        {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/erm'}

configs_clear10_irm =        {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'group_size': 1, 'irm_lambda': 1.0, 'irm_penalty_anneal_iters': 0, 'method': 'irm', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/irm'}

configs_clear10_coral =      {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'group_size': 1, 'coral_lambda': 0.9, 'method': 'coral', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/coral'}

configs_clear10_erm_mixup  = {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'mixup': True, 'mix_alpha': 2.0, 'cut_mix': False, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/erm_mixup'}

configs_clear10_erm_lisa =   {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'lisa': True, 'lisa_intra_domain': False, 'lisa_start_time': 0, 'method': 'erm', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/erm_lisa'}

configs_clear10_cdot =       {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'cdot', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/cdot'}

configs_clear10_cida =       {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'cida', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/cida'}

configs_clear10_lssae =      {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'lssae_coeff_y': 1.0, 'lssae_coeff_ts': 0.1, 'lssae_coeff_w': 1.0, 'lssae_zc_dim': 64, 'lssae_zw_dim': 64, 'method': 'lssae', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/lssae'}

configs_clear10_gi =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'time_dim': 8, 'time_append_dim': 32, 'gi_finetune_bs': 64, 'gi_finetune_epochs': 5, 'gi_start_to_finetune': None, 'method': 'gi', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/gi'}

configs_clear10_ft =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'K': 1, 'method': 'ft', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/ft'}

configs_clear10_simclr =     {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'simclr', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/simclr'}

configs_clear10_swav =       {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'swav', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/swav'}

configs_clear10_ewc =        {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'ewc_lambda': 0.5, 'gamma': 1.0, 'online': True, 'fisher_n': None, 'emp_FI': False, 'method': 'ewc', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/ewc'}

configs_clear10_si =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'si', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/si'}

configs_clear10_agem =       {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 1e-5, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'buffer_size': 1000, 'method': 'agem', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/agem'}

configs_clear10_drain =      {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, "hidden_dim": 64, "latent_dim": 64, "num_rnn_layers": 10, "num_layer_to_replace": 1, "window_size": 3, "lambda_forgetting": 0.0, 'method': 'drain', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/drain'}

configs_clear10_evos =       {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'dim_head': 8, 'num_head': 64, 'scale': 3, 'truncate': 1.0, 'tradeoff_adv': 1.0, 'hidden_discriminator': 1024, 'warm_max_iters': None, 'warm_multiply': 10.0, 'method': 'evos', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/evos'}

configs_clear10_swad =        {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'swad', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/swad', 'swad_method': "LossValley", 'n_converge':3, 'n_tolerance': 6, 'tolerance_ratio': 0.3, 'swad_eval_iter': 100, 'ref_bn_model_path':'/projectnb/ivc-ml/amliu/tdg/EvoS/ckpts/decrease_all/tsi_clear10_40/128_1_0.0001/base/step_4038.pth'}

configs_clear10_sep =        {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'sep', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/swa', 'epochs_sep':1, 'lr_sep': 5e-5}

configs_clear10_tsi =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'tsi', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/tsi'}

configs_clear10_stft =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'K': 1, 'si_c': 0.1, 'epsilon': 0.001, 'method': 'stft', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/stft'}

configs_clear10_diwa =         {'dataset': 'clear10', 'device': 0, 'random_seed': 1, 'dim_bottleneck_f': 256, 'epochs': 25, 'lr': 5e-4, 'eval_fix': True, 'init_timestamp': 0, 'split_time': 4, 'method': 'diwa', 'data_dir': '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/', 'log_dir': './checkpoints/clear10/diwa', 'split_epochs': 10, 'diwa_ratio': 0.5}