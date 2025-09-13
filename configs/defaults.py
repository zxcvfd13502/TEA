from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.dataset = 'yearbook'  # choices =['yearbook' 'rmnist' 'fmow' 'arxiv' 'huffpost' 'clear10']
_C.method = 'erm'
_C.device = 0  # 'gpu id'
_C.random_seed = 1  # 'random seed number'

# Training hyperparameters
_C.epochs = 2              # training epochs for each timestamp
_C.lr = 0.01  # the base learning rate
_C.momentum = 0.9  # 'momentum'
_C.weight_decay = 0.0  # 'weight decay'
_C.mini_batch_size = 64  # mini batch size
_C.reduction = 'mean'
_C.online_switch = False     # works only for the methods that are not designed for continually training
_C.init_timestamp = 0
_C.dim_bottleneck_f = None  # dim for the bottlenecked features

# Evaluation
# todo: set value of split_time
_C.split_time = 0  # timestep to split
_C.eval_metric = 'acc'
_C.eval_fix = True
_C.eval_next_timestamps = 1    # number of future timesteps to evaluate on, works when specifying "eval_fix = False"

# FT
_C.K = 1  # 'number of previous timesteps to finetune on'

# LISA and Mixup
_C.lisa = False  # 'train with LISA'
_C.lisa_intra_domain = False  # 'train with LISA intra domain'
_C.lisa_start_time = 0  # 'lisa_start_time'
_C.mixup = False  # 'train with vanilla mixup'
_C.mix_alpha = 2.0  # 'mix alpha for LISA'
_C.cut_mix = False  # 'use cut mix up'

# CORAL, IRM
_C.group_size = 4  # 'window size for Invariant Learning baselines'

# EWC
_C.ewc_lambda = 1.0  # help ='how strong to weigh EWC-loss ("regularisation strength")'
_C.gamma = 1.0  # help ='decay-term for old tasks (contribution to quadratic term)'
_C.online = False  # help ='"online" ( =single quadratic term) or "offline" ( =quadratic term per task) EWC'
_C.fisher_n = None  # help ='sample size for estimating FI-matrix (if "None" full pass over dataset)'
_C.emp_FI = False  # help ='if True use provided labels to calculate FI ("empirical FI"); else predicted labels'

# A-GEM
_C.buffer_size = 100  # 'buffer size for A-GEM'

# CORAL
_C.coral_lambda = 1.0  # 'how strong to weigh CORAL loss'

# IRM
_C.irm_lambda = 1.0  # 'how strong to weigh IRM penalty loss'
_C.irm_penalty_anneal_iters = 0  # 'number of iterations after which we anneal IRM penalty loss'

# SI
_C.si_c = 0.1  # 'SI: regularisation strength'
_C.epsilon = 0.001  # 'dampening parameter: bounds "omega" when squared parameter-change goes to 0'

# SGP
_C.gpm_eps = 0.97
_C.gpm_eps_inc = 0.003
_C.scale_coff = 10

# Drain
_C.hidden_dim = 256
_C.latent_dim = 128
_C.num_rnn_layers = 1
_C.num_layer_to_replace = -1
_C.window_size = 3  # <= 0 means disable skip connection
_C.lambda_forgetting = 0.1

# GI
_C.time_dim = 8
_C.time_append_dim = 32
_C.gi_finetune_epochs = 5
_C.gi_finetune_bs = 64
_C.gi_start_to_finetune = None

#LSSAE
_C.lssae_coeff_y = 1.0
_C.lssae_coeff_ts = 1.0
_C.lssae_coeff_w = 1.0
_C.lssae_zc_dim = 64
_C.lssae_zw_dim = 64

# our method
_C.yearbook_group_size = None
_C.truncate = 1.0
_C.hidden_discriminator = 128
_C.warm_max_iters = None    # warm up iterations for domain discriminator
_C.warm_multiply = None
_C.num_head = 16
_C.dim_head = None
_C.scale = 1                # the number of scales used in attention head
_C.tradeoff_adv = 1.0
_C.memory_pool = None  # size of the available historical statistics, None means all of the historical ones


# Logging saving and testing options
_C.print_freq = 200   # print frequency
_C.data_dir = './WildTime/datasets'  # 'directory for datasets.'
_C.log_dir = './checkpoints'  # 'directory for summaries and checkpoints.'
_C.num_workers = 4 # 'number of workers in data generator'
_C.log_name = 'log.txt'  # name of log file

_C.swad_method = 'LossValley'   # print frequency
_C.n_converge = 3   # print frequency
_C.n_tolerance = 6   # print frequency
_C.tolerance_ratio = 0.3   # print frequency
_C.swad_eval_iter = 100

_C.epochs_sep = 1
_C.lr_sep = 1e-5
_C.base_dir = '/projectnb/ivc-ml/amliu/tdg/EvoS/ckpts/'
_C.sep_iters = -1

_C.rw_len = -1
_C.rw_freq = 10
_C.rw_iters = 1000

_C.optim_sep = 'adam'
_C.eval_num = -1
_C.model_select = 'last'
_C.lr_sep_decay = False
_C.num_experts = -1

_C.sep_avg = 'no'
_C.slope = 0.0
_C.order = 1

_C.desp = ''
_C.mode3_path = None
_C.split_ratio = 0.1
_C.split_epochs = 1
_C.split_save_num = 5
_C.split_base = 'prior'
_C.split_lr = 2e-5
_C.split_trail_num = 5
_C.eval_pca_path = None
_C.only_last = False
_C.holdout_ratio = 0.0

_C.use_r50 = False
_C.use_densenet = False

_C.freeze_start = 10000000
_C.pt_split_ts = -1

_C.freeze_lr_ratio = 0.5
_C.avg_last_ts = False

_C.memory_num = 1000

_C.ref_bn_model_path = None

_C.diwa_ratio = 1.0
_C.all_experts_eval_path = None

_C.reverse_avg = False
_C.chft = False
# _C.data_dir = '/projectnb/ivc-ml/amliu/clear/CLEAR10/train_image_only/labeled_images/'
_C.partial_ft = None
_C.partial_ft_ratio = 1.0
_C.eval_pe = None
_C.eval_pe_ratio = 1.0
_C.tgt_alpha = None
_C.shuffle_path = None
_C.shuffle_seed = 1