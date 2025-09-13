import argparse
import os
import random
import numpy as np
import torch
import json
import PIL
import torchvision
from torch import cuda
import pdb

from configs import cfg
from utils import setup_logger, get_current_time
from baseline_trainer import trainer_init
from methods.cdot.cdot import CDOT
from methods.cida.cida import CIDA
from methods.agem.agem import AGEM
from methods.coral.coral import DeepCORAL
from methods.erm.erm import ERM
from methods.sep.sep import SEP
from methods.ewc.ewc import EWC
from methods.ft.ft import FT
from methods.irm.irm import IRM
from methods.si.si import SI
from methods.simclr.simclr import SimCLR
from methods.swav.swav import SwaV
from methods.ours.evos import EvoS
from methods.drain.drain import Drain
from methods.GI.gi import GI
from methods.lssae import LSSAETrainer
from methods.swad.swad import SWADTrainer
from methods.diwa.diwa import DiWA
from methods.tsi.tsi import TSI
from methods.stsi.stsi import STSI
from methods.stft.stft import STFT
from configs.eval_fix import configs_yearbook, configs_fmow, configs_arxiv, configs_huffpost, configs_rmnist, configs_clear10, configs_clear100


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wild-Time')
    parser.add_argument('--dataset', default='yearbook', choices=['arxiv', 'huffpost', 'fmow', 'yearbook', 'rmnist', 'clear10', 'clear100'])
    parser.add_argument('--method', default='ft', help='name of method', type=str)
    parser.add_argument("opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER,
                        )
    args = parser.parse_args()


    args_dict = globals()[f'configs_{args.dataset}'].__dict__['configs_' + args.dataset + '_' + args.method]
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key)
        args_list.append(value)
    args_list.extend(args.opts)

    # pdb.set_trace()
    print(args_list)
    cfg.merge_from_list(args_list)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    torch.set_num_threads(1)  # limiting the usage of cpu

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.cuda.manual_seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    if not os.path.isdir(cfg.data_dir):
        os.makedirs(cfg.data_dir)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    if cfg.method in ['groupdro', 'irm']:
        cfg.reduction = 'none'
    else:
        cfg.reduction = 'mean'

    cfg.freeze()

    logger = setup_logger("main", cfg.log_dir, 0, filename=get_current_time() + "_" + cfg.log_name)
    logger.info("PTL.version = {}".format(PIL.__version__))
    logger.info("torch.version = {}".format(torch.__version__))
    logger.info("torchvision.version = {}".format(torchvision.__version__))
    logger.info("Running with config:\n{}".format(cfg))


    dataset, criterion, network, optimizer, scheduler = trainer_init(cfg)
    # total_samples = 0
    # for ts in dataset.ENV:
    #     total_samples += len(dataset.datasets[ts][2]['images'])
    # logger.info("Total samples in the dataset: {}".format(total_samples))
    param_info = count_parameters(network)
    logger.info("Total parameters in the model: {}".format(param_info['Total']))
    

    if cfg.mode3_path is not None:
        # for ts in range(0,15):
        #     # pdb.set_trace()
        #     print(ts, "org len", len(dataset.datasets[ts][0]['image_idxs']), len(dataset.datasets[ts][1]['image_idxs']), len(dataset.datasets[ts][2]['image_idxs']))
        dataset.split_and_save_indices(cfg.mode3_path, cfg.split_ratio)
        # for ts in range(0,15):
        #     print(ts, "split len", len(dataset.datasets[ts][0]['image_idxs']), len(dataset.datasets[ts][1]['image_idxs']), len(dataset.datasets[ts][2]['image_idxs']), len(dataset.datasets[ts][3]['image_idxs']))
        # pdb.set_trace()
    if cfg.shuffle_path is not None:
        dataset.shuffle_dataset_by_year()
        
    if cfg.method == 'coral': trainer = DeepCORAL(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'irm': trainer = IRM(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'ft': trainer = FT(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'erm': trainer = ERM(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'ewc': trainer = EWC(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'agem': trainer = AGEM(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'si': trainer = SI(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'simclr': trainer = SimCLR(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == 'swav': trainer = SwaV(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "drain": trainer = Drain(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "gi": trainer = GI(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "lssae": trainer = LSSAETrainer(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "cdot": trainer = CDOT(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "cida": trainer = CIDA(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "evos": trainer = EvoS(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "swad": trainer = SWADTrainer(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "sep": trainer = SEP(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "tsi": trainer = TSI(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "stsi": trainer = STSI(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "stft": trainer = STFT(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    elif cfg.method == "diwa": trainer = DiWA(cfg, logger, dataset, network, criterion, optimizer, scheduler)
    else:
        raise ValueError

    trainer.run()





