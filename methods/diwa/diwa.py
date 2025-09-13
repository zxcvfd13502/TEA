import os
import copy
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import pdb
import os
import pandas as pd
import csv
from pathlib import Path
import torch.nn as nn
from ..dataloaders import FastDataLoader, InfiniteDataLoader
from ..utils import prepare_data, forward_pass, get_collate_functions, MetricLogger
import random
from ..base_trainer import BaseTrainer

def freeze_normalization_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # print(f"Freezing normalization layer: {name}")
            module.eval()
            for param in module.parameters():
                param.requires_grad = False



class DiWA(BaseTrainer):
    """
    Empirical Risk Minimization
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

    def simple_average_weight(self, weight_ls):
        count = len(weight_ls)
        if count == 1:
            return weight_ls[0]
        avg_weights = None
        for idx, state_dict in enumerate(weight_ls):
            weight = 1/count
            if avg_weights is None:
                avg_weights = {k: v.clone() * weight for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * weight

        return avg_weights

    def __str__(self):
        if self.args.lisa:
            return f'DiWA-LISA-no-domainid-{self.base_trainer_str}'
        elif self.args.mixup:
            return f'DiWA-Mixup-no-domainid-{self.base_trainer_str}'
        return f'DiWA-{self.base_trainer_str}'
    def train_offline(self):
        weight_ls = []
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_id_dataloader)
        best_id_acc = self.eval_all_id()
        org_network = copy.deepcopy(self.network)
        weight_ls.append(org_network.state_dict())
        cur_lr_rdev = 0.2
        for ide in range(self.args.split_epochs):
            cur_lr = self.args.lr + (-1)**ide * cur_lr_rdev *(ide//2 + 1) * self.args.lr
            print("cur_lr", cur_lr)
            self.network = copy.deepcopy(org_network)
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=cur_lr, weight_decay=self.args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
            self.train_step(train_id_dataloader, freeze_norm=True, user_epochs=1)
            weight_ls.append(self.network.state_dict())
            avg_weights = self.simple_average_weight(weight_ls)
            self.network.load_state_dict(avg_weights)
            cur_id_acc = self.eval_all_id()
            if cur_id_acc < best_id_acc:
                weight_ls.pop()
            else:
                best_id_acc = cur_id_acc
        avg_weights = self.simple_average_weight(weight_ls)
        self.network.load_state_dict(avg_weights)

    def eval_all_id(self):
        self.train_dataset.mode = 1
        test_id_dataloader = FastDataLoader(dataset=self.train_dataset,
                                            batch_size=self.mini_batch_size,
                                            num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        id_acc = self.network_evaluation(test_id_dataloader)
        self.train_dataset.mode = 0
        return id_acc

    def train_step(self, dataloader, freeze_norm=False, user_epochs = None):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        if freeze_norm:
            freeze_normalization_layers(self.network)
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        if user_epochs is not None:
            stop_iters = int((user_epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1) * self.args.diwa_ratio)
            print("==========Epochs: {}==========".format(user_epochs))
        else:
            stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
            print("==========Epochs: {}==========".format(self.args.epochs))
        eval_num = 5
        if self.args.eval_num > 0:
            eval_num = self.args.eval_num
        for step, (x, y) in enumerate(dataloader):
            
            if step % (stop_iters // eval_num) == 0:
                timestamp = self.train_dataset.current_time
                id_acc = self.eval_id(timestamp)
                if freeze_norm:
                    freeze_normalization_layers(self.network)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, id_acc * 100.0))
                if step == 0:
                    self.init_id_acc = id_acc
            
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                               self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            #-----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=(loss).item())
            if step % self.args.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "timestamp: {timestamp}",
                            f"[iter: {step}/{stop_iters}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        eta=eta_string,
                        timestamp=self.train_dataset.current_time,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )

            # if step % (stop_iters // 5) == 0:
            
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))