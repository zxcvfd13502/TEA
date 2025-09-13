import os

from ..base_trainer import BaseTrainer

import copy
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import pdb
import pandas as pd
import csv
from pathlib import Path
from ..dataloaders import FastDataLoader, InfiniteDataLoader
from ..utils import prepare_data, forward_pass, get_collate_functions, MetricLogger
import random

import torch.nn as nn

def freeze_normalization_layers(model):
    """
    将模型中的所有归一化层设置为不训练且不更新，并打印调试信息。
    
    Args:
        model (nn.Module): PyTorch 模型。
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # print(f"Freezing normalization layer: {name}")
            module.eval()
            # for param in module.parameters():
            #     param.requires_grad = False
            # frozen = not any(param.requires_grad for param in module.parameters())
            # print(f"Layer {name} frozen: {frozen}")
    # pdb.set_trace()

def update_table(df, step, rw_ts, id_acc):
    if rw_ts not in df.columns:
        df[rw_ts] = np.nan
    if step not in df.index:
        df.loc[step, :] = np.nan
    df.loc[step, rw_ts] = id_acc
    return df

class FT(BaseTrainer):
    """
    Fine-tuning
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.args = args
        self.K = args.K

    def __str__(self):
        return f'FT-K={self.K}-{self.base_trainer_str}'
    
    def train_step(self, dataloader, freeze_norm=False, rw_configs=None, user_iters = -1, user_lr_decay = False,ts = -1, idd = -1):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        # if freeze_norm:
        #     freeze_normalization_layers(self.network)
        loss_all = []
        if idd > self.args.freeze_start:
            freeze_normalization_layers(self.network)
        base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}_{self.args.freeze_start}" / f"{self.args.mini_batch_size}_stsi_{self.args.lr}"/ f"{ts}"
        base_path.mkdir(parents=True, exist_ok=True)
        split_table = pd.DataFrame()
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        if user_iters > 0:
            stop_iters = user_iters
        if user_lr_decay:
            user_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stop_iters, eta_min=1e-5)
        print("==========Epochs: {}==========".format(self.args.epochs))
        for step, (x, y) in enumerate(dataloader):
            # print("step: ", step, "ts: ", ts, "idd: ", idd)
        
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                               self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if user_lr_decay:
                user_scheduler.step()
            

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
            if step % (max(1, stop_iters // 6)) == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, acc * 100.0))
                if idd > self.args.freeze_start:
                    out_acc = self.eval_id(self.split_time+1, mode = 2)
                    freeze_normalization_layers(self.network)
                    split_table = update_table(split_table, step, ts, acc * 100.0)
                    split_table = update_table(split_table, step, self.split_time + 1, out_acc * 100.0)
                    split_table = split_table.round(2)
                    # print("org out acc: ", self.out_acc_org * 100.0)
                    print(split_table)
                    cur_params = self.network.state_dict()
                    param_path = base_path / f"step_{step}.pth"
                    torch.save(cur_params, param_path)
                    split_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}_{self.args.freeze_start}" / f"{self.args.mini_batch_size}_stsi_{self.args.lr}"/f"{ts}.csv")

            # if step % (stop_iters // 5) == 0:
            
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))
        
    
    def train_online(self):
        self.train_dataset.mode = 0
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == (self.split_time + 1):
                break
            else:
                if self.args.pt_split_ts >= 0:
                    # self.args.defrost()
                    # self.args.freeze_start = self.args.pt_split_ts
                    # self.args.freeze()
                    if timestamp < self.args.pt_split_ts:
                        print("accumulating samples at domain:", timestamp)
                        self.train_dataset.mode = 0
                        self.train_dataset.update_current_timestamp(timestamp)
                        self.train_dataset.update_historical(i + 1)
                        self.train_dataset.mode = 1
                        self.train_dataset.update_current_timestamp(timestamp)
                        self.train_dataset.update_historical(i + 1, data_del=True)
                    elif timestamp == self.args.pt_split_ts:
                        self.train_dataset.mode = 0
                        self.train_dataset.update_current_timestamp(timestamp)
                        if self.args.method in ['simclr', 'swav']:
                            self.train_dataset.ssl_training = True
                        train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                                batch_size=self.mini_batch_size,
                                                                num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                        self.train_step(train_id_dataloader)
                    elif timestamp > self.args.pt_split_ts:
                        if i == (self.args.freeze_start + 1):
                            print("freezing normalization layers", i, 'lr', self.args.lr)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] *= self.args.freeze_lr_ratio
                                print("freezing normalization layers", i, 'lr', param_group['lr'])
                        if self.args.lisa and i == self.args.lisa_start_time:
                            self.lisa = True
                        self.train_dataset.update_current_timestamp(timestamp)
                        if self.args.method in ['simclr', 'swav']:
                            self.train_dataset.ssl_training = True
                        train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                            num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                        self.train_step(train_dataloader, ts = timestamp, idd = i)

                        self.eval_dataset.mode = 1
                        self.eval_dataset.update_current_timestamp(timestamp)
                        test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                            batch_size=self.mini_batch_size,
                                                            num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                        acc = self.network_evaluation(test_id_dataloader)
                        self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                else:
                    if i == (self.args.freeze_start + 1):
                        print("freezing normalization layers", i, 'lr', self.args.lr)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= 1/4
                            print("freezing normalization layers", i, 'lr', param_group['lr'])
                    if self.args.lisa and i == self.args.lisa_start_time:
                        self.lisa = True
                    self.train_dataset.update_current_timestamp(timestamp)
                    if self.args.method in ['simclr', 'swav']:
                        self.train_dataset.ssl_training = True
                    train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                        num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                    self.train_step(train_dataloader, ts = timestamp, idd = i)

                    self.eval_dataset.mode = 1
                    self.eval_dataset.update_current_timestamp(timestamp)
                    test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                        batch_size=self.mini_batch_size,
                                                        num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                    acc = self.network_evaluation(test_id_dataloader)
                    self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

