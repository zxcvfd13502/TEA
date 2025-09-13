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
from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions, MetricLogger
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
            for param in module.parameters():
                param.requires_grad = False
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

def maintain_sorted_list(model_list, id_acc, file_path, max_size):
    model_list.append((id_acc, file_path))
    model_list.sort(key=lambda x: x[0], reverse=True)
    if len(model_list) > max_size:
        removed_model = model_list.pop()
        if os.path.exists(removed_model[1]):
            os.remove(removed_model[1])

class BaseTrainer:
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2
        self.num_classes = dataset.num_classes
        self.org_dataset = copy.deepcopy(dataset)
        self.org_dataset.mode = 0
        # self.num_tasks = dataset.num_tasks
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.lisa = args.lisa
        self.epochs = args.epochs
        self.mixup = args.mixup
        self.cut_mix = args.cut_mix
        self.mix_alpha = args.mix_alpha
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.split_time
        self.eval_next_timestamps = args.eval_next_timestamps
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'epochs={self.epochs}-lr={self.args.lr}-' \
                                f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.lisa:
            base_trainer_str += f'-lisa-mix_alpha={self.mix_alpha}'
        elif self.mixup:
            base_trainer_str += f'-mixup-mix_alpha={self.mix_alpha}'
        if self.cut_mix:
            base_trainer_str += f'-cut_mix'
        if self.args.eval_fix:
            base_trainer_str += f'-eval_fix'
        else:
            base_trainer_str += f'-eval_stream'
        return base_trainer_str
    
    def select_partial_domains(self):
        if self.args.partial_ft is None:
            return self.split_set.ENV[:-1]

        # 获取所有 domain 列表，排除最后一个
        all_domains = self.split_set.ENV[:-1]

        # 只保留 split_time 之前的 domain
        eligible_domains = [d for d in all_domains if d <= self.split_time]

        num_domains = len(eligible_domains)
        ratio = self.args.partial_ft_ratio
        num_selected = max(1, int(round(ratio * num_domains)))  # 至少选1个

        if self.args.partial_ft == "last":
            selected_domains = eligible_domains[-num_selected:]

        elif self.args.partial_ft == "uniform":
            import random
            selected_domains = sorted(random.sample(eligible_domains, num_selected))

        else:
            raise ValueError(f"Unknown partial_ft type: {self.args.partial_ft}")

        # 保存或处理选中的 domain
        self.selected_domains = selected_domains
        print(f"Selected domains ({self.args.partial_ft}):", selected_domains)
        return sorted(selected_domains)

    
    def average_and_save_weights(self, step, rw_freq, rw_queue, rw_len, rw_iters):
        avg_weights = None
        count = len(rw_queue)
        for state_dict in rw_queue:
            if avg_weights is None:
                avg_weights = {k: v.clone() for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k]
        for k in avg_weights.keys():
            if avg_weights[k].dtype in (torch.float32, torch.float64):
                avg_weights[k] /= count
            else:
                # print(k,rw_queue[-1][k], avg_weights[k].dtype)
                avg_weights[k] = rw_queue[-1][k].clone()
        # save_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{rw_len}_{rw_freq}_{rw_iters}" / f"{step}.pth"
        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # torch.save(avg_weights, save_path)
        # self.logger.info(f"Averaged weights saved to {save_path}")
        return avg_weights
    
    def average_weights(self, weight_ls):
        avg_weights = None
        count = len(weight_ls)
        for state_dict in weight_ls:
            if avg_weights is None:
                avg_weights = {k: v.clone() for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k]
        for k in avg_weights.keys():
            if avg_weights[k].dtype in (torch.float32, torch.float64):
                avg_weights[k] /= count
            else:
                # print(k,rw_queue[-1][k], avg_weights[k].dtype)
                avg_weights[k] = weight_ls[-1][k].clone()
        return avg_weights
    
    def eval_id(self, timestamp, mode = 1):
        
        self.eval_dataset.mode = mode
        self.eval_dataset.update_current_timestamp(timestamp)
        test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                            batch_size=self.mini_batch_size,
                                            num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        id_acc = self.network_evaluation(test_id_dataloader)
        return id_acc

    def train_step(self, dataloader, freeze_norm=False, rw_configs=None, user_iters = -1, user_lr_decay = False):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        if freeze_norm:
            freeze_normalization_layers(self.network)
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        eval_num = 5
        if self.args.eval_num > 0:
            eval_num = self.args.eval_num
        if rw_configs is not None:
            rw_len, rw_freq, rw_iters, rw_evals = rw_configs
            stop_iters = rw_iters
            rw_queue = deque(maxlen=rw_len)
            rw_table = pd.DataFrame()
            rw_avg_table = pd.DataFrame()
        if user_iters > 0:
            stop_iters = user_iters
        if user_lr_decay:
            user_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stop_iters, eta_min=1e-5)
        max_id_acc = 0.0
        max_weights = None
        max_weight_ls = []
        weight_acc_records = []
        max_weights_accs = []
        print("==========Epochs: {}==========".format(self.args.epochs))
        rw_models = {}
        for step, (x, y) in enumerate(dataloader):
            
            if step % (stop_iters // eval_num) == 0:
                timestamp = self.train_dataset.current_time
                id_acc = self.eval_id(timestamp)
                if id_acc > max_id_acc:
                    max_id_acc = id_acc
                    max_weights = copy.deepcopy(self.network.state_dict())
                weight_acc_records.append((id_acc, copy.deepcopy(self.network.state_dict())))
                    # max_weights_accs.append(id_acc * 100.0)
                if freeze_norm:
                    freeze_normalization_layers(self.network)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, id_acc * 100.0))
                if step == 0:
                    self.init_id_acc = id_acc
            
            if rw_configs is not None:
                rw_queue.append(self.network.state_dict())
                # if step % rw_freq == 0 and step != 0:
                #     org_weights = copy.deepcopy(self.network.state_dict())
                #     avg_weights = self.average_and_save_weights(step, rw_freq, rw_queue, rw_len, rw_iters)
                if step % rw_freq == 0:
                    for rw_ts, mode in rw_evals:
                        id_acc = self.eval_id(rw_ts, mode)
                        freeze_normalization_layers(self.network)
                        rw_table = update_table(rw_table, step, rw_ts, id_acc * 100.0)
                        rw_table = rw_table.round(2)
                        if self.args.num_experts > 0 and rw_ts <= self.split_time:
                            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"rw_{self.args.mini_batch_size}_{rw_freq}_{rw_iters}_{self.args.lr}/rw_ts/{rw_ts}"
                            base_path.mkdir(parents=True, exist_ok=True)
                            weight_file_path = base_path / f"step_{step}.pt"
                            torch.save(self.network.state_dict(), weight_file_path)
                            if rw_ts not in rw_models:
                                rw_models[rw_ts] = []
                            maintain_sorted_list(rw_models[rw_ts], id_acc, weight_file_path, self.args.num_experts)
                    print(rw_table)
                    rw_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"rw_{self.args.mini_batch_size}_{rw_freq}_{rw_iters}_{self.args.lr}.csv")

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

            # if step % (stop_iters // 5) == 0:
            
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))
        # if rw_configs is not None:
            
            # rw_avg_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"rw_avg_{rw_len}_{rw_freq}_{rw_iters}.csv")
        if self.args.model_select == 'best':
            self.network.load_state_dict(max_weights)
            self.logger.info("Load the best model with ID acc {:.3f}".format(max_id_acc * 100.0))
        if self.args.model_select == 'avg':
            for acc, weights in weight_acc_records:
                if acc >= 0.95 * max_id_acc:
                    max_weight_ls.append(weights)
                    max_weights_accs.append(acc * 100.0)
            avg_max_weight = self.average_weights(max_weight_ls)
            self.network.load_state_dict(avg_max_weight)
            self.logger.info("Load the avgered model of {} with max ID accs".format(max_weights_accs))
        
        if self.args.model_select == 'swa':
            weight_accs = random.sample(weight_acc_records, k=5)
            for acc, weights in weight_accs:
                max_weight_ls.append(weights)
                max_weights_accs.append(acc * 100.0)
            avg_max_weight = self.average_weights(max_weight_ls)
            self.network.load_state_dict(avg_max_weight)
            self.logger.info("Load the avgered model of {} with random".format(max_weights_accs))
    
    def train_step_domain_expert(self, dataloader, freeze_norm=False, rw_configs=None, user_iters = -1, user_lr_decay = False):
        self.logger.info("-------------------start training domain expert on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        eval_num = 5
        if self.args.eval_num > 0:
            eval_num = self.args.eval_num
        # if rw_configs is not None:
        #     rw_len, rw_freq, rw_iters, rw_evals = rw_configs
        #     stop_iters = rw_iters
        #     rw_queue = deque(maxlen=rw_len)
        #     rw_table = pd.DataFrame()
        #     rw_avg_table = pd.DataFrame()
        if user_iters > 0:
            stop_iters = user_iters
        if user_lr_decay:
            user_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, stop_iters, eta_min=1e-5)
        max_id_acc = 0.0
        max_weights = None
        max_weight_ls = []
        weight_acc_records = []
        max_weights_accs = []
        print("==========Epochs: {}==========".format(self.args.epochs))
        rw_models = {}
        for step, (x, y) in enumerate(dataloader):
            
            if step % (stop_iters // eval_num) == 0:
                timestamp = self.train_dataset.current_time
                id_acc = self.eval_id(timestamp)
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

            
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))
        split_epoch = self.args.split_epochs
        out_acc_org = self.eval_id(self.split_time + 1, mode=2)
        freeze_normalization_layers(self.network)
        org_weights = copy.deepcopy(self.network.state_dict())
        if self.args.split_base == 'org':
            self.args.defrost()
            self.args.base_dir += '/' + self.args.split_base
            self.args.freeze()
        for ts in range(self.split_time + 1):
            self.split_set.mode = 3
            self.split_set.update_current_timestamp(ts)
            split_id_dataloader = InfiniteDataLoader(dataset=self.split_set, weights=None, batch_size=self.mini_batch_size,
                                                        num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            stop_iters = split_epoch * (self.split_set.__len__() // self.args.mini_batch_size) - 1
            save_iter = stop_iters // self.args.split_save_num + 1
            # pdb.set_trace()
            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.lr}"/ f"{ts}"
            base_path.mkdir(parents=True, exist_ok=True)
            split_table = pd.DataFrame()
            if self.args.split_base == 'org':
                self.network.load_state_dict(org_weights)
            freeze_normalization_layers(self.network)
            for step, ((x, y), (split_x, split_y)) in enumerate(zip(dataloader, split_id_dataloader)):
                x = torch.cat([x, split_x], dim=0)
                y = torch.cat([y, split_y], dim=0)
                x, y = prepare_data(x, y, str(self.train_dataset))

                loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                                self.cut_mix, self.mix_alpha)
                loss_all.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(step, save_iter)
                if (step + 1) % save_iter == 0:
                    weight_file_path = base_path / f"step_{step}.pt"
                    torch.save(self.network.state_dict(), weight_file_path)
                    id_acc = self.eval_id(ts, mode=1)
                    out_acc = self.eval_id(self.split_time + 1, mode=2)
                    freeze_normalization_layers(self.network)
                    split_table = update_table(split_table, step, ts, id_acc * 100.0)
                    split_table = update_table(split_table, step, self.split_time + 1, out_acc * 100.0)
                    split_table = split_table.round(2)
                    print("org out acc: ", out_acc_org * 100.0)
                    print(split_table)
                    split_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.lr}"/f"{ts}.csv")
                if step == stop_iters:
                    break

    def train_online(self):
        self.train_dataset.mode = 0
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == (self.split_time + 1):
                break
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)

                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

    def train_offline(self):
        if self.args.method in ['simclr', 'swav']:
            self.train_dataset.ssl_training = True
        if self.args.mode3_path is not None:
            self.split_set = copy.deepcopy(self.train_dataset)
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
                # pdb.set_trace()
                if self.args.mode3_path is not None:
                    self.train_step_domain_expert(train_id_dataloader)
                else:
                    self.train_step(train_id_dataloader)
                break
    
    def train_sample_weights(self, rw_configs = None):
        self.train_dataset = copy.deepcopy(self.org_dataset)
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(self.org_dataset)
        self.eval_dataset.mode = 1
        rw_evals = []
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp <= self.split_time:
                rw_evals.append((timestamp, 1))
            else:
                rw_evals.append((timestamp, 2))
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                # rw_evals.append(timestamp)
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
                # pdb.set_trace()
                if rw_configs is not None:
                    rw_configs.append(rw_evals)
                self.train_step(train_id_dataloader, freeze_norm=True, rw_configs=rw_configs)
                break
        
    
    
    def train_sep_domains(self):
        self.network_base_weights = copy.deepcopy(self.network.state_dict())
        self.sep_record = []
        self.args.defrost()
        org_epochs = self.args.epochs
        self.args.epochs = self.args.epochs_sep
        self.args.freeze()
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == (self.split_time + 1):
                break
            else:
                cur_record = [timestamp,]
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset = copy.deepcopy(self.org_dataset)
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.eval_dataset = copy.deepcopy(self.org_dataset)
                self.eval_dataset.mode = 1
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.network.load_state_dict(self.network_base_weights)
                
                if self.args.optim_sep == "sgd":
                    self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.args.lr_sep, weight_decay=self.args.weight_decay, momentum=0.9, nesterov=True)
                else:
                    self.optimizer = torch.optim.Adam((self.network.parameters()), lr=self.args.lr_sep, weight_decay=self.args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
                self.train_step(train_dataloader, freeze_norm=True, user_iters = self.args.sep_iters, user_lr_decay = self.args.lr_sep_decay)
                cur_record.append(self.init_id_acc * 100.0)

                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                cur_record.append(acc * 100.0)
                print("######################## Sep Train Domain: {}########################".format(timestamp))
                self.logger.info("ID timestamp = {}: \t final model {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                self.evaluate_offline()
                cur_record = cur_record + self.cur_metrics

                cur_ckpt = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{org_epochs}" / f"{timestamp}_ft{self.args.epochs_sep}.ckpt"
                torch.save(self.network.state_dict(), cur_ckpt)
                self.sep_record.append(cur_record)
        # print(self.sep_record)
        for record in self.sep_record:
            self.logger.info("Timestamp: {}  Init ID Acc: {:.3f}  FT ID Acc: {:.3f} OOD Next Acc: {:.3f} OOD Avg Acc: {:.3f}  OOD Worst Acc: {:.3f}".format(*record))
        
        csv_file = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{org_epochs}" / f"sep_records_ft{self.args.epochs_sep}_{self.args.optim_sep}_{self.args.sep_iters}_{self.args.lr_sep}_{self.args.model_select}_{self.args.eval_num}.csv"
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Init ID Acc", "FT ID Acc", "OOD Next Acc", "OOD Avg Acc", "OOD Worst Acc"])
            for record in self.sep_record:
                writer.writerow(record)
        self.logger.info(f"Records saved to {csv_file}")

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()
        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        self.network.train()
        return metric

    def evaluate_stream(self, start):
        self.network.eval()
        metrics = []
        for i in range(start, min(start + self.eval_next_timestamps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.mode = 2
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.mini_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = self.network_evaluation(test_time_dataloader)
            metrics.append(metric * 100.0)

        avg_metric, worst_metric, best_metric = np.mean(metrics), np.min(metrics), np.max(metrics)

        print(
            f'Timestamp = {start - 1}'
            f'\t Average {self.eval_metric}: {avg_metric}'
            f'\t Worst {self.eval_metric}: {worst_metric}'
            f'\t Best {self.eval_metric}: {best_metric}'
            f'\t Performance over all timestamps: {metrics}\n'
        )
        self.network.train()
        return avg_metric, worst_metric, best_metric, metrics

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                self.logger.info("Merged ID test {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        self.cur_metrics = metrics
        if len(metrics) >= 2:
            self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        if (self.args.method in ['agem', 'ewc', 'ft', 'si', 'drain', 'evos', 'tsi', 'stsi','stft']) or self.args.online_switch:
            # pdb.set_trace()
            if self.args.chft:
                print("training along temporal order")
                self.train_online_order()
            else:
                self.train_online()
        else:
            self.train_offline()
        self.evaluate_offline()

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...\n")
        self.train_dataset.mode = 0
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, timestamp in enumerate(self.train_dataset.ENV[:end]):
            if self.args.lisa and i == self.args.lisa_start_time:
                self.lisa = True
            #----------train on the training set of current domain---------
            self.train_dataset.update_current_timestamp(timestamp)
            if self.args.method in ['simclr', 'swav']:
                self.train_dataset.ssl_training = True
            train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader)

            # -------evaluate on the validation set of current domain-------
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(timestamp)
            test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            acc = self.network_evaluation(test_id_dataloader)
            self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

            # -------evaluate on the next K domains-------
            avg_metric, worst_metric, best_metric, all_metrics = self.evaluate_stream(i + 1)
            self.task_accuracies[timestamp] = avg_metric
            self.worst_time_accuracies[timestamp] = worst_metric
            self.best_time_accuracies[timestamp] = best_metric

            self.logger.info("acc of next {} domains: \t {}".format(self.eval_next_timestamps, all_metrics))
            self.logger.info("avg acc of next {} domains  : \t {:.3f}".format(self.eval_next_timestamps, avg_metric))
            self.logger.info("worst acc of next {} domains: \t {:.3f}".format(self.eval_next_timestamps, worst_metric))

        for key, value in self.task_accuracies.items():
             self.logger.info("timestamp {} : avg acc = \t {}".format(key, value))

        for key, value in self.worst_time_accuracies.items():
             self.logger.info("timestamp {} : worst acc = \t {}".format(key, value))

        self.logger.info("\naverage of avg acc list: \t {:.3f}".format(np.array(list(self.task_accuracies.values())).mean()))
        self.logger.info("average of worst acc list: \t {:.3f}".format(np.array(list(self.worst_time_accuracies.values())).mean()))

        import csv
        with open(self.args.log_dir+'/avg_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.task_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
        with open(self.args.log_dir+'/worst_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.worst_time_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        # pdb.set_trace()
        if self.args.eval_fix:
            self.run_eval_fix()
        else:
            self.run_eval_stream()
        runtime = time.time() - start_time
        runtime = runtime / 60 / 60
        self.logger.info(f'Runtime: {runtime:.2f} h\n')
