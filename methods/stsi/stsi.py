import numpy as np
import time
import datetime
import torch.utils.data

from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..utils import prepare_data, forward_pass, MetricLogger
from ..dataloaders import FastDataLoader
import copy
import torch.nn as nn
import pandas as pd
from pathlib import Path
from collections import deque
import pdb
import random
import os

def freeze_normalization_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # print(f"Freezing normalization layer: {name}")
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

def update_table(df, step, rw_ts, id_acc):
    if rw_ts not in df.columns:
        df[rw_ts] = np.nan
    if step not in df.index:
        df.loc[step, :] = np.nan
    df.loc[step, rw_ts] = id_acc
    return df

class STSI(BaseTrainer):
    """
    Synaptic Intelligence

    Original paper:
        @inproceedings{zenke2017continual,
            title={Continual learning through synaptic intelligence},
            author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
            booktitle={International Conference on Machine Learning},
            pages={3987--3995},
            year={2017},
            organization={PMLR}
        }

    Code adapted from https://github.com/GMvandeVen/continual-learning.
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.si_c = args.si_c            #-> hyperparam: how strong to weigh STSI-loss ("regularisation strength")
        self.epsilon = args.epsilon      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

    def __str__(self):
        str_all = f'SI-si_c={self.si_c}-epsilon={self.epsilon}-{self.base_trainer_str}'
        return str_all

    def _device(self):
        return next(self.network.parameters()).device

    def _is_on_cuda(self):
        return next(self.network.parameters()).is_cuda

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

                # Find/calculate new values for quadratic penalty on parameters
                # print(n)
                p_prev = getattr(self.network, '{}_SI_prev_task'.format(n))
                p_current = p.detach().clone()
                p_change = p_current - p_prev
                omega_add = W[n] / (p_change ** 2 + epsilon)
                try:
                    omega = getattr(self.network, '{}_SI_omega'.format(n))
                except AttributeError:
                    omega = p.detach().clone().zero_()
                omega_new = omega + omega_add

                # Store these new values in the model
                self.network.register_buffer('{}_SI_prev_task'.format(n), p_current)
                self.network.register_buffer('{}_SI_omega'.format(n), omega_new)
        # pdb.set_trace()

    def surrogate_loss(self):
        """
        Calculate STSI's surrogate loss.
        """
        try:
            losses = []
            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.network, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.network, '{}_SI_omega'.format(n))
                    # Calculate STSI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)

        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())
    
    def eval_id(self, timestamp, mode = 1):
        
        self.eval_dataset.mode = mode
        self.eval_dataset.update_current_timestamp(timestamp)
        test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                            batch_size=self.mini_batch_size,
                                            num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        id_acc = self.network_evaluation(test_id_dataloader)
        return id_acc

    def train_step(self, dataloader, split_epoch = 1, ts = 0, cl_flag = False):
        # Prepare <dicts> to store running importance estimates and parameter-values before update
        W = {}
        p_old = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()

        self.network.train()
        loss_all = []
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))

        if cl_flag:
            freeze_normalization_layers(self.network)
            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.split_lr}"/ f"{ts}"
            base_path.mkdir(parents=True, exist_ok=True)
            split_table = pd.DataFrame()
            self.logger.info("-------------------start training on timestamp {}-------------------".format(self.split_set.current_time))
            self.logger.info("self.split_set.len = {} x {} = {} samples".format(self.split_set.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.split_set.__len__()))
            stop_iters = split_epoch * (self.split_set.__len__() // self.args.mini_batch_size) - 1
        else:
            self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
            stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        
        for step, (x, y) in enumerate(dataloader):

            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss = loss + self.si_c * self.surrogate_loss()
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if not cl_flag:
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # Update running parameter importance estimates in W
                    for n, p in self.network.named_parameters():
                        if p.requires_grad:
                            # n = "network." + n
                            n = n.replace('.', '__')
                            if p.grad is not None:
                                W[n].add_(-p.grad * (p.detach() - p_old[n]))
                            p_old[n] = p.detach().clone()
                    self.update_omega(W, self.epsilon)
                break
            # -----------------print log infromation------------
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
                
            if step % (max(1, stop_iters // 6)) == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, acc * 100.0))
                if cl_flag:
                    out_acc = self.eval_id(self.split_time+1, mode = 2)
                    # if idd > self.args.freeze_start:
                    freeze_normalization_layers(self.network)
                    split_table = update_table(split_table, step, ts, acc * 100.0)
                    split_table = update_table(split_table, step, self.split_time + 1, out_acc * 100.0)
                    split_table = split_table.round(2)
                    # print("org out acc: ", self.out_acc_org * 100.0)
                    print(split_table)
                    cur_params = self.network.state_dict()
                    param_path = base_path / f"step_{step}.pth"
                    torch.save(cur_params, param_path)
                    
                    split_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.split_lr}"/ f"{ts}.csv")
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))

    def train_online(self):
        # Register starting param-values (needed for "intelligent synapses").
        self.train_dataset.mode = 0
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        self.split_set = copy.deepcopy(self.train_dataset)
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and t == (self.split_time + 1):
                break
            else:
                self.train_dataset.update_current_timestamp(t)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader, i, t)
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(t)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(t, self.eval_metric, acc * 100.0))
        
        assert self.args.mode3_path is not None
        
        self.out_acc_org = self.eval_id(self.split_time+1, mode = 2)
        org_network = copy.deepcopy(self.network)
        for i, t in enumerate(reversed(self.split_set.ENV[:-1])):
            assert self.args.eval_fix
            if self.args.eval_fix and t >= (self.split_time + 1):
                pass
            else:
                print("fine_tuning on timestamp: ", t)
                # self.network = copy.deepcopy(org_network)
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.split_lr, weight_decay=self.args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
                # if self.args.mode3_path is not None:
                #     self.split_set.mode = 3
                # else:
                self.split_set.mode = 3
                self.split_set.update_current_timestamp(t)
                split_dataloader = InfiniteDataLoader(dataset=self.split_set, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                freeze_normalization_layers(self.network)
                self.train_step(split_dataloader, cl_flag = True, split_epoch = self.args.split_epochs, ts = t)
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(t)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(t, self.eval_metric, acc * 100.0))

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...\n")
        self.train_dataset.mode = 0
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps

        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        for i, t in enumerate(self.train_dataset.ENV[:end]):
            self.train_dataset.update_current_timestamp(t)
            train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader, i, t)

            # -------evaluate on the testing set of current domain-------
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(t)
            test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            acc = self.network_evaluation(test_id_dataloader)
            self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(t, self.eval_metric, acc * 100.0))

            # -------evaluate on the next K domains-------
            avg_metric, worst_metric, best_metric, all_metrics = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_metric
            self.worst_time_accuracies[t] = worst_metric
            self.best_time_accuracies[t] = best_metric

            self.logger.info("acc of next {} domains: \t {}".format(self.eval_next_timestamps, all_metrics))
            self.logger.info("avg acc of next {} domains  : \t {:.3f}".format(self.eval_next_timestamps, avg_metric))
            self.logger.info( "worst acc of next {} domains: \t {:.3f}".format(self.eval_next_timestamps, worst_metric))

        for key, value in self.task_accuracies.items():
            self.logger.info("timestamp {} : avg acc = \t {}".format(key + self.args.init_timestamp, value))

        for key, value in self.worst_time_accuracies.items():
            self.logger.info("timestamp {} : worst acc = \t {}".format(key + self.args.init_timestamp, value))

        self.logger.info("\naverage of avg acc list: \t {:.3f}".format(np.array(list(self.task_accuracies.values())).mean()))
        self.logger.info("average of worst acc list: \t {:.3f}".format(np.array(list(self.worst_time_accuracies.values())).mean()))

        import csv
        with open(self.args.log_dir + '/avg_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.task_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
        with open(self.args.log_dir + '/worst_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.worst_time_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
