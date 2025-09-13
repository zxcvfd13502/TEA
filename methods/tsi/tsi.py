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
from .pca_utils import check_model_weight_difference, pca_for_domain_experts, visualize_pca, reconstruct_weights_with_avg, reconstruct_weights_from_all_param, get_weights_with_dists,transport_weights
from .pca_estimate import compute_avg_vectors, fit_and_forecast_varmax, plot_var_results, fit_and_forecast_var, fit_and_forecast_univariate_arima, fit_and_forecast_univariate_auto_arima, fit_and_forecast_univariate_linreg

def freeze_normalization_layers(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LayerNorm, nn.GroupNorm)):
            # print(f"Freezing normalization layer: {name}")
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

def filter_weights_by_network(weight, network):
    """
    过滤 weight 字典，去掉那些不在 network.state_dict() 中的 key-value。

    :param weight: dict, 预训练模型的权重字典
    :param network: torch.nn.Module, 目标网络
    :return: dict, 经过筛选后的权重字典
    """
    network_keys = set(network.state_dict().keys())  # 获取网络的 key
    filtered_weight = {k: v for k, v in weight.items() if k in network_keys}
    return filtered_weight

def update_table(df, step, rw_ts, id_acc):
    if rw_ts not in df.columns:
        df[rw_ts] = np.nan
    if step not in df.index:
        df.loc[step, :] = np.nan
    df.loc[step, rw_ts] = id_acc
    return df

def get_each_coeffs(adw):
    all_coffs = []
    for experiments in adw:
        all_params = [exp['all_param'] for exp in experiments]
        all_coffs.append(all_params)
    return np.array(all_coffs)

def load_latest_checkpoint(base_path):
    # 转换为Path对象便于处理
    path = Path(base_path)
    
    # 检查文件夹是否存在
    if not path.exists():
        return None
    
    # 检查文件夹是否为空
    checkpoint_files = list(path.glob("step_*.pth"))
    if not checkpoint_files:
        return None
    
    # 提取所有的step数并找到最大的
    steps = []
    for file in checkpoint_files:
        try:
            # 从文件名中提取step数
            step = int(file.stem.split('_')[1])
            steps.append((step, file))
        except (ValueError, IndexError):
            continue
    
    if not steps:
        return None
    
    # 获取最大step的文件路径
    max_step, latest_checkpoint = max(steps, key=lambda x: x[0])
    
    # 加载checkpoint
    try:
        checkpoint = torch.load(latest_checkpoint)
        print(f"成功加载checkpoint: {latest_checkpoint}")
        return checkpoint
    except Exception as e:
        raise RuntimeError(f"加载checkpoint失败: {e}")

class TSI(BaseTrainer):
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
        self.si_c = args.si_c            #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = args.epsilon      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

    def __str__(self):
        str_all = f'TSI-si_c={self.si_c}-epsilon={self.epsilon}-{self.base_trainer_str}'
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
    
    def dist_average_weight(self, weight_dist_ls, idt, alpha=1.0, reverse=False):
        """
        基于距离加权平均的权重聚合方法，使用minmax标准化
        
        参数:
        weight_dist_ls: 权重和距离元组的列表，每个元组包含(权重字典, 距离列表)
        idt: 测试样本索引，用于从每个距离列表中选择对应的距离
        alpha: 控制权重分布的指数参数，较大的值使分布更锐利
        
        返回:
        加权平均后的权重字典
        """
        # 提取每个权重对应的距离
        weights = [item[0] for item in weight_dist_ls]
        distances = [item[1][idt] for item in weight_dist_ls]
        
        # 如果所有距离都相同，则使用均匀权重
        if len(set(distances)) == 1:
            count = len(weights)
            uniform_weight = 1.0 / count
            weights_normalized = [uniform_weight] * count
        else:
            # MinMax标准化距离: (dist - min_dist) / (max_dist - min_dist)
            min_dist = min(distances)
            max_dist = max(distances)
            normalized_distances = [(d - min_dist) / (max_dist - min_dist) for d in distances]
            
            # 反转标准化距离，使距离越小权重越大: 1 - norm_dist
            inverted_distances = [1 - nd for nd in normalized_distances]
            
            # 应用指数参数来控制权重分布的锐度
            if alpha != 1.0:
                powered_distances = [d ** alpha for d in inverted_distances]
            else:
                powered_distances = inverted_distances
            
            # 标准化权重使其总和为1
            sum_distances = sum(powered_distances)
            weights_normalized = [d / sum_distances for d in powered_distances]
        
            weights_array = np.array(weights_normalized)

            # Step 1: 找到 25% 分位数的阈值
            threshold = np.percentile(weights_array, self.args.holdout_ratio * 100)  # bottom 25%
            print("holding out", self.args.holdout_ratio * 100)

            # Step 2: 小于或等于这个阈值的都设为 0
            weights_array = np.where(weights_array <= threshold, 0.0, weights_array)

            # Step 3: 重新归一化剩下的非零部分
            total = weights_array.sum()
            if total > 0:
                weights_normalized = weights_array / total
            else:
                raise ValueError("All weights are zero after thresholding. Can't normalize.")
        # 初始化加权平均的权重字典
        avg_weights = None
        if reverse:
            print("reversing!!!!")
            weights.reverse()
        
        # 根据计算出的权重进行加权平均
        # pdb.set_trace()
        for idx, state_dict in enumerate(weights):
            weight = weights_normalized[idx]
            if avg_weights is None:
                avg_weights = {k: v.clone() * weight for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * weight
        
        return avg_weights, weights_normalized
    
    def trans_dist_average_weight(self, weight_dist_ls, idt, alpha=1.0):
        """
        基于距离加权平均的权重聚合方法，使用minmax标准化
        
        参数:
        weight_dist_ls: 权重和距离元组的列表，每个元组包含(权重字典, 距离列表)
        idt: 测试样本索引，用于从每个距离列表中选择对应的距离
        alpha: 控制权重分布的指数参数，较大的值使分布更锐利
        
        返回:
        加权平均后的权重字典
        """
        # 提取每个权重对应的距离
        weights = [item[0][idt] for item in weight_dist_ls]
        distances = [item[1][idt] for item in weight_dist_ls]
        
        # 如果所有距离都相同，则使用均匀权重
        if len(set(distances)) == 1:
            count = len(weights)
            uniform_weight = 1.0 / count
            weights_normalized = [uniform_weight] * count
        else:
            # MinMax标准化距离: (dist - min_dist) / (max_dist - min_dist)
            min_dist = min(distances)
            max_dist = max(distances)
            normalized_distances = [(d - min_dist) / (max_dist - min_dist) for d in distances]
            
            # 反转标准化距离，使距离越小权重越大: 1 - norm_dist
            inverted_distances = [1 - nd for nd in normalized_distances]
            
            # 应用指数参数来控制权重分布的锐度
            if alpha != 1.0:
                powered_distances = [d ** alpha for d in inverted_distances]
            else:
                powered_distances = inverted_distances
            
            # 标准化权重使其总和为1
            sum_distances = sum(powered_distances)
            weights_normalized = [d / sum_distances for d in powered_distances]

            # weights_array = np.array(weights_normalized)

            # # Step 1: 找到 25% 分位数的阈值
            # threshold = np.percentile(weights_array, self.args.holdout_ratio * 100)  # bottom 25%

            # # Step 2: 小于或等于这个阈值的都设为 0
            # weights_array = np.where(weights_array <= threshold, 0.0, weights_array)

            # # Step 3: 重新归一化剩下的非零部分
            # total = weights_array.sum()
            # if total > 0:
            #     weights_normalized = weights_array / total
            # else:
            #     raise ValueError("All weights are zero after thresholding. Can't normalize.")
        
        # 初始化加权平均的权重字典
        avg_weights = None
        
        # 根据计算出的权重进行加权平均
        # pdb.set_trace()
        for idx, state_dict in enumerate(weights):
            weight = weights_normalized[idx]
            if avg_weights is None:
                avg_weights = {k: v.clone() * weight for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * weight
        
        return avg_weights, weights_normalized
    
    def surrogate_loss(self):
        """
        Calculate SI's surrogate loss.
        """
        try:
            losses = []
            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.network, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.network, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)

        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

    def train_step(self, dataloader, cl_flag=False, split_epoch = 1, ts = 0):
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
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        
        
        
        if cl_flag:
            freeze_normalization_layers(self.network)
            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.split_lr}"/ f"{ts}"
            base_path.mkdir(parents=True, exist_ok=True)
            split_table = pd.DataFrame()
            self.logger.info("-------------------start training on timestamp {}-------------------".format(self.split_set.current_time))
            self.logger.info("self.split_set.len = {} x {} = {} samples".format(self.split_set.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.split_set.__len__()))
            stop_iters = split_epoch * (self.split_set.__len__() // self.args.mini_batch_size) - 1
        else:
            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.split_lr}"/ "base"
            base_path.mkdir(parents=True, exist_ok=True)
            self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
            self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
            stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1

        # if not cl_flag:
        #     trail_queue = deque(maxlen=self.args.split_trail_num)
        #     prior_last_ckpt = load_latest_checkpoint(base_path)
        #     if prior_last_ckpt is not None:
        #         self.network.load_state_dict(prior_last_ckpt)
        #         self.logger.info(f"Loaded checkpoint from {base_path}")
        #         self.
        start_of_last_5_percent = int(0.95 * stop_iters)
        last_5_percent_steps = range(start_of_last_5_percent, stop_iters)

        # 随机选择其中的一部分 steps 来放入队列
        num_checkpoints = int(len(last_5_percent_steps) * 0.01)  # 假设只放入 5% 的 step
        chosen_steps = set(random.sample(last_5_percent_steps, num_checkpoints))  # 确保均等概率选择
        
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            if cl_flag:
                loss = loss + self.si_c * self.surrogate_loss()
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step in chosen_steps and not cl_flag:
                # org_params = copy.deepcopy(self.network.state_dict())
                # for n, p in self.network.named_parameters():
                #     if p.requires_grad:
                #         # n = "network." + n
                #         n = n.replace('.', '__')
                #         if p.grad is not None:
                #             W[n].add_(-p.grad * (p.detach() - p_old[n]))
                #         p_old[n] = p.detach().clone()
                # self.update_omega(W, self.epsilon)
                cur_params = self.network.state_dict()
                param_path = base_path / f"step_{step}.pth"
                torch.save(cur_params, param_path)
                # self.network.load_state_dict(org_params)
                # W = {}
                print(f"Added checkpoint for step {step} to trail_queue.")
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
                    freeze_normalization_layers(self.network)
                    split_table = update_table(split_table, step, ts, acc * 100.0)
                    split_table = update_table(split_table, step, self.split_time + 1, out_acc * 100.0)
                    split_table = split_table.round(2)
                    print("org out acc: ", self.out_acc_org * 100.0)
                    print(split_table)
                    cur_params = self.network.state_dict()
                    param_path = base_path / f"step_{step}.pth"
                    torch.save(cur_params, param_path)
                    split_table.to_csv(Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"{self.args.mini_batch_size}_{split_epoch}_{self.args.split_lr}"/f"{ts}.csv")

        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))

    def train_online_order(self):
        # Register starting param-values (needed for "intelligent synapses").

        if len(self.args.desp) > 0:
            self.args.defrost()
            self.args.base_dir += '/' + self.args.desp
            self.args.freeze()
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        # if self.args.mode3_path is not None:
        print("using the new train online function")
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
                # pdb.set_trace() 
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                # pdb.set_trace()
                self.train_step(train_id_dataloader)
            else:
                break
        self.out_acc_org = self.eval_id(self.split_time+1, mode = 2)
        org_network = copy.deepcopy(self.network)

        for i, t in enumerate(self.split_set.ENV[:-1]):
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
                self.split_set.mode = 0
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
    def train_online(self):
        # Register starting param-values (needed for "intelligent synapses").

        if len(self.args.desp) > 0:
            self.args.defrost()
            self.args.base_dir += '/' + self.args.desp
            self.args.freeze()
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        # if self.args.mode3_path is not None:
        print("using the new train online function")
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
                # pdb.set_trace() 
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                # pdb.set_trace()
                self.train_step(train_id_dataloader)
            else:
                break
        self.out_acc_org = self.eval_id(self.split_time+1, mode = 2)
        org_network = copy.deepcopy(self.network)

        selected_domains = self.select_partial_domains()
        # pdb.set_trace()
        for i, t in enumerate(reversed(selected_domains)):
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
                self.split_set.mode = 0
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
    def train_online_or(self):
        # Register starting param-values (needed for "intelligent synapses").

        if len(self.args.desp) > 0:
            self.args.defrost()
            self.args.base_dir += '/' + self.args.desp
            self.args.freeze()
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        # if self.args.mode3_path is not None:
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
                # pdb.set_trace() 
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                # pdb.set_trace()
                self.train_step(train_id_dataloader)
            else:
                break
        self.out_acc_org = self.eval_id(self.split_time+1, mode = 2)
        org_network = copy.deepcopy(self.network)

        for i, t in enumerate(self.split_set.ENV[:-1]):
            if self.args.eval_fix and t == (self.split_time + 1):
                break
            else:
                self.network = copy.deepcopy(org_network)
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.split_lr, weight_decay=self.args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
                if self.args.mode3_path is not None:
                    self.split_set.mode = 3
                else:
                    self.split_set.mode = 0
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
            self.train_step(train_dataloader)

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
    def get_avg_starts(self):
        ENV = self.eval_dataset.ENV
        split_time = self.split_time
        split_idx = np.searchsorted(ENV, split_time, side='right')
        if split_idx == 0:
            raise ValueError("split_time is smaller than all timestamps in ENV")
        t_start = ENV[0]
        t_end = ENV[split_idx]
        duration = t_end - t_start
        ratios = [0, 1/3, 1/2, 2/3, 3/4]
        start_times = [t_start + r * duration for r in ratios]

        start_indices = []
        for t in start_times:
            idx = np.searchsorted(ENV[:split_idx], t, side='right') - 1
            start_indices.append(max(idx, 0))
        start_ts_list = [ENV[idx] for idx in start_indices]
        return start_ts_list

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        eval_res = {}
        if self.args.eval_pca_path is not None:
            weight_pca_dict = self.load_pca_weights()
            # pdb.set_trace()
            domain_ids = sorted(list(weight_pca_dict.keys()))
            if self.args.eval_pe is not None:
                print("use partial experts!")

                num_total = len(domain_ids)
                num_select = max(1, int(round(num_total * self.args.eval_pe_ratio)))  # 至少选一个

                if self.args.eval_pe == "last":
                    selected_domains = domain_ids[-num_select:]

                elif self.args.eval_pe == "uniform":
                    import random
                    selected_domains = random.sample(domain_ids, num_select)

                else:
                    raise ValueError(f"Unknown eval_pe type: {self.args.eval_pe}")

                # 只保留被选中的 domain 的权重
                weight_pca_dict = {
                    k: v for k, v in weight_pca_dict.items() if k in selected_domains
                }
                domain_ids = sorted(list(weight_pca_dict.keys()))
            print("left partial experts!", domain_ids)
            if self.args.all_experts_eval_path is not None:
                results = []
                for ts in weight_pca_dict:
                    for depth in range(len(weight_pca_dict[ts])):
                        self.network.load_state_dict(weight_pca_dict[ts][depth][1])
                        acc_values = {}  # Dictionary to store accuracy values for each eval timestamp
                        for cur_eval_ts in self.eval_dataset.ENV:
                            if cur_eval_ts > self.split_time:
                                cur_acc_expert = self.eval_id(cur_eval_ts, mode=2)
                                acc_values[cur_eval_ts] = cur_acc_expert
                        row = {'ts': ts, 'depth': depth}
                        row.update(acc_values)  # Add all accuracy values with eval_ts as keys
                        results.append(row)
                results_df = pd.DataFrame(results)
                results_df.to_csv(self.args.all_experts_eval_path, index=False)
                return
            # for depth in range(len(weight_pca_dict[0])):
            avg_starts = self.get_avg_starts()
            # for start_ts in avg_starts:
            #     for depth in [0,-1]:
            #         try:
            #             weight_depth_ls = [weight_pca_dict[ts][depth][1] for ts in range(start_ts, self.split_time+1)]
            #         except:
            #             pdb.set_trace()
            #         cur_avg_weight = self.simple_average_weight(weight_depth_ls)
            #         self.network.load_state_dict(cur_avg_weight)
            #         acc1_recon_all_avg = self.eval_id(self.split_time+1, mode=2)
            #         print("start_ts",start_ts, "depth:",depth,"start", self.split_time+1, 'recon all avg', acc1_recon_all_avg)
            # for ts in weight_pca_dict:
            #     cur_weight_ls = [weight_pca_dict[ts][depth][1] for depth in range(1,6)]
            #     cur_avg_weight = self.simple_average_weight(cur_weight_ls)
            #     self.network.load_state_dict(cur_avg_weight)
            #     acc1_recon_all_avg = self.eval_id(self.split_time+1, mode=2)
            #     print("single ts:",ts, 'recon all avg', acc1_recon_all_avg)
            try:
                key_params_all = check_model_weight_difference(weight_pca_dict[self.eval_dataset.ENV[0]][-1][1], weight_pca_dict[domain_ids[-1]][-1][1])
            except:
                # pdb.set_trace()
                first_ts = sorted(list(weight_pca_dict.keys()))[0]
                key_params_all = check_model_weight_difference(weight_pca_dict[first_ts][-1][1], weight_pca_dict[domain_ids[-1]][-1][1])
            kp_weights = {}
            for ts in weight_pca_dict:
                kp_weights[ts] = []
                for acc, weights in weight_pca_dict[ts]:
                    kp_weights[ts].append(weights)
            # base_weight = weight_pca_dict[self.split_time][0][1]
            base_weight = None
            print("doing PCA")
            avg_weights, pca_results, adw = pca_for_domain_experts(kp_weights, n_components = 10, key_list = key_params_all, base_weight = base_weight)
            key_params = ['all_param',]
            # pdb.set_trace()
            print("doing PCA visualization")
            visualize_pca(adw, key_params, self.args.eval_pca_path, smooth_window=None, domain_ids = domain_ids)
            print("doing PCA fitting and estimation")
            mean_coff_vec = compute_avg_vectors(adw)
            # try:
            # (data: np.ndarray, t: int, order=(1,1,1))
            # fitted, forecast = fit_and_forecast_univariate_auto_arima(mean_coff_vec, t=10)
            # fitted, forecast = fit_and_forecast_var(mean_coff_vec, t=5)
            # fitted, forecast = fit_and_forecast_univariate_linreg(mean_coff_vec, 5)
            fitted, forecast = fit_and_forecast_univariate_arima(mean_coff_vec, t=10, order=(1,1,1), domain_ids = domain_ids)
            # except:
            #     pdb.set_trace()
            plot_var_results(mean_coff_vec, fitted, forecast, base_path=self.args.eval_pca_path)
            print(key_params.pop(-1))
            key_params_all.pop(-1)
            
            all_weights = []
            for ts in weight_pca_dict:
                if self.args.avg_last_ts and ts != self.split_time:
                    print("only using the last ts weights, skipping", ts)
                    continue
                for depth in range(len(weight_pca_dict[ts])):
                    all_weights.append(weight_pca_dict[ts][depth][1])
            all_avg_weight = self.simple_average_weight(all_weights)
            all_avg_weight.pop("all_param")
            
            # all_ckpt_coffs = get_each_coeffs(adw)
            weight_dist_ls, weight_ts = get_weights_with_dists(forecast, adw, weight_pca_dict, self.args.avg_last_ts)
            # try:
            #     trans_weight_dist_ls = transport_weights(forecast, adw, weight_pca_dict, pca_results, weight_pca_dict[self.eval_dataset.ENV[0]][-1][1], key_params_all, 0.5)
            # except:
            #     first_ts = sorted(list(weight_pca_dict.keys()))[0]
            #     trans_weight_dist_ls = transport_weights(forecast, adw, weight_pca_dict, pca_results, weight_pca_dict[first_ts][-1][1], key_params_all, 0.5)
            alphas = [0.5, 1.0, 5.0, 15]
            if self.args.tgt_alpha is not None:
                alphas = [self.args.tgt_alpha,]
            for cur_eval_ts in self.eval_dataset.ENV:
                if cur_eval_ts > self.split_time:
                    
                    # self.network.load_state_dict(all_avg_weight)
                    # acc1_recon_all_avg = self.eval_id(cur_eval_ts, mode=2)
                    # print("avg all ts","eval on", cur_eval_ts, 'recon all avg', acc1_recon_all_avg)

                    wid = cur_eval_ts - self.split_time - 1
                    for alpha in alphas:
                        if alpha not in eval_res:
                            eval_res[alpha] = []
                        davg_weight, davgw_normalized = self.dist_average_weight(weight_dist_ls, wid, alpha=alpha, reverse = self.args.reverse_avg)
                        davg_weight.pop("all_param")
                        self.network.load_state_dict(davg_weight)
                        acc1_davg = self.eval_id(cur_eval_ts, mode=2)
                        # print(davgw_normalized)
                        cur_ts_weight = {}
                        for idw in range(len(davgw_normalized)):
                            cwts = weight_ts[idw]
                            if cwts not in cur_ts_weight:
                                cur_ts_weight[cwts] = davgw_normalized[idw]
                            else:
                                cur_ts_weight[cwts] += davgw_normalized[idw]
                        print(cur_ts_weight)
                        print("davg avg","eval on", cur_eval_ts, "alpha:", alpha,  'recon all avg', acc1_davg)
                        eval_res[alpha].append(acc1_davg)

                    # tavg_weight, tavgw_normalized = self.trans_dist_average_weight(trans_weight_dist_ls, wid, alpha=0)
                    # tavg_weight.pop("all_param")
                    # self.network.load_state_dict(tavg_weight)
                    # acc1_tavg = self.eval_id(cur_eval_ts, mode=2)
                    # print("tavg avg","eval on", cur_eval_ts, "transport_ratio", 0.5, 'recon all avg', acc1_tavg)

                # tdavg_weight, tdavgw_normalized = self.trans_dist_average_weight(trans_weight_dist_ls, 0, alpha=1)
                # tdavg_weight.pop("all_param")
                # self.network.load_state_dict(tdavg_weight)
                # acc1_tdavg = self.eval_id(self.split_time + 1, mode=2)
                # print("transport_ratio", transport_ratio, "tdavg avg","eval on", self.split_time + 1, 'recon all avg', acc1_tdavg)
            for alpha in alphas:
                print("all res:", alpha, eval_res[alpha], "T+1:",eval_res[alpha][0],"avg:", np.mean(eval_res[alpha]),"worst:", np.min(eval_res[alpha]))
            pdb.set_trace()
            return
        if (self.args.method in ['agem', 'ewc', 'ft', 'si', 'drain', 'evos', 'tsi']) or self.args.online_switch:
            # pdb.set_trace()
            if self.args.chft:
                print("training along temporal order")
                self.train_online_order()
            else:
                self.train_online()
        else:
            self.train_offline()
        self.evaluate_offline()
    
    def load_pca_weights(self):
        weight_dict = {}

        for ts in range(self.eval_dataset.ENV[0], self.split_time + 1):  # 遍历 ts 文件夹
            ts_path = os.path.join(self.args.eval_pca_path, str(ts))
            if not os.path.isdir(ts_path):
                print("Path", ts_path, "does not exist.")
                continue  # 跳过不存在的文件夹
            
            weights = []
            for file in os.listdir(ts_path):
                if file.endswith(".pth"):  # 只处理 .pth 文件
                    step = int(file.split(".")[0][5:])  # 提取 step
                    weight_path = os.path.join(ts_path, file)
                    # if step == 0:
                    #     print("skipping the step 0 weights", file)
                    #     continue
                    try:
                        weight = torch.load(weight_path, map_location="cpu")  # 加载权重
                        weight = filter_weights_by_network(weight, self.network)
                        weights.append((step, weight))  # 存入列表
                    except Exception as e:
                        print(f"Error loading {weight_path}: {e}")
            
            weights.sort(key=lambda x: x[0])  # 按 step 排序
            if len(weights) > 5:
                weights = weights[1:6]
            weight_dict[ts] = weights  # 存入字典
        if self.args.only_last:
            # pdb.set_trace()
            cur_ts = self.eval_dataset.ENV[0]
            next_weight = weight_dict[cur_ts][0:1]
            weight_dict[cur_ts] = weight_dict[cur_ts][-1:]
            print("for ts", cur_ts, "keeping", weight_dict[cur_ts][0][0])
            for cur_ts in range(self.eval_dataset.ENV[1], self.split_time + 1):
                temp_weight = weight_dict[cur_ts][0:1]
                weight_dict[cur_ts] = next_weight
                next_weight = temp_weight
                print("for ts", cur_ts, "keeping", weight_dict[cur_ts][0][0])
        return weight_dict
    
