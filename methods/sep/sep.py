import os
import torch
from pathlib import Path
from ..base_trainer import BaseTrainer, update_table
import copy
import pandas as pd
import pdb
import random
from torch.nn.functional import cosine_similarity
from .wpca import pca_for_domain_experts, visualize_pca, reconstruct_weights_with_avg, reconstruct_weights_from_all_param
from .sample import run_sampling_and_prediction


def compare_model_weights(base_weights, ft_weights, threshold=1e-3):
    results = {}
    for name in base_weights:
        # print(name)
        if name not in ft_weights:
            print(f"Parameter {name} missing in ft_weights.")
            continue
        base_param = base_weights[name]
        if base_param.dtype not in (torch.float32, torch.float64):
            # print("skipping ", name, base_param.dtype)
            continue
        ft_param = ft_weights[name]
        diff = ft_param - base_param
        l1_diff = diff.abs().sum().item()
        l2_diff = diff.pow(2).sum().sqrt().item()
        # results["total_l1_diff"] += l1_diff
        # results["total_l2_diff"] += l2_diff
        base_norm = base_param.norm(2).item()
        relative_diff = l2_diff / base_norm if base_norm > 1e-9 else 0.0
        changed_num = (diff.abs() > threshold).sum().item()
        total_num = diff.numel()
        changed_ratio = (changed_num / total_num) * 100
        base_param = base_weights[name].view(-1)
        ft_param = ft_weights[name].view(-1)

        avg_l2_diff = l2_diff / total_num
        
        # 计算 Cosine Similarity
        if len(base_param) == 1:
            cos_sim = 1.0
        cos_sim = 1-cosine_similarity(base_param.unsqueeze(0), ft_param.unsqueeze(0)).item()
        results[name] = avg_l2_diff
        # results[name] = l2_diff
        # results[name] = cos_sim

    return results

def compute_param_ratio(ts1_weights, key_param):

    if isinstance(key_param, str):
        key_param = [key_param]  # 转为列表处理单个参数名

    # 检查 key_param 是否在 ts1_weights 中
    for key in key_param:
        if key not in ts1_weights:
            raise ValueError(f"Parameter {key} not found in ts1_weights.")
    
    # 计算 key_param 的参数数量
    key_param_count = sum(ts1_weights[key].numel() for key in key_param)
    
    # 计算总参数数量
    total_param_count = sum(param.numel() for param in ts1_weights.values())
    
    # 计算比例
    ratio = (key_param_count / total_param_count) * 100
    
    return ratio


def replace_weight(src_weight, tgt_weight, key_param):
    if isinstance(key_param, str):
        key_param = [key_param]
    for key in key_param:
        if key not in src_weight:
            raise ValueError(f"Parameter {key} not found in src_weight.")
        if key not in tgt_weight:
            raise ValueError(f"Parameter {key} not found in tgt_weight.")

    updated_weight = src_weight.copy()
    for key in key_param:
        updated_weight[key] = tgt_weight[key].clone()
    
    return updated_weight

def get_top_relative_diff(results, num_param=5):
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    if num_param == -1:
        return [item for item in sorted_results if item[1] > 0]
    return sorted_results[:num_param]

from collections import Counter

def count_parameter_frequency(top_change_dict):
    param_counter = Counter()
    
    for key, param_list in top_change_dict.items():
        # 只提取参数名称，统计出现次数
        param_names = [param[0] for param in param_list]
        param_counter.update(param_names)
    
    # 将 Counter 对象转换为字典并按频率排序
    sorted_param_freq = dict(param_counter.most_common())
    
    return sorted_param_freq


class SEP(BaseTrainer):
    """
    Empirical Risk Minimization
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

    def __str__(self):
        if self.args.lisa:
            return f'SEP-LISA-no-domainid-{self.base_trainer_str}'
        elif self.args.mixup:
            return f'SEP-Mixup-no-domainid-{self.base_trainer_str}'
        return f'SEP-{self.base_trainer_str}'
    
    def average_weights(self, weight_ls, slope=0):
        if not weight_ls:
            raise ValueError("weight_ls is empty.")
        # Calculate weights for averaging
        weight_ls.sort(key=lambda x: x[0])

        count = len(weight_ls)
        if count == 1:
            return weight_ls[0][1]  # Only one weight, return it directly

        # Generate weights linearly based on slope
        if slope == 0:
            linear_weights = [1 / count] * count  # Equal weights
        else:
            max_weight = 1
            min_weight = max(0, 1 - slope * (count - 1))
            linear_weights = [min_weight + i * slope for i in range(count)]
            total_weight = sum(linear_weights)
            linear_weights = [w / total_weight for w in linear_weights]  # Normalize to sum to 1
        # print("current weights: ", linear_weights)

        # Initialize averaged weights
        avg_weights = None
        for idx, (_, state_dict) in enumerate(weight_ls):
            weight = linear_weights[idx]
            if avg_weights is None:
                avg_weights = {k: v.clone() * weight for k, v in state_dict.items()}
            else:
                for k in avg_weights.keys():
                    avg_weights[k] += state_dict[k] * weight

        return avg_weights
    
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
    
    def average_experts_by_distance(self, experts, tgt_ts, slope=0, order = 1):

        if not experts:
            raise ValueError("experts 字典为空，无法加权平均。")

        ts_avg_dicts = {}  # { ts: avg_state_dict }
        for ts, acc_weight_ls in experts.items():
            if not acc_weight_ls:
                raise ValueError(f"experts[{ts}] 对应的列表为空。")
            count = len(acc_weight_ls)
            sum_dict = None
            for _, state_dict in acc_weight_ls:
                if sum_dict is None:
                    sum_dict = {k: v.clone() for k, v in state_dict.items()}
                else:
                    for k in sum_dict:
                        sum_dict[k] += state_dict[k]

            # 计算平均
            for k in sum_dict:
                if sum_dict[k].dtype in (torch.float32, torch.float64):
                    sum_dict[k] /= count
                else:
                    # print(sum_dict[k])
                    org_dtype = sum_dict[k].dtype
                    sum_dict[k] = (sum_dict[k].float()/count).to(org_dtype)
                    # pdb.set_trace()

            ts_avg_dicts[ts] = sum_dict

        dist_list = []
        for ts, avg_dict in ts_avg_dicts.items():
            dist = abs(ts - tgt_ts)
            dist_list.append((dist, avg_dict))

        dist_list.sort(key=lambda x: x[0])  # x[0] 是距离

        count = len(dist_list)
        if count == 1:
            return dist_list[0][1]
        if slope == 0:
            linear_weights = [1.0 / count] * count
        else:
            max_weight = 1.0
            min_weight = max(0.0, 1.0 - slope * (count - 1))
            linear_weights = [min_weight + (i * slope)**order for i in range(count)]
            total_weight = sum(linear_weights)
            linear_weights = [w / total_weight for w in linear_weights]
        final_avg = None
        print(tgt_ts, linear_weights)
        for idx, (_, avg_dict) in enumerate(dist_list):
            w = linear_weights[idx]
            if final_avg is None:
                final_avg = {k: v.clone() * w for k, v in avg_dict.items()}
            else:
                for k in final_avg:
                    final_avg[k] += avg_dict[k] * w
        return final_avg
    
    def run_eval_fix(self):
        
        if len(self.args.desp) > 0:
            self.args.defrost()
            self.args.base_dir += '/' + self.args.desp
            self.args.freeze()
        if self.args.sep_avg in ['in', 'out', 'comp']:
            base_path = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"rw_{self.args.mini_batch_size}_{self.args.rw_freq}_{self.args.rw_iters}_{self.args.lr}"
            record_csv = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / f"rw_{self.args.mini_batch_size}_{self.args.rw_freq}_{self.args.rw_iters}_{self.args.lr}.csv"
            record_df = pd.read_csv(record_csv)
            ts_weights = {}
            # pdb.set_trace()
            results = {'rw_ts': [], 'max_acc': [], 'acc_avg': [], 'model_avg': []}
            for rw_ts in record_df.columns[1:]:
                if int(rw_ts) > self.split_time:
                    continue
                ts_data = record_df.iloc[:, [0, int(rw_ts) + 1]]  # 包含第 0 列和当前时间片列
                ts_data.columns = ['step_index', 'performance'] 
                ts_folder = base_path / f"rw_ts/{rw_ts}"
                if not ts_folder.exists():
                    continue
                ts_weights[rw_ts] = []
                for weight_file in ts_folder.glob("*.pt"):
                    step = int(weight_file.stem.split('_')[1])  # 假设文件名格式为 "step_<step>.pt"
                    acc_row = ts_data[ts_data['step_index'] == step]
                    if not acc_row.empty:
                        acc = acc_row['performance'].values[0]  # 假设 performance 列名为 'performance'
                        weights = torch.load(weight_file)
                        ts_weights[rw_ts].append((acc, weights))
                ts_weights[rw_ts].sort(key=lambda x: x[0], reverse=True)
                # random.shuffle(ts_weights[rw_ts])
                ts_weights[rw_ts] = ts_weights[rw_ts][:self.args.num_experts]
            if self.args.sep_avg == 'in':
                for rw_ts in ts_weights:
                    accs = [acc for acc, _ in ts_weights[rw_ts]]
                    avg_weights = self.average_weights(ts_weights[rw_ts], self.args.slope)
                    self.network.load_state_dict(avg_weights)
                    avg_acc = self.eval_id(int(rw_ts), mode=1)
                    # print(f"TS {rw_ts}: AVG ID test {avg_acc}, org max accs: {max(accs)}")
                    results['rw_ts'].append(int(rw_ts))
                    results['max_acc'].append(max(accs) if accs else None)
                    results['acc_avg'].append(sum(accs)/len(accs))
                    results['model_avg'].append(avg_acc*100)
                results_df = pd.DataFrame(results).round(2).T
                print(results_df)
            if self.args.sep_avg == 'out':
                experts = {}
                for rw_ts in ts_weights:
                    experts[int(rw_ts)] = ts_weights[rw_ts]
                new_res_df = record_df.head(1)
                for rw_ts in experts:
                    avg_expert_weights = self.average_experts_by_distance(experts, rw_ts, slope = self.args.slope, order = self.args.order)
                    self.network.load_state_dict(avg_expert_weights)
                    id_acc = self.eval_id(rw_ts, mode = 1)
                    new_res_df = update_table(new_res_df, 'avg', str(rw_ts), id_acc * 100.0)
                    print(new_res_df)
                for rw_ts in record_df.columns[1:]:
                    rw_ts = int(rw_ts)
                    if rw_ts > self.split_time:
                        avg_expert_weights = self.average_experts_by_distance(experts, rw_ts, slope = self.args.slope, order = self.args.order)
                        self.network.load_state_dict(avg_expert_weights)
                        out_acc = self.eval_id(rw_ts, mode = 2)
                        new_res_df = update_table(new_res_df, 'avg', str(rw_ts), out_acc * 100.0)
                        print(new_res_df)
                print(new_res_df)
            if self.args.sep_avg == 'comp':
                comp_res_dict = {}
                top_change_dict = {}
                for ts1 in ts_weights:
                    for ts2 in ts_weights:
                        if ts1 == ts2:
                            continue
                        tw1 = ts_weights[ts1][0]
                        tw2 = ts_weights[ts2][0]
                        comp_res_dict[(ts1, ts2)] = compare_model_weights(tw1[1], tw2[1])
                        top_change_dict[(int(ts1), int(ts2))] = get_top_relative_diff(comp_res_dict[(ts1, ts2)], num_param = -1)
                param_freq = count_parameter_frequency(top_change_dict)
                key_param = [key for key in param_freq if param_freq[key] > len(top_change_dict) * 0.0]
                print(len(key_param),len(tw2[1]), 'ratio:', compute_param_ratio(tw2[1], key_param))

                kp_weights = {}
                for ts in ts_weights:
                    kp_weights[int(ts)] = []
                    for acc, weights in ts_weights[ts]:
                        kp_weights[int(ts)].append(weights)
                base_ckpt = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / "base.ckpt"
                base_weight = torch.load(base_ckpt)
                avg_weights, pca_results, adw = pca_for_domain_experts(kp_weights, n_components = 30, key_list = key_param, base_weight = base_weight)
                print(key_param.pop(-1))
                predict_pca_res, clean_res = run_sampling_and_prediction(adw, N=10, num_fd=3, alpha=0.5, beta=1.0,gamma=0.1)
                temp_weights = ts_weights['0'][0][1]
                temp_weights.pop("all_param")
                sample_weights = []
                for ts1 in ts_weights:
                    # print(len(ts_weights[ts1]))
                    ts1_weights = ts_weights[ts1][0][1]
                    if "all_param" in ts1_weights:
                        ts1_weights.pop("all_param")
                    sample_weights.append(ts1_weights)
                cur_avg_weight = self.simple_average_weight(sample_weights)
                self.network.load_state_dict(cur_avg_weight)
                acc1_recon_all_avg = self.eval_id(self.split_time+1, mode=2)
                print("start", self.split_time+1, 'recon all avg', acc1_recon_all_avg)
                sample_weights = [cur_avg_weight,]
                for iter in range(10):
                    random_numbers = random.choices(range(len(clean_res[0])), k=len(clean_res))
                    components_coeffs = []
                    for idc in range(len(clean_res)):
                        idd = random_numbers[idc]
                        components_coeffs.append(clean_res[idc][idd][self.split_time+1])
                    components_coeffs = torch.tensor(components_coeffs, dtype=torch.float32).to(adw[0][0]['all_param'].device)
                    
                    recon_weight_all = reconstruct_weights_from_all_param(pca_results, components_coeffs, key_param, avg_weights, temp_weights)
                    sample_weights.append(recon_weight_all)
                    self.network.load_state_dict(recon_weight_all)
                    acc1_recon_all = self.eval_id(self.split_time+1, mode=2)
                    cur_avg_weight = self.simple_average_weight(sample_weights)
                    self.network.load_state_dict(cur_avg_weight)
                    acc1_recon_all_avg = self.eval_id(self.split_time+1, mode=2)
                    print(iter, self.split_time+1, "recon all", acc1_recon_all, 'recon all avg', acc1_recon_all_avg)
                for ts1 in ts_weights:
                    print(len(ts_weights[ts1]))
                    ts1_weights = ts_weights[ts1][0][1]
                    if "all_param" in ts1_weights:
                        ts1_weights.pop("all_param")
                    self.network.load_state_dict(ts1_weights)
                    acc1_org = self.eval_id(self.split_time+1, mode=2)
                    print("select best", ts1, "original acc:", acc1_org)
                pdb.set_trace()
                for ts1 in ts_weights:
                    print(len(ts_weights[ts1]))
                    ts1_weights = ts_weights[ts1][0][1]
                    ts1_weights.pop("all_param")
                    components_coeffs = adw[int(ts1)][1]
                    recon_weight = reconstruct_weights_with_avg(pca_results, components_coeffs, key_param, avg_weights, ts1_weights)
                    recon_weight_all = reconstruct_weights_from_all_param(pca_results, components_coeffs["all_param"], key_param, avg_weights, ts1_weights)
                    self.network.load_state_dict(ts1_weights)
                    acc1_org = self.eval_id(int(ts1), mode=1)
                    acc1_org_out = self.eval_id(self.split_time + 1, mode=2)
                    self.network.load_state_dict(recon_weight)
                    acc1_recon = self.eval_id(int(ts1), mode=1)
                    acc1_recom_out = self.eval_id(self.split_time + 1, mode=2)
                    self.network.load_state_dict(recon_weight_all)
                    acc1_recon_all = self.eval_id(int(ts1), mode=1)
                    acc1_recom_out_all = self.eval_id(self.split_time + 1, mode=2)
                    print(ts1, "original:", acc1_org, "recon:", acc1_recon, 'recon all', acc1_recon_all)
                    print(self.split_time + 1, "original:", acc1_org_out, "recon:", acc1_recom_out, 'recon all', acc1_recom_out_all)
                pdb.set_trace()
        else:
            base_ckpt = Path(self.args.base_dir) / f"{self.args.method}_{self.args.dataset}_{self.epochs}" / "base.ckpt"

            # if base_ckpt.exists():
            #     # 如果 checkpoint 存在，加载到网络并运行 evaluate_offline
            #     print(f"Checkpoint found at {base_ckpt}. Loading network...")
            #     self.network.load_state_dict(torch.load(base_ckpt))  # 加载模型权重
            #     # self.evaluate_offline()  # 离线评估
            # else:
            # 如果 checkpoint 不存在，运行父类方法
            print("Checkpoint not found. Running super().run_eval_fix()...")
            super().run_eval_fix()

            # 确保保存路径的上级文件夹存在
            base_ckpt.parent.mkdir(parents=True, exist_ok=True)

            # 保存网络到 checkpoint
            print(f"Saving network to {base_ckpt}...")
            torch.save(self.network.state_dict(), base_ckpt)
            # rw_len, rw_freq, rw_iters = rw_configs
            if self.args.rw_len > 0:
                rw_configs = [self.args.rw_len, self.args.rw_freq, self.args.rw_iters]
                self.train_sample_weights(rw_configs)

        # 继续训练领域分离
        # self.train_sep_domains()