import torch
import random
import math
import numpy as np
import copy
import pdb

def normalize_probs(probs):
    """将概率列表进行归一化，如果全为0则返回均匀分布"""
    s = sum(probs)
    if s > 1e-12:
        return [p / s for p in probs]
    else:
        # 如果出现全部为0的情况，改用均匀分布
        n = len(probs)
        return [1.0 / n] * n

def distance_weight(c1, c2, alpha):
    """
    距离越大，概率权重越小的函数。
    示例：使用高斯型衰减 e^{-alpha*(|c1 - c2|^2)}。
    你可以根据需求调整为 e^{-alpha*|c1 - c2|} 或其他。
    """
    d = abs(c1 - c2)
    return math.exp(-alpha * (d ** 2))



def weighted_linear_fit(time_points, values, weights):
    """加权线性拟合，返回拟合参数 (a, b)"""
    coeffs = np.polyfit(time_points, values, 1, w=weights)
    b, a = coeffs  # polyfit 返回的是最高次幂系数在前
    return a, b

def run_sampling_and_prediction(
    adw,
    N=10,           # 采样次数
    num_fd=5,       # 预测未来 num_fd 个时间步
    alpha=0.5,      # 距离-概率衰减
    beta=0.9,       # 时间权重衰减
    gamma=0.9       # 先验衰减
):
    """
    主函数：根据给定的 adw 数据结构，针对每个主成分做随机采样、加权拟合和预测。
    返回 all_results，其中包含每次迭代得到的路径和预测结果。
    """
    # 1) 准备先验分布 P_prior[ts][exp_id]
    n_components=len(adw[0][0]['all_param'])
    P_prior_org = {}
    # pdb.set_trace() 
    for ts in range(len(adw)):
        P_prior_org[ts] = {}
        exp_ids = np.arange(len(adw[ts]))
        # 简单初始化为均匀先验
        for exp_id in exp_ids:
            P_prior_org[ts][exp_id] = 1.0 / len(exp_ids)
    
    # 2) 确定时间戳的排序 (假设 ts 是可以排序的数值)
    ts_list = np.arange(len(adw))
    ts_min, ts_max = ts_list[0], ts_list[-1]

    # 结果存储结构: all_results[idc] -> list of iterations -> { 'chosen_path':..., 'future_preds':... }
    all_results = {idc: [] for idc in range(n_components)}

    # 3) 开始对每个主成分，重复 N 次采样 & 拟合
    for idc in range(n_components):
        P_prior = copy.deepcopy(P_prior_org)
        exp_ids_max = np.arange(len(adw[ts_max]))
        probs_max = [P_prior[ts_max][e] for e in exp_ids_max]
        probs_max = normalize_probs(probs_max)
        max_c_val = max([adw[ts_max][exp_id]['all_param'][idc] for exp_id in exp_ids_max])
        for iteration in range(N):
            # pdb.set_trace()
            exp_id_chosen_max = random.choices(exp_ids_max, weights=probs_max, k=1)[0]
            c_ts_max = adw[ts_max][exp_id_chosen_max]['all_param'][idc]

            # 用字典记录本次采样路径： chosen_path[ts] = (exp_id, coeff)
            chosen_path = {ts_max: (exp_id_chosen_max, c_ts_max.item())}
            a = c_ts_max
            b = 0.0
            for i in range(len(ts_list) - 2, -1, -1):
                ts_curr = ts_list[i+1]  # 上一时刻
                ts_next = ts_list[i]    # 更早时刻
                # c_ts_curr = chosen_path[ts_curr][1]
                pred_c_ts_next = a + b * ts_next

                exp_ids_next = np.arange(len(adw[ts_next]))
                raw_probs = []
                for e in exp_ids_next:
                    c_val = adw[ts_next][e]['all_param'][idc]
                    dist_w = distance_weight(c_val/max_c_val, pred_c_ts_next/max_c_val, alpha)
                    # print(c_val/max_c_val, pred_c_ts_next/max_c_val, dist_w)
                    # 结合先验
                    p = dist_w * P_prior[ts_next][e]
                    raw_probs.append(p)

                probs_next = normalize_probs(raw_probs)
                # print(ts_next, probs_next)
                exp_id_chosen_next = random.choices(exp_ids_next, weights=probs_next, k=1)[0]
                c_ts_next = adw[ts_next][exp_id_chosen_next]['all_param'][idc].item()
                chosen_path[ts_next] = (exp_id_chosen_next, c_ts_next)

                time_points = []
                values = []
                weights = []
                for ts0 in sorted(chosen_path.keys()):
                    exp0, c0 = chosen_path[ts0]
                    time_points.append(float(ts0))  # 确保是 float
                    values.append(float(c0))
                    w_ts = (beta ** (ts_max - ts0)) if (ts_max - ts0) >= 0 else 1.0
                    weights.append(w_ts)
                a, b = weighted_linear_fit(time_points, values, weights)

            # ---- (D) 利用拟合结果预测未来时间 (ts_max+1, ..., ts_max+num_fd) ----
            future_preds = {}
            for k in range(1, num_fd + 1):
                ts_future = ts_max + k
                c_pred = a + b * ts_future
                future_preds[ts_future] = float(c_pred)

            # ---- (E) 存储本次采样结果 ----
            all_results[idc].append({
                'chosen_path': chosen_path,
                'fitted_model': (a, b),        # y = a + b*x
                'future_preds': future_preds
            })

            # ---- (F) 更新先验: 对本次被选中的 exp_id 做衰减，再归一化 ----
            for ts0, (exp0, c0) in chosen_path.items():
                P_prior[ts0][exp0] *= gamma

            # 归一化先验
            for ts0 in ts_list:
                sum_p = sum(P_prior[ts0].values())
                if sum_p > 1e-12:
                    for e in P_prior[ts0]:
                        P_prior[ts0][e] /= sum_p
                else:
                    # 若出现全部变 0，则回到均匀分布
                    num_e = len(P_prior[ts0])
                    for e in P_prior[ts0]:
                        P_prior[ts0][e] = 1.0 / num_e
        clean_res = []
        for idc in range(n_components):
            clean_res.append([])
            for idr in range(len(all_results[idc])):
                clean_res[idc].append([])
                chosen_path = all_results[idc][idr]['chosen_path']
                for ts in ts_list:
                    clean_res[idc][idr].append(chosen_path[ts][1])
                for ts in range(ts_max+1, ts_max+num_fd+1):
                    clean_res[idc][idr].append(all_results[idc][idr]['future_preds'][ts])

    return all_results, np.array(clean_res)


    # results = run_sampling_and_prediction(
    #     adw_example,
    #     n_components=2,  # 这里有2个主成分
    #     N=5,             # 采样5次
    #     num_fd=2,        # 预测未来2个时间点
    #     alpha=0.5,
    #     beta=0.9,
    #     gamma=0.9
    # )

    # # 打印结果 (仅做示例)
    # for idc in results:
    #     print(f"\n主成分 idc = {idc}")
    #     for i, r in enumerate(results[idc]):
    #         chosen_path_str = ", ".join([f"ts={ts} -> (exp={r['chosen_path'][ts][0]}, c={r['chosen_path'][ts][1]:.3f})"
    #                                      for ts in sorted(r['chosen_path'].keys())])
    #         a, b = r['fitted_model']
    #         future_preds = r['future_preds']
    #         print(f"  采样第 {i+1} 次:")
    #         print(f"    路径: {chosen_path_str}")
    #         print(f"    拟合线性函数: y = {a:.3f} + {b:.3f} * x")
    #         print(f"    预测未来: {future_preds}")