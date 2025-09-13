import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pdb

def moving_average(data, window_size):
    """
    计算移动平均值，仅返回平滑有效的部分。
    
    参数：
        data (list): 原始数据。
        window_size (int): 移动平均窗口大小。
    
    返回：
        tuple: 平滑后的数据和对应的有效索引。
    """
    if window_size < 2:
        return data, list(range(len(data)))
    averaged_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    valid_indices = list(range(window_size - 1, len(data)))  # 有效的横坐标索引
    return averaged_data, valid_indices


def pca_for_domain_experts(weight_ls, n_components=2, key_list=None, base_weight = None):
    """
    对多领域专家模型的权重执行 PCA，并将压缩后的系数存回每个子列表。

    参数：
        weight_ls (list of list of dict): 外层列表为领域，内层列表为每个领域的专家权重。
        n_components (int): PCA 的主成分数。
        key_list (list of str or None): 要执行 PCA 的参数名列表。如果为 None，则选择所有参数。

    返回：
        tuple:
            avg_weights (dict): 所有领域的平均权重。
            pca_results (dict): 每个参数的 PCA 结果，包括压缩系数。
    """
    def compute_avg_weights_for_domains(weight_ls, keys):
        """
        计算每个领域的平均权重，并将其插入到每个领域的专家列表的最前面。
        """
        for domain, experts in weight_ls.items():
            domain_avg = {key: torch.mean(torch.stack([expert[key] for expert in experts]), dim=0) for key in keys}
            weight_ls[domain].insert(0, domain_avg)
        return weight_ls
    
    def compute_avg_weights(weight_ls, keys):
        """
        计算所有领域的平均权重。
        """
        avg_weights = {key: 0 for key in keys}
        total_count = 0
        for domain_idx, domain in weight_ls.items():
            for expert in domain:
                for key in keys:
                    avg_weights[key] += expert[key]
                total_count += 1
        for key in keys:
            avg_weights[key] /= total_count
        return avg_weights

    def compute_delta_weights(domain, avg_weights, keys):
        """
        计算 delta_w = w - avg_w。
        """
        delta_weights = []
        for expert in domain:
            delta = {key: expert[key] - avg_weights[key] for key in keys}
            delta_weights.append(delta)
        return delta_weights

    def extract_parameters(delta_weights, keys):
        """
        提取所有 delta_w 并展平为矩阵，每一行是一个专家的展平 delta 参数。
        """
        param_matrices = {key: [] for key in keys}
        for delta in delta_weights:
            for key in keys:
                param_matrices[key].append(delta[key].flatten())
        for key in keys:
            param_matrices[key] = torch.stack(param_matrices[key])
        return param_matrices
    
    def merge_params_into_all_param(weight_ls, key_list):
        for ts, experts in weight_ls.items():
            for expert in experts:
                # 拼接 key_list 中所有参数为一个 1D 向量
                all_param = torch.cat([expert[key].flatten() for key in key_list])
                expert["all_param"] = all_param
        key_list.append("all_param")
        return weight_ls, key_list

    def perform_pca(param_matrix, n_components):
        """
        对参数矩阵执行 PCA 降维。
        """
        # 计算均值并中心化
        mean = param_matrix.mean(dim=0)
        centered_matrix = param_matrix - mean

        # 使用 SVD 进行 PCA
        U, S, V = torch.svd(centered_matrix)
        # pdb.set_trace()

        # 选取前 n_components 个主成分
        principal_components = V[:, :n_components]
        reduced = torch.mm(centered_matrix, principal_components)
        return reduced, principal_components, mean

    # 获取所有参数的 key
    # pdb.set_trace()
    if key_list is None:
        key_list = weight_ls[0][0].keys()

    weight_ls, key_list = merge_params_into_all_param(weight_ls, key_list)
    # Step 1: 计算平均权重 avg_w
    weight_ls = compute_avg_weights_for_domains(weight_ls, key_list)
    if base_weight is not None:
        base_dict, key_list = merge_params_into_all_param({0:[base_weight,]}, key_list[:-1])
        avg_weights = base_dict[0][0]
    else:
        avg_weights = compute_avg_weights(weight_ls, key_list)

    # Step 2: 对每个领域的专家计算 delta_w
    all_delta_weights = []
    for domain_idx, domain in weight_ls.items():
        delta_weights = compute_delta_weights(domain, avg_weights, key_list)
        all_delta_weights.append(delta_weights)

    # Step 3: 对每个参数的 delta_w 执行 PCA
    pca_results = {}
    for key in key_list:
        # 提取所有领域的 delta_w 并拼接
        all_params = []
        for delta_weights in all_delta_weights:
            all_params.extend([dw[key] for dw in delta_weights])
        param_matrix = torch.stack([param.flatten() for param in all_params])
        
        # 执行 PCA
        reduced_params, principal_components, mean = perform_pca(param_matrix, n_components=n_components)
        print(f"PCA for {key}: {reduced_params.shape}")
        
        # 分领域存储降维结果
        idx = 0
        for domain_delta_weights in all_delta_weights:
            for dw in domain_delta_weights:
                dw[key] = reduced_params[idx]  # 替换为 PCA 压缩系数
                idx += 1
        
        # 存储 PCA 的主成分信息
        pca_results[key] = {
            "principal_components": principal_components,
            "mean": mean
        }

    # res_coff = {}
    # for domain_idx, domain in weight_ls.items():
    #     for expert_idx, expert in enumerate(domain):
    #         for key in key_list:
    #             expert[key] = all_delta_weights[domain_idx][expert_idx][key]

    return avg_weights, pca_results, all_delta_weights

def pca_for_domain_experts_all(weight_ls, n_components=2, key_list=None, base_weight = None):
    """
    对多领域专家模型的权重执行 PCA，并将压缩后的系数存回每个子列表。

    参数：
        weight_ls (list of list of dict): 外层列表为领域，内层列表为每个领域的专家权重。
        n_components (int): PCA 的主成分数。
        key_list (list of str or None): 要执行 PCA 的参数名列表。如果为 None，则选择所有参数。

    返回：
        tuple:
            avg_weights (dict): 所有领域的平均权重。
            pca_results (dict): 每个参数的 PCA 结果，包括压缩系数。
    """
    def compute_avg_weights_for_domains(weight_ls, keys):
        """
        计算每个领域的平均权重，并将其插入到每个领域的专家列表的最前面。
        """
        for domain, experts in weight_ls.items():
            domain_avg = {key: torch.mean(torch.stack([expert[key] for expert in experts]), dim=0) for key in keys}
            weight_ls[domain].insert(0, domain_avg)
        return weight_ls
    
    def compute_avg_weights(weight_ls, keys):
        """
        计算所有领域的平均权重。
        """
        avg_weights = {key: 0 for key in keys}
        total_count = 0
        for domain_idx, domain in weight_ls.items():
            for expert in domain:
                for key in keys:
                    avg_weights[key] += expert[key]
                total_count += 1
        for key in keys:
            avg_weights[key] /= total_count
        return avg_weights

    def compute_delta_weights(domain, avg_weights, keys):
        """
        计算 delta_w = w - avg_w。
        """
        delta_weights = []
        for expert in domain:
            delta = {key: expert[key] - avg_weights[key] for key in keys}
            delta_weights.append(delta)
        return delta_weights

    def extract_parameters(delta_weights, keys):
        """
        提取所有 delta_w 并展平为矩阵，每一行是一个专家的展平 delta 参数。
        """
        param_matrices = {key: [] for key in keys}
        for delta in delta_weights:
            for key in keys:
                param_matrices[key].append(delta[key].flatten())
        for key in keys:
            param_matrices[key] = torch.stack(param_matrices[key])
        return param_matrices

    def perform_pca(param_matrix, n_components):
        """
        对参数矩阵执行 PCA 降维。
        """
        # 计算均值并中心化
        mean = param_matrix.mean(dim=0)
        centered_matrix = param_matrix - mean

        # 使用 SVD 进行 PCA
        U, S, V = torch.svd(centered_matrix)

        # 选取前 n_components 个主成分
        principal_components = V[:, :n_components]
        reduced = torch.mm(centered_matrix, principal_components)
        return reduced, principal_components, mean

    # 获取所有参数的 key
    if key_list is None:
        key_list = weight_ls[0][0].keys()

    # Step 1: 计算平均权重 avg_w
    weight_ls = compute_avg_weights_for_domains(weight_ls, key_list)
    if base_weight is not None:
        avg_weights = base_weight
    else:
        avg_weights = compute_avg_weights(weight_ls, key_list)

    # Step 2: 对每个领域的专家计算 delta_w
    all_delta_weights = []
    for domain_idx, domain in weight_ls.items():
        delta_weights = compute_delta_weights(domain, avg_weights, key_list)
        all_delta_weights.append(delta_weights)

    # Step 3: 合并所有参数为 "all_param"
    for domain_idx, domain_delta_weights in enumerate(all_delta_weights):
        for delta in domain_delta_weights:
            all_params = torch.cat([delta[key].flatten() for key in key_list])
            delta["all_param"] = all_params

    # 对 "all_param" 执行 PCA
    all_params_matrix = []
    for domain_delta_weights in all_delta_weights:
        all_params_matrix.extend([dw["all_param"] for dw in domain_delta_weights])
    all_params_matrix = torch.stack(all_params_matrix)
    
    reduced_all_param, all_param_components, all_param_mean = perform_pca(all_params_matrix, n_components=n_components)

    # 将 PCA 压缩结果存入 delta_weights
    idx = 0
    for domain_delta_weights in all_delta_weights:
        for delta in domain_delta_weights:
            delta["all_param"] = reduced_all_param[idx]
            idx += 1

    # 存储 "all_param" 的 PCA 信息
    pca_results = {"all_param": {"principal_components": all_param_components, "mean": all_param_mean}}

    return avg_weights, pca_results, all_delta_weights

def visualize_pca(adw, key_param, args, smooth_window=None):
    """
    可视化 PCA 的结果并保存图片。
    
    参数：
        adw: 包含 PCA 数据的结构，adw[ts][0][key] 是平均值，adw[ts][1:][key] 是不同专家的值。
        key_param: 需要绘制的参数列表。
        args: 包含路径信息的参数对象，需包含 base_dir, method, dataset, epochs 等。
        smooth_window: 移动平均窗口大小。如果为 None，则不进行平滑。
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    # 构造保存路径
    save_path = Path(args.base_dir) / f"{args.method}_{args.dataset}_{args.epochs}" / \
                f"rw_{args.mini_batch_size}_{args.rw_freq}_{args.rw_iters}_{args.lr}_pca_vis"
    save_path.mkdir(parents=True, exist_ok=True)  # 创建路径

    # 遍历每个 key，创建一个图
    for key in key_param:
        print(f"Processing key: {key}")

        # 子图布局参数
        num_dimensions = len(adw[0][0][key])
        rows = 5  # 每列 5 张图
        cols = (num_dimensions + rows - 1) // rows  # 动态计算列数
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 15))
        axes = axes.flatten()  # 将二维数组展平以便索引

        for dim_idx in range(num_dimensions):
            ts_values = list(range(len(adw)))  # 原始横坐标 ts

            # 中心线数据：adw[ts][0][key]
            try:
                center_line = [adw[ts][0][key][dim_idx].item() for ts in ts_values]
            except:
                pdb.set_trace()

            # 专家数据：adw[ts][1:][key]
            expert_scatter = [[adw[ts][exp_idx][key][dim_idx].item() for exp_idx in range(1, len(adw[ts]))] for ts in ts_values]

            # 平滑数据
            if smooth_window:
                smoothed_center_line, valid_indices = moving_average(center_line, smooth_window)
                smoothed_ts_values = [ts_values[i] for i in valid_indices]
            else:
                smoothed_center_line = center_line
                smoothed_ts_values = ts_values

            # 绘制子图
            axes[dim_idx].plot(ts_values, center_line, marker='o', alpha=0.8, label=f'Center Line Dimension {dim_idx + 1}')
            axes[dim_idx].plot(smoothed_ts_values, smoothed_center_line, marker='o', label=f'Smoothed Center Line Dim {dim_idx + 1}')
            
            # 绘制散点图
            for ts, expert_values in zip(ts_values, expert_scatter):
                axes[dim_idx].scatter([ts] * len(expert_values), expert_values, alpha=0.5, label=f'Experts at ts={ts}' if ts == 0 else "")  # 防止重复图例
            
            axes[dim_idx].set_title(f'{key} - Dimension {dim_idx + 1}', fontsize=12)
            axes[dim_idx].set_xlabel('Time Step (ts)', fontsize=10)
            axes[dim_idx].set_ylabel('PCA Value', fontsize=10)
            axes[dim_idx].legend()

        # 清空多余的子图
        for dim_idx in range(num_dimensions, len(axes)):
            fig.delaxes(axes[dim_idx])

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path / f"{key}_pca_visualization.png")
        plt.close(fig)
        print(f"Saved visualization for {key} at {save_path / f'{key}_pca_visualization.png'}")

def reconstruct_weights_with_avg(pca_results, components_coeffs, key_list, avg_weights, org_weights):
    """
    根据 PCA 的主成分和系数重建权重，并加回平均权重 avg_weights。
    
    参数：
        pca_results (dict): 每个参数的 PCA 结果，包括主成分和均值。
            结构为：
                {
                    key1: {
                        "principal_components": Tensor,  # 主成分矩阵
                        "mean": Tensor,                # 均值向量
                    },
                    ...
                }
        components_coeffs (dict): 每个参数对应的降维系数。
            结构为：
                {
                    key1: Tensor (shape: [n_samples, n_components]),
                    key2: Tensor (shape: [n_samples, n_components]),
                    ...
                }
        key_list (list of str): 重建的参数名称列表。
        avg_weights (dict): 平均权重，结构为 {key: Tensor}。

    返回：
        reconstructed_weights (dict): 重建后的权重。
            结构为：
                {
                    key1: Tensor (shape: [n_samples, original_dim]),
                    key2: Tensor (shape: [n_samples, original_dim]),
                    ...
                }
    """
    reconstructed_weights = {}
    
    for key in key_list:
        if key not in pca_results:
            raise ValueError(f"Key '{key}' not found in PCA results.")
        if key not in avg_weights:
            raise ValueError(f"Key '{key}' not found in avg_weights.")

        # 获取 PCA 结果中的主成分矩阵和均值向量
        principal_components = pca_results[key]["principal_components"]
        mean = pca_results[key]["mean"]

        # 获取该参数的降维系数
        coeffs = components_coeffs[key]  # shape: [n_samples, n_components]

        # 使用主成分和均值重建原始 delta 参数
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0)
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        
        reconstructed_delta = torch.mm(coeffs, principal_components.T) + mean
        
        
        # 加回平均权重
        target_shape = avg_weights[key].shape
        reconstructed_delta = reconstructed_delta.view(target_shape)
        try:
            reconstructed = reconstructed_delta + avg_weights[key]
        except:
            pdb.set_trace()
        reconstructed_weights[key] = reconstructed  # shape: [n_samples, original_dim]
    for key in org_weights:
        if key not in key_list:
            reconstructed_weights[key] = org_weights[key]
    
    return reconstructed_weights

def reconstruct_weights_from_all_param(pca_results, components_coeffs, key_list, avg_weights, org_weights):
    """
    根据 `all_param` 重建所有参数权重，从 `org_weights` 提取参数形状。
    
    参数：
        pca_results (dict): PCA 的结果，包括主成分和均值。
        components_coeffs (Tensor): `all_param` 的降维系数，shape: [n_samples, n_components]。
        key_list (list of str): 参数名称列表。
        avg_weights (dict): 平均权重，结构为 {key: Tensor}。
        org_weights (dict): 原始权重，包含各参数的张量。

    返回：
        reconstructed_weights (dict): 重建后的权重，结构为 {key: Tensor}。
    """
    reconstructed_weights = {}

    # 获取 `all_param` 的 PCA 结果
    all_param_pca = pca_results["all_param"]
    principal_components = all_param_pca["principal_components"]
    mean = all_param_pca["mean"]

    # 确保降维系数是二维
    # pdb.set_trace()
    if components_coeffs.dim() == 1:
        components_coeffs = components_coeffs.unsqueeze(0)  # shape: [1, n_components]
    if mean.dim() == 1:
        mean = mean.unsqueeze(0)  # shape: [1, original_dim]

    # 重建展平的 `all_param`
    reconstructed_all_param = torch.mm(components_coeffs, principal_components.T) + mean  # shape: [n_samples, original_dim]

    # 从展平的 `all_param` 分割出每个参数
    start_idx = 0
    for key in key_list:
        # 获取参数的原始形状和总元素数
        shape = org_weights[key].shape
        num_elements = torch.prod(torch.tensor(shape)).item()

        # 提取参数的展平切片并 reshape
        flattened_param = reconstructed_all_param[:, start_idx:start_idx + num_elements]
        reconstructed_param = flattened_param.view(*shape)

        # 加回平均权重
        reconstructed_weights[key] = reconstructed_param + avg_weights[key]
        # print(reconstructed_weights[key].shape)

        # 更新切片的起始索引
        start_idx += num_elements

    # 对于不在 key_list 中的参数，直接使用原始权重
    for key in org_weights:
        if key not in key_list:
            reconstructed_weights[key] = org_weights[key]

    return reconstructed_weights
