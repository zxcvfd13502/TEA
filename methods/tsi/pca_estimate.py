import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.arima.model import ARIMA  # statsmodels>=0.12.0
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

import pmdarima as pm

def compute_avg_vectors(adw):
    result = []
    for experiments in adw:
        all_params = np.array([exp['all_param'] for exp in experiments])
        avg_vector = np.mean(all_params, axis=0)
        result.append(avg_vector)
    return np.vstack(result)

def get_each_coeffs(adw):
    all_coffs = []
    for experiments in adw:
        all_params = [exp['all_param'] for exp in experiments]
        all_coffs.append(all_params)
    return np.array(all_coffs)

def fit_and_forecast_varmax(data: np.ndarray, t: int):
    df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
    # model = VARMAX(df, order=(1, 1), enforce_stationarity=False, enforce_invertibility=False, trend='c')
    # fit_res = model.fit(maxiter=100, disp=False)
    model = VAR(df)
    fit_res = model.fit(maxlags=1)
    
    fitted = fit_res.fittedvalues.values
    forecast = fit_res.forecast(steps=t).values
    
    return fitted, forecast

def fit_and_forecast_var(data: np.ndarray, t: int):
    """
    data: (N, K) 的 numpy 数组, N=时间步, K=维度数
    t: 预测未来 t 步
    返回:
        fitted: (N - lag, K) 的训练集拟合值
        forecast: (t, K) 的预测值
    """
    df = pd.DataFrame(data, columns=[f'var_{i}' for i in range(data.shape[1])])
    
    # 拟合 VAR 模型（这里默认 maxlags=1，可以自行调大）
    model = VAR(df)
    fit_res = model.fit(maxlags=1)
    
    # 拟合值: 大小一般是 (N - lag, K)
    fitted = fit_res.fittedvalues.values
    
    # 预测未来 t 步：从最后 lag 条数据开始往后
    # 如果 lag=1，则 last_values.shape=(1, K)
    last_values = df.values[-fit_res.k_ar:]
    forecast = fit_res.forecast(last_values, steps=t)  # shape = (t, K)
    
    return fitted, forecast

def fit_and_forecast_univariate_arima(data: np.ndarray, t: int, domain_ids: np.ndarray, order=(1,1,1)):
    """
    对数据的每个维度各自做 ARIMA 拟合并预测。
    
    参数:
        data: shape = (N, K), N 为时间序列长度, K 为维度数
        t: 预测步数
        order: (p,d,q)，ARIMA 模型阶数
        
    返回:
        fitted: shape=(N, K) 的拟合值 (in-sample prediction)
        forecast: shape=(t, K) 的 out-of-sample 预测
    """
    N, K = data.shape
    assert len(domain_ids) == N

    # 构造相对时间索引
    base_time = np.min(domain_ids)
    relative_time_index = domain_ids - base_time

    # 创建 pandas DataFrame 以便带索引建模
    time_index = pd.Index(relative_time_index, name="time")

    fitted_values = np.zeros((relative_time_index[-1] + 1, K))
    forecast_values = np.zeros((t, K))

    for k in range(K):
        series_k = pd.Series(data[:, k], index=time_index)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            model = ARIMA(series_k, order=order)
            result = model.fit()

            # 获取 in-sample 拟合值（与输入时间戳顺序一致）
            fitted_series = result.predict(start=relative_time_index[0], end=relative_time_index[-1])
            fitted_values[:, k] = fitted_series.values

            # 预测未来 t 个相对时间点
            last_time = relative_time_index[-1]
            forecast_series = result.predict(start=last_time + 1, end=last_time + t)
            forecast_values[:, k] = forecast_series.values

    return fitted_values, forecast_values

def fit_and_forecast_univariate_auto_arima(data: np.ndarray, t: int):
    N, K = data.shape
    fitted_values = np.zeros((N, K))
    forecast_values = np.zeros((t, K))
    
    for k in range(K):
        series_k = data[:, k]
        
        # 自动选阶
        model = pm.auto_arima(series_k, start_p=0, max_p=2, start_q=0, max_q=2, max_d = 2,
                              seasonal=False,  # 如果你的数据没有季节性
                              stepwise=True, trace=False)
        
        # 拟合 in-sample
        d = model.order[1]
        print("using order:", d)
        fitted_series = model.predict_in_sample(start=d, end=N-1)
        fitted_values[d:, k] = fitted_series
        
        # 预测未来 t 步
        forecast_series = model.predict(n_periods=t)
        forecast_values[:, k] = forecast_series
        
    return fitted_values, forecast_values


def fit_and_forecast_univariate_linreg(data: np.ndarray, t: int):
    """
    使用最简单的线性回归(时间 -> 数值) 拟合每个维度，
    然后对未来 t 步做外推预测。

    参数:
        data: shape = (N, K)
              N = 时间步, K = 维度数
        t   : 需要预测的未来步数

    返回:
        fitted_values : shape = (N, K)
                        训练集上的拟合值（每个维度一条直线）
        forecast_values: shape = (t, K)
                         对未来 t 步的预测
    """
    N, K = data.shape
    fitted_values = np.zeros((N, K))
    forecast_values = np.zeros((t, K))

    # X 轴就是 [0, 1, 2, ..., N-1]
    x = np.arange(N)

    for k in range(K):
        y = data[:, k]
        # 用 np.polyfit(x, y, 1) 做一次线性回归 (degree=1)
        # polyfit 返回 [slope, intercept]
        slope, intercept = np.polyfit(x, y, 1)
        
        # 训练集上的拟合值
        fitted_values[:, k] = slope * x + intercept
        
        # 预测未来 t 步：对应 x = N ~ N+t-1
        x_future = np.arange(N, N + t)
        forecast_values[:, k] = slope * x_future + intercept

    return fitted_values, forecast_values

def plot_var_results(data: np.ndarray, fitted: np.ndarray, forecast: np.ndarray, base_path: str):
    """
    将原始 data、拟合值 fitted、预测值 forecast 分 20 个子图画在一张大图上。
    然后存到  Path(base_path) / "pca_vis" / "all_param_fit_pred.png"
    """
    num_vars = data.shape[1]       # K
    N = data.shape[0]              # 原始序列长度
    fitted_len = fitted.shape[0]   # 拟合值序列长度 (通常 N - lag)
    t = forecast.shape[0]          # 预测步数
    
    # 拟合值的 x 轴起点
    # 若 lag=1, fitted_len= N-1, 则从 1 到 N-1
    x_fitted_start = N - fitted_len
    x_fitted = np.arange(x_fitted_start, N)
    
    # 预测值的 x 轴从 N 到 N + t - 1
    x_forecast = np.arange(N, N + t)

    fig, axes = plt.subplots(5, 4, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(num_vars):
        ax = axes[i]

        # 原始数据
        ax.plot(np.arange(N), data[:, i], label='Original', color='black')
        
        # 拟合值
        ax.plot(x_fitted, fitted[:, i], label='Fitted', linestyle='--', color='blue')
        
        # 预测值
        ax.plot(x_forecast, forecast[:, i], label='Forecast', linestyle=':', color='red')
        
        ax.set_title(f'Dimension {i}')
        ax.legend()
        ax.grid(True)
    
    # 如果 num_vars < 20，就删除多余子图
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # 保存图像
    save_path = Path(base_path) / "pca_vis"
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / "all_param_fit_pred.png", dpi=300)
    plt.close(fig)
