import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
import time
import warnings
from joblib import Parallel, delayed # Para paralelização
from tqdm import tqdm


def histogram_entropy_shannon_calculate(asset_returns, n_total_min, k_par=0):
    def set_histogram(asset_returns_local, k_local):
        n = len(asset_returns_local)
        b = np.max(asset_returns_local) - np.min(asset_returns_local)
        

        if k_local <= 0: k_local = 1 
        
        bins, bin_edges = np.histogram(asset_returns_local, k_local)
        
        h_real = 0.0
        if len(bin_edges) > 1:
            h_real = bin_edges[1] - bin_edges[0]
            
        pacote = [bins, bin_edges, n, h_real, k_local, asset_returns_local]
        return pacote
    
    def get_k(asset_returns_local, method=0):
        if method == 0:
            b = np.max(asset_returns_local) - np.min(asset_returns_local)
            std_dev = np.std(asset_returns_local)
            if std_dev == 0:
                return 1

            denominator = ((24.0 * np.pi**0.5 / asset_returns_local.size)**(1.0 / 3.0) * std_dev)
            if denominator == 0:
                return 1
            k_float = np.ceil(b / denominator)
            return int(k_float)

    def _histogram_entropy_shannon_vetorizada(bins, k, n, h):
        if h <= 0:
            return -np.inf
        vj_filtrado = bins[bins > 0]
        if vj_filtrado.size == 0:
             return -np.inf
        f_x = vj_filtrado / (n * h)
        soma_vetorizada = np.sum(vj_filtrado * np.log(f_x))
        return -soma_vetorizada / n

    asset_returns_cleaned = np.asarray(asset_returns)
    asset_returns_cleaned = asset_returns_cleaned[~np.isnan(asset_returns_cleaned)]
    n_total = len(asset_returns_cleaned)


    if n_total < n_total_min:
        return np.nan 
    
    if np.max(asset_returns_cleaned) == np.min(asset_returns_cleaned):
        return -np.inf
    
    if k_par > 0:
        k = int(k_par)
    else:
        k = get_k(asset_returns_cleaned)
    if k < 2:
        k = 2
        
    try:
        pacote_ar = set_histogram(asset_returns_cleaned, k)
    except ValueError:
        return np.nan
    bins_count = pacote_ar[0]
    n_val = pacote_ar[2]
    h_val = pacote_ar[3]
    k_val = pacote_ar[4]
    
    entropy = _histogram_entropy_shannon_vetorizada(bins_count, k_val, n_val, h_val)
    return entropy

def calcular_risk_estimator(asset_returns, risk_free_returns, n_total_min):

    asset_returns = np.asarray(asset_returns)
    risk_free_returns = np.asarray(risk_free_returns)
    
    valid_indices = ~np.isnan(asset_returns) & ~np.isnan(risk_free_returns)
    
    if np.sum(valid_indices) < n_total_min:
        return np.nan 
        
    excess_returns = asset_returns[valid_indices] - risk_free_returns[valid_indices]

    H = histogram_entropy_shannon_calculate(excess_returns, 
                                            n_total_min=n_total_min)
    
    kappa_H = np.exp(H)
    return kappa_H

def asymmetry_entropy_calculate(asset_returns, market_returns, c_level, n_total_min):


    def clean_and_standardize(x, y):
        data = np.vstack([x, y])
        data_clean = data[:, ~np.any(np.isnan(data), axis=0)]
        x_clean, y_clean = data_clean[0, :], data_clean[1, :]
        n_total = len(x_clean)
        x_mean, x_std = np.mean(x_clean), np.std(x_clean)
        y_mean, y_std = np.mean(y_clean), np.std(y_clean)
        x_std_scores = (x_clean - x_mean) / x_std if x_std > 0 else np.zeros_like(x_clean)
        y_std_scores = (y_clean - y_mean) / y_std if y_std > 0 else np.zeros_like(y_clean)
        return x_std_scores, y_std_scores, n_total

    def filter_scenarios(asset_z, market_z, c_level_local):
        up_indexes = np.where((asset_z > c_level_local) & (market_z > c_level_local))
        up_data = np.vstack([asset_z[up_indexes], market_z[up_indexes]])
        down_indexes = np.where((asset_z < -c_level_local) & (market_z < -c_level_local))
        down_data = np.vstack([asset_z[down_indexes], market_z[down_indexes]])
        return up_data, down_data

    def rotate_data(data):
        return -data

    #REVISAR
    def estimate_joint_pdf(data):
        if data.shape[1] < 2:
            return None
        try:
            kde = gaussian_kde(data)
            return kde.pdf
        except (np.linalg.LinAlgError, ValueError):
            return None

    def calculate_hellinger(pdf_up, pdf_down):
        def integrand(y, x): 
            f_up_val = pdf_up([x, y])[0]
            f_down_val = pdf_down([x, y])[0]
            sqrt_f_up = np.sqrt(np.maximum(0, f_up_val))
            sqrt_f_down = np.sqrt(np.maximum(0, f_down_val))
            return (sqrt_f_up - sqrt_f_down)**2

        if pdf_up is None:
          pdf_up = lambda coords: np.array([0.0]) # <-- Retorna array
        if pdf_down is None:
          pdf_down = lambda coords: np.array([0.0]) # <-- Retorna array
        
        lim = 8.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            integral_result, _ = dblquad(integrand, -lim, lim, -lim, lim)

        S_rho = 0.5 * integral_result
        return np.clip(S_rho, 0, 1)

    asset_z, market_z, n_total = clean_and_standardize(asset_returns, market_returns)
    
    if n_total < n_total_min: 
        return 0.0, 0.0, 0.0 

    up_data, down_data = filter_scenarios(asset_z, market_z, c_level)

    n_up = up_data.shape[1]
    n_down = down_data.shape[1]
    
    LQP = n_down / n_total
    UQP = n_up / n_total

    rotated_down_data = rotate_data(down_data)
    pdf_up = estimate_joint_pdf(up_data)
    pdf_down = estimate_joint_pdf(rotated_down_data)
    S_rho_c = calculate_hellinger(pdf_up, pdf_down)

    return S_rho_c, LQP, UQP

def downside_asymmetry_entropy_calculate(asset_returns, market_returns,c_level, n_total_min):

    S_rho, LQP, UQP = asymmetry_entropy_calculate(asset_returns,market_returns, c_level, n_total_min)
    signal = np.sign(LQP - UQP)
    down_asy_score = signal * S_rho
    return down_asy_score



def get_rebalance_dates(df_retornos: pd.DataFrame, 
                        start_date: str, 
                        end_date: str, 
                        freq: str) -> pd.DatetimeIndex:
    """Cria o "relógio" do backtest."""
    mask = (df_retornos.index >= start_date) & (df_retornos.index <= end_date)
    all_dates = df_retornos.loc[mask].index
    rebal_dates = pd.date_range(start_date, end_date, freq=freq)
    rebal_dates_in_index = all_dates.searchsorted(rebal_dates, side='right') - 1
    return all_dates[rebal_dates_in_index].unique()

