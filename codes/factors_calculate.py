import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
import time
import warnings
from joblib import Parallel, delayed # Para paralelização
from tqdm import tqdm
from toolbox import *

import pandas as pd
import numpy as np
import warnings
import os 
import time
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from joblib import Parallel, delayed # Para paralelização
from tqdm import tqdm # Para barra de progresso



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

def run_factor_calculation_for_date(t_date: pd.Timestamp, 
                                    df_retornos: pd.DataFrame, 
                                    tickers: list,
                                    config: dict) -> pd.DataFrame:
    """
    O "Trabalho" a ser paralelizado: Calcula os fatores para 1 data.
    """
    # Extrai parâmetros do config para esta tarefa
    lookback_days = config['lookback_days']
    benchmark_ticker = config['benchmark_ticker']
    risk_free_ticker = config['risk_free_ticker']
    c_level = config['c_level']
    n_min = config['n_min'] # <- Usa a sua assinatura 'n_min'
    
    end_date = t_date
    start_date = end_date - pd.DateOffset(days=lookback_days)
    
    window_df = df_retornos.loc[start_date:end_date]
    
    market_returns_np = window_df[benchmark_ticker].to_numpy()
    rf_returns_np = window_df[risk_free_ticker].to_numpy()
    
    all_factors = []
    
    for ticker in tickers:
        asset_returns_np = window_df[ticker].to_numpy()
        
        # Chama a Função Core 1 (Paper 1)
        fator_risco = calcular_risk_estimator(asset_returns_np, 
                                              rf_returns_np,
                                              n_min)
        
        # Chama a Função Core 2 (Paper 2)
        fator_assimetria = downside_asymmetry_entropy_calculate(asset_returns_np, 
                                                              market_returns_np, 
                                                              c_level,
                                                              n_min)
        
        all_factors.append({
            'ticker': ticker,
            'fator_risco': fator_risco,
            'fator_assimetria': fator_assimetria
        })
        
    df_factors = pd.DataFrame(all_factors)
    df_factors['date'] = t_date # Adiciona a data para a montagem final
    return df_factors


def main_calculate_factors():

    
    config = {
        'lookback_days': 252,    # Janela de 1 ano
        'rebal_frequency': 'BM', # Fim do Mês de Negócios
        'start_date': '2023-01-01', # Início do período de cálculo dos fatores
        'end_date': '2024-12-31',   # Fim do período
        'c_level': 0.0,          # Nível 'c' para o DOWN_ASY
        'n_min': 30,             # Limite de robustez (N mínimo de dias)
        'benchmark_ticker': 'IBOV',
        'risk_free_ticker': 'CDI',
        'input_file': 'retornos.csv',
        'output_file': 'fatores.csv'
    }
    
    print(f"--- ESTÁGIO 1: CÁLCULO DE FATORES (PARALELIZADO) ---")
    print(f"Configuração: {config}")
    print(f"Carregando dados de '{config['input_file']}'...")
    try:

        df_retornos = pd.read_csv(config['input_file'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Arquivo 'retornos.csv' não encontrado. Gerando mockup...")
        return
    
    universo_tickers = [t for t in df_retornos.columns if t not in [config['benchmark_ticker'], config['risk_free_ticker']]]
    
    rebalance_dates = get_rebalance_dates(df_retornos, 
                                          config['start_date'], 
                                          config['end_date'], 
                                          config['rebal_frequency'])
    
    print(f"Iniciando cálculo de fatores para {len(universo_tickers)} ativos em {len(rebalance_dates)} datas...")
    start_loop = time.perf_counter()

    # 4. Loop Principal (PARALELIZADO)
    # n_jobs=-1 usa todos os núcleos do CPU
    # 'tqdm' cria a barra de progresso
    parallel_results = Parallel(n_jobs=-1)(
        delayed(run_factor_calculation_for_date)(
            t_date, df_retornos, universo_tickers, config
        ) 
        for t_date in tqdm(rebalance_dates) # tqdm envolve o iterável
    )
    
    end_loop = time.perf_counter()
    print(f"\nCálculo em paralelo concluído em {end_loop - start_loop:.2f} segundos.")

    print("Montando DataFrame final de fatores...")
    all_factors_df = pd.concat(parallel_results)
    
    # Cria o MultiIndex (date, ticker) para lookups rápidos no Estágio 2
    all_factors_df = all_factors_df.set_index(['date', 'ticker'])
    
    all_factors_df.to_csv(config['output_file'])
    print(f"Fatores salvos com sucesso em '{config['output_file']}'!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_calculate_factors()