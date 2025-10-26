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
import os

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

# --- [NOVA FUNÇÃO "TRABALHADORA"] ---
def process_single_task(task: tuple, 
                        df_retornos: pd.DataFrame, 
                        config: dict) -> dict:
    """
    A "Unidade de Trabalho" para a paralelização.
    Calcula os dois fatores para 1 ATIVO em 1 DATA.
    """
    t_date, ticker = task # Desempacota a tarefa
    
    # Extrai parâmetros do config
    lookback_days = config['lookback_days']
    benchmark_ticker = config['benchmark_ticker']
    risk_free_ticker = config['risk_free_ticker']
    c_level = config['c_level']
    n_min = config['n_min'] 
    
    # Define a janela de lookback
    end_date = t_date
    start_date = end_date - pd.DateOffset(days=lookback_days)
    
    # Fatia o DataFrame (esta é a operação mais pesada depois do cálculo)
    window_df = df_retornos.loc[start_date:end_date]
    
    # Prepara os arrays numpy
    asset_returns_np = window_df[ticker].to_numpy()
    market_returns_np = window_df[benchmark_ticker].to_numpy()
    rf_returns_np = window_df[risk_free_ticker].to_numpy()
    
    # Chama a Função Core 1 (Paper 1)
    fator_risco = calcular_risk_estimator(asset_returns_np, 
                                          rf_returns_np,
                                          n_min)
    
    # Chama a Função Core 2 (Paper 2)
    fator_assimetria = downside_asymmetry_entropy_calculate(asset_returns_np, 
                                                          market_returns_np, 
                                                          c_level,
                                                          n_min)
    
    # Retorna um dicionário (que será facilmente convertido em DataFrame)
    return {
        'date': t_date,
        'ticker': ticker,
        'fator_risco': fator_risco,
        'fator_assimetria': fator_assimetria
    }

# --- BLOCO 3: EXECUÇÃO PRINCIPAL (ESTÁGIO 1) ---

def main_calculate_factors():
    """
    Orquestrador do Estágio 1: Carrega retornos, calcula todos os fatores 
    em paralelo e salva em 'fatores.csv'.
    """
    
    # 1. Definição de Parâmetros
    config = {
        'lookback_days': 252,    
        'rebal_frequency': 'BM', 
        'start_date': '2023-12-01', 
        'end_date': '2024-12-31',   
        'c_level': 0.0,          
        'n_min': 30,             
        'benchmark_ticker': 'IBOV',
        'risk_free_ticker': 'CDI',
        'input_file': 'retornos_copy.csv',
        'output_file': 'fatores.csv'
    }
    
    print(f"--- ESTÁGIO 1: CÁLCULO DE FATORES (OTIMIZADO) ---")
    print(f"Configuração: {config}")
    
    # 2. Carregar Dados Brutos
    print(f"Carregando dados de '{config['input_file']}'...")
    try:
        df_retornos = pd.read_csv(config['input_file'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Arquivo 'retornos.csv' não encontrado. Gerando mockup...")
        return
    
    universo_tickers = [t for t in df_retornos.columns if t not in [config['benchmark_ticker'], config['risk_free_ticker']]]
    
    # 3. Obter Datas de Rebalanceamento
    rebalance_dates = get_rebalance_dates(df_retornos, 
                                          config['start_date'], 
                                          config['end_date'], 
                                          config['rebal_frequency'])
    
    # --- [MUDANÇA PRINCIPAL 1: Criar a Lista de Tarefas] ---
    # Em vez de 120 tarefas, criamos N_datas * N_tickers tarefas
    all_tasks = [(t_date, ticker) 
                 for t_date in rebalance_dates 
                 for ticker in universo_tickers]
    
    print(f"\nIniciando cálculo para {len(all_tasks)} tarefas (Ativo x Data)...")
    start_loop = time.perf_counter()

    # --- [MUDANÇA PRINCIPAL 2: Informar Núcleos] ---
    # Tenta obter o número de núcleos, se falhar, usa 1.
    try:
        n_cores = os.cpu_count()
        print(f"Utilizando {n_cores} núcleos de CPU para paralelização.")
    except NotImplementedError:
        n_cores = 1
        print("Não foi possível detectar o número de núcleos. Usando 1.")

    # --- [MUDANÇA PRINCIPAL 3: Loop Paralelo Otimizado] ---
    # O loop agora itera sobre 'all_tasks' (10.800+) 
    # e o tqdm mostrará um progresso e ETA muito mais coerentes.
    
    # (backend="multiprocessing" é o padrão, mas é bom ser explícito)
    parallel_results = Parallel(n_jobs=n_cores, backend="multiprocessing")(
        delayed(process_single_task)(
            task, df_retornos, config
        ) 
        for task in tqdm(all_tasks) # tqdm agora envolve as 10.800 tarefas
    )
    
    end_loop = time.perf_counter()
    print(f"\nCálculo em paralelo concluído em {end_loop - start_loop:.2f} segundos.")

    # 5. Montagem e Salvamento
    print("Montando DataFrame final de fatores...")
    # 'parallel_results' é agora uma lista de dicionários
    all_factors_df = pd.DataFrame(parallel_results)
    
    all_factors_df = all_factors_df.set_index(['date', 'ticker'])
    
    all_factors_df.to_csv(config['output_file'])
    print(f"Fatores salvos com sucesso em '{config['output_file']}'!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_calculate_factors()