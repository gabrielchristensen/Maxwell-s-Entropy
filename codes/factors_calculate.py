import pandas as pd
import numpy as np
import warnings
import os 
import time
from joblib import Parallel, delayed
from tqdm import tqdm 

# Importa as suas 4 funções de cálculo (Risk Estimator e DOWN_ASY Entropia)
# (Assumindo que elas estão num ficheiro chamado 'toolbox.py')
try:
    from toolbox import (
        calcular_risk_estimator, 
        downside_asymmetry_entropy_calculate
    )
except ImportError:
    print("Erro: Não foi possível encontrar o 'toolbox.py'.")
    print("Certifique-se de que o ficheiro com as 4 funções de cálculo está na mesma pasta.")
    exit()

# --- BLOCO 1: FUNÇÃO "TRABALHADORA" (A mesma de antes) ---

def process_single_task(task: tuple, 
                        df_retornos: pd.DataFrame, # Recebe o DF MESTRE
                        config: dict) -> dict:
    """
    A "Unidade de Trabalho" para a paralelização.
    Calcula os dois fatores para 1 ATIVO em 1 DATA (lookback de 1 ano).
    """
    t_date, ticker = task 
    
    lookback_days = config['lookback_days']
    benchmark_ticker = config['benchmark_ticker']
    risk_free_ticker = config['risk_free_ticker']
    c_level = config['c_level']
    n_min = config['n_min'] 
    
    # Define a janela de lookback (ex: 252 dias antes de t_date)
    end_date = t_date
    start_date = end_date - pd.DateOffset(days=lookback_days)
    
    try:
        window_df = df_retornos.loc[start_date:end_date]
    except KeyError:
        return {
            'date': t_date, 'ticker': ticker,
            'fator_risco': np.nan, 'fator_assimetria': np.nan
        }

    if ticker not in window_df.columns:
        return {'date': t_date, 'ticker': ticker, 'fator_risco': np.nan, 'fator_assimetria': np.nan}
        
    asset_returns_np = window_df[ticker].to_numpy()
    market_returns_np = window_df[benchmark_ticker].to_numpy()
    rf_returns_np = window_df[risk_free_ticker].to_numpy()
    
    fator_risco = calcular_risk_estimator(asset_returns_np, 
                                          rf_returns_np,
                                          n_min)
    
    fator_assimetria = downside_asymmetry_entropy_calculate(asset_returns_np, 
                                                          market_returns_np, 
                                                          c_level,
                                                          n_min)
    
    return {
        'date': t_date,
        'ticker': ticker,
        'fator_risco': fator_risco,
        'fator_assimetria': fator_assimetria
    }

# --- BLOCO 2: FUNÇÕES AUXILIARES ---

def get_rebalance_dates(df_index: pd.DatetimeIndex, 
                        start_date: str, 
                        end_date: str, 
                        freq: str) -> pd.DatetimeIndex:
    """Cria o "relógio" do backtest alinhado ao índice de retornos."""
    mask = (df_index >= start_date) & (df_index <= end_date)
    all_dates = df_index[mask]
    
    # Gera as datas de rebalanceamento "ideais"
    rebal_dates_ideal = pd.date_range(start_date, end_date, freq=freq)
    
    # Encontra as datas de negociação reais mais próximas (anteriores)
    rebal_dates_in_index = all_dates.searchsorted(rebal_dates_ideal, side='right') - 1
    
    # Garante que não haja índices negativos e pega as datas reais
    valid_rebal_dates = all_dates[rebal_dates_in_index[rebal_dates_in_index >= 0]].unique()
    
    # Filtra novamente para garantir que está dentro do start_date (searchsorted pode pegar data anterior)
    return valid_rebal_dates[valid_rebal_dates >= pd.Timestamp(start_date)]

def get_universe_for_date(t_date: pd.Timestamp, universe_ranges: list) -> list:
    """
    Encontra a lista de tickers líquidos válidos para uma data de rebalanceamento específica.
    """
    for window in universe_ranges:
        # Verifica se a data de rebalanceamento está DENTRO do período de validade da janela
        if t_date >= window['start'] and t_date <= window['end']:
            return window['universe'] # Retorna a lista de tickers
    
    # Se nenhuma janela cobrir esta data, retorna uma lista vazia
    return []

# --- BLOCO 3: EXECUÇÃO PRINCIPAL (ESTÁGIO 1 - UNIVERSO DINÂMICO) ---

def main_calculate_factors_dynamic_universe():
    """
    Orquestrador do Estágio 1 (Corrigido):
    Carrega todos os dados de retorno para criar um 'master' e um 'mapa de universo'.
    Executa o cálculo de fator MENSALMENTE usando o universo dinâmico.
    """
    
    # 1. Definição de Parâmetros
    config = {
        'lookback_days': 252,    # O lookback rolling (ex: 1 ano)
        'rebal_frequency': 'BM', # Frequência de cálculo MENSAL
        'start_date': '2011-01-01', # Início do cálculo de fatores
        'end_date': '2024-12-31',   # Fim do período
        'n_min': 30,             # Limite de robustez (N mínimo de dias)
        'c_level': 0.0,          # Nível 'c' para o DOWN_ASY
        'benchmark_ticker': '^BVSP',
        'risk_free_ticker': 'CDI',
        
        'num_windows': 166, # Vai de 0 a 27
        'returns_input_prefix': r'data/retornos_diarios_janela_',
        'output_file': r'fatores/fatores_master.csv' # <-- Saída é UM ÚNICO ficheiro
    }
    
    print(f"--- ESTÁGIO 1: CÁLCULO MENSAL (UNIVERSO DINÂMICO) ---")
    print(f"Configuração: {config}")

    # --- 2. Montagem dos Dados Mestres ---
    
    print(f"Carregando {config['num_windows']} ficheiros de retorno para montar dados mestres...")
    all_returns_list = []
    universe_ranges = [] # Lista de dicionários: [{'start', 'end', 'universe'}]
    
    for i in range(config['num_windows']):
        returns_file = f"{config['returns_input_prefix']}{i}.csv"
        
        try:
            df_return_window = pd.read_csv(returns_file, index_col=0, parse_dates=True)
            if df_return_window.empty:
                print(f"Aviso: {returns_file} está vazio. Pulando.")
                continue
                
            all_returns_list.append(df_return_window)
            
            # [LÓGICA CORRIGIDA] Mapeia o universo e seu período de validade
            start_date = df_return_window.index.min()
            end_date = df_return_window.index.max()
            universe_tickers = [t for t in df_return_window.columns if t not in [config['benchmark_ticker'], config['risk_free_ticker']]]
            
            universe_ranges.append({
                'start': start_date,
                'end': end_date,
                'universe': universe_tickers
            })
            
        except FileNotFoundError:
            print(f"Aviso: Arquivo de retorno '{returns_file}' não encontrado. Pulando.")
    
    if not all_returns_list:
        print("Erro Crítico: Nenhum ficheiro de retorno foi carregado. Abortando.")
        return
        
    # Concatena todos os retornos e remove duplicados (mantém o mais recente)
    df_retornos_master = pd.concat(all_returns_list)
    df_retornos_master = df_retornos_master.sort_index()
    df_retornos_master = df_retornos_master.loc[~df_retornos_master.index.duplicated(keep='last')]
    
    print(f"DataFrame Mestre de Retornos criado com {len(df_retornos_master)} linhas.")
    print(f"Mapa de Universo criado com {len(universe_ranges)} períodos.")
    
    # Salva o master de retornos para o Estágio 2 usar
    df_retornos_master.to_csv(r"resultados/retornos_master.csv")
    print("DataFrame Mestre de Retornos salvo em 'retornos_master.csv'")

    # --- 3. Geração de Tarefas (Lógica Dinâmica) ---
    
    print("Gerando datas de rebalanceamento mensais...")
    rebalance_dates = get_rebalance_dates(df_retornos_master.index, 
                                          config['start_date'], 
                                          config['end_date'], 
                                          config['rebal_frequency'])
    
    print(f"Gerando lista de tarefas para {len(rebalance_dates)} datas...")
    all_tasks = []
    for t_date in rebalance_dates:
        # Encontra o universo líquido para esta data de rebalanceamento
        liquid_tickers = get_universe_for_date(t_date, universe_ranges)
        
        if liquid_tickers:
            for ticker in liquid_tickers:
                all_tasks.append((t_date, ticker))
        else:
            print(f"Aviso: Nenhum universo encontrado para a data {t_date.date()}. Pulando data.")
            
    if len(all_tasks) == 0:
        print("Erro: Nenhuma tarefa a ser processada (verifique o mapa de universo e as datas).")
        return

    # --- 4. Execução Paralela ---
    try:
        n_cores = os.cpu_count()
    except NotImplementedError:
        n_cores = 1
    
    print("\n--- RESUMO DA EXECUÇÃO ---")
    print(f"  - Período de Cálculo: {config['start_date']} a {config['end_date']}")
    print(f"  - Frequência: {config['rebal_frequency']} (Rolling)")
    print(f"  - Lookback: {config['lookback_days']} dias")
    print(f"  - Núcleos de CPU: {n_cores}")
    print(f"  - TOTAL DE TAREFAS (Data x Ticker Dinâmico): {len(all_tasks)}")
    print("---------------------------------")
    print("Iniciando cálculo em paralelo...")
    
    start_loop = time.perf_counter()

    parallel_results = Parallel(n_jobs=n_cores, backend="multiprocessing")(
        delayed(process_single_task)(
            task, 
            df_retornos_master, # Passa o DF MESTRE
            config
        ) 
        for task in tqdm(all_tasks)
    )
    
    end_loop = time.perf_counter()
    print(f"\nCálculo em paralelo concluído em {end_loop - start_loop:.2f} segundos.")

    # --- 5. Montagem e Salvamento ---
    print("Montando DataFrame final de fatores...")
    all_factors_df = pd.DataFrame(parallel_results)
    
    all_factors_df = all_factors_df.set_index(['date', 'ticker'])
    
    all_factors_df.to_csv(config['output_file'])
    print(f"Fatores salvos com sucesso em '{config['output_file']}'!")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_calculate_factors_dynamic_universe()