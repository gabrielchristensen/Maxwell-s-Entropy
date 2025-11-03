import pandas as pd
import numpy as np
import warnings
import os 
import time
from joblib import Parallel, delayed # Para paralelização
from tqdm import tqdm # Para barra de progresso

# Importa as suas 4 funções de cálculo (Risk Estimator e DOWN_ASY Entropia)
# (Assumindo que elas estão num ficheiro chamado 'toolbox.py')
try:
    from toolbox import (
        calcular_risk_estimator, 
        downside_asymmetry_entropy_calculate,
        histogram_entropy_shannon_calculate, # Importado por risk_estimator
        asymmetry_entropy_calculate # Importado por downside_asymmetry
    )
except ImportError:
    print("Erro: Não foi possível encontrar o 'toolbox.py'.")
    print("Certifique-se de que o ficheiro com as 4 funções de cálculo está na mesma pasta.")
    exit()

# --- BLOCO 2: FUNÇÃO "TRABALHADORA" PARA PARALELIZAÇÃO ---

def process_single_task_window(task: tuple, 
                               df_window: pd.DataFrame, 
                               config: dict) -> dict:
    """
    A "Unidade de Trabalho" para a paralelização DENTRO de uma janela.
    Calcula os dois fatores para 1 ATIVO usando o DataFrame da JANELA INTEIRA.
    """
    # Desempacota a tarefa
    t_date, ticker = task 
    
    # Extrai parâmetros do config
    benchmark_ticker = config['benchmark_ticker']
    risk_free_ticker = config['risk_free_ticker']
    c_level = config['c_level']
    n_min = config['n_min'] 
    
    # --- LÓGICA MODIFICADA ---
    # Não há mais fatiamento de lookback. O df_window É o lookback.
    
    # Prepara os arrays numpy
    asset_returns_np = df_window[ticker].to_numpy()
    market_returns_np = df_window[benchmark_ticker].to_numpy()
    rf_returns_np = df_window[risk_free_ticker].to_numpy()
    
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
        'date': t_date, # Esta é a data de rebalanceamento (fim da janela)
        'ticker': ticker,
        'fator_risco': fator_risco,
        'fator_assimetria': fator_assimetria
    }

# --- BLOCO 3: EXECUÇÃO PRINCIPAL (ESTÁGIO 1 - POR JANELA) ---

def main_calculate_factors_by_window():
    """
    Orquestrador do Estágio 1 (Modificado):
    Itera por cada ficheiro de 'janela', calcula os fatores em paralelo
    para os ativos daquela janela, e salva num ficheiro de fatores
    correspondente.
    """
    
    # 1. Definição de Parâmetros
    config = {
        # O 'lookback_days' não é mais usado para fatiar,
        # mas 'n_min' ainda é crucial.
        'n_min': 30,             # Limite de robustez (N mínimo de dias)
        'c_level': 0.0,          # Nível 'c' para o DOWN_ASY
        'benchmark_ticker': '^BVSP',
        'risk_free_ticker': 'CDI',
        
        # --- [NOVO] Parâmetros de Nomenclatura de Ficheiros ---
        'num_windows': 165,       # Número de ficheiros de janela 
        'input_prefix': r'data_mamaco/retornos_diarios_janela_',
        'output_prefix': r'fatores/fatores_janela_'
    }
    
    print(f"--- ESTÁGIO 1: CÁLCULO DE FATORES (POR JANELA) ---")
    print(f"Configuração: {config}")

    try:
        n_cores = os.cpu_count()
        print(f"Utilizando {n_cores} núcleos de CPU para paralelização interna.")
    except NotImplementedError:
        n_cores = 1
        print("Não foi possível detectar o número de núcleos. Usando 1.")
    
    total_start_time = time.perf_counter()

    # 2. Loop Principal (Itera pelos ficheiros de janela)
    for i in range(config['num_windows']):
        input_file = f"{config['input_prefix']}{i}.csv"
        output_file = f"{config['output_prefix']}{i}.csv"
        
        print(f"\n--- Processando Janela {i}: {input_file} ---")

        # 2a. Carregar Dados da Janela
        try:
            df_window = pd.read_csv(input_file, index_col=0, parse_dates=True)
            if df_window.empty:
                print(f"Aviso: {input_file} está vazio. Pulando.")
                continue
        except FileNotFoundError:
            print(f"Aviso: Arquivo {input_file} não encontrado. Pulando.")
            continue
            
        # 2b. Definir Universo e Data de Rebalanceamento
        # A data de rebalanceamento é o ÚLTIMO dia desta janela
        rebalance_date = df_window.index.max()
        
        # O universo são todos os ativos neste ficheiro (exceto IBOV/CDI)
        universo_tickers = [t for t in df_window.columns if t not in [config['benchmark_ticker'], config['risk_free_ticker']]]
        
        if not universo_tickers:
            print(f"Aviso: Nenhum ticker encontrado em {input_file} (além de IBOV/CDI). Pulando.")
            continue

        # 2c. Criar Lista de Tarefas (para esta janela)
        # A tarefa é (data_de_rebalanceamento, ticker_a_calcular)
        all_tasks_window = [(rebalance_date, ticker) for ticker in universo_tickers]

        print(f"Calculando {len(all_tasks_window)} tarefas para a data {rebalance_date.date()}...")
        start_loop = time.perf_counter()

        # 2d. Execução Paralela (DENTRO da janela)
        parallel_results = Parallel(n_jobs=n_cores, backend="multiprocessing")(
            delayed(process_single_task_window)(
                task, 
                df_window, # Passa o DataFrame da janela para o trabalhador
                config
            ) 
            for task in tqdm(all_tasks_window)
        )
        
        end_loop = time.perf_counter()
        print(f"Janela {i} concluída em {end_loop - start_loop:.2f} segundos.")

        # 2e. Salvar Resultados (para esta janela)
        if not parallel_results:
            print(f"Aviso: Nenhum resultado gerado para a janela {i}.")
            continue
            
        df_results_window = pd.DataFrame(parallel_results)
        df_results_window = df_results_window.set_index(['date', 'ticker'])
        df_results_window.to_csv(output_file)
        print(f"Resultados salvos em {output_file}")

    total_end_time = time.perf_counter()
    print(f"\n--- Processamento de todas as janelas concluído em {total_end_time - total_start_time:.2f} segundos ---")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_calculate_factors_by_window()