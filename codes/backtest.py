import pandas as pd
import numpy as np
import warnings
import os 
import time
from tqdm import tqdm 
from toolbox import *


def run_portfolio_construction(df_factors: pd.DataFrame,
                             asymmetry_percentile: float,
                             risk_percentile: float,
                             max_assets: int,
                             allocation_method: str) -> pd.Series:
    """
    A "Lógica": Aplica o Filtro Duplo e a Alocação, usando os fatores dados.
    Retorna a série de pesos.
    """
    df_factors_clean = df_factors.dropna()

    if df_factors_clean.empty:
        return pd.Series(dtype=float)

    assym_threshold = df_factors_clean['fator_assimetria'].quantile(asymmetry_percentile)
    grupo_filtrado_1 = df_factors_clean[df_factors_clean['fator_assimetria'] >= assym_threshold]

    if grupo_filtrado_1.empty:
        return pd.Series(dtype=float)

    risk_threshold = grupo_filtrado_1['fator_risco'].quantile(risk_percentile)
    portifolio_final_df = grupo_filtrado_1[grupo_filtrado_1['fator_risco'] <= risk_threshold]

    if portifolio_final_df.empty:
        return pd.Series(dtype=float)

    if max_assets > 0 and len(portifolio_final_df) > max_assets:
        portifolio_final_df = portifolio_final_df.nsmallest(max_assets, 'fator_risco')

    if allocation_method == 'inverse_entropy':
        epsilon_risk = 1e-10
        inv_risk = 1.0 / (portifolio_final_df['fator_risco'] + epsilon_risk)
        weights = inv_risk / inv_risk.sum()
    elif allocation_method == 'equal_weight':
        n_assets = len(portifolio_final_df)
        weights = pd.Series(1.0 / n_assets, index=portifolio_final_df.index)
    else:
        raise ValueError(f"Método de alocação '{allocation_method}' desconhecido.")

    return weights

def run_backtest_simulation(df_retornos: pd.DataFrame,
                            # [MODIFICADO] Recebe detalhes completos (pesos + fatores)
                            portfolio_details: dict,
                            benchmark_ticker: str,
                            risk_free_ticker: str): # <- Retorna df_diagnostics
    """
    O "Simulador": Calcula retornos da estratégia E GERA DADOS DE DIAGNÓSTICO.
    """
    portfolio_returns = []
    diagnostics_data = [] # <- Lista para guardar dados de diagnóstico
    dates = sorted(portfolio_details.keys())

    if not dates:
        print("Aviso: Nenhum portfólio foi construído. Retornando resultados vazios.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # Itera pelos períodos ENTRE as datas de rebalanceamento
    for i in range(len(dates) - 1):
        start_date = dates[i] # Data da decisão
        end_date = dates[i+1] # Próxima data de decisão

        # Pega os detalhes (pesos e fatores) definidos em start_date
        details = portfolio_details[start_date]
        weights = details['weights']
        factors_at_decision = details['factors'] # Fatores dos ativos selecionados

        # Seleciona os retornos DENTRO do período de holding
        period_mask = (df_retornos.index > start_date) & (df_retornos.index <= end_date)
        period_returns = df_retornos.loc[period_mask]

        # Se não houver pesos ou não houver dias de retorno no período
        if weights.empty or period_returns.empty:
            empty_returns_index = period_returns.index
            if not empty_returns_index.empty:
                 portfolio_returns.append(pd.Series(0.0, index=empty_returns_index))
            # Adiciona entrada vazia ao diagnóstico se o portfólio estava vazio
            if weights.empty:
                 diagnostics_data.append({
                     'rebalance_date': start_date,
                     'ticker': None, 'weight': 0.0, 'fator_risco': np.nan,
                     'fator_assimetria': np.nan, 'holding_period_return': 0.0
                 })
            continue

        # Calcula o retorno diário do portfólio (Produto escalar)
        returns_subset = period_returns.reindex(columns=weights.index)
        daily_portfolio_returns = returns_subset.fillna(0).dot(weights)
        portfolio_returns.append(daily_portfolio_returns)

        # --- [NOVO] Cálculo de Diagnóstico para este período ---
        for ticker in weights.index:
            # Pega os retornos APENAS deste ticker no período
            ticker_returns_period = period_returns[ticker].dropna()

            # Calcula o retorno geométrico TOTAL do ticker no período
            if not ticker_returns_period.empty:
                holding_period_return = (1 + ticker_returns_period).prod() - 1
            else:
                holding_period_return = 0.0 # Ou NaN se preferir

            # Pega os fatores deste ticker na data da decisão
            fator_risco = factors_at_decision.loc[ticker, 'fator_risco']
            fator_assimetria = factors_at_decision.loc[ticker, 'fator_assimetria']

            # Guarda os dados
            diagnostics_data.append({
                'rebalance_date': start_date,
                'ticker': ticker,
                'weight': weights.loc[ticker],
                'fator_risco': fator_risco,
                'fator_assimetria': fator_assimetria,
                'holding_period_return': holding_period_return
            })
        # --- Fim do Bloco de Diagnóstico ---

    if not portfolio_returns:
        print("Aviso: A série de retornos do portfólio está vazia após a simulação.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    strategy_returns_series = pd.concat(portfolio_returns)
    strategy_returns_series.name = "Estrategia_Duplo_Fator"

    # --- Análise de Resultados (como antes) ---
    benchmark_returns = df_retornos.loc[strategy_returns_series.index][benchmark_ticker]
    strategy_cumulative = (1 + strategy_returns_series).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    results_df = pd.DataFrame({
        'Estrategia': strategy_returns_series, 'Benchmark': benchmark_returns,
        'Estrategia_Acum': strategy_cumulative, 'Benchmark_Acum': benchmark_cumulative
    })
    rf_diario = df_retornos.loc[strategy_returns_series.index, risk_free_ticker].mean()
    n_days_strategy = len(strategy_returns_series)
    if n_days_strategy == 0:
        return strategy_returns_series, results_df, pd.DataFrame(diagnostics_data) # Retorna o que tiver
    strategy_cagr = (strategy_cumulative.iloc[-1] ** (252 / n_days_strategy)) - 1
    benchmark_cagr = (benchmark_cumulative.iloc[-1] ** (252 / n_days_strategy)) - 1
    strategy_vol = strategy_returns_series.std() * np.sqrt(252)
    benchmark_vol = benchmark_returns.std() * np.sqrt(252)
    epsilon = 1e-10
    strategy_sharpe = (strategy_returns_series.mean() - rf_diario) / (strategy_returns_series.std() + epsilon) * np.sqrt(252)
    benchmark_sharpe = (benchmark_returns.mean() - rf_diario) / (benchmark_returns.std() + epsilon) * np.sqrt(252)

    # Imprime Métricas (como antes)
    print("\n--- RESULTADOS DO BACKTEST ---")
    print("\nMétricas da Estratégia (Duplo Fator):")
    print(f"Retorno Anualizado (CAGR): {strategy_cagr:.2%}")
    print(f"Volatilidade Anualizada:   {strategy_vol:.2%}")
    print(f"Índice Sharpe:            {strategy_sharpe:.2f}")
    print(f"\nMétricas do Benchmark ({benchmark_ticker}):")
    print(f"Retorno Anualizado (CAGR): {benchmark_cagr:.2%}")
    print(f"Volatilidade Anualizada:   {benchmark_vol:.2%}")
    print(f"Índice Sharpe:            {benchmark_sharpe:.2f}")

    # --- [NOVO] Converte a lista de diagnóstico num DataFrame ---
    df_diagnostics = pd.DataFrame(diagnostics_data)
    # Define o índice para facilitar a análise
    if not df_diagnostics.empty:
        df_diagnostics = df_diagnostics.set_index(['rebalance_date', 'ticker'])

    # --- [NOVO] Calcula e imprime métricas de diagnóstico adicionais ---
    if not df_diagnostics.empty:
        # Conta o número de ativos por data (reset_index transforma o índice em coluna)
        holdings_count = df_diagnostics.reset_index().groupby('rebalance_date')['ticker'].nunique()
        avg_holdings = holdings_count.mean()
        print(f"\nNúmero Médio de Ativos na Carteira por Período: {avg_holdings:.1f}")

        # Calcula o retorno médio simples das ações selecionadas POR PERÍODO
        avg_holding_return = df_diagnostics['holding_period_return'].mean()
        print(f"Retorno Médio Simples por Ativo por Período de Holding: {avg_holding_return:.2%}")


    # Retorna a série de retornos, o df de resultados E o df de diagnóstico
    return strategy_returns_series, results_df, df_diagnostics

# --- BLOCO 2: EXECUÇÃO PRINCIPAL (MAIN - ESTÁGIO 2) ---

def main_run_backtest():
    """
    Orquestrador do Estágio 2: Carrega fatores e retornos,
    testa a lógica da estratégia e simula os resultados.
    """

    # --- 1. Definição de Parâmetros da ESTRATÉGIA ---
    config = {
        'rebal_frequency': 'BM',
        'start_date': '2015-12-01', # <-- Ajuste para coincidir com seu teste anterior
        'end_date': '2024-12-31',
        'asymmetry_percentile': 0.5,
        'risk_percentile': 0.5,
        'max_assets': 20,
        'allocation_method': 'inverse_entropy',
        'benchmark_ticker': 'IBOV',
        'risk_free_ticker': 'CDI',
        'input_returns_file': 'retornos.csv',
        'input_factors_file': 'fatores_10anos.csv', # <-- Ajuste para seu nome
        'output_diagnostics_file': 'diagnostico_detalhado.csv' # <-- Nome do novo arquivo
    }

    print("--- ESTÁGIO 2: EXECUÇÃO DO BACKTEST (COM DIAGNÓSTICO) ---")
    print(f"Configuração da Estratégia:")
    print(f"  - Frequência: {config['rebal_frequency']}")
    print(f"  - Período: {config['start_date']} a {config['end_date']}")
    print(f"  - Filtro Assimetria (>) : {config['asymmetry_percentile']:.0%}")
    print(f"  - Filtro Risco (<)      : {config['risk_percentile']:.0%}")
    print(f"  - Máximo de Ativos      : {config['max_assets'] if config['max_assets'] > 0 else 'Sem Limite'}")
    print(f"  - Alocação              : {config['allocation_method']}")

    # --- 2. Carregar Dados ---
    print(f"\nCarregando retornos de '{config['input_returns_file']}'...")
    try:
        df_retornos = pd.read_csv(config['input_returns_file'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de retornos '{config['input_returns_file']}' não encontrado.")
        return

    print(f"Carregando fatores pré-calculados de '{config['input_factors_file']}'...")
    try:
        all_factors_df = pd.read_csv(config['input_factors_file'], index_col=[0, 1], parse_dates=[0])
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo de fatores '{config['input_factors_file']}' não encontrado.")
        return

    # --- 3. Obter Datas de Rebalanceamento ---
    rebalance_dates = get_rebalance_dates(df_retornos,
                                          config['start_date'],
                                          config['end_date'],
                                          config['rebal_frequency'])

    if len(rebalance_dates) == 0:
        print("Erro: Nenhuma data de rebalanceamento encontrada no período de simulação.")
        return

    # [MODIFICADO] Guarda pesos E fatores
    portfolio_details = {}

    print(f"\nIniciando loop de {len(rebalance_dates)} datas para construção do portfólio...")
    start_loop = time.perf_counter()

    # --- 4. Loop Principal (RÁPIDO: lookup, construção, guarda detalhes) ---
    for t_date in tqdm(rebalance_dates):

        try:
            df_factors_for_date = all_factors_df.loc[t_date]
        except KeyError:
            portfolio_details[t_date] = {'weights': pd.Series(dtype=float), 'factors': pd.DataFrame()}
            continue

        weights = run_portfolio_construction(df_factors_for_date,
                                           config['asymmetry_percentile'],
                                           config['risk_percentile'],
                                           config['max_assets'],
                                           config['allocation_method'])

        # Guarda os pesos E os fatores dos ativos selecionados
        if not weights.empty:
            factors_subset = df_factors_for_date.loc[weights.index]
            portfolio_details[t_date] = {'weights': weights, 'factors': factors_subset}
        else:
            portfolio_details[t_date] = {'weights': weights, 'factors': pd.DataFrame()}


    end_loop = time.perf_counter()
    print(f"\nLoop de construção de portfólio concluído em {end_loop - start_loop:.2f} segundos.")

    # --- 5. Simulação e Análise de Retornos (Agora com Diagnóstico) ---
    print("Iniciando simulação final de retornos e diagnóstico...")
    strategy_returns, results_df, df_diagnostics = run_backtest_simulation(
                                                         df_retornos,
                                                         portfolio_details, # Passa os detalhes completos
                                                         config['benchmark_ticker'],
                                                         config['risk_free_ticker'])

    # --- 6. Salvar Resultados / Plotar ---
    if not results_df.empty:
        results_df.to_csv("resultados_backtest.csv")
        print("Resultados principais salvos em 'resultados_backtest.csv'")

    # --- [NOVO] Salvar Diagnóstico Detalhado ---
    if not df_diagnostics.empty:
        df_diagnostics.to_csv(config['output_diagnostics_file'])
        print(f"Diagnóstico detalhado salvo em '{config['output_diagnostics_file']}'")


    print("\nBacktest concluído.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_run_backtest()