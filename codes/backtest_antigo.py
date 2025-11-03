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

    RISK_FACTOR_FLOOR = 1e-5 #Mudança Temporária
    df_factors_clean = df_factors_clean[df_factors_clean['fator_risco'] > RISK_FACTOR_FLOOR]
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

def calculate_performance_metrics(returns_series: pd.Series, 
                                  rf_daily_mean: float, 
                                  var_quantile: float = 0.05) -> dict:
    """
    Calcula um conjunto abrangente de métricas de performance.
    """
    metrics = {}
    epsilon = 1e-10 # Para evitar divisão por zero

    # --- Lidar com retornos vazios ---
    if returns_series.empty or len(returns_series) < 2:
        keys = ['CAGR', 'Volatilidade', 'Sharpe Ratio', 'Max Drawdown', 
                'Sortino Ratio', 'Calmar Ratio', 'VaR Histórico (95%)', 'Rachev Ratio (95%)']
        return {k: 0.0 for k in keys}
        
    n_days = len(returns_series)
    
    # --- Métricas Básicas (CAGR, Vol, Sharpe) ---
    cumulative = (1 + returns_series).cumprod()
    metrics['CAGR'] = (cumulative.iloc[-1] ** (252 / n_days)) - 1
    metrics['Volatilidade'] = returns_series.std() * np.sqrt(252)
    metrics['Sharpe Ratio'] = (returns_series.mean() - rf_daily_mean) / (returns_series.std() + epsilon) * np.sqrt(252)
    
    # --- Max Drawdown ---
    running_max = cumulative.rolling(window=n_days, min_periods=1).max()
    drawdown = (cumulative / running_max) - 1
    metrics['Max Drawdown'] = drawdown.min()
    
    # --- Calmar Ratio ---
    metrics['Calmar Ratio'] = metrics['CAGR'] / (abs(metrics['Max Drawdown']) + epsilon)
    
    # --- Sortino Ratio ---
    # (Calculado usando retornos diários)
    negative_returns = returns_series[returns_series < 0]
    downside_dev = negative_returns.std() # Desvio padrão diário dos retornos negativos
    metrics['Sortino Ratio'] = (returns_series.mean() - rf_daily_mean) / (downside_dev + epsilon) * np.sqrt(252)
    
    # --- VaR (Value at Risk) Histórico ---
    # (Ex: 0.05 para 95% VaR)
    metrics['VaR Histórico (95%)'] = returns_series.quantile(var_quantile)
    
    # --- Rachev Ratio ---
    # (Ratio de Ganhos na Cauda (CVaR Ganho) vs Perdas na Cauda (CVaR Perda))
    var_loss = metrics['VaR Histórico (95%)']
    cvar_loss = returns_series[returns_series <= var_loss].mean() # CVaR (Expected Shortfall)
    
    var_gain = returns_series.quantile(1 - var_quantile) # 95º percentil
    cvar_gain = returns_series[returns_series >= var_gain].mean() # CVaR de Ganhos
    
    metrics['Rachev Ratio (95%)'] = cvar_gain / (abs(cvar_loss) + epsilon)
    
    return metrics
# --- [FIM DA NOVA FUNÇÃO] ---


def run_backtest_simulation(df_retornos: pd.DataFrame,
                            portfolio_details: dict,
                            benchmark_ticker: str,
                            risk_free_ticker: str): 
    """
    O "Simulador": Calcula retornos E GERA MÉTRICAS DE DIAGNÓSTICO DETALHADAS.
    """
    portfolio_returns = []
    diagnostics_data = [] 
    dates = sorted(portfolio_details.keys())

    if not dates:
        print("Aviso: Nenhum portfólio foi construído. Retornando resultados vazios.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    for i in range(len(dates) - 1):
        start_date = dates[i] 
        end_date = dates[i+1] 

        details = portfolio_details[start_date]
        weights = details['weights']
        factors_at_decision = details['factors'] 

        period_mask = (df_retornos.index > start_date) & (df_retornos.index <= end_date)
        period_returns = df_retornos.loc[period_mask]

        if weights.empty or period_returns.empty:
            empty_returns_index = period_returns.index
            if not empty_returns_index.empty:
                 portfolio_returns.append(pd.Series(0.0, index=empty_returns_index))
            if weights.empty:
                 diagnostics_data.append({
                     'rebalance_date': start_date, 'ticker': None, 'weight': 0.0, 
                     'fator_risco': np.nan, 'fator_assimetria': np.nan, 'holding_period_return': 0.0
                 })
            continue

        returns_subset = period_returns.reindex(columns=weights.index)
        daily_portfolio_returns = returns_subset.fillna(0).dot(weights)
        portfolio_returns.append(daily_portfolio_returns)

        # --- Cálculo de Diagnóstico (como antes) ---
        for ticker in weights.index:
            ticker_returns_period = period_returns[ticker].dropna()
            if not ticker_returns_period.empty:
                holding_period_return = (1 + ticker_returns_period).prod() - 1
            else:
                holding_period_return = 0.0 
            
            fator_risco = factors_at_decision.loc[ticker, 'fator_risco']
            fator_assimetria = factors_at_decision.loc[ticker, 'fator_assimetria']
            
            diagnostics_data.append({
                'rebalance_date': start_date, 'ticker': ticker, 'weight': weights.loc[ticker],
                'fator_risco': fator_risco, 'fator_assimetria': fator_assimetria,
                'holding_period_return': holding_period_return
            })
        # --- Fim do Bloco de Diagnóstico ---

    if not portfolio_returns:
        print("Aviso: A série de retornos do portfólio está vazia após a simulação.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    strategy_returns_series = pd.concat(portfolio_returns)
    strategy_returns_series.name = "Estrategia_Duplo_Fator"

    # --- [MODIFICADO] Análise de Resultados ---
    
    # 1. Alinha todos os retornos (Estratégia, Benchmark, CDI)
    benchmark_returns = df_retornos.loc[strategy_returns_series.index, benchmark_ticker]
    cdi_returns = df_retornos.loc[strategy_returns_series.index, risk_free_ticker] # <-- REQUISIÇÃO 1
    rf_diario = cdi_returns.mean() # Usa a média real do período
    
    # 2. Calcula métricas usando a nova função helper
    strategy_metrics = calculate_performance_metrics(strategy_returns_series, rf_diario)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns, rf_diario)
    
    # 3. Monta DataFrame de resultados (com CDI) <-- REQUISIÇÃO 1
    strategy_cumulative = (1 + strategy_returns_series).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    cdi_cumulative = (1 + cdi_returns).cumprod() # <-- REQUISIÇÃO 1
    
    results_df = pd.DataFrame({
        'Estrategia': strategy_returns_series, 
        'Benchmark': benchmark_returns,
        'CDI': cdi_returns, # <-- REQUISIÇÃO 1
        'Estrategia_Acum': strategy_cumulative, 
        'Benchmark_Acum': benchmark_cumulative,
        'CDI_Acum': cdi_cumulative # <-- REQUISIÇÃO 1
    })

    # 4. Imprime Métricas Detalhadas <-- REQUISIÇÃO 2
    print("\n--- RESULTADOS DO BACKTEST ---")
    
    def print_metrics(metrics_dict, title):
        """Função auxiliar para imprimir métricas de forma limpa."""
        print(f"\n{title}:")
        print(f"  - Retorno Anualizado (CAGR): {metrics_dict['CAGR']:.2%}")
        print(f"  - Volatilidade Anualizada:   {metrics_dict['Volatilidade']:.2%}")
        print(f"  - Índice Sharpe:             {metrics_dict['Sharpe Ratio']:.2f}")
        print(f"  - Índice Sortino:            {metrics_dict['Sortino Ratio']:.2f}")
        print(f"  - Max Drawdown:              {metrics_dict['Max Drawdown']:.2%}")
        print(f"  - Calmar Ratio:              {metrics_dict['Calmar Ratio']:.2f}")
        print(f"  - VaR Histórico (95%):       {metrics_dict['VaR Histórico (95%)']:.2%}")
        print(f"  - Rachev Ratio (95%):        {metrics_dict['Rachev Ratio (95%)']:.2f}")

    print_metrics(strategy_metrics, "Métricas da Estratégia (Duplo Fator)")
    print_metrics(benchmark_metrics, f"Métricas do Benchmark ({benchmark_ticker})")

    # --- Diagnóstico (como antes) ---
    df_diagnostics = pd.DataFrame(diagnostics_data)
    if not df_diagnostics.empty:
        df_diagnostics = df_diagnostics.set_index(['rebalance_date', 'ticker'])
        holdings_count = df_diagnostics.reset_index().groupby('rebalance_date')['ticker'].nunique()
        avg_holdings = holdings_count.mean()
        avg_holding_return = df_diagnostics['holding_period_return'].mean()
        print(f"\nNúmero Médio de Ativos na Carteira por Período: {avg_holdings:.1f}")
        print(f"Retorno Médio Simples por Ativo por Período de Holding: {avg_holding_return:.2%}")

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
        'start_date': '2015-12-01', 
        'end_date': '2024-12-31',
        'asymmetry_percentile': 0.9,
        'risk_percentile': 0.5,
        'max_assets': 20,
        'allocation_method': 'equal_weight', # Mantido 'equal_weight'
        'benchmark_ticker': 'IBOV',
        'risk_free_ticker': 'CDI',
        'input_returns_file': 'retornos.csv',
        'input_factors_file': 'fatores_10anos.csv', 
        'output_diagnostics_file': 'diagnostico_detalhado.csv' 
    }

    print("--- ESTÁGIO 2: EXECUÇÃO DO BACKTEST (COM MÉTRICAS AVANÇADAS) ---")
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

    portfolio_details = {}

    print(f"\nIniciando loop de {len(rebalance_dates)} datas para construção do portfólio...")
    start_loop = time.perf_counter()

    # --- 4. Loop Principal (RÁPIDO) ---
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
        
        if not weights.empty:
            factors_subset = df_factors_for_date.loc[weights.index]
            portfolio_details[t_date] = {'weights': weights, 'factors': factors_subset}
        else:
            portfolio_details[t_date] = {'weights': weights, 'factors': pd.DataFrame()}

    end_loop = time.perf_counter()
    print(f"\nLoop de construção de portfólio concluído em {end_loop - start_loop:.2f} segundos.")

    # --- 5. Simulação e Análise de Retornos ---
    print("Iniciando simulação final de retornos e diagnóstico...")
    strategy_returns, results_df, df_diagnostics = run_backtest_simulation(
                                                         df_retornos,
                                                         portfolio_details, 
                                                         config['benchmark_ticker'],
                                                         config['risk_free_ticker'])

    # --- 6. Salvar Resultados ---
    if not results_df.empty:
        results_df.to_csv("resultados_backtest.csv")
        print("Resultados principais (com CDI) salvos em 'resultados_backtest.csv'")

    if not df_diagnostics.empty:
        df_diagnostics.to_csv(config['output_diagnostics_file'])
        print(f"Diagnóstico detalhado salvo em '{config['output_diagnostics_file']}'")

    print("\nBacktest concluído.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_run_backtest()