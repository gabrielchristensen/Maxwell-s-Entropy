import pandas as pd
import numpy as np
import warnings
import os 
import time
from tqdm import tqdm 

# --- BLOCO 1: FUNÇÕES DO BACKTEST (O "MOTOR") ---

# (run_portfolio_construction e calculate_performance_metrics não mudam)
# (Colei-as aqui para o script ficar completo)

def run_portfolio_construction(df_factors: pd.DataFrame, 
                             asymmetry_percentile: float,
                             risk_percentile: float,
                             max_assets: int,
                             allocation_method: str) -> (pd.Series, pd.DataFrame):
    """
    A "Lógica": Aplica o Filtro Duplo e a Alocação,
    com o filtro de quantil para robustez.
    """
    df_factors_clean = df_factors.dropna()

    RISK_FACTOR_QUANTILE_FLOOR = 0.01 
    
    if not df_factors_clean.empty and len(df_factors_clean) > 1:
        risk_floor_value = df_factors_clean['fator_risco'].quantile(RISK_FACTOR_QUANTILE_FLOOR)
        df_factors_clean = df_factors_clean[df_factors_clean['fator_risco'] > risk_floor_value]
    
    if df_factors_clean.empty:
        return pd.Series(dtype=float), df_factors_clean 
    
    assym_threshold = df_factors_clean['fator_assimetria'].quantile(asymmetry_percentile)
    grupo_filtrado_1 = df_factors_clean[df_factors_clean['fator_assimetria'] >= assym_threshold]

    if grupo_filtrado_1.empty:
        return pd.Series(dtype=float), df_factors_clean

    risk_threshold = grupo_filtrado_1['fator_risco'].quantile(risk_percentile)
    portifolio_final_df = grupo_filtrado_1[grupo_filtrado_1['fator_risco'] <= risk_threshold]

    if portifolio_final_df.empty:
        return pd.Series(dtype=float), df_factors_clean

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

    return weights, df_factors_clean


def calculate_performance_metrics(returns_series: pd.Series, 
                                  rf_daily_mean: float, 
                                  var_quantile: float = 0.05) -> dict:
    metrics = {}
    epsilon = 1e-10 
    if returns_series.empty or len(returns_series) < 2:
        keys = ['CAGR', 'Volatilidade', 'Sharpe Ratio', 'Max Drawdown', 
                'Sortino Ratio', 'Calmar Ratio', 'VaR Histórico (95%)', 'Rachev Ratio (95%)']
        return {k: 0.0 for k in keys}
        
    n_days = len(returns_series)
    cumulative = (1 + returns_series).cumprod()
    metrics['CAGR'] = (cumulative.iloc[-1] ** (252 / n_days)) - 1
    metrics['Volatilidade'] = returns_series.std() * np.sqrt(252)
    excess_return_mean = returns_series.mean() - rf_daily_mean
    metrics['Sharpe Ratio'] = excess_return_mean / (returns_series.std() + epsilon) * np.sqrt(252)
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    metrics['Max Drawdown'] = drawdown.min()
    metrics['Calmar Ratio'] = metrics['CAGR'] / (abs(metrics['Max Drawdown']) + epsilon)
    negative_returns = returns_series[returns_series < rf_daily_mean] 
    downside_dev = negative_returns.std() 
    metrics['Sortino Ratio'] = excess_return_mean / (downside_dev + epsilon) * np.sqrt(252)
    metrics['VaR Histórico (95%)'] = returns_series.quantile(var_quantile)
    var_loss = metrics['VaR Histórico (95%)']
    cvar_loss = returns_series[returns_series <= var_loss].mean() 
    var_gain = returns_series.quantile(1 - var_quantile) 
    cvar_gain = returns_series[returns_series >= var_gain].mean() 
    metrics['Rachev Ratio (95%)'] = cvar_gain / (abs(cvar_loss) + epsilon)
    
    return metrics


# [MODIFICADO] A função agora retorna 'summary_metrics_df'
def run_backtest_simulation(df_retornos: pd.DataFrame,
                            portfolio_details: dict,
                            benchmark_universo_details: dict, 
                            benchmark_ticker: str,
                            risk_free_ticker: str) -> (pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame): # <- NOVO RETORNO
    """
    O "Simulador": Calcula retornos E GERA MÉTRICAS DE DIAGNÓSTICO.
    """
    
    def run_simulation_loop(details_dict: dict) -> (pd.Series, pd.DataFrame):
        """Executa o loop de simulação de P&L para um dado dicionário de pesos."""
        portfolio_returns = []
        diagnostics_data = [] 
        dates = sorted(details_dict.keys())

        if not dates:
            return pd.Series(dtype=float), pd.DataFrame()

        for i in range(len(dates) - 1):
            start_date = dates[i] 
            end_date = dates[i+1] 

            details = details_dict[start_date]
            weights = details['weights']
            factors_at_decision = details['factors'] 

            period_mask = (df_retornos.index > start_date) & (df_retornos.index <= end_date)
            period_returns = df_retornos.loc[period_mask]

            if weights.empty or period_returns.empty:
                empty_returns_index = period_returns.index
                if not empty_returns_index.empty:
                     portfolio_returns.append(pd.Series(0.0, index=empty_returns_index))
                if weights.empty and not factors_at_decision.empty: 
                     diagnostics_data.append({
                         'rebalance_date': start_date, 'ticker': None, 'weight': 0.0, 
                         'fator_risco': np.nan, 'fator_assimetria': np.nan, 'holding_period_return': 0.0
                     })
                continue

            returns_subset = period_returns.reindex(columns=weights.index)
            daily_portfolio_returns = returns_subset.fillna(0).dot(weights)
            portfolio_returns.append(daily_portfolio_returns)

            if not factors_at_decision.empty:
                for ticker in weights.index:
                    ticker_returns_period = period_returns[ticker].dropna()
                    holding_period_return = (1 + ticker_returns_period).prod() - 1 if not ticker_returns_period.empty else 0.0
                    
                    fator_risco = factors_at_decision.loc[ticker, 'fator_risco']
                    fator_assimetria = factors_at_decision.loc[ticker, 'fator_assimetria']
                    
                    diagnostics_data.append({
                        'rebalance_date': start_date, 'ticker': ticker, 'weight': weights.loc[ticker],
                        'fator_risco': fator_risco, 'fator_assimetria': fator_assimetria,
                        'holding_period_return': holding_period_return
                    })

        if not portfolio_returns:
            return pd.Series(dtype=float), pd.DataFrame()
        
        final_returns_series = pd.concat(portfolio_returns)
        df_diagnostics = pd.DataFrame(diagnostics_data)
        
        return final_returns_series, df_diagnostics
    # --- FIM DA SUB-FUNÇÃO ---

    # 1. Simula a Estratégia
    strategy_returns_series, df_diagnostics = run_simulation_loop(portfolio_details)
    strategy_returns_series.name = "Estrategia"
    
    # 2. Simula o novo Benchmark-Universo
    benchmark_universo_returns, _ = run_simulation_loop(benchmark_universo_details)
    benchmark_universo_returns.name = "Benchmark_Universo"

    # --- Análise de Resultados ---
    
    # 3. Alinha todos os retornos
    benchmark_ibov_returns = df_retornos.loc[strategy_returns_series.index, benchmark_ticker]
    cdi_returns = df_retornos.loc[strategy_returns_series.index, risk_free_ticker] 
    benchmark_universo_returns = benchmark_universo_returns.reindex(strategy_returns_series.index, fill_value=0.0)
    
    rf_diario = cdi_returns.mean() 
    
    # 4. Calcula métricas para todos
    strategy_metrics = calculate_performance_metrics(strategy_returns_series, rf_diario)
    benchmark_universo_metrics = calculate_performance_metrics(benchmark_universo_returns, rf_diario)
    benchmark_ibov_metrics = calculate_performance_metrics(benchmark_ibov_returns, rf_diario)
    cdi_metrics = calculate_performance_metrics(cdi_returns, rf_diario)
    
    # --- [NOVO] Bloco de criação do DataFrame de Métricas ---
    # 5. Compila métricas em um DataFrame
    summary_metrics_df = pd.DataFrame({
        'Estrategia_Maxwell': strategy_metrics,
        'Benchmark_Universo': benchmark_universo_metrics,
        'Benchmark_IBOV': benchmark_ibov_metrics,
        'CDI': cdi_metrics
    })
    
    # Transpõe o DataFrame para o formato Tabela (como no seu analyzer.py)
    # Colunas = Métricas, Linhas = Estratégias
    summary_metrics_df = summary_metrics_df.T 
    # --- Fim do Novo Bloco ---
    
    # 6. Monta DataFrame de resultados (Time Series)
    strategy_cumulative = (1 + strategy_returns_series).cumprod()
    benchmark_universo_cumulative = (1 + benchmark_universo_returns).cumprod()
    benchmark_ibov_cumulative = (1 + benchmark_ibov_returns).cumprod()
    cdi_cumulative = (1 + cdi_returns).cumprod() 
    
    results_df = pd.DataFrame({
        'Estrategia': strategy_returns_series, 
        'Benchmark_Universo': benchmark_universo_returns, 
        'Benchmark_IBOV': benchmark_ibov_returns, 
        'CDI': cdi_returns,
        'Estrategia_Acum': strategy_cumulative, 
        'Benchmark_Universo_Acum': benchmark_universo_cumulative,
        'Benchmark_IBOV_Acum': benchmark_ibov_cumulative,
        'CDI_Acum': cdi_cumulative
    })

    # 7. Imprime Métricas Detalhadas (ainda útil para o log)
    print("\n--- RESULTADOS DO BACKTEST ---")
    
    def print_metrics(metrics_dict, title):
        print(f"\n{title}:")
        print(f"  - Retorno Anualizado (CAGR): {metrics_dict['CAGR']:.2%}")
        print(f"  - Volatilidade Anualizada:   {metrics_dict['Volatilidade']:.2%}")
        print(f"  - Índice Sharpe:             {metrics_dict['Sharpe Ratio']:.2f}")
        # ... (restante da função print_metrics)
        print(f"  - Índice Sortino:            {metrics_dict['Sortino Ratio']:.2f}")
        print(f"  - Max Drawdown:              {metrics_dict['Max Drawdown']:.2%}")
        print(f"  - Calmar Ratio:              {metrics_dict['Calmar Ratio']:.2f}")
        print(f"  - VaR Histórico (95%):       {metrics_dict['VaR Histórico (95%)']:.2%}")
        print(f"  - Rachev Ratio (95%):        {metrics_dict['Rachev Ratio (95%)']:.2f}")

    print_metrics(strategy_metrics, "Métricas da Estratégia (Duplo Fator)")
    print_metrics(benchmark_universo_metrics, "Métricas do Benchmark-Universo (Grupo de Controle)") 
    print_metrics(benchmark_ibov_metrics, f"Métricas do Benchmark-Mercado ({benchmark_ticker})")
    print_metrics(cdi_metrics, f"Métricas do Custo de Oportunidade ({risk_free_ticker})")

    if not df_diagnostics.empty:
        df_diagnostics = df_diagnostics.set_index(['rebalance_date', 'ticker'])
        holdings_count = df_diagnostics.reset_index().groupby('rebalance_date')['ticker'].nunique()
        avg_holdings = holdings_count.mean()
        avg_holding_return = df_diagnostics['holding_period_return'].mean()
        print(f"\nNúmero Médio de Ativos na Carteira (Estratégia): {avg_holdings:.1f}")
        print(f"Retorno Médio Simples por Ativo (Estratégia): {avg_holding_return:.2%}")

    # Retorna todos os 4 DataFrames
    return strategy_returns_series, results_df, df_diagnostics, summary_metrics_df

# --- BLOCO 2: EXECUÇÃO PRINCIPAL (MAIN - ESTÁGIO 2) ---

def main_run_backtest():
    """
    Orquestrador do Estágio 2: Carrega os ficheiros MESTRES de fatores e retornos
    e simula os resultados.
    """

    # [MODIFICADO] Adicionando novo arquivo de saída para as métricas
    config = {
        'start_date': '2011-01-01', 
        'end_date': '2024-12-31',   
        'asymmetry_percentile': 0.5,
        'risk_percentile': 0.5,
        'max_assets': 20,
        'allocation_method': 'equal_weight',
        'benchmark_ticker': '^BVSP', # Usando ^BVSP como no seu config
        'risk_free_ticker': 'CDI',
        
        'input_returns_file': r'resultados/retornos_master.csv', 
        'input_factors_file': r'fatores/fatores_master.csv', 

        'output_results_file': r'resultados/resultados_backtest.csv', # <- Nome do arquivo de séries
        'output_diagnostics_file': r'resultados/diagnostico_detalhado.csv',
        'output_summary_file': r'resultados/summary_metrics.csv' # <- NOVO ARQUIVO DE SAÍDA
    }

    print("--- ESTÁGIO 2: EXECUÇÃO DO BACKTEST (com Benchmark de Universo) ---")
    print(f"Configuração da Estratégia: {config}")

    # --- 2. Carregar Dados ---
    try:
        df_retornos = pd.read_csv(config['input_returns_file'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo MASTER de retornos '{config['input_returns_file']}' não encontrado.")
        return
        
    try:
        all_factors_df = pd.read_csv(config['input_factors_file'], index_col=[0, 1], parse_dates=[0])
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo MASTER de fatores '{config['input_factors_file']}' não encontrado.")
        return
    print(f"Total de {len(all_factors_df)} linhas de fator carregadas.")
    
    # Renomeia o ticker do IBOV para corresponder ao config
    if config['benchmark_ticker'] not in df_retornos.columns and '^BVSP' in df_retornos.columns:
        df_retornos = df_retornos.rename(columns={'^BVSP': config['benchmark_ticker']})
    elif config['benchmark_ticker'] not in df_retornos.columns:
         print(f"Erro Crítico: Benchmark Ticker '{config['benchmark_ticker']}' não encontrado em {config['input_returns_file']}")
         return

    # --- 3. Obter Datas de Rebalanceamento ---
    all_rebalance_dates = all_factors_df.index.get_level_values('date').unique().sort_values()
    rebalance_dates = all_rebalance_dates[
        (all_rebalance_dates >= pd.Timestamp(config['start_date'])) &
        (all_rebalance_dates <= pd.Timestamp(config['end_date']))
    ]
    
    if len(rebalance_dates) == 0:
        print("Erro: Nenhuma data de rebalanceamento encontrada no período de simulação.")
        return

    portfolio_details = {} 
    benchmark_universo_details = {} 

    print(f"\nIniciando loop de {len(rebalance_dates)} datas para construção dos portfólios...")
    start_loop = time.perf_counter()

    # --- 4. Loop Principal (MODIFICADO) ---
    for t_date in tqdm(rebalance_dates): 
        try:
            df_factors_for_date = all_factors_df.loc[t_date]
        except KeyError:
            portfolio_details[t_date] = {'weights': pd.Series(dtype=float), 'factors': pd.DataFrame()}
            benchmark_universo_details[t_date] = {'weights': pd.Series(dtype=float), 'factors': pd.DataFrame()}
            continue
        
        # 1. CHAMA A SUA FUNÇÃO ORIGINAL (que faz tudo)
        weights_strategy, df_factors_clean = run_portfolio_construction(
                                                   df_factors_for_date, 
                                                   config['asymmetry_percentile'],
                                                   config['risk_percentile'],
                                                   config['max_assets'],
                                                   config['allocation_method'])
        
        # 2. Armazena os detalhes da ESTRATÉGIA
        if not weights_strategy.empty:
            factors_subset = df_factors_clean.loc[weights_strategy.index]
            portfolio_details[t_date] = {'weights': weights_strategy, 'factors': factors_subset}
        else:
            portfolio_details[t_date] = {'weights': weights_strategy, 'factors': pd.DataFrame()}

        # 3. CALCULA O BENCHMARK-UNIVERSO (O "GRUPO DE CONTROLE")
        if not df_factors_clean.empty:
            n_assets_bench = len(df_factors_clean)
            weights_bench = pd.Series(1.0 / n_assets_bench, index=df_factors_clean.index)
            benchmark_universo_details[t_date] = {'weights': weights_bench, 'factors': pd.DataFrame()}
        else:
            benchmark_universo_details[t_date] = {'weights': pd.Series(dtype=float), 'factors': pd.DataFrame()}
        

    end_loop = time.perf_counter()
    print(f"\nLoop de construção de portfólio concluído em {end_loop - start_loop:.2f} segundos.")

    # --- 5. Simulação e Análise de Retornos ---
    print("Iniciando simulação final de retornos e diagnóstico...")
    # [MODIFICADO] Recebe 4 DataFrames
    strategy_returns, results_df, df_diagnostics, summary_metrics_df = run_backtest_simulation(
                                                         df_retornos,
                                                         portfolio_details, 
                                                         benchmark_universo_details, 
                                                         config['benchmark_ticker'],
                                                         config['risk_free_ticker'])

    # --- 6. Salvar Resultados ---
    # [MODIFICADO] Salva os 3 arquivos
    if not results_df.empty:
        results_df.to_csv(config['output_results_file'])
        print(f"Resultados (séries temporais) salvos em '{config['output_results_file']}'")

    if not df_diagnostics.empty:
        df_diagnostics.to_csv(config['output_diagnostics_file'])
        print(f"Diagnóstico detalhado salvo em '{config['output_diagnostics_file']}'")

    if not summary_metrics_df.empty:
        summary_metrics_df.to_csv(config['output_summary_file'])
        print(f"Métricas resumidas salvas em '{config['output_summary_file']}'")

    print("\nBacktest concluído.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_run_backtest()