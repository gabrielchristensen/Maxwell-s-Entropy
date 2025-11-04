import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
import warnings
from tqdm import tqdm

# --- CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')

# --- PALETA DE CORES "MAXWELL" ---
COLOR_STRATEGY = "#67cdfc"
COLOR_BENCHMARK_IBOV = "#464141"
COLOR_BENCHMARK_UNIVERSO = "#416d8a"
COLOR_CDI = "#F70909"
COLOR_FILL_STRATEGY = '#a9c4db'
# ------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100
pd.options.display.float_format = '{:.4f}'.format

# (calculate_performance_metrics não muda)
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

# --- [MODIFICADO] De Decil (10) para Quartil (4) ---
def calculate_quartile_performance(df_factors: pd.DataFrame, 
                                   df_returns: pd.DataFrame, 
                                   factor_name: str):
    """
    Calcula a performance (retorno, sharpe) para 4 quartis
    de um determinado fator.
    """
    print(f"\nIniciando análise de quartis para: {factor_name}...")
    rebal_dates = df_factors.index.get_level_values('date').unique().sort_values()
    
    # [MODIFICADO] De D1-D10 para Q1-Q4
    quartile_daily_returns_dict = {f'Q{i}': [] for i in range(1, 5)}
    daily_rf = df_returns['CDI'].mean() if 'CDI' in df_returns.columns else 0.0

    for i in tqdm(range(len(rebal_dates) - 1), desc=f"Testando {factor_name}"):
        t_date = rebal_dates[i]
        t_next_date = rebal_dates[i+1]
        
        factors_today = df_factors.loc[t_date].dropna(subset=[factor_name])
        
        # [MODIFICADO] Precisa de pelo menos 4 ativos para 4 quartis
        if factors_today.empty or len(factors_today) < 4:
            continue
            
        try:
            # [MODIFICADO] pd.qcut para 4 grupos
            quartile_labels = [f'Q{i}' for i in range(1, 5)]
            factors_today['quartile'] = pd.qcut(factors_today[factor_name], 4, labels=quartile_labels, duplicates='drop')
        except ValueError:
            continue
            
        returns_period = df_returns.loc[(df_returns.index > t_date) & (df_returns.index <= t_next_date)]
        if returns_period.empty:
            continue
            
        for q_name in quartile_daily_returns_dict.keys():
            tickers_in_q = factors_today[factors_today['quartile'] == q_name].index
            valid_tickers = [t for t in tickers_in_q if t in returns_period.columns]
            
            if valid_tickers:
                ret_q_daily = returns_period[valid_tickers].mean(axis=1)
                quartile_daily_returns_dict[q_name].append(ret_q_daily)

    if not any(quartile_daily_returns_dict.values()):
        print(f"Erro: Nenhum retorno de quartil pôde ser calculado para {factor_name}.")
        return pd.Series(), pd.DataFrame(), pd.Series()

    df_quartile_daily_list = []
    for q_name, returns_list in quartile_daily_returns_dict.items():
        if returns_list:
            df_q = pd.concat(returns_list)
            df_q.name = q_name
            df_quartile_daily_list.append(df_q)
        else:
            df_quartile_daily_list.append(pd.Series(name=q_name, dtype=float))
            
    df_quartile_daily = pd.concat(df_quartile_daily_list, axis=1).fillna(0)

    avg_monthly_ret = df_quartile_daily.resample('M').apply(lambda x: (1 + x).prod() - 1).mean()
    df_quartile_cum = (1 + df_quartile_daily).cumprod()
    mean_excess_ret = df_quartile_daily.mean() - daily_rf
    std_ret = df_quartile_daily.std()
    sharpe_per_q = (mean_excess_ret / (std_ret + 1e-10)) * np.sqrt(252)
    
    return avg_monthly_ret, df_quartile_cum, sharpe_per_q


# (calculate_turnover não muda)
def calculate_turnover(df_diag: pd.DataFrame) -> pd.Series:
    print("Calculando turnover da carteira...")
    holdings = df_diag.reset_index().groupby('rebalance_date')['ticker'].apply(set)
    holdings_prev = holdings.shift(1)
    df_turnover = pd.DataFrame({'current': holdings, 'previous': holdings_prev}).dropna()
    
    def calc_turnover_row(row):
        sold = row['previous'] - row['current']
        bought = row['current'] - row['previous']
        if len(row['previous']) == 0:
            return 1.0 
        return (len(sold) + len(bought)) / (2 * len(row['previous']))

    turnover_series = df_turnover.apply(calc_turnover_row, axis=1)
    turnover_series.name = "Monthly Turnover"
    return turnover_series


# --- FUNÇÕES DE ANÁLISE (IMPRESSÃO) ---

# [BUG CORRIGIDO] A função agora usa 'alpha_fator' corretamente
def print_overall_analysis(df_results):
    """Fase 1: Imprime a análise de performance agregada."""
    print("\n\n" + "="*80)
    print(" FASE 1: ANÁLISE DE PERFORMANCE AGREGADA (O 'QUÊ?')")
    print("="*80)

    rf_daily = df_results['CDI'].mean()
    strat_metrics = calculate_performance_metrics(df_results['Estrategia'], rf_daily)
    universo_metrics = calculate_performance_metrics(df_results['Benchmark_Universo'], rf_daily)
    ibov_metrics = calculate_performance_metrics(df_results['Benchmark_IBOV'], rf_daily)
    cdi_metrics = calculate_performance_metrics(df_results['CDI'], rf_daily)

    print(f"\n--- MÉTRICAS (Estratégia 'Maxwell') ---")
    print(f"CAGR: {strat_metrics['CAGR']:.2%} | Volatilidade: {strat_metrics['Volatilidade']:.2%} | Sharpe: {strat_metrics['Sharpe Ratio']:.2f}")
    
    print(f"\n--- MÉTRICAS (Benchmark-Universo / Grupo de Controle) ---")
    print(f"CAGR: {universo_metrics['CAGR']:.2%} | Volatilidade: {universo_metrics['Volatilidade']:.2%} | Sharpe: {universo_metrics['Sharpe Ratio']:.2f}")

    print(f"\n--- MÉTRICAS (Benchmark-Mercado / IBOV) ---")
    print(f"CAGR: {ibov_metrics['CAGR']:.2%} | Volatilidade: {ibov_metrics['Volatilidade']:.2%} | Sharpe: {ibov_metrics['Sharpe Ratio']:.2f}")
    
    print(f"\n--- MÉTRICAS (Custo de Oportunidade / CDI) ---")
    print(f"CAGR: {cdi_metrics['CAGR']:.2%} | Volatilidade: {cdi_metrics['Volatilidade']:.2%}")
    
    print("\n" + "-"*80)
    print(" VEREDITO DA ANÁLISE DE ALFA (Retorno)")
    print("-" * 80)
    
    alpha_total = strat_metrics['CAGR'] - ibov_metrics['CAGR']
    alpha_universo = universo_metrics['CAGR'] - ibov_metrics['CAGR']
    
    # --- [BUG CORRIGIDO] ---
    # O seu código original estava: 
    # alpha_fator = strat_metrics['CAGR'] - ibov_metrics['CAGR']
    # A versão correta (Estratégia vs. Universo) está abaixo:
    alpha_fator = strat_metrics['CAGR'] - universo_metrics['CAGR']
    # --- FIM DA CORREÇÃO ---

    print(f"Alpha Total (Estratégia vs. IBOV): {alpha_total:+.2%}")
    print(f"  ↳ Alpha de Construção (Universo vs. IBOV): {alpha_universo:+.2%}")
    print(f"  ↳ Alpha Verdadeiro (Estratégia vs. Universo): {alpha_fator:+.2%}")

    if alpha_fator > 0:
        print(f"\n✅ TESE 2 (ASSIMETRIA) VALIDADA: Os fatores de Entropia geraram {alpha_fator:+.2%}% a.a. de retorno")
        print(f"   acima do grupo de controle (Benchmark-Universo).")
    else:
        print(f"\n❌ TESE 2 (ASSIMETRIA) INVÁLIDA: Os fatores de Entropia não geraram Alpha.")
        print(f"   Todo o ganho de {alpha_total:+.2%}% veio da construção do universo.")
        
    print("\n" + "-"*80)
    print(" VEREDITO DA ANÁLISE DE RISCO (Volatilidade)")
    print("-" * 80)
    
    vol_reducao_fator = universo_metrics['Volatilidade'] - strat_metrics['Volatilidade']

    if vol_reducao_fator > 0:
        print(f"✅ TESE 1 (RISCO) VALIDADA: O filtro 'fator_risco' (baixa entropia) reduziu a volatilidade em {vol_reducao_fator:.2f}%")
        print(f"   em relação ao Benchmark-Universo (ambos com mesmo N e peso).")
    else:
        print(f"❌ TESE 1 (RISCO) INVÁLIDA: O filtro 'fator_risco' aumentou a volatilidade em {abs(vol_reducao_fator):.2f}%")
        print(f"   em relação ao Benchmark-Universo.")
        
    print("\n--- VEREDITO (RISCO DE CAUDA) ---")
    if strat_metrics['Max Drawdown'] < ibov_metrics['Max Drawdown']:
         print(f"✅ BÔNUS DEFENSIVO: A estratégia foi mais defensiva, com um Drawdown menor ({strat_metrics['Max Drawdown']:.2%}) que o IBOV ({ibov_metrics['Max Drawdown']:.2%}).")
    else:
        print(f"⚠️ RISCO CONFIRMADO: A estratégia teve um Drawdown pior ({strat_metrics['Max Drawdown']:.2%}) que o IBOV ({ibov_metrics['Max Drawdown']:.2%}).")


# --- [MODIFICADO] De Decil (10) para Quartil (4) ---
def print_quartile_analysis_returns(avg_ret: pd.Series, factor_name: str):
    print("\n\n" + "="*80)
    print(f" FASE 3: VALIDAÇÃO DO FATOR '{factor_name}' (FOCO: RETORNO)")
    print("="*80)
    print("Retorno Médio Mensal por Quartil (Q1 = Baixo, Q4 = Alto):")
    print(avg_ret.to_string())
    
    # [MODIFICADO] De D10-D1 para Q4-Q1
    spread = avg_ret['Q4'] - avg_ret['Q1']
    print(f"\nSpread (Q4 - Q1): {spread:.4%}")
    if spread > 0:
        print(f"✅ TESE VALIDADA: O quartil superior (Q4) rendeu mais que o inferior (Q1).")
    else:
        print(f"❌ TESE INVÁLIDA: O quartil superior (Q4) rendeu menos que o inferior (Q1).")
    
    monotonic = (avg_ret.diff().dropna() > 0).all()
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O retorno aumentou a cada quartil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear, mas o spread Q4-Q1 é o que importa.")

# --- [MODIFICADO] De Decil (10) para Quartil (4) ---
def print_quartile_analysis_sharpe(sharpe_q: pd.Series, factor_name: str):
    print("\n\n" + "="*80)
    print(f" FASE 3: VALIDAÇÃO DO FATOR '{factor_name}' (FOCO: SHARPE)")
    print("="*80)
    print("Sharpe Ratio Anualizado por Quartil (Q1 = Baixo, Q4 = Alto):")
    print(sharpe_q.to_string())
    
    # [MODIFICADO] De D1-D10 para Q1-Q4
    spread = sharpe_q['Q1'] - sharpe_q['Q4']
    print(f"\nSpread (Q1 - Q4): {spread:.2f}")
    if spread > 0:
        print(f"✅ TESE VALIDADA: O quartil inferior (Q1, baixo risco) teve Sharpe maior que o superior (Q4, alto risco).")
    else:
        print(f"❌ TESE INVÁLIDA: O quartil inferior (Q1) teve Sharpe pior que o superior (Q4).")
    
    monotonic = (sharpe_q.diff().dropna() < 0).all()
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O Sharpe diminuiu a cada quartil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear.")

# (print_implementation_analysis não muda)
def print_implementation_analysis(turnover: pd.Series, df_diag: pd.DataFrame):
    print("\n\n" + "="*80)
    print(" FASE 4: ANÁLISE DE IMPLEMENTAÇÃO (O 'COMO?')")
    print("="*80)
    holdings_count = df_diag.reset_index().groupby('rebalance_date')['ticker'].nunique()
    mean_holdings = holdings_count.mean()
    mean_turnover = turnover.mean()
    mean_hit_rate = df_diag['holding_period_return'].mean()
    print(f"Número Médio de Ativos na Carteira: {mean_holdings:.1f}")
    if mean_holdings < 10:
        print(f"   ↳ ⚠️ ALERTA DE CONCENTRAÇÃO: A média de ativos é muito baixa.")
    else:
        print(f"   ↳ ✅ Nível de diversificação saudável (próximo do limite de 20).")
    print(f"\nTurnover Médio Mensal: {mean_turnover:.1%}")
    if mean_turnover > 0.4: 
        print(f"   ↳ ⚠️ ALERTA DE CUSTOS: O giro da carteira é alto.")
    else:
        print(f"   ↳ ✅ Giro da carteira gerenciável.")
    print(f"\nRetorno Médio por 'Aposta' (Hit Rate): {mean_hit_rate:.2%}")
    if mean_hit_rate > 0:
         print(f"   ↳ ✅ Positivo. Em média, as ações selecionadas tiveram performance positiva.")
    else:
         print(f"   ↳ ❌ Negativo. A seleção de fatores está, em média, a escolher ações perdedoras.")


# --- FUNÇÕES DE PLOTAGEM (MODIFICADAS) ---

# (plot_cumulative_returns, plot_drawdowns, plot_rolling_sharpe não mudam)
def plot_cumulative_returns(df_results: pd.DataFrame, output_dir: str):
    print("Plotando: 1. Retorno Acumulado...")
    plt.figure()
    
    df_results['Estrategia_Acum'].plot(label='Estratégia "Maxwell" (Fatores)', color=COLOR_STRATEGY, linewidth=2.5, zorder=4)
    df_results['Benchmark_Universo_Acum'].plot(label='Benchmark-Universo (Controle)', color=COLOR_BENCHMARK_UNIVERSO, linestyle='-', linewidth=1.5, zorder=3)
    df_results['Benchmark_IBOV_Acum'].plot(label='IBOV (Mercado)', color=COLOR_BENCHMARK_IBOV, linestyle='--', linewidth=1.5, zorder=2)
    #df_results['CDI_Acum'].plot(label='CDI (Custo de Oportunidade)', color=COLOR_CDI, linestyle=':', linewidth=1.5, zorder=1)
    
    plt.title('Performance Acumulada da Estratégia vs. Benchmark', fontsize=16)
    plt.ylabel('Retorno Acumulado (Percentual)')
    plt.xlabel('Data')
    plt.yscale('log')
    
    ax = plt.gca()
    def log_percent_formatter(x, pos):
        percent_value = (x - 1) * 100
        return f"{percent_value:+.0f}%"
    formatter = FuncFormatter(log_percent_formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)
    
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_performance_acumulada.png"))
    plt.close()

def plot_drawdowns(df_results: pd.DataFrame, output_dir: str):
    print("Plotando: 2. Drawdowns...")
    
    def calc_drawdown(cum_returns):
        running_max = cum_returns.cummax()
        return (cum_returns / running_max) - 1

    dd_strategy = calc_drawdown(df_results['Estrategia_Acum'])
    dd_universo = calc_drawdown(df_results['Benchmark_Universo_Acum'])
    dd_benchmark_ibov = calc_drawdown(df_results['Benchmark_IBOV_Acum'])
    
    plt.figure()
    dd_strategy.plot(label='Estratégia "Maxwell"', color=COLOR_STRATEGY, kind='area', alpha=0.5, zorder=3)
    dd_universo.plot(label='Benchmark-Universo', color=COLOR_BENCHMARK_UNIVERSO, linestyle='-', linewidth=1.5, zorder=2)
    dd_benchmark_ibov.plot(label='IBOV (Mercado)', color=COLOR_BENCHMARK_IBOV, linestyle='--', linewidth=1.5, zorder=1)
    
    plt.title('Drawdowns da Estratégia vs. Benchmarks', fontsize=16)
    plt.ylabel('Queda Percentual do Pico')
    plt.xlabel('Data')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_drawdowns.png"))
    plt.close()

def plot_rolling_sharpe(df_results: pd.DataFrame, output_dir: str, window: int = 252):
    print("Plotando: 3. Sharpe Ratio Rolante...")
    
    excess_returns_strat = df_results['Estrategia'] - df_results['CDI']
    excess_returns_universo = df_results['Benchmark_Universo'] - df_results['CDI']
    excess_returns_ibov = df_results['Benchmark_IBOV'] - df_results['CDI']
    
    rolling_sharpe_strat = (excess_returns_strat.rolling(window).mean() / (excess_returns_strat.rolling(window).std() + 1e-10)) * np.sqrt(252)
    rolling_sharpe_universo = (excess_returns_universo.rolling(window).mean() / (excess_returns_universo.rolling(window).std() + 1e-10)) * np.sqrt(252)
    rolling_sharpe_ibov = (excess_returns_ibov.rolling(window).mean() / (excess_returns_ibov.rolling(window).std() + 1e-10)) * np.sqrt(252)
    
    plt.figure()
    rolling_sharpe_strat.plot(label='Estratégia "Maxwell"', color=COLOR_STRATEGY, linewidth=2.5, zorder=3)
    rolling_sharpe_universo.plot(label='Benchmark-Universo (Controle)', color=COLOR_BENCHMARK_UNIVERSO, linestyle='-', linewidth=1.5, zorder=2)
    rolling_sharpe_ibov.plot(label='IBOV (Mercado)', color=COLOR_BENCHMARK_IBOV, linestyle='--', linewidth=1.5, zorder=1)
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    
    plt.title(f'Sharpe Ratio Rolante ({window} dias)', fontsize=16)
    plt.ylabel('Sharpe Ratio Anualizado')
    plt.xlabel('Data')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_rolling_sharpe.png"))
    plt.close()


# --- [MODIFICADO] De Decil (10) para Quartil (4) ---
def plot_quartile_returns(avg_ret: pd.Series, cum_ret: pd.DataFrame, title: str, output_dir: str, filename: str):
    print(f"Plotando: {filename}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 2]})
    
    avg_ret.plot(kind='bar', ax=ax1, color=COLOR_STRATEGY, alpha=0.7)
    ax1.set_title(f'Retorno Médio Mensal por Quartil - {title}', fontsize=14)
    ax1.set_ylabel('Retorno Médio Mensal')
    ax1.set_xlabel('Quartil (Q1 = Baixo, Q4 = Alto)') # Modificado
    ax1.tick_params(axis='x', rotation=0)
    
    cum_ret.plot(ax=ax2, linewidth=2, colormap='Blues') # Modificado
    ax2.set_title(f'Performance Acumulada por Quartil - {title}', fontsize=14)
    ax2.set_ylabel('Retorno Acumulado (Base 1)')
    ax2.set_xlabel('Data')
    ax2.set_yscale('log')
    ax2.legend(title='Quartil') # Modificado
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# --- [MODIFICADO] De Decil (10) para Quartil (4) ---
def plot_quartile_sharpe(sharpe_q: pd.Series, title: str, output_dir: str, filename: str):
    print(f"Plotando: {filename}...")
    plt.figure()
    sharpe_q.plot(kind='bar', color=COLOR_STRATEGY, alpha=0.7) 
    plt.title(f'Sharpe Ratio Anualizado por Quartil - {title}', fontsize=16) # Modificado
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Quartil (Q1 = Baixo, Q4 = Alto)') # Modificado
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    plt.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# (plot_holdings_over_time, plot_turnover, etc. não mudam, apenas cores)
def plot_holdings_over_time(df_diag: pd.DataFrame, output_dir: str):
    print("Plotando: 6. Número de Ativos na Carteira...")
    holdings_count = df_diag.reset_index().groupby('rebalance_date')['ticker'].nunique()
    plt.figure()
    holdings_count.plot(kind='line', color=COLOR_STRATEGY, label='Nº de Ativos')
    plt.axhline(holdings_count.mean(), color='red', linestyle='--', label=f'Média ({holdings_count.mean():.1f})')
    plt.title('Número de Ativos na Carteira por Período', fontsize=16)
    plt.ylabel('Contagem de Ativos')
    plt.xlabel('Data')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_holdings_over_time.png"))
    plt.close()

def plot_turnover(turnover_series: pd.Series, output_dir: str):
    print("Plotando: 7. Turnover Mensal...")
    plt.figure()
    turnover_series.plot(kind='area', color=COLOR_FILL_STRATEGY, alpha=0.4)
    mean_turnover = turnover_series.mean()
    plt.axhline(mean_turnover, color='red', linestyle='--', label=f'Média ({mean_turnover:.1%})')
    plt.xlabel('Período de Rebalanceamento')
    plt.ylabel('Turnover Mensal (%)')
    plt.title('Turnover Mensal da Carteira', fontsize=16)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "7_turnover.png"))
    plt.close()

def plot_hit_rate_distribution(df_diag: pd.DataFrame, output_dir: str):
    print("Plotando: 8. Distribuição de Retorno por Ativo...")
    plt.figure()
    sns.histplot(df_diag['holding_period_return'], kde=True, bins=100, color=COLOR_STRATEGY)
    median_ret = df_diag['holding_period_return'].median()
    mean_ret = df_diag['holding_period_return'].mean()
    plt.axhline(mean_ret, color='red', linestyle='--', label=f'Média ({mean_ret:.2%})')
    plt.axhline(median_ret, color='green', linestyle=':', label=f'Mediana ({median_ret:.2%})')
    plt.title('Distribuição do Retorno Mensal por Ativo ("Hit Rate")', fontsize=16)
    plt.xlabel('Retorno no Período de Holding')
    plt.ylabel('Frequência')
    plt.xlim(-0.5, 1.0) 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8_hit_rate_distribution.png"))
    plt.close()

def plot_portfolio_factor_exposure(df_diag: pd.DataFrame, output_dir: str):
    print("Plotando: 9. Exposição aos Fatores...")
    avg_factors = df_diag.reset_index().groupby('rebalance_date')[['fator_risco', 'fator_assimetria']].mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    avg_factors['fator_risco'].plot(ax=ax1, color=COLOR_STRATEGY, label='Fator de Risco Médio da Carteira')
    ax1.set_title('Exposição Média ao Fator de Risco (Baixa Entropia)', fontsize=14)
    ax1.set_ylabel('Score Médio Fator Risco')
    ax1.legend(loc='upper left')
    avg_factors['fator_assimetria'].plot(ax=ax2, color=COLOR_BENCHMARK_UNIVERSO, label='Fator de Assimetria Médio da Carteira') 
    ax2.set_title('Exposição Média ao Fator de Assimetria (DOWN_ASY)', fontsize=14)
    ax2.set_ylabel('Score Médio Fator Assimetria')
    ax2.set_xlabel('Data')
    ax2.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "9_factor_exposure_over_time.png"))
    plt.close()


# --- FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---

def main_analysis():
    """
    Função principal para carregar os dados e executar todas as análises.
    """
    
    FILE_RESULTS = r'resultados/resultados_backtest.csv'
    FILE_DIAGNOSTICS = r'resultados/diagnostico_detalhado.csv'
    FILE_RETURNS_MASTER = r'resultados/retornos_master.csv'
    FILE_FACTORS_MASTER = r'fatores/fatores_master.csv' 
    OUTPUT_DIR = r'resultados/output_analysis'

    df_results = None
    df_diag = None
    df_returns = None
    df_factors = None
    turnover = None
    
    # [MODIFICADO] Renomeado para 'quartil' (q)
    avg_ret_asy_q = cum_ret_asy_q = sharpe_asy_q = None
    avg_ret_risk_q = cum_ret_risk_q = sharpe_risk_q = None

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- INICIANDO ANÁLISE PROFUNDA DA ESTRATÉGIA ---")
    print(f"Resultados serão salvos em: '{OUTPUT_DIR}'")

    # --- 2. Carregamento de Dados ---
    try:
        print(f"Carregando: {FILE_RESULTS}...")
        df_results = pd.read_csv(FILE_RESULTS, index_col=0, parse_dates=True)
        
        expected_cols = ['Estrategia_Acum', 'Benchmark_Universo_Acum', 'Benchmark_IBOV_Acum', 'CDI_Acum']
        if not all(col in df_results.columns for col in expected_cols):
            print(f"Erro: O arquivo '{FILE_RESULTS}' não contém as colunas esperadas.")
            print(f"Esperado: {expected_cols}")
            print(f"Encontrado: {df_results.columns.tolist()}")
            return

        print(f"Carregando: {FILE_DIAGNOSTICS}...")
        df_diag = pd.read_csv(FILE_DIAGNOSTICS, index_col=[0,1], parse_dates=[0])
        
        print(f"Carregando: {FILE_RETURNS_MASTER}...")
        df_returns = pd.read_csv(FILE_RETURNS_MASTER, index_col=0, parse_dates=True)
        
        if 'IBOV' not in df_returns.columns:
            if '^BVSP' in df_returns.columns:
                df_returns = df_returns.rename(columns={'^BVSP': 'IBOV'})
            
    except FileNotFoundError as e:
        print(f"\n--- ERRO ---")
        print(f"Arquivo não encontrado: {e.filename}")
        return
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    # --- 3. Execução das Análises e Gráficos ---
    
    # FASE 2: Análise Temporal
    try:
        plot_cumulative_returns(df_results, OUTPUT_DIR)
        plot_drawdowns(df_results, OUTPUT_DIR)
        plot_rolling_sharpe(df_results, OUTPUT_DIR, window=252) 
    except Exception as e:
        print(f"Erro ao plotar Fases 1 & 2: {e}")

    # FASE 4: Diagnóstico de Implementação
    try:
        turnover = calculate_turnover(df_diag)
        plot_holdings_over_time(df_diag, OUTPUT_DIR)
        plot_turnover(turnover, OUTPUT_DIR)
        plot_hit_rate_distribution(df_diag, OUTPUT_DIR)
        plot_portfolio_factor_exposure(df_diag, OUTPUT_DIR)
    except Exception as e:
        print(f"Erro ao plotar Fase 4: {e}")

    # FASE 3: Validação dos Fatores (O "Porquê?")
    # --- [MODIFICADO] Chamando funções de QUARTIL ---
    try:
        print(f"Carregando: {FILE_FACTORS_MASTER} (necessário para Teste de Quartil)...")
        df_factors = pd.read_csv(FILE_FACTORS_MASTER, index_col=[0,1], parse_dates=[0])
        
        avg_ret_asy_q, cum_ret_asy_q, sharpe_asy_q = calculate_quartile_performance(df_factors, df_returns, 'fator_assimetria')
        plot_quartile_returns(avg_ret_asy_q, cum_ret_asy_q, 'Fator Assimetria (DOWN_ASY)', OUTPUT_DIR, "4_quartil_assimetria_RETORNO")
        
        avg_ret_risk_q, cum_ret_risk_q, sharpe_risk_q = calculate_quartile_performance(df_factors, df_returns, 'fator_risco')
        plot_quartile_sharpe(sharpe_risk_q, 'Fator Risco (Risk Estimator)', OUTPUT_DIR, "5_quartil_risco_SHARPE")

    except FileNotFoundError:
        print(f"\n--- AVISO ---")
        print(f"Arquivo '{FILE_FACTORS_MASTER}' não encontrado.")
        print("A Fase 3 (Validação de Fatores / Teste de Quartil) será pulada.")
    except Exception as e:
        print(f"Erro ao executar a Fase 3: {e}")
        
    # --- 4. Impressão da Análise Final ---
    # --- [MODIFICADO] Chamando funções de QUARTIL ---
    if df_results is not None:
        print_overall_analysis(df_results)
    
    if avg_ret_asy_q is not None:
        print_quartile_analysis_returns(avg_ret_asy_q, "Fator Assimetia (DOWN_ASY)")
    
    if sharpe_risk_q is not None:
        print_quartile_analysis_sharpe(sharpe_risk_q, "Fator Risco (Risk Estimator)")
    
    if turnover is not None and df_diag is not None:
        print_implementation_analysis(turnover, df_diag)
        
    print(f"\n--- ANÁLISE CONCLUÍDA ---")
    print(f"Gráficos salvos em: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main_analysis()