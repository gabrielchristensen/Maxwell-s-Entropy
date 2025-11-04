import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from tqdm import tqdm

# --- CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')

# --- [MODIFICADO] PALETA DE CORES "MAXWELL" ---
# Baseado na paleta da apresentação (Azul escuro, Preto/Cinza, Branco)
COLOR_STRATEGY = "#67cdfc"        # Azul "Maxwell" (Principal)
COLOR_BENCHMARK_IBOV = '#333333'   # Preto/Cinza Escuro (Benchmark de Mercado)
COLOR_BENCHMARK_UNIVERSO = "#416d8a" # Cinza Médio (Benchmark de Controle)
COLOR_CDI = "#F70909"              # Cinza Claro (Custo de Oportunidade)
COLOR_FILL_STRATEGY = '#a9c4db'   # Azul Claro (Preenchimento de área)
# ------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100
pd.options.display.float_format = '{:.4f}'.format

# --- [MODIFICADO] Importa a função de métricas do run_backtest.py ---
# (É melhor ter a função em um local, mas vamos copiá-la para
# manter o analyzer.py autocontido e garantir que as métricas
# sejam calculadas exatamente da mesma maneira)

def calculate_performance_metrics(returns_series: pd.Series, 
                                  rf_daily_mean: float, 
                                  var_quantile: float = 0.05) -> dict:
    """
    Calcula um conjunto abrangente de métricas de performance.
    """
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

# (Funções calculate_quintile_performance e calculate_turnover não mudam)
def calculate_quintile_performance(df_factors: pd.DataFrame, 
                                     df_returns: pd.DataFrame, 
                                     factor_name: str):
    print(f"\nIniciando análise de quintis para: {factor_name}...")
    rebal_dates = df_factors.index.get_level_values('date').unique().sort_values()
    quintile_daily_returns_dict = {f'Q{i}': [] for i in range(1, 6)}
    daily_rf = df_returns['CDI'].mean() if 'CDI' in df_returns.columns else 0.0

    for i in tqdm(range(len(rebal_dates) - 1), desc=f"Testando {factor_name}"):
        t_date = rebal_dates[i]
        t_next_date = rebal_dates[i+1]
        
        factors_today = df_factors.loc[t_date].dropna(subset=[factor_name])
        if factors_today.empty or len(factors_today) < 5:
            continue
            
        try:
            factors_today['quintile'] = pd.qcut(factors_today[factor_name], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except ValueError:
            continue
            
        returns_period = df_returns.loc[(df_returns.index > t_date) & (df_returns.index <= t_next_date)]
        if returns_period.empty:
            continue
            
        for q_name in quintile_daily_returns_dict.keys():
            tickers_in_q = factors_today[factors_today['quintile'] == q_name].index
            valid_tickers = [t for t in tickers_in_q if t in returns_period.columns]
            
            if valid_tickers:
                ret_q_daily = returns_period[valid_tickers].mean(axis=1)
                quintile_daily_returns_dict[q_name].append(ret_q_daily)

    if not any(quintile_daily_returns_dict.values()):
        print(f"Erro: Nenhum retorno de quintil pôde ser calculado para {factor_name}.")
        return pd.Series(), pd.DataFrame(), pd.Series()

    df_quintile_daily_list = []
    for q_name, returns_list in quintile_daily_returns_dict.items():
        if returns_list:
            df_q = pd.concat(returns_list)
            df_q.name = q_name
            df_quintile_daily_list.append(df_q)
        else:
            df_quintile_daily_list.append(pd.Series(name=q_name, dtype=float))
        
    df_quintile_daily = pd.concat(df_quintile_daily_list, axis=1).fillna(0)

    avg_monthly_ret = df_quintile_daily.resample('M').apply(lambda x: (1 + x).prod() - 1).mean()
    df_quintile_cum = (1 + df_quintile_daily).cumprod()
    mean_excess_ret = df_quintile_daily.mean() - daily_rf
    std_ret = df_quintile_daily.std()
    sharpe_per_q = (mean_excess_ret / (std_ret + 1e-10)) * np.sqrt(252)
    
    return avg_monthly_ret, df_quintile_cum, sharpe_per_q


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

# [MODIFICADO] A função agora imprime a nova narrativa de 3 camadas
def print_overall_analysis(df_results):
    """Fase 1: Imprime a análise de performance agregada."""
    print("\n\n" + "="*80)
    print(" FASE 1: ANÁLISE DE PERFORMANCE AGREGADA (O 'QUÊ?')")
    print("="*80)

    # Recalcula as métricas
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
    alpha_fator = strat_metrics['CAGR'] - universo_metrics['CAGR']

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
    
    # Compara a Estratégia com seu controle (Universo)
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


# (Funções print_quintile... e print_implementation... não mudam)
def print_quintile_analysis_returns(avg_ret: pd.Series, factor_name: str):
    print("\n\n" + "="*80)
    print(f" FASE 3: VALIDAÇÃO DO FATOR '{factor_name}' (FOCO: RETORNO)")
    print("="*80)
    print("Retorno Médio Mensal por Quintil (Q1 = Baixo, Q5 = Alto):")
    print(avg_ret.to_string())
    spread = avg_ret['Q5'] - avg_ret['Q1']
    print(f"\nSpread (Q5 - Q1): {spread:.4%}")
    if spread > 0:
        print(f"✅ TESE VALIDADA: O quintil superior (Q5) rendeu mais que o inferior (Q1).")
    else:
        print(f"❌ TESE INVÁLIDA: O quintil superior (Q5) rendeu menos que o inferior (Q1).")
    monotonic = (avg_ret.diff().dropna() > 0).all()
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O retorno aumentou a cada quintil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear, mas o spread Q5-Q1 é o que importa.")

def print_quintile_analysis_sharpe(sharpe_q: pd.Series, factor_name: str):
    print("\n\n" + "="*80)
    print(f" FASE 3: VALIDAÇÃO DO FATOR '{factor_name}' (FOCO: SHARPE)")
    print("="*80)
    print("Sharpe Ratio Anualizado por Quintil (Q1 = Baixo, Q5 = Alto):")
    print(sharpe_q.to_string())
    spread = sharpe_q['Q1'] - sharpe_q['Q5']
    print(f"\nSpread (Q1 - Q5): {spread:.2f}")
    if spread > 0:
        print(f"✅ TESE VALIDADA: O quintil inferior (Q1, baixo risco) teve Sharpe maior que o superior (Q5, alto risco).")
    else:
        print(f"❌ TESE INVÁLIDA: O quintil inferior (Q1) teve Sharpe pior que o superior (Q5).")
    monotonic = (sharpe_q.diff().dropna() < 0).all()
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O Sharpe diminuiu a cada quintil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear.")

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

# [MODIFICADO] Plota 4 linhas (EstratégIA, Universo, IBOV, CDI)
def plot_cumulative_returns(df_results: pd.DataFrame, output_dir: str):
    """Fase 1: Plota o gráfico de retorno acumulado (O "Filme")."""
    print("Plotando: 1. Retorno Acumulado...")
    plt.figure()
    
    df_results['Benchmark_Universo_Acum'].plot(label='Benchmark-Universo (Controle)', color=COLOR_BENCHMARK_UNIVERSO, linestyle='-', linewidth=1.5, zorder=3)
    df_results['Benchmark_IBOV_Acum'].plot(label='IBOV (Mercado)', color=COLOR_BENCHMARK_IBOV, linestyle='--', linewidth=1.5, zorder=2)
    
    plt.title('Performance Acumulada da Estratégia vs. Benchmarks', fontsize=16)
    plt.ylabel('Retorno Acumulado')
    plt.yscale('log')
    plt.xlabel('Data')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_performance_acumulada.png"))
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
    OUTPUT_DIR = 'imamge_paulo'

    df_results = None
    df_diag = None
    df_returns = None
    df_factors = None
    turnover = None
    avg_ret_asy = cum_ret_asy = sharpe_asy = None
    avg_ret_risk = cum_ret_risk = sharpe_risk = None

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- INICIANDO ANÁLISE PROFUNDA DA ESTRATÉGIA ---")
    print(f"Resultados serão salvos em: '{OUTPUT_DIR}'")

    # --- 2. Carregamento de Dados ---
    try:
        print(f"Carregando: {FILE_RESULTS}...")
        df_results = pd.read_csv(FILE_RESULTS, index_col=0, parse_dates=True)
        
        # [MODIFICADO] Verifica se as colunas esperadas existem
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
    
    except Exception as e:
        print(f"Erro ao plotar Fases 1 & 2: {e}")

        
    print(f"\n--- ANÁLISE CONCLUÍDA ---")
    print(f"Gráficos salvos em: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main_analysis()