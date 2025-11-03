import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from tqdm import tqdm
from backtest import calculate_performance_metrics

# --- CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100
pd.options.display.float_format = '{:.4f}'.format # Formatação limpa para DataFrames

# --- FUNÇÕES DE CÁLCULO AUXILIARES ---

def calculate_quintile_performance(df_factors: pd.DataFrame, 
                                   df_returns: pd.DataFrame, 
                                   factor_name: str) -> (pd.Series, pd.DataFrame, pd.Series):
    """
    [CORRIGIDO] Executa o teste de quintis para um fator. 
    Esta função agora agrega os retornos de quintil corretamente.
    """
    print(f"\nIniciando análise de quintis para: {factor_name}...")
    
    rebal_dates = df_factors.index.get_level_values('date').unique().sort_values()
    
    # Dicionário para armazenar listas de séries de retorno diário por quintil
    quintile_daily_returns_dict = {f'Q{i}': [] for i in range(1, 6)}

    # Pega a média diária do CDI para o cálculo do Sharpe
    # Trata o caso de 'CDI' não estar presente (ex: teste rápido)
    daily_rf = df_returns['CDI'].mean() if 'CDI' in df_returns.columns else 0.0

    for i in tqdm(range(len(rebal_dates) - 1), desc=f"Testando {factor_name}"):
        t_date = rebal_dates[i]
        t_next_date = rebal_dates[i+1]
        
        # 1. Pegar fatores do dia e criar quintis
        factors_today = df_factors.loc[t_date].dropna(subset=[factor_name])
        if factors_today.empty or len(factors_today) < 5: # Precisa de min 5 ativos
            continue
            
        try:
            factors_today['quintile'] = pd.qcut(factors_today[factor_name], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except ValueError:
            continue
            
        # 2. Pegar retornos do período seguinte
        returns_period = df_returns.loc[(df_returns.index > t_date) & (df_returns.index <= t_next_date)]
        if returns_period.empty:
            continue
            
        # 3. Calcular retorno por quintil
        for q_name in quintile_daily_returns_dict.keys():
            tickers_in_q = factors_today[factors_today['quintile'] == q_name].index
            valid_tickers = [t for t in tickers_in_q if t in returns_period.columns]
            
            if valid_tickers:
                # Retorno diário médio do portfólio do quintil (equal weight)
                ret_q_daily = returns_period[valid_tickers].mean(axis=1)
                # Adiciona a série de retornos diários à lista correta
                quintile_daily_returns_dict[q_name].append(ret_q_daily)

    # [CORREÇÃO] Verifica se *pelo menos uma* lista de quintil tem dados
    if not any(quintile_daily_returns_dict.values()):
        print(f"Erro: Nenhum retorno de quintil pôde ser calculado para {factor_name}.")
        return pd.Series(), pd.DataFrame(), pd.Series()

    # 4. Criar o DataFrame de Retornos Diários por Quintil
    df_quintile_daily_list = []
    for q_name, returns_list in quintile_daily_returns_dict.items():
        if returns_list:
            # [CORREÇÃO] Concatena todas as séries para um quintil. 
            # O groupby().mean() é desnecessário pois os períodos não se sobrepõem.
            df_q = pd.concat(returns_list)
            df_q.name = q_name
            df_quintile_daily_list.append(df_q)
        else:
            # [CORREÇÃO] Adiciona uma Series vazia se um quintil não teve dados
            df_quintile_daily_list.append(pd.Series(name=q_name, dtype=float))
        
    # Concatena os 5 DFs de quintil em colunas, preenchendo dias faltantes com 0
    df_quintile_daily = pd.concat(df_quintile_daily_list, axis=1).fillna(0)

    # 5. Calcular Métricas
    
    # Retorno médio mensal
    avg_monthly_ret = df_quintile_daily.resample('M').apply(lambda x: (1 + x).prod() - 1).mean()
    
    # Retorno acumulado
    df_quintile_cum = (1 + df_quintile_daily).cumprod()
    
    # Sharpe Ratio por quintil
    mean_excess_ret = df_quintile_daily.mean() - daily_rf
    std_ret = df_quintile_daily.std()
    sharpe_per_q = (mean_excess_ret / (std_ret + 1e-10)) * np.sqrt(252)
    
    return avg_monthly_ret, df_quintile_cum, sharpe_per_q


def calculate_turnover(df_diag: pd.DataFrame) -> pd.Series:
    """
    Calcula o turnover (giro) mensal da carteira.
    """
    print("Calculando turnover da carteira...")
    holdings = df_diag.reset_index().groupby('rebalance_date')['ticker'].apply(set)
    holdings_prev = holdings.shift(1)
    df_turnover = pd.DataFrame({'current': holdings, 'previous': holdings_prev}).dropna()
    
    def calc_turnover_row(row):
        sold = row['previous'] - row['current']
        bought = row['current'] - row['previous']
        if len(row['previous']) == 0:
            return 1.0 # Carteira anterior estava vazia, comprou tudo (100% turnover)
        # Definição: (Valor Comprado + Valor Vendido) / (2 * Valor Total)
        # Em equal weight, Valor ~ N.
        return (len(sold) + len(bought)) / (2 * len(row['previous']))

    turnover_series = df_turnover.apply(calc_turnover_row, axis=1)
    turnover_series.name = "Monthly Turnover"
    return turnover_series


# --- FUNÇÕES DE ANÁLISE (IMPRESSÃO) ---

def print_overall_analysis(df_results):
    """Fase 1: Imprime a análise de performance agregada."""
    print("\n\n" + "="*80)
    print(" FASE 1: ANÁLISE DE PERFORMANCE AGREGADA (O 'QUÊ?')")
    print("="*80)

    # Recalcula as métricas para garantir precisão
    rf_daily = df_results['CDI'].mean()
    strat_metrics = calculate_performance_metrics(df_results['Estrategia'], rf_daily)
    ibov_metrics = calculate_performance_metrics(df_results['Benchmark'], rf_daily)
    cdi_metrics = calculate_performance_metrics(df_results['CDI'], rf_daily)

    print(f"\n--- MÉTRICAS (Estratégia) ---")
    print(f"CAGR: {strat_metrics['CAGR']:.2%} | Volatilidade: {strat_metrics['Volatilidade']:.2%} | Sharpe: {strat_metrics['Sharpe Ratio']:.2f}")
    
    print(f"\n--- MÉTRICAS (IBOV) ---")
    print(f"CAGR: {ibov_metrics['CAGR']:.2%} | Volatilidade: {ibov_metrics['Volatilidade']:.2%} | Sharpe: {ibov_metrics['Sharpe Ratio']:.2f}")
    
    print(f"\n--- MÉTRICAS (CDI) ---")
    print(f"CAGR: {cdi_metrics['CAGR']:.2%} | Volatilidade: {cdi_metrics['Volatilidade']:.2%}")
    
    print("\n--- VEREDITO (TESE 1: FATOR DE RISCO) ---")
    if strat_metrics['Volatilidade'] < ibov_metrics['Volatilidade']:
        print(f"✅ SUCESSO: A estratégia foi {ibov_metrics['Volatilidade'] - strat_metrics['Volatilidade']:.2f}% menos volátil que o IBOV.")
        print(f"   ↳ TESE VALIDADA: O filtro 'fator_risco' (baixa entropia) foi eficaz em reduzir o risco, mesmo com uma carteira concentrada.")
    else:
        print(f"❌ FALHA: A estratégia foi {strat_metrics['Volatilidade'] - ibov_metrics['Volatilidade']:.2f}% mais volátil que o IBOV.")
        print(f"   ↳ TESE INVÁLIDA: A seleção de 'fator_risco' não foi suficiente para superar o risco de concentração.")
        
    print("\n--- VEREDITO (TESE 2: FATOR DE ASSIMETRIA) ---")
    if strat_metrics['CAGR'] > ibov_metrics['CAGR']:
        if strat_metrics['CAGR'] > cdi_metrics['CAGR']:
             print(f"✅ SUCESSO: A estratégia gerou Alpha positivo, superando o IBOV em {strat_metrics['CAGR'] - ibov_metrics['CAGR']:.2f}% a.a. e o CDI em {strat_metrics['CAGR'] - cdi_metrics['CAGR']:.2f}% a.a.")
             print(f"   ↳ TESE VALIDADA: O filtro 'fator_assimetria' (DOWN_ASY) capturou um prémio de risco que resultou em retornos superiores.")
        else:
             print(f"⚠️ ATENÇÃO: A estratégia superou o IBOV, mas falhou em superar o custo de oportunidade (CDI).")
    else:
        print(f"❌ FALHA: A estratégia teve performance inferior ao IBOV em {ibov_metrics['CAGR'] - strat_metrics['CAGR']:.2f}% a.a.")
        print(f"   ↳ TESE INVÁLIDA: O prémio de risco 'fator_assimetria' não se materializou ou não foi capturado.")
        
    print("\n--- VEREDITO (RISCO DE CAUDA) ---")
    if strat_metrics['Max Drawdown'] > ibov_metrics['Max Drawdown']:
        print(f"⚠️ RISCO CONFIRMADO: A estratégia teve um Drawdown pior ({strat_metrics['Max Drawdown']:.2%}) que o IBOV ({ibov_metrics['Max Drawdown']:.2%}).")
        print(f"   ↳ Isto é esperado pela Tese 2 (seleção de alto DOWN_ASY).")
        if strat_metrics['Calmar Ratio'] > ibov_metrics['Calmar Ratio']:
            print(f"✅ COMPENSAÇÃO: O Calmar Ratio foi superior ({strat_metrics['Calmar Ratio']:.2f} vs {ibov_metrics['Calmar Ratio']:.2f}), indicando que o retorno maior compensou a queda.")
        else:
            print(f"❌ PROBLEMA: O Calmar Ratio foi inferior. O retorno extra NÃO compensou o risco de queda maior.")
    else:
        print(f"✅ BÔNUS DEFENSIVO: A estratégia foi mais defensiva, com um Drawdown menor ({strat_metrics['Max Drawdown']:.2%}) que o IBOV ({ibov_metrics['Max Drawdown']:.2%}).")
        print(f"   ↳ O poder do 'fator_risco' foi mais forte que o risco do 'fator_assimetria'.")


def print_quintile_analysis_returns(avg_ret: pd.Series, factor_name: str):
    """Fase 3: Imprime a análise de quintis (foco em RETORNO)."""
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
        
    # Verifica monotonicidade
    monotonic = (avg_ret.diff().dropna() > 0).all()
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O retorno aumentou a cada quintil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear, mas o spread Q5-Q1 é o que importa.")

def print_quintile_analysis_sharpe(sharpe_q: pd.Series, factor_name: str):
    """Fase 3: Imprime a análise de quintis (foco em SHARPE)."""
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
        
    monotonic = (sharpe_q.diff().dropna() < 0).all() # Espera-se que caia
    if monotonic:
        print("✅ MONOTÔNICO: Perfeito! O Sharpe diminuiu a cada quintil.")
    else:
        print("⚠️ NÃO MONOTÔNICO: A relação não é linear.")

def print_implementation_analysis(turnover: pd.Series, df_diag: pd.DataFrame):
    """Fase 4: Imprime a análise de implementação."""
    print("\n\n" + "="*80)
    print(" FASE 4: ANÁLISE DE IMPLEMENTAÇÃO (O 'COMO?')")
    print("="*80)
    
    holdings_count = df_diag.reset_index().groupby('rebalance_date')['ticker'].nunique()
    mean_holdings = holdings_count.mean()
    mean_turnover = turnover.mean()
    mean_hit_rate = df_diag['holding_period_return'].mean()
    
    print(f"Número Médio de Ativos na Carteira: {mean_holdings:.1f}")
    if mean_holdings < 10:
        print(f"   ↳ ⚠️ ALERTA DE CONCENTRAÇÃO: A média de ativos é muito baixa. O risco idiossincrático pode estar dominando a estratégia.")
    else:
        print(f"   ↳ ✅ Nível de diversificação saudável (próximo do limite de 20).")
        
    print(f"\nTurnover Médio Mensal: {mean_turnover:.1%}")
    if mean_turnover > 0.4: # 40% ao mês
        print(f"   ↳ ⚠️ ALERTA DE CUSTOS: O giro da carteira é alto. Os custos de transação (não modelados) podem consumir uma parte significativa do Alpha.")
    else:
        print(f"   ↳ ✅ Giro da carteira gerenciável.")
        
    print(f"\nRetorno Médio por 'Aposta' (Hit Rate): {mean_hit_rate:.2%}")
    if mean_hit_rate > 0:
         print(f"   ↳ ✅ Positivo. Em média, as ações selecionadas tiveram performance positiva no mês seguinte.")
    else:
         print(f"   ↳ ❌ Negativo. A seleção de fatores está, em média, a escolher ações perdedoras.")


# --- FUNÇÕES DE PLOTAGEM (CORRIGIDAS) ---

def plot_cumulative_returns(df_results: pd.DataFrame, output_dir: str):
    """Fase 1: Plota o gráfico de retorno acumulado (O "Filme")."""
    print("Plotando: 1. Retorno Acumulado...")
    plt.figure()
    
    df_results['Estrategia_Acum'].plot(label='Estratégia (Fator Duplo)', color='blue', linewidth=2)
    df_results['Benchmark_Acum'].plot(label='IBOV (Benchmark)', color='black', linestyle='--', linewidth=1.5)
    df_results['CDI_Acum'].plot(label='CDI (Custo de Oportunidade)', color='green', linestyle=':', linewidth=1.5)
    
    plt.title('Performance Acumulada da Estratégia vs. Benchmarks (Log Scale)', fontsize=16)
    plt.ylabel('Retorno Acumulado (Base 1)')
    plt.xlabel('Data')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_performance_acumulada.png"))
    plt.close()

def plot_drawdowns(df_results: pd.DataFrame, output_dir: str):
    """Fase 2: Plota os drawdowns da Estratégia vs. IBOV."""
    print("Plotando: 2. Drawdowns...")
    
    def calc_drawdown(cum_returns):
        running_max = cum_returns.cummax() # Use cummax() para eficiência
        return (cum_returns / running_max) - 1

    dd_strategy = calc_drawdown(df_results['Estrategia_Acum'])
    dd_benchmark = calc_drawdown(df_results['Benchmark_Acum'])
    
    plt.figure()
    dd_strategy.plot(label='Estratégia (Fator Duplo)', color='blue', kind='area', alpha=0.5)
    dd_benchmark.plot(label='IBOV (Benchmark)', color='black', linestyle='--', linewidth=1.5)
    
    plt.title('Drawdowns da Estratégia vs. IBOV', fontsize=16)
    plt.ylabel('Queda Percentual do Pico')
    plt.xlabel('Data')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_drawdowns.png"))
    plt.close()

def plot_rolling_sharpe(df_results: pd.DataFrame, output_dir: str, window: int = 252):
    """Fase 2: Plota o Sharpe Ratio rolante (Análise de consistência)."""
    print("Plotando: 3. Sharpe Ratio Rolante...")
    
    excess_returns_strat = df_results['Estrategia'] - df_results['CDI']
    excess_returns_bench = df_results['Benchmark'] - df_results['CDI']
    
    rolling_sharpe_strat = (excess_returns_strat.rolling(window).mean() / (excess_returns_strat.rolling(window).std() + 1e-10)) * np.sqrt(252)
    rolling_sharpe_bench = (excess_returns_bench.rolling(window).mean() / (excess_returns_bench.rolling(window).std() + 1e-10)) * np.sqrt(252)
    
    plt.figure()
    rolling_sharpe_strat.plot(label='Estratégia (Fator Duplo)', color='blue', linewidth=2)
    rolling_sharpe_bench.plot(label='IBOV (Benchmark)', color='black', linestyle='--', linewidth=1.5)
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    
    plt.title(f'Sharpe Ratio Rolante ({window} dias)', fontsize=16)
    plt.ylabel('Sharpe Ratio Anualizado')
    plt.xlabel('Data')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_rolling_sharpe.png"))
    plt.close()

def plot_quintile_returns(avg_ret: pd.Series, cum_ret: pd.DataFrame, title: str, output_dir: str, filename: str):
    """Fase 3: Plota os resultados de quintil (foco em Retorno). [CORRIGIDO]"""
    print(f"Plotando: {filename}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 2]})
    
    avg_ret.plot(kind='bar', ax=ax1, color='blue', alpha=0.7)
    ax1.set_title(f'Retorno Médio Mensal por Quintil - {title}', fontsize=14)
    ax1.set_ylabel('Retorno Médio Mensal')
    ax1.set_xlabel('Quintil (Q1 = Baixo, Q5 = Alto)')
    ax1.tick_params(axis='x', rotation=0)
    
    cum_ret.plot(ax=ax2, linewidth=2, colormap='coolwarm')
    ax2.set_title(f'Performance Acumulada por Quintil - {title}', fontsize=14)
    ax2.set_ylabel('Retorno Acumulado (Base 1)')
    ax2.set_xlabel('Data')
    ax2.set_yscale('log')
    ax2.legend(title='Quintil')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

def plot_quintile_sharpe(sharpe_q: pd.Series, title: str, output_dir: str, filename: str):
    """Fase 3: Plota os resultados de quintil (foco em Sharpe). [NOVO]"""
    print(f"Plotando: {filename}...")
    plt.figure()
    
    sharpe_q.plot(kind='bar', color='blue', alpha=0.7)
    plt.title(f'Sharpe Ratio Anualizado por Quintil - {title}', fontsize=16)
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Quintil (Q1 = Baixo, Q5 = Alto)')
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    plt.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

def plot_holdings_over_time(df_diag: pd.DataFrame, output_dir: str):
    """Fase 4: Plota o número de ativos na carteira ao longo do tempo."""
    print("Plotando: 6. Número de Ativos na Carteira...")
    holdings_count = df_diag.reset_index().groupby('rebalance_date')['ticker'].nunique()
    
    plt.figure()
    holdings_count.plot(kind='line', color='blue', label='Nº de Ativos')
    plt.axhline(holdings_count.mean(), color='red', linestyle='--', label=f'Média ({holdings_count.mean():.1f})')
    
    plt.title('Número de Ativos na Carteira por Período', fontsize=16)
    plt.ylabel('Contagem de Ativos')
    plt.xlabel('Data')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "6_holdings_over_time.png"))
    plt.close()

def plot_turnover(turnover_series: pd.Series, output_dir: str):
    """Fase 4: Plota o turnover (giro) da carteira."""
    print("Plotando: 7. Turnover Mensal...")
    plt.figure()
    # [CORREÇÃO] Gráfico de barra é ilegível. Mudar para 'area'.
    turnover_series.plot(kind='area', color='blue', alpha=0.4)
    
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
    """Fase 4: Plota a distribuição dos retornos de cada "aposta"."""
    print("Plotando: 8. Distribuição de Retorno por Ativo...")
    plt.figure()
    sns.histplot(df_diag['holding_period_return'], kde=True, bins=100, color='blue')
    
    median_ret = df_diag['holding_period_return'].median()
    mean_ret = df_diag['holding_period_return'].mean()
    plt.axvline(mean_ret, color='red', linestyle='--', label=f'Média ({mean_ret:.2%})')
    plt.axvline(median_ret, color='green', linestyle=':', label=f'Mediana ({median_ret:.2%})')
    
    plt.title('Distribuição do Retorno Mensal por Ativo ("Hit Rate")', fontsize=16)
    plt.xlabel('Retorno no Período de Holding')
    plt.ylabel('Frequência')
    plt.xlim(-0.5, 1.0) 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8_hit_rate_distribution.png"))
    plt.close()

def plot_portfolio_factor_exposure(df_diag: pd.DataFrame, output_dir: str):
    """Fase 4: Plota a exposição média aos fatores ao longo do tempo."""
    print("Plotando: 9. Exposição aos Fatores...")
    
    avg_factors = df_diag.reset_index().groupby('rebalance_date')[['fator_risco', 'fator_assimetria']].mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    avg_factors['fator_risco'].plot(ax=ax1, color='blue', label='Fator de Risco Médio da Carteira')
    ax1.set_title('Exposição Média ao Fator de Risco (Baixa Entropia)', fontsize=14)
    ax1.set_ylabel('Score Médio Fator Risco')
    ax1.legend(loc='upper left')
    
    avg_factors['fator_assimetria'].plot(ax=ax2, color='red', label='Fator de Assimetria Médio da Carteira')
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
    
    # --- 1. Definição de Ficheiros ---
    FILE_RESULTS = r'resultados/resultados_backtest.csv'
    FILE_DIAGNOSTICS = r'resultados/diagnostico_detalhado.csv'
    FILE_RETURNS_MASTER = r'resultados/retornos_master.csv'
    FILE_FACTORS_MASTER = r'fatores/fatores_master.csv' 
    OUTPUT_DIR = r'resultados/output_analysis'

    # [CORREÇÃO] Inicializa variáveis de análise como None
    df_results = None
    df_diag = None
    df_returns = None
    df_factors = None
    turnover = None
    avg_ret_asy = cum_ret_asy = sharpe_asy = None
    avg_ret_risk = cum_ret_risk = sharpe_risk = None

    # Cria o diretório de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- INICIANDO ANÁLISE PROFUNDA DA ESTRATÉGIA ---")
    print(f"Resultados serão salvos em: '{OUTPUT_DIR}'")

    # --- 2. Carregamento de Dados ---
    try:
        print(f"Carregando: {FILE_RESULTS}...")
        df_results = pd.read_csv(FILE_RESULTS, index_col=0, parse_dates=True)
        
        print(f"Carregando: {FILE_DIAGNOSTICS}...")
        df_diag = pd.read_csv(FILE_DIAGNOSTICS, index_col=[0,1], parse_dates=[0])
        
        print(f"Carregando: {FILE_RETURNS_MASTER}...")
        df_returns = pd.read_csv(FILE_RETURNS_MASTER, index_col=0, parse_dates=True)
        
        # Renomeia o benchmark se necessário (ex: ^BVSP para IBOV)
        if '^BVSP' in df_returns.columns:
            df_returns = df_returns.rename(columns={'^BVSP': 'IBOV'})
            
    except FileNotFoundError as e:
        print(f"\n--- ERRO ---")
        print(f"Arquivo não encontrado: {e.filename}")
        print("Certifique-se que 'resultados_backtest.csv', 'diagnostico_detalhado.csv', e 'retornos_master.csv' estão na pasta.")
        return

    # --- 3. Execução das Análises e Gráficos ---
    
    # FASE 2: Análise Temporal
    try:
        plot_cumulative_returns(df_results, OUTPUT_DIR)
        plot_drawdowns(df_results, OUTPUT_DIR)
        plot_rolling_sharpe(df_results, OUTPUT_DIR, window=252) # 1 ano rolante
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
    try:
        print(f"Carregando: {FILE_FACTORS_MASTER} (necessário para Teste de Quintis)...")
        df_factors = pd.read_csv(FILE_FACTORS_MASTER, index_col=[0,1], parse_dates=[0])
        
        # Teste de Quintil - Fator Assimetria (Tese 2)
        avg_ret_asy, cum_ret_asy, sharpe_asy = calculate_quintile_performance(df_factors, df_returns, 'fator_assimetria')
        plot_quintile_returns(avg_ret_asy, cum_ret_asy, 'Fator Assimetria (DOWN_ASY)', OUTPUT_DIR, "4_quintil_assimetria_RETORNO")
        
        # Teste de Quintil - Fator Risco (Tese 1)
        avg_ret_risk, cum_ret_risk, sharpe_risk = calculate_quintile_performance(df_factors, df_returns, 'fator_risco')
        plot_quintile_sharpe(sharpe_risk, 'Fator Risco (Risk Estimator)', OUTPUT_DIR, "5_quintil_risco_SHARPE")

    except FileNotFoundError:
        print(f"\n--- AVISO ---")
        print(f"Arquivo '{FILE_FACTORS_MASTER}' não encontrado.")
        print("A Fase 3 (Validação de Fatores / Teste de Quintis) será pulada.")
    except Exception as e:
        print(f"Erro ao executar a Fase 3: {e}")
        
    # --- 4. Impressão da Análise Final ---
    # [CORREÇÃO] Move as impressões para dentro de verificações 'if not None'
    
    if df_results is not None:
        print_overall_analysis(df_results)
    
    if avg_ret_asy is not None:
        print_quintile_analysis_returns(avg_ret_asy, "Fator Assimetria (DOWN_ASY)")
    
    if sharpe_risk is not None:
        print_quintile_analysis_sharpe(sharpe_risk, "Fator Risco (Risk Estimator)")
    
    if turnover is not None and df_diag is not None:
        print_implementation_analysis(turnover, df_diag)
        
    print(f"\n--- ANÁLISE CONCLUÍDA ---")
    print(f"Gráficos salvos em: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main_analysis()