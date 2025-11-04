import pandas as pd
import numpy as np
import warnings
import os 
import time
from tqdm import tqdm 
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÕES GLOBAIS DE PLOTAGEM ---
warnings.filterwarnings('ignore')

# --- [MODIFICADO] PALETA DE CORES "MAXWELL" ---
# Baseado na paleta da apresentação (Azul escuro, Preto/Cinza, Branco)
COLOR_STRATEGY = '#004a7c'        # Azul "Maxwell" (Principal, Entropia)
COLOR_BENCHMARK_IBOV = '#333333'   # Preto/Cinza Escuro (Comparação, Volatilidade)
# ------------------------------------------------

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100

def plot_analytics_comparison(df_analytics, correlation, output_file_path):
    """
    Plota as duas métricas de risco (Entropia vs. Volatilidade) ao longo do tempo.
    Usa padronização (Z-score) para comparar as duas séries em uma
    escala comum, já que suas unidades (nats vs. %) são diferentes.
    """
    
    print(f"Plotando gráfico de comparação...")
    
    # 1. Padroniza os dados (Z-score) para comparação visual
    df_standardized = pd.DataFrame(index=df_analytics.index)
    df_standardized['Entropia (Z-score)'] = (df_analytics['avg_entropy_factor'] - df_analytics['avg_entropy_factor'].mean()) / df_analytics['avg_entropy_factor'].std()
    df_standardized['Volatilidade (Z-score)'] = (df_analytics['avg_volatility'] - df_analytics['avg_volatility'].mean()) / df_analytics['avg_volatility'].std()

    # 2. Plota as séries temporais padronizadas
    plt.figure()
    
    # [MODIFICADO] Cores alteradas para a paleta da apresentação
    df_standardized['Entropia (Z-score)'].plot(label='Média do Fator Risco (Entropia)', color=COLOR_STRATEGY, linewidth=2)
    df_standardized['Volatilidade (Z-score)'].plot(label='Média da Volatilidade (Tradicional)', color=COLOR_BENCHMARK_IBOV, linestyle='--', linewidth=2)
    
    plt.title(f'Comparação: Risco (Entropia) vs. Risco (Volatilidade) no Universo Elegível\nCorrelação: {correlation:.2f}', fontsize=16)
    plt.xlabel('Data')
    plt.ylabel('Valor Padronizado (Z-score)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(output_file_path)
    plt.close()


def main_calculate_analytics():
    """
    Orquestrador principal:
    Para cada data de rebalanceamento, define o universo elegível e calcula
    a média do Fator de Risco (Entropia) e da Volatilidade (Desvio Padrão)
    para esse universo.
    """

    # --- 1. Configuração ---
    config = {
        'lookback_days': 252,
        'RISK_FACTOR_QUANTILE_FLOOR': 0.01,
        
        'input_returns_file': r'resultados/retornos_master.csv', 
        'input_factors_file': r'fatores/fatores_master.csv', 
        
        'output_analytics_file': r'resultados/universe_analytics.csv',
        'output_plot_dir': r'resultados/output_analysis' # <- [Corrigido] Salva gráficos na mesma pasta do analyzer
    }

    print("--- CÁLCULO DE MÉTRICAS DO UNIVERSO (Entropia vs. Volatilidade) ---")

    # Cria o diretório de saída do gráfico se não existir
    if not os.path.exists(config['output_plot_dir']):
        os.makedirs(config['output_plot_dir'])

    # --- 2. Carregar Dados ---
    try:
        print(f"Carregando retornos de '{config['input_returns_file']}'...")
        df_retornos = pd.read_csv(config['input_returns_file'], index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo MASTER de retornos '{config['input_returns_file']}' não encontrado.")
        return
        
    try:
        print(f"Carregando fatores de '{config['input_factors_file']}'...")
        all_factors_df = pd.read_csv(config['input_factors_file'], index_col=[0, 1], parse_dates=[0])
    except FileNotFoundError:
        print(f"Erro Crítico: Arquivo MASTER de fatores '{config['input_factors_file']}' não encontrado.")
        return
    print(f"Total de {len(all_factors_df)} linhas de fator carregadas.")

    # --- 3. Obter Datas de Rebalanceamento ---
    all_rebalance_dates = all_factors_df.index.get_level_values('date').unique().sort_values()
    
    analytics_data = [] 

    print(f"\nIniciando loop de {len(all_rebalance_dates)} datas para calcular métricas do universo...")
    start_loop = time.perf_counter()

    # --- 4. Loop Principal ---
    for t_date in tqdm(all_rebalance_dates): 
        try:
            df_factors_for_date = all_factors_df.loc[t_date]
        except KeyError:
            continue

        # --- Etapa A: Definir o Universo Elegível ---
        df_factors_clean = df_factors_for_date.dropna()

        if not df_factors_clean.empty and len(df_factors_clean) > 1:
            risk_floor_value = df_factors_clean['fator_risco'].quantile(config['RISK_FACTOR_QUANTILE_FLOOR'])
            df_factors_clean = df_factors_clean[df_factors_clean['fator_risco'] > risk_floor_value]
        
        if df_factors_clean.empty:
            continue
            
        eligible_assets = df_factors_clean.index

        # --- Etapa B: Calcular Métricas para o Universo Elegível ---
        avg_entropy_factor = df_factors_clean['fator_risco'].mean()

        end_date = t_date
        # Ajusta para o período de 252 dias corridos, não dias úteis
        start_date = end_date - pd.DateOffset(days=config['lookback_days'] * (365/252) + 10) 
        
        window_returns = df_retornos.loc[start_date:end_date]
        eligible_returns = window_returns.reindex(columns=eligible_assets)
        
        if eligible_returns.empty:
            avg_volatility = np.nan
        else:
            # Calcula a volatilidade anualizada para CADA ativo
            volatilities = eligible_returns.tail(config['lookback_days']).std(skipna=True) * np.sqrt(252)
            avg_volatility = volatilities.mean(skipna=True)

        # --- Etapa C: Armazenar Resultados ---
        analytics_data.append({
            'date': t_date,
            'n_eligible_assets': len(eligible_assets),
            'avg_entropy_factor': avg_entropy_factor,
            'avg_volatility': avg_volatility
        })

    end_loop = time.perf_counter()
    print(f"\nLoop de cálculo concluído em {end_loop - start_loop:.2f} segundos.")

    # --- 5. Salvar Resultados e Plotar ---
    if not analytics_data:
        print("Nenhum dado de análise foi gerado.")
        return

    df_analytics = pd.DataFrame(analytics_data)
    df_analytics = df_analytics.set_index('date')
    df_analytics = df_analytics.dropna()

    df_analytics.to_csv(config['output_analytics_file'])
    print(f"Análise do Universo (Entropia vs. Volatilidade) salva em: '{config['output_analytics_file']}'")

    # --- [NOVO] Calcular Correlação e Chamar Plotagem ---
    if not df_analytics.empty:
        try:
            correlation = df_analytics['avg_entropy_factor'].corr(df_analytics['avg_volatility'])
            print(f"\nCorrelação (Entropia vs. Volatilidade): {correlation:.4f}")
            
            plot_file_path = os.path.join(config['output_plot_dir'], '11_comparacao_entropia_vs_volatilidade.png')
            plot_analytics_comparison(df_analytics, correlation, plot_file_path)
            print(f"Gráfico de comparação salvo em: '{plot_file_path}'")
            
        except Exception as e:
            print(f"Erro ao calcular correlação ou plotar gráfico: {e}")
            
    print("\nCálculo de análise do universo concluído.")


if __name__ == "__main__":
    main_calculate_analytics()