import pandas as pd
import numpy as np
import warnings
import os 
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad

# --- CONFIGURAÇÕES GLOBAIS ---
warnings.filterwarnings('ignore')

# === INÍCIO DA MODIFICAÇÃO DE ESTILO (PALETA ANANKE) ===

# 1. Definição da Paleta "Ananke"
PRETO_FUNDO = "#0A0A0A"      # Preto suave (Fundo da Figura e Eixos)
AZUL_PRINCIPAL = "#00BFFF"   # Azul elétrico (Estratégia, Títulos)
BRANCO_TEXTO = "#F0F0F0"     # Branco/Cinza claro (Texto, Eixos)
CINZA_GRID = "#333333"       # Grid sutil no fundo preto
VERMELHO_NEG = "#FF4136"     # Vermelho (Drawdowns, Média 2, Fator 2)
VERDE_POS = "#2ECC40"        # Verde (CDI, Média 3)
CINZA_BENCH = "#BBBBBB"       # Cinza claro (Benchmark IBOV)

# 2. Dicionário rc_params para o tema "Ananke" (Corrigido)
ANANKE_THEME_RC = {
    # --- Fundo ---
    'figure.facecolor': PRETO_FUNDO,
    'axes.facecolor': PRETO_FUNDO,
    
    # --- Texto e Títulos ---
    'text.color': BRANCO_TEXTO,
    'axes.labelcolor': BRANCO_TEXTO,
    'xtick.color': BRANCO_TEXTO,
    'ytick.color': BRANCO_TEXTO,
    'axes.titlecolor': AZUL_PRINCIPAL, # Títulos em azul
    'axes.titleweight': 'bold',
    'axes.titlesize': '16',
    
    # --- Eixos (Spines) e Grid ---
    'axes.edgecolor': CINZA_GRID,     # Borda do gráfico
    'grid.color': CINZA_GRID,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,

    # --- Legenda ---
    'legend.facecolor': PRETO_FUNDO,
    'legend.edgecolor': CINZA_GRID,
    'legend.frameon': True,
    'legend.title_fontsize': '12',
    'legend.fontsize': '10',
    'legend.labelcolor': BRANCO_TEXTO, # Colore os itens da legenda
}

# 3. Aplicar o Tema
sns.set_theme(style="darkgrid", rc=ANANKE_THEME_RC)

# 4. Configurações Globais de Figura
plt.rcParams['figure.figsize'] = (16, 8) 
plt.rcParams['figure.dpi'] = 100

# === FIM DA MODIFICAÇÃO DE ESTILO ===


# --- BLOCO 1: FUNÇÕES DE CÁLCULO (DO SEU TOOLBOX) ---
# (Lógica 100% original mantida)

def clean_and_standardize(x, y):
    data = np.vstack([x, y])
    data_clean = data[:, ~np.any(np.isnan(data), axis=0)]
    x_clean, y_clean = data_clean[0, :], data_clean[1, :]
    n_total = len(x_clean)
    x_mean, x_std = np.mean(x_clean), np.std(x_clean)
    y_mean, y_std = np.mean(y_clean), np.std(y_clean)
    x_std_scores = (x_clean - x_mean) / x_std if x_std > 0 else np.zeros_like(x_clean)
    y_std_scores = (y_clean - y_mean) / y_std if y_std > 0 else np.zeros_like(y_clean)
    return x_std_scores, y_std_scores, n_total

def filter_scenarios(asset_z, market_z, c_level_local):
    up_indexes = np.where((asset_z > c_level_local) & (market_z > c_level_local))
    up_data = np.vstack([asset_z[up_indexes], market_z[up_indexes]])
    down_indexes = np.where((asset_z < -c_level_local) & (market_z < -c_level_local))
    down_data = np.vstack([asset_z[down_indexes], market_z[down_indexes]])
    return up_data, down_data

def gaussian_kde_2d(data):
    try:
        kde = gaussian_kde(data)
        return kde
    except (np.linalg.LinAlgError, ValueError):
        return None

def gaussian_kde_2d_Hellinger_distance(kde1, kde2):
    min_kde_1 = np.min(kde1.dataset, axis=1)
    max_kde_1 = np.max(kde1.dataset, axis=1)
    min_kde_2 = np.min(kde2.dataset, axis=1)
    max_kde_2 = np.max(kde2.dataset, axis=1)
    
    min_x = np.min([min_kde_1[0], min_kde_2[0]]) - 2 
    max_x = np.max([max_kde_1[0], max_kde_2[0]]) + 2 
    min_y = np.min([min_kde_1[1], min_kde_2[1]]) - 2 
    max_y = np.max([max_kde_1[1], max_kde_2[1]]) + 2 
    
    def integrand(x, y):
        p1 = kde1.evaluate([x, y])[0]
        p2 = kde2.evaluate([x, y])[0]
        return (np.sqrt(p1) - np.sqrt(p2)) ** 2
    
    try:
        integral, _ = dblquad(integrand, min_x, max_x, lambda y: min_y, lambda y: max_y)
        return 0.5 * integral
    except Exception:
        return 1.0 

def asymmetry_entropy_calculate(up_data_local, down_data_local):
    down_data_rotated = down_data_local * -1
    kde_up = gaussian_kde_2d(up_data_local)
    kde_down_rotated = gaussian_kde_2d(down_data_rotated)
    if kde_up is None or kde_down_rotated is None:
        return 0.0
    S_rho = gaussian_kde_2d_Hellinger_distance(kde_up, kde_down_rotated)
    return S_rho

def downside_asymmetry_entropy_calculate(asset_returns: np.ndarray, 
                                         market_returns: np.ndarray, 
                                         c_level: float,
                                         n_min: int) -> float:
    asset_z, market_z, n_total = clean_and_standardize(asset_returns, market_returns)
    if n_total < n_min:
        return np.nan 

    up_data, down_data = filter_scenarios(asset_z, market_z, c_level)
    LQP = len(down_data[0]) / n_total if n_total > 0 else 0
    UQP = len(up_data[0]) / n_total if n_total > 0 else 0
    
    min_points_kde = max(10, n_min // 5)
    if up_data.shape[1] < min_points_kde or down_data.shape[1] < min_points_kde:
        return np.nan
        
    S_rho = asymmetry_entropy_calculate(up_data, down_data)
    DOWN_ASY = np.sign(LQP - UQP) * S_rho
    return DOWN_ASY

def calculate_correlation_asymmetry(asset_returns: np.ndarray, 
                                      market_returns: np.ndarray, 
                                      c_level: float,
                                      n_min: int) -> float:
    asset_z, market_z, n_total = clean_and_standardize(asset_returns, market_returns)
    if n_total < n_min:
        return np.nan 

    up_data, down_data = filter_scenarios(asset_z, market_z, c_level)

    min_points_corr = max(10, n_min // 5) 
    corr_up = 0.0
    corr_down = 0.0

    if up_data.shape[1] >= min_points_corr:
        corr_matrix_up = np.corrcoef(up_data[0], up_data[1])
        if not np.any(np.isnan(corr_matrix_up)):
            corr_up = corr_matrix_up[0, 1]

    if down_data.shape[1] >= min_points_corr:
        corr_matrix_down = np.corrcoef(down_data[0], down_data[1])
        if not np.any(np.isnan(corr_matrix_down)):
            corr_down = corr_matrix_down[0, 1]
            
    down_asy_corr = corr_down - corr_up
    return down_asy_corr

# --- BLOCO 2: FUNÇÃO DE BUSCA (LÓGICA CORRIGIDA) ---
# (Lógica 100% original mantida)

def find_best_example_asset(df_factors: pd.DataFrame, 
                            df_returns: pd.DataFrame, 
                            config: dict):
    """
    [CORRIGIDO] Encontra o ativo que é "correlação-simétrico" (Corr ~ 0)
    mas "entropia-assimétrico" (Entropia alta).
    """
    print("Iniciando busca pelo 'Ativo de Ouro' (Correlação vs. Entropia)...")
    
    lookback_days = 252 
    n_min = 30
    c_level = 0.0
    benchmark_ticker = "IBOV"
    
    paired_factors = []
    factor_index = df_factors.index
    
    for t_date, ticker in tqdm(factor_index, desc="Calculando Fatores Pareados"):
        
        fator_entropia = df_factors.loc[(t_date, ticker), 'fator_assimetria']
        
        if pd.isna(fator_entropia) or abs(fator_entropia) < 0.1: 
            continue
            
        end_date = t_date
        start_date = end_date - pd.DateOffset(days=lookback_days)
        
        try:
            window_df = df_returns.loc[start_date:end_date]
        except KeyError:
            continue

        if ticker not in window_df.columns or benchmark_ticker not in window_df.columns:
            continue
            
        asset_returns_np = window_df[ticker].to_numpy()
        market_returns_np = window_df[benchmark_ticker].to_numpy()

        fator_correlacao = calculate_correlation_asymmetry(asset_returns_np, 
                                                           market_returns_np, 
                                                           c_level,
                                                           n_min)
        
        if pd.isna(fator_correlacao):
            continue
            
        paired_factors.append({
            'date': t_date,
            'ticker': ticker,
            'fator_entropia': fator_entropia,
            'fator_correlacao': fator_correlacao
        })

    if not paired_factors:
        print("Erro: Nenhum dado pareado pôde ser calculado.")
        return None, None

    df_paired = pd.DataFrame(paired_factors)
    
    correlation_threshold = 0.1 
    df_candidates = df_paired[df_paired['fator_correlacao'].abs() < correlation_threshold].copy()
    
    if df_candidates.empty:
        print(f"Aviso: Nenhum ativo 'Correlação-Simétrico' (fator < {correlation_threshold}) foi encontrado.")
        print("O script não pode provar a tese. Tente aumentar o 'correlation_threshold'.")
        return None, None
    else:
        best_example = df_candidates.loc[df_candidates['fator_entropia'].abs().idxmax()]
    
    return best_example.to_dict(), df_paired


# --- BLOCO 3: FUNÇÃO DE PLOTAGEM (ESTÉTICA CORRIGIDA) ---
# (Seu código original de plotagem, agora com as variáveis de cor definidas globalmente)

# [FUNÇÃO ATUALIZADA COM A PALETA ANANKE]
# Esta função assume que as variáveis de cor 
# (AZUL_PRINCIPAL, VERMELHO_NEG, BRANCO_TEXTO)
# foram definidas no escopo global do script.

def plot_asymmetry_comparison(best_example: dict, 
                             df_returns: pd.DataFrame, 
                             config: dict, 
                             output_dir: str):
    """
    [CORRIGIDO] Recria o gráfico do artigo (KDE 2D) para o ativo e data vencedores.
    - Colormaps 'viridis' e 'inferno' (substituindo 'jet')
    - Erro de digitação 'up_mean_Y' corrigido para 'up_mean_y'
    - [ESTILO ANANKE] Cores atualizadas para paleta dark mode.
    """
    
    t_date = best_example['date']
    ticker = best_example['ticker']
    fator_entropia = best_example['fator_entropia']
    fator_correlacao = best_example['fator_correlacao']
    
    print(f"\n--- ATIVO DE OURO ENCONTRADO ---")
    print(f"Ticker: {ticker}")
    print(f"Data: {t_date.date()}")
    print(f"Fator Entropia (DOWN_ASY): {fator_entropia:.4f} (Alto)")
    print(f"Fator Correlação (Corr_D-U): {fator_correlacao:.4f} (Próximo de Zero)")
    print(f"Plotando os gráficos de densidade...")

    lookback_days = 252
    n_min = 30
    c_level = 0.0
    benchmark_ticker = "IBOV"
    
    end_date = t_date
    start_date = end_date - pd.DateOffset(days=lookback_days)
    window_df = df_returns.loc[start_date:end_date]
    
    asset_returns_np = window_df[ticker].to_numpy()
    market_returns_np = window_df[benchmark_ticker].to_numpy()

    asset_z, market_z, _ = clean_and_standardize(asset_returns_np, market_returns_np)
    up_data, down_data = filter_scenarios(asset_z, market_z, c_level)

    corr_up = np.corrcoef(up_data[0], up_data[1])[0, 1]
    corr_down = np.corrcoef(down_data[0], down_data[1])[0, 1]

    up_mean_x, up_mean_y = up_data[0].mean(), up_data[1].mean()
    down_mean_x, down_mean_y = down_data[0].mean(), down_data[1].mean()

    # O tema global (set_theme) já cuida dos fundos e da cor do texto/eixo
    fig = plt.figure(figsize=(16, 8))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # --- [ESTILO ANANKE] ---
    
    # Gráfico 1: Quadrante Cima-Cima (Tons de Azul)
    sns.kdeplot(x=up_data[0], y=up_data[1], 
              cmap="Blues",       # [ESTILO] Colormap azul
              levels=15,          
              fill=False,         
              linewidths=0.7,
              ax=ax1)
    
    # [ESTILO] Ponto central em BRANCO para contraste
    ax1.plot(up_mean_x, up_mean_y, 
             marker='o', 
             color=BRANCO_TEXTO, # <-- Mudança de 'ro'
             markersize=6, 
             label=f'Centro ({up_mean_x:.1f}, {up_mean_y:.1f})')
    
    # Título do eixo herdará a cor de 'text.color' (BRANCO_TEXTO) do rc_params
    ax1.set_title(f"Quadrante 'Cima-Cima' (Q1)\nCorrelação Pearson = {corr_up:.3f}", fontsize=14)
    ax1.set_xlabel(f"{ticker} (Z-score)")
    ax1.set_ylabel(f"{benchmark_ticker} (Z-score)")
    ax1.legend(loc='upper right') # Legenda será estilizada pelo rc_params

    # Gráfico 2: Quadrante Baixo-Baixo (Tons de Vermelho)
    sns.kdeplot(x=down_data[0], y=down_data[1], 
              cmap="Reds",        # [ESTILO] Colormap vermelho
              levels=15, 
              fill=False, 
              linewidths=0.7,
              ax=ax2)
    
    # [ESTILO] Ponto central em VERMELHO (cor de perigo)
    ax2.plot(down_mean_x, down_mean_y, 
             marker='o',
             color=VERMELHO_NEG, # <-- Mudança de 'ro'
             markersize=6, 
             label=f'Centro ({down_mean_x:.1f}, {down_mean_y:.1f})')

    ax2.set_title(f"Quadrante 'Baixo-Baixo' (Q3)\nCorrelação Pearson = {corr_down:.3f}", fontsize=14)
    ax2.set_xlabel(f"{ticker} (Z-score)")
    ax2.set_ylabel(f"{benchmark_ticker} (Z-score)")
    ax2.legend(loc='lower left')

    # [ESTILO] Título principal em AZUL (herdado de 'axes.titlecolor' se no rc_params, 
    # ou forçado aqui para garantir)
    fig.suptitle(f"Análise de Assimetria: {ticker} em {t_date.date()}\n"
                 f"Fator Correlação (Q3-Q1) = {fator_correlacao:.3f} (Próximo de 0) | Fator Entropia (DOWN_ASY) = {fator_entropia:.3f} (Alto)", 
                 fontsize=18, y=1.03, 
                 color=AZUL_PRINCIPAL) # <-- Garante o título em azul
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "10_comparacao_entropia_vs_correlacao.png"))
    plt.close()
    print(f"Gráfico de comparação salvo em: '{output_dir}/10_comparacao_entropia_vs_correlacao.png'")


# --- BLOCO 4: EXECUÇÃO PRINCIPAL ---
# (Lógica 100% original mantida)

def main():
    """
    Orquestrador: Carrega dados, encontra o melhor exemplo e plota.
    """
    
    FILE_FACTORS_MASTER = r'fatores/fatores_master.csv'
    FILE_RETURNS_MASTER = r'resultados/retornos_master.csv'
    OUTPUT_DIR = r'resultados/countour'

    config = {
        'lookback_days': 252,
        'n_min': 30,
        'c_level': 0.0,
        'benchmark_ticker': 'IBOV',
    }
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"--- PROVA DA TESE: ENTROPIA VS. CORRELAÇÃO ---")

    try:
        print(f"Carregando: {FILE_FACTORS_MASTER}...")
        df_factors = pd.read_csv(FILE_FACTORS_MASTER, index_col=[0,1], parse_dates=[0])
        
        print(f"Carregando: {FILE_RETURNS_MASTER}...")
        df_returns = pd.read_csv(FILE_RETURNS_MASTER, index_col=0, parse_dates=True)
        
        if 'IBOV' not in df_returns.columns:
            if '^BVSP' in df_returns.columns:
                df_returns = df_returns.rename(columns={'^BVSP': 'IBOV'})
            else:
                raise ValueError("Benchmark 'IBOV' ou '^BVSP' não encontrado em retornos_master.csv")

    except FileNotFoundError as e:
        print(f"\n--- ERRO ---")
        print(f"Arquivo não encontrado: {e.filename}")
        print("Certifique-se que 'fatores_master.csv' e 'retornos_master.csv' estão na pasta.")
        return
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return

    try:
        best_example, df_paired = find_best_example_asset(df_factors, df_returns, config)
        
        if best_example is None:
            print("Não foi possível encontrar um bom exemplo nos dados.")
            return
            
    except Exception as e:
        print(f"Erro durante a busca pelo ativo: {e}")
        return

    try:
        plot_asymmetry_comparison(best_example, df_returns, config, OUTPUT_DIR)
    except Exception as e:
        print(f"Erro ao plotar o gráfico de comparação: {e}")
        import traceback
        traceback.print_exc()

    print("\nAnálise de prova da tese concluída.")


if __name__ == "__main__":
    main()