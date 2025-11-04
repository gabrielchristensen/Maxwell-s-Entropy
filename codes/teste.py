import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# --- PALETA DE CORES "MAXWELL" ---
# Define uma paleta customizada no estilo da apresentação
MAXWELL_PALETTE_LIST = [
    '#004a7c',     # 1. Estratégia (Azul Principal)
    "#15AD07",     # 2. Benchmark 1 (Preto/Cinza Escuro)
    "#002CEE",     # 3. Benchmark 2 (Cinza Médio)
    "#04B471",     # 4. Benchmark 3 (Cinza Claro)
    "#333333"      # 5. Benchmark 4 (Outro Cinza)
]
# ------------------------------------------------

# --- 1. Carregar e Preparar os Dados ---

FILE_NAME = 'comp.csv'

try:
    # Carregar o 'comp.csv'
    data = pd.read_csv(
        FILE_NAME,
        sep=';',
        decimal=',',
        encoding='latin-1',
        parse_dates=[4],        # Usar o ÍNDICE da coluna de data
        dayfirst=True,
        index_col=4             # Usar o ÍNDICE da coluna de data como índice
    )
    
    data.index.name = 'Data'
    
    print(f"Arquivo '{FILE_NAME}' carregado com sucesso.")
    print("Colunas originais encontradas:", data.columns.tolist())
    
    cumulative_returns = data.copy()
    
    # --- [MODIFICAÇÃO PARA MUDAR A LEGENDA] ---
    #
    # Para alterar os textos da legenda, renomeie as colunas aqui.
    # O Seaborn usará esses 'novos_nomes' para os gráficos.
    #
    # ATENÇÃO: Os nomes à esquerda ('NOME ANTIGO') devem ser EXATAMENTE
    # iguais aos nomes das colunas impressos acima.
    #
    novos_nomes = {
        'ï»¿Maxwell': 'Maxwell',
        'Assimetria Only': 'Assimetria 100%',
        'Entropia Only': 'Entropia 100%',
        'Maxwell com Asy Alta': 'Estratégia com Assimetria Alta',
        'IBOV': 'IBOV'
    }
    
    # Aplica a renomeação ao DataFrame
    cumulative_returns.rename(columns=novos_nomes, inplace=True)
    
    print("\nColunas renomeadas para:", cumulative_returns.columns.tolist())
    # ---------------------------------------------
    
    
    # Calcular retornos diários (agora com colunas renomeadas)
    daily_returns = cumulative_returns.pct_change().dropna()

    # --- 2. Gráfico de Retorno Acumulado (Seaborn) ---

    # Agora o 'melt' usará os nomes novos
    cumulative_long = cumulative_returns.reset_index().melt(
        'Data', 
        var_name='Estratégia/Benchmark', 
        value_name='Retorno Acumulado'
    )

    # Plotar
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid") # Estilo de fundo branco
    
    ax1 = sns.lineplot(
        data=cumulative_long,
        x='Data',
        y='Retorno Acumulado',
        hue='Estratégia/Benchmark',
        palette=MAXWELL_PALETTE_LIST, # Paleta de cores personalizada
        linewidth=2
    )
    
    ax1.set_title('Performance Comparativa (Retorno Acumulado)', fontsize=16)
    ax1.set_xlabel('Data', fontsize=12)
    ax1.set_ylabel('Retorno Acumulado (Índice)', fontsize=12)
    
    # --- MELHORIA DA LEGENDA (Gráfico 1) ---
    ax1.legend(
        title='Ativo',
        loc='upper center',          # Ancora a legenda pela parte superior central
        bbox_to_anchor=(0.5, -0.1),  # Posiciona 10% abaixo do gráfico, no centro (0.5)
        ncol=5,                      # 5 colunas (para os 5 ativos)
        frameon=False                # Remove a moldura
    )
    
    # Adiciona espaço na parte inferior para a legenda caber
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    
    print("\nExibindo Gráfico 1: Retorno Acumulado...")
    plt.show()

    # --- 3. Análise de Max Drawdown ---

    # Os cálculos de drawdown também herdarão os novos nomes
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # O melt aqui também usará os nomes novos
    drawdown_long = drawdown.reset_index().melt(
        'Data',
        var_name='Estratégia/Benchmark',
        value_name='Drawdown'
    )

    # Plotar o Drawdown ao longo do tempo
    plt.figure(figsize=(14, 7))
    
    ax2 = sns.lineplot(
        data=drawdown_long,
        x='Data',
        y='Drawdown',
        hue='Estratégia/Benchmark',
        palette=MAXWELL_PALETTE_LIST, # Paleta de cores personalizada
        linewidth=1.5,
        alpha=0.8
    )

    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.set_title('Drawdown Comparativo ao Longo do Tempo', fontsize=16)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.set_ylabel('Drawdown', fontsize=12)
    
    # --- MELHORIA DA LEGENDA (Gráfico 2) ---
    ax2.legend(
        title='Ativo',
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
        frameon=False
    )
    
    # Adiciona espaço na parte inferior para a legenda caber
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    
    print("\nExibindo Gráfico 2: Drawdown ao Longo do Tempo...")
    plt.show()

    # --- 4. Calcular Métricas (CAGR e Sharpe Ratio) ---
    
    num_days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
    num_years = num_days / 365.25

    start_value = cumulative_returns.iloc[0]
    end_value = cumulative_returns.iloc[-1]
    cagr = (end_value / start_value) ** (1 / num_years) - 1

    trading_days = 252
    risk_free_rate = 0
    
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    sharpe_ratio_daily = (mean_daily_return - risk_free_rate) / std_daily_return
    sharpe_ratio_anual = sharpe_ratio_daily * np.sqrt(trading_days)

    # --- 5. Criar e Salvar CSV com Métricas ---

    # O DataFrame de métricas também usará os nomes novos
    metrics_df = pd.DataFrame({
        'CAGR': cagr,
        'Sharpe_Ratio_Anual': sharpe_ratio_anual,
        'Max_Drawdown': max_drawdown
    })

    metrics_df = metrics_df.round(4)
    
    output_filename = 'metricas_comparativas.csv'
    metrics_df.to_csv(
        output_filename,
        sep=';',
        decimal=',',
        encoding='utf-8-sig'
    )

    print(f"\n--- Métricas Calculadas (incluindo Benchmark) ---")
    print(metrics_df)
    print(f"\nMétricas salvas com sucesso no arquivo: '{output_filename}'")

except FileNotFoundError:
    print(f"Erro: O arquivo '{FILE_NAME}' não foi encontrado.")
except KeyError:
    print(f"Erro: Não foi possível encontrar a coluna de data.")
    print("Verifique se a coluna de data é realmente a 5ª coluna (índice 4) no arquivo.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")