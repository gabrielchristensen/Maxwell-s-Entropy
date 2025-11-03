import pandas as pd
import investpy as inv
import yfinance as yf
import warnings
from bcb import sgs  # Importa a biblioteca do Banco Central

# Ignorar avisos futuros do Pandas/YFinance para uma saída mais limpa
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define o período globalmente
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

# --- 1. Obtenção da Lista de Tickers ---
print("Buscando lista de tickers do investpy...")
try:
    lista_tickers = inv.get_stocks_list("brazil")
except Exception as e:
    print(f"Erro ao buscar lista do Investpy: {e}")
    lista_tickers = []

if not lista_tickers:
    print("Nenhum ticker encontrado. Encerrando.")
    exit()

lista_tickers = [ticker + '.SA' for ticker in lista_tickers]
lista_tickers.append('^BVSP') # Adicionando o Ibovespa
print(f"Total de {len(lista_tickers)} tickers a serem baixados.")

# --- 2. Download dos Dados do yfinance ---
print("Baixando dados históricos do yfinance... Isso pode demorar vários minutos.")
cotacoes_ibovespa = yf.download(lista_tickers, 
                                start=START_DATE, 
                                end=END_DATE, 
                                group_by="ticker",
                                auto_adjust=False,
                                threads=True)

if cotacoes_ibovespa.empty:
    print("Download do yfinance falhou ou retornou dados vazios.")
    exit()

base_yahoo_completa = cotacoes_ibovespa.copy()

# --- 3. Download e Processamento do CDI em Chunks ---
print("Baixando dados do CDI do Banco Central (SGS)...")
cdi_index_aligned = None
try:
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    max_years_per_request = 9 
    lista_df_cdi = []
    
    current_start_dt = start_dt
    while current_start_dt <= end_dt:
        current_end_dt = current_start_dt + pd.DateOffset(years=max_years_per_request)
        if current_end_dt > end_dt:
            current_end_dt = end_dt
            
        print(f"  > Baixando CDI de {current_start_dt.date()} até {current_end_dt.date()}...")
        
        cdi_chunk = sgs.get({'CDI': 12}, 
                            start=current_start_dt.strftime('%Y-%m-%d'), 
                            end=current_end_dt.strftime('%Y-%m-%d'))
        
        if not cdi_chunk.empty:
            lista_df_cdi.append(cdi_chunk)
        
        current_start_dt = current_end_dt + pd.DateOffset(days=1)
        
    if not lista_df_cdi:
        raise Exception("Nenhum dado de CDI foi retornado pelo BCB.")
        
    cdi_df = pd.concat(lista_df_cdi)
    cdi_df = cdi_df[~cdi_df.index.duplicated(keep='first')]

    # Processamento do CDI (criação do ÍNDICE CUMULATIVO)
    cdi_df['CDI'] = cdi_df['CDI'] / 100
    cdi_diario_fator = (1 + cdi_df['CDI'])**(1/252)
    cdi_index = cdi_diario_fator.cumprod() # Gera o índice
    
    # Alinha o índice do CDI com o índice das ações (B3)
    cdi_index_aligned = cdi_index.reindex(base_yahoo_completa.index).ffill().bfill()
    
    print("Dados do CDI processados com sucesso.")

except Exception as e:
    print(f"Aviso: Falha ao baixar ou processar dados do CDI: {e}")
    print("O script continuará sem os dados do CDI.")

# --- 4. Função de Filtro de Ticker Robusta ---
def filtro_ticker_robusto(col_name):
    """Verifica se o nome da coluna é uma ação ON/PN ou Unit válida."""
    if not isinstance(col_name, str):
        return False
    if len(col_name) == 8 and col_name.endswith('.SA'): 
        return True
    if len(col_name) == 9 and col_name.endswith('.SA') and col_name[-5:-3] == '11': 
        return True
    return False

# --- 5. Seleção Correta de Preço e Volume ---
print("Processando dados baixados do yfinance...")
try:
    # df_valor_full conterá PREÇOS, pois é necessário para o filtro de liquidez
    df_valor_full = base_yahoo_completa.xs('Close', level=1, axis=1).astype(float)
    volume_de_negociacao_full = base_yahoo_completa.xs('Volume', level=1, axis=1).astype(float)
except KeyError as e:
    print(f"Erro ao extrair colunas. Verifique se 'Close' e 'Volume' existem: {e}")
    exit()
    
if '^BVSP' in df_valor_full.columns:
    bvsp_precos = df_valor_full[['^BVSP']]
else:
    bvsp_precos = None
    print("Aviso: ^BVSP não encontrado nos dados de preço.")

volume_de_negociacao_full = volume_de_negociacao_full.drop(columns=['^BVSP'], errors='ignore')

# --- 6. Loop Dinâmico (Rolling Window) ---
print("Iniciando processamento em janelas (filtro de liquidez)...")
d1, d2 = 252, 251
num_windows = (df_valor_full.shape[0] - d1) // d2 + 1

for i in range(num_windows):
    start_idx = d2 * i
    end_idx = d2 * i + d1
    
    # Seleciona a janela de PREÇOS para o filtro
    volume_intermediario = volume_de_negociacao_full.iloc[start_idx:end_idx]
    df_valor_intermediario = df_valor_full.iloc[start_idx:end_idx] # PREÇOS

    # Calcular volume financeiro (Volume * PREÇO)
    volume_em_reais_intermediario = volume_intermediario * df_valor_intermediario.drop(columns=['^BVSP'], errors='ignore')
    
    # --- Aplicar Filtros (baseado em Preços e Volume) ---
    volume_em_reais_intermediario.dropna(axis=1, how='all', inplace=True)
    filtro_liquidez = volume_em_reais_intermediario.mean() >= 10**7
    volume_em_reais_intermediario = volume_em_reais_intermediario.loc[:, filtro_liquidez]
    colunas_filtradas_nome = [col for col in volume_em_reais_intermediario.columns if filtro_ticker_robusto(col)]
    
    # --- Preparar Lista Final ---
    lista_tickers_final_janela = sorted(list(set(colunas_filtradas_nome)))
    if bvsp_precos is not None:
        lista_tickers_final_janela.append('^BVSP')

    # --- (MUDANÇA AQUI) Preparar PREÇOS/ÍNDICES para calcular retornos ---
    
    # 1. Selecionar os PREÇOS dos tickers aprovados na janela
    preco_de_fechamento_intermediario = df_valor_intermediario[lista_tickers_final_janela]
    
    # 2. Adicionar o ÍNDICE CDI à janela
    if cdi_index_aligned is not None:
        cdi_janela = cdi_index_aligned.iloc[start_idx:end_idx]
        preco_de_fechamento_intermediario = preco_de_fechamento_intermediario.join(cdi_janela)

    # 3. Limpar Duplicatas e Preencher NaNs (dos PREÇOS/ÍNDICES)
    preco_de_fechamento_intermediario = preco_de_fechamento_intermediario.loc[:, ~preco_de_fechamento_intermediario.columns.duplicated()]
    preco_de_fechamento_intermediario = preco_de_fechamento_intermediario.interpolate(method='linear', axis=0)
    preco_de_fechamento_intermediario = preco_de_fechamento_intermediario.ffill().bfill() 

    # --- (NOVO) Calcular Retornos Diários ---
    # Agora calculamos o pct_change() sobre o DataFrame de preços/índices limpo
    retornos_intermediario = preco_de_fechamento_intermediario.pct_change()
    
    # Remove a primeira linha (que será NaN após o pct_change())
    retornos_intermediario = retornos_intermediario.iloc[1:]

    # --- 7. Salvar CSV (agora de RETORNOS) ---
    num_acoes = len(colunas_filtradas_nome)
    tem_cdi_str = " (com CDI)" if cdi_index_aligned is not None else " (sem CDI)"
    output_filename = f'retornos_diarios_janela_{i}.csv' # Nome do arquivo alterado
    
    print(f"Salvando {output_filename} com {num_acoes} ações + Índices{tem_cdi_str}...")
    retornos_intermediario.to_csv(output_filename)

print("Processamento concluído.")