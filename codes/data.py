import yfinance as yf
import pandas as pd

# --- 1. Parâmetros da Estratégia ---
tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'MGLU3.SA', 'BBDC4.SA', 'WEGE3.SA', 'ABEV3.SA']
start_date = '2010-01-01'
end_date = '2024-12-31'
window_size = 252
min_liquidity = 10_000_000  # R$ 10 milhões

# --- 2. Coleta de Dados ---
print(f"Baixando dados para {tickers} de {start_date} até {end_date}...")
try:
    # CORREÇÃO: Usar auto_adjust=True.
    # Isso baixa os preços JÁ AJUSTADOS (para dividendos/splits) 
    # diretamente para a coluna 'Close'.
    # A coluna 'Adj Close' NÃO existirá mais.
    data = yf.download(tickers, 
                       start=start_date, 
                       end=end_date, 
                       auto_adjust=True,  # <--- MUDANÇA PRINCIPAL
                       progress=False)
    
    if data.empty:
        print("Nenhum dado retornado. Verifique os tickers e o período.")
        exit()
except Exception as e:
    print(f"Erro ao baixar dados: {e}")
    exit()

print("Download concluído.")

# --- 3. Pré-Tratamento e Cálculo da Liquidez ---

# O yfinance se comporta de forma diferente para 1 ou N tickers.
# Este bloco normaliza os dados para que o resto do script funcione sempre.
if len(tickers) > 1:
    # Múltiplos tickers: data.columns é MultiIndex ('Close', 'PETR4.SA')
    df_close = data['Close']
    df_volume = data['Volume']
else:
    # Ticker único: data.columns é Index ('Close', 'Volume')
    # Se for ticker único, df_close e df_volume seriam Series.
    # Convertemos para DataFrame para manter a consistência.
    df_close = data[['Close']]
    df_volume = data[['Volume']]
    
    # Renomeia as colunas para o nome do ticker
    df_close.columns = tickers
    df_volume.columns = tickers

# NOTA: 'df_close' agora contém os preços AJUSTADOS.
# O cálculo da liquidez usará (Volume * Preço Ajustado),
# que é uma proxy perfeitamente aceitável para o volume financeiro.
df_financial_volume = df_close * df_volume
df_financial_volume = df_financial_volume.fillna(0)

# --- 4. Lógica da Janela Móvel (Elegibilidade) ---
# Esta parte permanece idêntica

# 1. Calcular a média móvel de 'window_size' dias
rolling_mean_liquidity = df_financial_volume.rolling(window=window_size).mean()

# 2. Aplicar .shift(1) para usar dados de t-1
avg_liquidity_t_minus_1 = rolling_mean_liquidity.shift(1)


# --- 5. Criar a "Máscara" de Ativos Elegíveis ---
df_eligible = (avg_liquidity_t_minus_1 >= min_liquidity)

# --- 6. Resultados ---
print("\n--- Amostra do Volume Financeiro Diário (R$) ---")
print(df_financial_volume.tail())

print(f"\n--- Amostra da Média Móvel de Liquidez ({window_size}d, em t-1) ---")
print(avg_liquidity_t_minus_1.tail())

print("\n--- Máscara de Elegibilidade (True = Pode Investir) ---")
print(df_eligible.tail())

# --- 7. Exemplo de Uso ---

# CORREÇÃO: Não precisamos mais de 'df_adj_close'.
# Os preços ajustados que queremos já estão em 'df_close'.
precos_elegiveis = df_close[df_eligible]

print("\n--- Preços de Fechamento Ajustado (Apenas Elegíveis) ---")
print(precos_elegiveis.tail())