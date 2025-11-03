import pandas as pd
from bcb import sgs
import warnings
import numpy as np

# Ignorar avisos futuros (apenas para limpeza)
warnings.simplefilter(action='ignore', category=FutureWarning)

INPUT_FILE = "retornos_master.csv"
OUTPUT_FILE = "retornos_master_CDI_CORRIGIDO.csv"

print(f"Iniciando a correção do CDI para o arquivo: {INPUT_FILE}")

# --- 1. Carregar o arquivo CSV mestre ---
try:
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado.")
    exit()
except Exception as e:
    print(f"ERRO ao ler o CSV: {e}")
    exit()

if 'CDI' not in df.columns:
    print(f"ERRO: A coluna 'CDI' não foi encontrada no arquivo {INPUT_FILE}.")
    exit()

print("Arquivo CSV carregado com sucesso.")

if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# --- 2. Identificar o período ---
start_dt = df.index.min()
end_dt = df.index.max()
print(f"Período identificado no CSV: {start_dt.date()} até {end_dt.date()}")

# --- 3. Baixar os dados corretos (SGS 12 - Taxa DIÁRIA) ---
print("Baixando dados corretos do CDI (SGS 12) do Banco Central...")
cdi_returns_correct = None

try:
    max_years_per_request = 9
    lista_df_cdi = []
    
    current_start_dt = start_dt
    while current_start_dt <= end_dt:
        current_end_dt = current_start_dt + pd.DateOffset(years=max_years_per_request)
        if current_end_dt > end_dt:
            current_end_dt = end_dt
            
        print(f"  > Baixando de {current_start_dt.date()} até {current_end_dt.date()}...")
        
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
    
    # --- 4. CORREÇÃO PRINCIPAL: Processamento Correto do CDI ---
    
    # VERIFICAÇÃO ROBUSTA: O SGS 12 retorna a taxa DIÁRIA em percentual
    # Exemplo: se CDI = 0.12% ao dia, retorna 0.12
    print(f"CDI - Amostra bruta: {cdi_df['CDI'].head()}")
    
    # Converter de percentual para decimal (0.12% → 0.0012)
    cdi_decimal = cdi_df['CDI'] / 100
    
    # ✅ CORREÇÃO: O CDI JÁ É TAXA DIÁRIA, usar diretamente
    cdi_returns_raw = cdi_decimal  # Este já é o retorno diário!
    
    # Alinhar com as datas do DataFrame
    cdi_returns_correct = cdi_returns_raw.reindex(df.index)
    
    print("Retornos diários do CDI calculados corretamente.")

    # --- 5. Verificação de Sanidade ---
    print("\n--- Verificação das Métricas do CDI ---")
    
    # Dados limpos para cálculo
    cdi_limpo = cdi_returns_correct.dropna()
    
    if not cdi_limpo.empty:
        # Período coberto
        data_inicio = cdi_limpo.index.min()
        data_fim = cdi_limpo.index.max()
        num_dias = len(cdi_limpo)
        num_anos = num_dias / 252
        
        # Estatísticas básicas
        retorno_medio_diario = cdi_limpo.mean()
        vol_diaria = cdi_limpo.std()
        
        # Calcular acumulado para verificação
        acumulado = (1 + cdi_limpo).cumprod()
        retorno_total = acumulado.iloc[-1] - 1
        cagr = (acumulado.iloc[-1] ** (1/num_anos)) - 1
        
        print(f"Período: {data_inicio.date()} a {data_fim.date()}")
        print(f"Dias úteis: {num_dias} (~{num_anos:.1f} anos)")
        print(f"Retorno médio diário: {retorno_medio_diario:.6f} ({retorno_medio_diario:.4%})")
        print(f"Volatilidade diária: {vol_diaria:.6f}")
        print(f"Retorno total acumulado: {retorno_total:.2%}")
        print(f"CAGR (anualizado): {cagr:.2%}")
        
        # Verificação de valores típicos
        print(f"\nValores típicos do CDI diário:")
        print(f"Mínimo: {cdi_limpo.min():.6f} | Máximo: {cdi_limpo.max():.6f}")
        print(f"Percentil 25%: {cdi_limpo.quantile(0.25):.6f} | Mediana: {cdi_limpo.median():.6f}")
        
    else:
        print("AVISO: Não há dados válidos de CDI para verificação!")
        
    print("-------------------------------------------------\n")
        
except Exception as e:
    print(f"ERRO: Falha ao processar dados do CDI: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 6. Substituir a coluna 'CDI' ---
print("Substituindo a coluna 'CDI'...")

if cdi_returns_correct is not None:
    # Preencher NaNs (usar 0 para dias sem CDI pode ser conservador)
    cdi_returns_correct.fillna(0.0, inplace=True)
    
    # Substituir a coluna
    df['CDI'] = cdi_returns_correct
    
    print("Coluna CDI substituída com sucesso!")
else:
    print("ERRO: Não foi possível calcular os retornos do CDI.")
    exit()

# --- 7. Salvar o novo arquivo ---
df.to_csv(OUTPUT_FILE)

print(f"\nProcesso concluído com sucesso!")
print(f"Arquivo salvo como: {OUTPUT_FILE}")

print("\nAmostra dos dados CORRIGIDOS:")
print("Primeiros 5 dias:")
print(df['CDI'].head())
print("\nÚltimos 5 dias:")
print(df['CDI'].tail())

print(f"\nEstatísticas finais do CDI corrigido:")
print(f"Média: {df['CDI'].mean():.6f}")
print(f"Desvio padrão: {df['CDI'].std():.6f}")
print(f"Dias com CDI = 0: {(df['CDI'] == 0).sum()}")