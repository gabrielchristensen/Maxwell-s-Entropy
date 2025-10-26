import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import requests
warnings.filterwarnings('ignore')

def create_data_directory():
    """Cria diretório para armazenar os dados se não existir"""
    directories = ['data/raw', 'data/processed', 'data/returns']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Diretório criado: {directory}")

def get_cdi_data(start_date='2010-01-01', end_date=None):
    """
    Obtém dados históricos do CDI com retornos diários corretos
    """
    if end_date is None:
        end_date = datetime.today().strftime("%d/%m/%Y")
    
    try:
        # API do Banco Central para CDI diário
        url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        params = {
            'formato': 'json',
            'dataInicial': start_date,
            'dataFinal': end_date
        }
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("Nenhum dado retornado pela API do BCB")
        
        # Converte para DataFrame
        cdi_df = pd.DataFrame(data)
        cdi_df['data'] = pd.to_datetime(cdi_df['data'], dayfirst=True)
        cdi_df['valor'] = cdi_df['valor'].astype(float) / 100  # Converte para decimal
        
        # Cria índice com todos os dias úteis no período
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business days
        
        # Cria DataFrame com dias úteis
        cdi_daily = pd.DataFrame(index=date_range)
        cdi_daily['CDI'] = np.nan
        
        # Preenche com os dados do CDI
        for _, row in cdi_df.iterrows():
            date = row['data']
            if date in cdi_daily.index:
                taxa_diaria = row['valor'] / 100  # Já está em decimal, divide por 100 para taxa diária
                cdi_daily.loc[date, 'CDI'] = taxa_diaria
        
        # Preenche valores faltantes com o último valor conhecido
        cdi_daily['CDI'] = cdi_daily['CDI'].ffill()
        
        print(f"✓ Dados do CDI obtidos com sucesso: {len(cdi_df)} registros")
        return cdi_daily
    
    except Exception as e:
        print(f"✗ Erro ao obter dados do CDI via API: {e}")
        print("Tentando método alternativo...")
        
        try:
            # Método alternativo: usar dados do Yahoo Finance ou simular CDI
            # Vamos simular o CDI baseado em uma taxa anual média histórica
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            cdi_daily = pd.DataFrame(index=date_range)
            
            # Taxas anuais históricas aproximadas do CDI
            cdi_annual_rates = {
                '2010': 0.1015, '2011': 0.1096, '2012': 0.0755, '2013': 0.0805,
                '2014': 0.1105, '2015': 0.1425, '2016': 0.1375, '2017': 0.0725,
                '2018': 0.0655, '2019': 0.0535, '2020': 0.0195, '2021': 0.0275,
                '2022': 0.1325, '2023': 0.1185, '2024': 0.1050
            }
            
            cdi_values = []
            for date in date_range:
                year = str(date.year)
                if year in cdi_annual_rates:
                    taxa_anual = cdi_annual_rates[year]
                else:
                    taxa_anual = 0.10  # Fallback 10% ao ano
                
                # Converte taxa anual para diária (base 252 dias úteis)
                taxa_diaria = (1 + taxa_anual) ** (1/252) - 1
                cdi_values.append(taxa_diaria)
            
            cdi_daily['CDI'] = cdi_values
            print("✓ CDI simulado com base em taxas anuais históricas")
            return cdi_daily
            
        except Exception as e2:
            print(f"✗ Erro no método alternativo: {e2}")
            print("Usando fallback: CDI constante de 10% ao ano")
            
            # Fallback final: CDI constante
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            cdi_daily = pd.DataFrame(index=date_range)
            taxa_anual = 0.10  # 10% ao ano
            taxa_diaria = (1 + taxa_anual) ** (1/252) - 1
            cdi_daily['CDI'] = taxa_diaria
            
            return cdi_daily

def get_ibov_tickers():
    """
    Obtém a lista atual de tickers que compõem o IBOV
    """
    # URL da composição do IBOV (B3)
    url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraQuadrimestre.aspx?Indice=IBOV&idioma=pt-br'
    
    try:
        # Tenta obter a lista atual da B3
        tables = pd.read_html(url, decimal=',', thousands='.')
        df_composicao = tables[0].copy()
        
        # Ajusta os tickers para o formato do yfinance
        tickers = [f"{ticker.strip()}.SA" for ticker in df_composicao['Código']]
        return tickers
        
    except:
        # Fallback: lista de tickers conhecidos (atualize periodicamente)
        print("Não foi possível obter a lista atual do IBOV. Usando lista fallback.")
        tickers_fallback = [
            'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'B3SA3.SA',
            'ABEV3.SA', 'WEGE3.SA', 'MGLU3.SA', 'RENT3.SA', 'BBAS3.SA',
            'BBDC3.SA', 'JBSS3.SA', 'ITSA4.SA', 'RADL3.SA', 'LREN3.SA',
            'SANB11.SA', 'EQTL3.SA', 'GGBR4.SA', 'CSAN3.SA', 'VBBR3.SA',
            'HAPV3.SA', 'BRFS3.SA', 'SBSP3.SA', 'RAIL3.SA', 'KLBN11.SA',
            'UGPA3.SA', 'TIMS3.SA', 'CCRO3.SA', 'AZUL4.SA', 'CYRE3.SA',
            'EMBR3.SA', 'ELET3.SA', 'ELET6.SA', 'GOAU4.SA', 'CSNA3.SA',
            'USIM5.SA', 'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA', 'PCAR3.SA',
            'QUAL3.SA', 'TOTS3.SA', 'VIVT3.SA', 'WIZS3.SA', 'YDUQ3.SA'
        ]
        return tickers_fallback

def get_ibov_data(start_date='2010-01-01', end_date=None, save_to_csv=True):
    """
    Obtém dados de preço de fechamento ajustado do IBOV e seus componentes
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Obtém tickers do IBOV
    ibov_tickers = get_ibov_tickers()
    
    # Adiciona o IBOV como benchmark
    all_tickers = ibov_tickers + ['^BVSP']
    
    print(f"Obtendo dados para {len(all_tickers)} ativos...")
    print(f"Período: {start_date} até {end_date}")
    
    # Baixa os dados
    data = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Cria dataframe com preços de fechamento ajustado
    close_prices = pd.DataFrame()
    
    for ticker in all_tickers:
        try:
            if ticker in data:
                # Usa Adjusted Close se disponível, senão usa Close
                if 'Adj Close' in data[ticker]:
                    close_prices[ticker] = data[ticker]['Adj Close']
                else:
                    close_prices[ticker] = data[ticker]['Close']
                print(f"✓ {ticker}")
            else:
                print(f"✗ {ticker} - Dados não encontrados")
        except Exception as e:
            print(f"✗ {ticker} - Erro: {e}")
    
    # Remove colunas vazias
    close_prices = close_prices.dropna(axis=1, how='all')
    
    # Renomeia a coluna do IBOV
    if '^BVSP' in close_prices.columns:
        close_prices = close_prices.rename(columns={'^BVSP': 'IBOV'})
    
    print(f"\nDados obtidos com sucesso para {len(close_prices.columns)} ativos")
    print(f"Período final: {close_prices.index[0].strftime('%Y-%m-%d')} até {close_prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Salva os dados brutos em CSV
    if save_to_csv:
        create_data_directory()
        filename = f"data/raw/ibov_close_prices_{start_date}_to_{end_date.replace('-', '')}.csv"
        close_prices.to_csv(filename)
        print(f"\nDados de preços salvos em: {filename}")
        
        # Salva também um arquivo com metadados
        metadata = {
            'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'periodo_inicio': start_date,
            'periodo_fim': end_date,
            'total_ativos': len(close_prices.columns),
            'dias_uteis': len(close_prices),
            'tickers_incluidos': list(close_prices.columns)
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_filename = f"data/raw/metadata_prices_{start_date}_to_{end_date.replace('-', '')}.csv"
        metadata_df.to_csv(metadata_filename, index=False)
        print(f"Metadados salvos em: {metadata_filename}")
    
    return close_prices

def calculate_returns_with_cdi(prices_df, cdi_df, save_to_csv=True):
    """
    Calcula retornos diários incluindo CDI de forma correta
    """
    print("\nCalculando retornos diários...")
    
    # Garante que estamos trabalhando apenas com dias úteis
    prices_df = prices_df.asfreq('B')
    cdi_df = cdi_df.asfreq('B')
    
    # Combina os dados
    combined_df = prices_df.copy()
    
    # Adiciona CDI aos preços (para cálculo consistente)
    combined_df['CDI'] = cdi_df['CDI']
    
    # Calcula retornos percentuais para ações e IBOV
    returns_df = pd.DataFrame()
    
    for column in combined_df.columns:
        if column == 'CDI':
            # Para CDI, já temos a taxa diária diretamente
            returns_df[column] = combined_df[column]
        else:
            # Para ações e IBOV, calcula retorno percentual
            returns_df[column] = combined_df[column].pct_change()
    
    # Remove a primeira linha com NaN
    returns_df = returns_df.iloc[1:]
    
    # Preenche eventuais NaN com 0
    returns_df = returns_df.fillna(0)
    
    # Trata retornos extremos apenas para ações (não para IBOV e CDI)
    extreme_returns_count = 0
    for column in returns_df.columns:
        if column not in ['IBOV', 'CDI']:
            extreme_returns = (returns_df[column] > 1.0) | (returns_df[column] < -0.8)
            extreme_returns_count += extreme_returns.sum()
            if extreme_returns.any():
                median_return = returns_df[column].median()
                returns_df.loc[extreme_returns, column] = median_return
    
    print(f"Retornos extremos tratados: {extreme_returns_count} valores")
    
    # Reorganiza colunas: IBOV e CDI primeiro
    column_order = []
    if 'IBOV' in returns_df.columns:
        column_order.append('IBOV')
    if 'CDI' in returns_df.columns:
        column_order.append('CDI')
    
    # Adiciona as outras colunas (ações)
    other_columns = [col for col in returns_df.columns if col not in ['IBOV', 'CDI']]
    column_order.extend(sorted(other_columns))
    
    returns_df = returns_df[column_order]
    
    # Verifica se o CDI tem valores não nulos
    if 'CDI' in returns_df.columns:
        cdi_non_zero = (returns_df['CDI'] != 0).sum()
        cdi_mean = returns_df['CDI'].mean()
        print(f"CDI - Valores não nulos: {cdi_non_zero}/{len(returns_df)}")
        print(f"CDI - Retorno médio diário: {cdi_mean:.6f} ({cdi_mean*100:.4f}%)")
    
    return returns_df

def get_complete_returns_data(start_date='2010-01-01', end_date=None, save_to_csv=True):
    """
    Função principal que obtém todos os dados: IBOV, ações e CDI
    e retorna dataframe com retornos diários
    """
    print("=== OBTENDO DADOS COMPLETOS ===")
    
    # Obtém dados do IBOV e ações
    print("\n1. Obtendo dados do IBOV e ações...")
    prices_df = get_ibov_data(start_date=start_date, end_date=end_date, save_to_csv=save_to_csv)
    
    # Obtém dados do CDI
    print("\n2. Obtendo dados do CDI...")
    cdi_df = get_cdi_data(start_date=start_date, end_date=end_date)
    
    # Combina os dados e calcula retornos
    print("\n3. Combinando dados e calculando retornos...")
    returns_df = calculate_returns_with_cdi(prices_df, cdi_df, save_to_csv=save_to_csv)
    
    # Salva dados finais de retornos
    if save_to_csv:
        create_data_directory()
        start_date_str = returns_df.index[0].strftime('%Y-%m-%d')
        end_date_str = returns_df.index[-1].strftime('%Y-%m-%d')
        
        filename = f"data/returns/complete_returns_{start_date_str}_to_{end_date_str.replace('-', '')}.csv"
        returns_df.to_csv(filename)
        print(f"\nDados completos de retornos salvos em: {filename}")
        
        # Salva estatísticas dos retornos
        stats = returns_df[['IBOV', 'CDI']].describe() if 'IBOV' in returns_df.columns and 'CDI' in returns_df.columns else returns_df.describe()
        stats_filename = f"data/returns/statistics_returns_{start_date_str}_to_{end_date_str.replace('-', '')}.csv"
        stats.to_csv(stats_filename)
        print(f"Estatísticas salvas em: {stats_filename}")
        
        # Salva metadados completos
        metadata = {
            'data_geracao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'periodo_inicio': start_date_str,
            'periodo_fim': end_date_str,
            'total_ativos': len(returns_df.columns),
            'dias_uteis': len(returns_df),
            'inclui_ibov': 'IBOV' in returns_df.columns,
            'inclui_cdi': 'CDI' in returns_df.columns,
            'retorno_medio_ibov': returns_df['IBOV'].mean() if 'IBOV' in returns_df.columns else None,
            'retorno_medio_cdi': returns_df['CDI'].mean() if 'CDI' in returns_df.columns else None,
            'acoes_incluidas': len([col for col in returns_df.columns if col not in ['IBOV', 'CDI']])
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_filename = f"data/returns/metadata_complete_{start_date_str}_to_{end_date_str.replace('-', '')}.csv"
        metadata_df.to_csv(metadata_filename, index=False)
        print(f"Metadados completos salvos em: {metadata_filename}")
    
    return returns_df

def load_saved_data(periodo=None, data_type='returns'):
    """
    Carrega dados salvos anteriormente
    """
    create_data_directory()
    
    if data_type == 'raw':
        directory = 'data/raw'
        prefix = 'ibov_close_prices'
    elif data_type == 'processed':
        directory = 'data/processed'
        prefix = 'ibov_prices_treated'
    elif data_type == 'returns':
        directory = 'data/returns'
        prefix = 'complete_returns'
    else:
        raise ValueError("data_type deve ser 'raw', 'processed' ou 'returns'")
    
    # Lista arquivos disponíveis
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.csv')]
    
    if not files:
        print(f"Nenhum arquivo encontrado em {directory}")
        return None
    
    if periodo:
        filename = f"{prefix}_{periodo}.csv"
        if filename not in files:
            print(f"Arquivo {filename} não encontrado. Arquivos disponíveis:")
            for f in files:
                print(f"  - {f}")
            return None
    else:
        # Pega o arquivo mais recente
        filename = sorted(files)[-1]
    
    filepath = os.path.join(directory, filename)
    print(f"Carregando dados de: {filepath}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

# Exemplo de uso
if __name__ == "__main__":
    # Obtém todos os dados completos (IBOV, ações e CDI)
    returns_complete = get_complete_returns_data(start_date='2012-01-01', save_to_csv=True)
    
    print("\n=== RESUMO FINAL ===")
    print(f"Dataframe de Retornos: {returns_complete.shape}")
    
    # Verifica se IBOV e CDI estão presentes e com dados
    if 'IBOV' in returns_complete.columns:
        ibov_stats = returns_complete['IBOV'].describe()
        print(f"\n✓ IBOV - Retorno médio diário: {returns_complete['IBOV'].mean():.6f}")
    
    if 'CDI' in returns_complete.columns:
        cdi_stats = returns_complete['CDI'].describe()
        print(f"✓ CDI - Retorno médio diário: {returns_complete['CDI'].mean():.6f}")
        print(f"✓ CDI - Dias com retorno não nulo: {(returns_complete['CDI'] != 0).sum()}/{len(returns_complete)}")
    
    print(f"\nPrimeiras 5 linhas (IBOV e CDI):")
    print(returns_complete[['IBOV', 'CDI']].head() if 'IBOV' in returns_complete.columns and 'CDI' in returns_complete.columns else returns_complete.head())
    
    # Exemplo de carregamento
    print("\n=== EXEMPLO DE CARREGAMENTO ===")
    dados_carregados = load_saved_data(data_type='returns')
    if dados_carregados is not None:
        print(f"Dados carregados: {dados_carregados.shape}")
        if 'CDI' in dados_carregados.columns:
            print(f"CDI - Retorno médio: {dados_carregados['CDI'].mean():.6f}")