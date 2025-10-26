from toolbox import *

print("--- 1. CASOS DE TESTE (ROBUSTEZ) ---")

# Setup: Taxa risk-free constante para os testes
rf_rate = 0.0001
rf_base = np.array([rf_rate] * 10) # Array de 10 dias

# Teste 1: Risco Zero (Excesso de Retorno Constante)
# O ativo rende exatamente 0.05% acima do risk-free, todos os dias.
# Incerteza é zero.
print("\n--- Teste 1: Risco Zero ---")
ativos_constantes = np.array([rf_rate + 0.0005] * 10)
risco_teste_1 = calcular_risk_estimator(ativos_constantes, rf_base, 10)
print(f"Entrada (Excesso): {[0.0005] * 3}...")
print(f"Resultado (Esperado: 0.0): {risco_teste_1}") # np.exp(-inf) == 0.0

# Teste 2: Dados Insuficientes
# Arrays com menos de 2 pontos de dados não podem ter entropia calculada.
print("\n--- Teste 2: Dados Insuficientes ---")
ativos_insuficientes = np.array([0.05])
rf_insuficientes = np.array([rf_rate])
risco_teste_2 = calcular_risk_estimator(ativos_insuficientes, rf_insuficientes, 10)
print(f"Entrada (Pontos): {len(ativos_insuficientes)}")
print(f"Resultado (Esperado: nan): {risco_teste_2}")

# Teste 3: Dados com NaN (Missing Data)
# A função deve ignorar os 'nan' e calcular com os 5 pontos restantes.
print("\n--- Teste 3: Dados com NaN ---")
ativos_com_nan = np.array([0.01, 0.02, np.nan, -0.01, 0.03, np.nan, 0.01])
rf_com_nan = np.array([rf_rate] * len(ativos_com_nan))
risco_teste_3 = calcular_risk_estimator(ativos_com_nan, rf_com_nan, 2)
print(f"Entrada (Pontos): 7 (com 2 NaN)")
print(f"Resultado (Esperado: um número, não NaN): {risco_teste_3:.4f}")

# Teste 4: Comparação Lógica (Volatilidade)
# Comparamos um ativo estável (B) com um volátil (A).
print("\n--- Teste 4: Comparação (Alto Risco vs. Baixo Risco) ---")
rf_longo = np.array([rf_rate] * 100)
# Excesso de retorno com std dev = 0.005 (baixo risco)
ativos_B = np.random.normal(loc=0.0001, scale=0.005, size=100) + rf_rate
# Excesso de retorno com std dev = 0.02 (alto risco)
ativos_A = np.random.normal(loc=0.0001, scale=0.02, size=100) + rf_rate

risco_teste_A = calcular_risk_estimator(ativos_A, rf_longo, 10)
risco_teste_B = calcular_risk_estimator(ativos_B, rf_longo, 10)
print(f"Estimador Risco A (Volátil): {risco_teste_A:.4f}")
print(f"Estimador Risco B (Estável): {risco_teste_B:.4f}")
print(f"Teste (Esperado: Risco A > Risco B): {risco_teste_A > risco_teste_B}")

# Teste 5: Seu Exemplo (que resulta em H negativa)
print("\n--- Teste 5: Seu Exemplo Específico ---")
r = np.array([0.02, -0.03, -0.02, 0.0, -0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1])
rf_teste_5 = np.array([rf_rate] * len(r))
# Primeiro, calculamos a entropia H
H_teste_5 = histogram_entropy_shannon_calculate(r - rf_teste_5, 10)
# Depois, o estimador de risco
risco_teste_5 = calcular_risk_estimator(r, rf_teste_5, 10)
print(f"Entropia (H) (Esperado: < 0): {H_teste_5:.4f}")
print(f"Estimador Risco (exp(H)) (Esperado: > 0): {risco_teste_5:.4f}")