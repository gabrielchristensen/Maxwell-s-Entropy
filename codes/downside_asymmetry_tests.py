from toolbox import *

if __name__ == "__main__":
    
    C_LEVEL_DEFAULT = 0.0
    N_SAMPLES = 2000
    
    print("--- A EXECUTAR CENÁRIOS DE TESTE PARA a FUNÇÃO DOWN_ASY ---")
    
    # --- Testes de Robustez (Lidar com "dados sujos") ---

    print("\n--- TESTES DE ROBUSTEZ (CASOS EXTREMOS) ---")
    
    # Teste 1: Dados Constantes
    test1_asset = np.full(N_SAMPLES, 0.01)
    test1_market = np.full(N_SAMPLES, 0.005)
    
    start_time = time.perf_counter()
    score1 = downside_asymmetry_entropy_calculate(test1_asset, test1_market, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration1 = end_time - start_time
    
    print(f"Teste 1: Dados Constantes (std_dev = 0)")
    print(f"Resultado: {score1} (Esperado: 0.0) (Tempo: {duration1:.4f}s)")

    # Teste 2: Dados Insuficientes (Abaixo do limite 'n_total < 10')
    test2_asset = np.random.randn(9)
    test2_market = np.random.randn(9)
    
    start_time = time.perf_counter()
    score2 = downside_asymmetry_entropy_calculate(test2_asset, test2_market, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration2 = end_time - start_time
    
    print(f"\nTeste 2: Dados Insuficientes (n=9)")
    print(f"Resultado: {score2} (Esperado: 0.0, acionando a 'guard clause') (Tempo: {duration2:.4f}s)")

    # Teste 3: Dados com NaNs (Que reduzem n_total < 10)
    test3_asset = np.random.randn(N_SAMPLES)
    test3_market = np.random.randn(N_SAMPLES)
    test3_asset[10:] = np.nan 
    test3_market[5:15] = np.nan 
    
    start_time = time.perf_counter()
    score3 = downside_asymmetry_entropy_calculate(test3_asset, test3_market, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration3 = end_time - start_time
    
    print(f"\nTeste 3: Dados com NaNs (n_total < 10)")
    print(f"Resultado: {score3} (Esperado: 0.0, acionando a 'guard clause') (Tempo: {duration3:.4f}s)")

    # Teste 4: Dados com NaNs (Mas n_total > 10)
    test4_asset = np.random.randn(N_SAMPLES)
    test4_market = np.random.randn(N_SAMPLES)
    test4_asset[:500] = np.nan # Remove 500 pontos
    
    start_time = time.perf_counter()
    score4 = downside_asymmetry_entropy_calculate(test4_asset, test4_market, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration4 = end_time - start_time
    
    print(f"\nTeste 4: Dados com NaNs (n_total > 10)")
    print(f"Resultado: {score4:.6f} (Esperado: um número, não 0 ou NaN) (Tempo: {duration4:.4f}s)")

    # Teste 5: Sem Eventos de "Baixa" (n_down = 0)
    test5_asset = np.abs(np.random.randn(N_SAMPLES)) # Ativo sempre positivo
    test5_market = np.random.randn(N_SAMPLES)
    
    start_time = time.perf_counter()
    score5 = downside_asymmetry_entropy_calculate(test5_asset, test5_market, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration5 = end_time - start_time
    
    print(f"\nTeste 5: Sem Eventos de Baixa (n_down = 0)")
    print(f"Resultado: {score5:.6f} (Esperado: < 0.0) (Tempo: {duration5:.4f}s)")

    # --- Testes Lógicos (Validar a Teoria) ---

    print("\n\n--- TESTES LÓGICOS (VALIDAR A TEORIA) ---")
    np.random.seed(42) # Seed para reprodutibilidade

    # Teste 6: Simetria Perfeita (Forma e Probabilidade)
    mean_sym = [0, 0]
    cov_sym = [[1, 0.5], [0.5, 1]] # corr = 0.5
    sym_data = np.random.multivariate_normal(mean_sym, cov_sym, N_SAMPLES).T
    X_sym = sym_data[0]
    Y_sym = sym_data[1]
    
    start_time = time.perf_counter()
    score6 = downside_asymmetry_entropy_calculate(X_sym, Y_sym, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration6 = end_time - start_time
    
    print(f"\nTeste 6: Simetria Perfeita (Forma e Probabilidade)")
    print(f"Resultado: {score6:.6f} (Esperado: próximo de 0.0) (Tempo: {duration6:.4f}s)")

    # Teste 7: Assimetria (Probabilidade de Baixa) - O Fator DOWN_ASY
    mean_up = [2, 2]; cov_up = [[1, 0.5], [0.5, 1]]
    data_up = np.random.multivariate_normal(mean_up, cov_up, 600)
    mean_down = [-2, -2]; cov_down = [[1, 0.8], [0.8, 1]]
    data_down = np.random.multivariate_normal(mean_down, cov_down, 1400)
    X_asym_down = np.hstack([data_up[:, 0], data_down[:, 0]])
    Y_asym_down = np.hstack([data_up[:, 1], data_down[:, 1]])

    start_time = time.perf_counter()
    score7 = downside_asymmetry_entropy_calculate(X_asym_down, Y_asym_down, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration7 = end_time - start_time
    
    print(f"\nTeste 7: Assimetria de Baixa (LQP > UQP e S_rho > 0)")
    print(f"Resultado: {score7:.6f} (Esperado: > 0.0, positivo) (Tempo: {duration7:.4f}s)")

    # Teste 8: Assimetria (Probabilidade de Alta)
    mean_up = [2, 2]; cov_up = [[1, 0.5], [0.5, 1]]
    data_up = np.random.multivariate_normal(mean_up, cov_up, 1400)
    mean_down = [-2, -2]; cov_down = [[1, 0.8], [0.8, 1]]
    data_down = np.random.multivariate_normal(mean_down, cov_down, 600)
    X_asym_up = np.hstack([data_up[:, 0], data_down[:, 0]])
    Y_asym_up = np.hstack([data_up[:, 1], data_down[:, 1]])

    start_time = time.perf_counter()
    score8 = downside_asymmetry_entropy_calculate(X_asym_up, Y_asym_up, C_LEVEL_DEFAULT,10)
    end_time = time.perf_counter()
    duration8 = end_time - start_time

    print(f"\nTeste 8: Assimetria de Alta (UQP > LQP e S_rho > 0)")
    print(f"Resultado: {score8:.6f} (Esperado: < 0.0, negativo) (Tempo: {duration8:.4f}s)")