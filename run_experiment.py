# ========================
# run_experiment.py (Ajustado para Dependência Monotônica)
# ========================

import os
import random
import soundfile as sf
import numpy as np
from signal_analyzer import SignalAnalyzer
from generate_signals import gerar_sinal_protocolo_A, gerar_sinal_protocolo_B

# --- Parâmetros do Experimento ---
# Lista de "causas" a serem testadas. O valor 0 é o nosso controle (Protocolo A).
LISTA_CHIRP_RATES = [0, 100, 250, 500, 750] 
N_TRIALS_POR_TAXA = 20  # Número de repetições para cada taxa de chirp

DURACAO_S = 10  # Duração de cada sinal (s)
FS = 48000

# Salva o arquivo de resultados no mesmo diretório do script
OUTPUT_CSV_PATH = "C:/Users/ATM2\Desktop/Leis fisicas com espectro/Revisão/Final_Revisão_erros/Rho(t)_Como_Prova_Gemini/segundo teste falsicabilidade/resultados_dependencia_monotonica.csv"
TEMP_WAV_DIR = "temp_signals"

# --- Controles de Variabilidade para Robustez ---
# Adiciona fases aleatórias ao sinal B para garantir que o efeito venha do chirp, não do alinhamento em t=0.
VARY_PHASES_B = True
# Adiciona um pequeno desvio aleatório ao chirp rate em cada trial para evitar artefatos de um valor específico.
CHIRP_JITTER_SIGMA = 25.0  # Desvio padrão do jitter em Hz/s. Use 0.0 para desligar.

# --- Configuração do Analisador ---
ANALYZER_CONFIG = {
    'FS': FS,
    'BLOCK': 4096,
    'BAND_MIN': 500,
    'BAND_MAX': 18000,
    'PEAK_THRESH': 6.0,
    'MAX_TRACKS': 50,
    'SMOOTH': 10,
    'RHO_WINDOW_SIZE': 25
}

def main():
    # Resetar CSV se já existir
    if os.path.exists(OUTPUT_CSV_PATH):
        os.remove(OUTPUT_CSV_PATH)
        print(f"Arquivo anterior '{OUTPUT_CSV_PATH}' removido.")

    # Criar diretório temporário
    if not os.path.exists(TEMP_WAV_DIR):
        os.makedirs(TEMP_WAV_DIR)
        print(f"Diretório '{TEMP_WAV_DIR}' criado.")
        
    # --- Criação da Lista de Tarefas ---
    tasks = []
    for rate in LISTA_CHIRP_RATES:
        for i in range(N_TRIALS_POR_TAXA):
            condition = 'A' if rate == 0 else 'B'
            # O trial_id é único para cada execução
            tasks.append({'trial_id': len(tasks) + 1, 'condition': condition, 'param_compressao': rate})
    
    random.shuffle(tasks)  # Randomizar a ordem de todas as execuções
    
    # --- Instanciar o Analisador ---
    analyzer = SignalAnalyzer(ANALYZER_CONFIG, OUTPUT_CSV_PATH)
    
    # --- Execução do Loop Experimental ---
    total_tasks = len(tasks)
    for i, task in enumerate(tasks):
        print("-" * 50)
        print(f"Executando tarefa {i+1}/{total_tasks}: Condição={task['condition']}, Chirp Rate Nominal={task['param_compressao']} Hz/s")
        
        trial_id = task['trial_id']
        condition = task['condition']
        param_nominal = task['param_compressao']
        
        if condition == 'A':
            sinal = gerar_sinal_protocolo_A(DURACAO_S, fs=FS)
            param_efetivo = 0
        else:  # condition == 'B'
            # Aplica o jitter ao chirp rate para maior robustez
            param_efetivo = param_nominal + np.random.normal(0, CHIRP_JITTER_SIGMA)
            # Garante que o chirp não seja negativo
            param_efetivo = max(0, param_efetivo)
            
            sinal = gerar_sinal_protocolo_B(
                DURACAO_S, fs=FS, chirp_rate=param_efetivo,
                vary_phases=VARY_PHASES_B
            )
            
        temp_wav_path = os.path.join(TEMP_WAV_DIR, f"temp_task_{i+1}.wav")
        sf.write(temp_wav_path, sinal, FS)
        
        # Processar o arquivo, salvando o chirp rate nominal para a análise
        analyzer.process_file(temp_wav_path, trial_id, condition, param_nominal)
        
        # Limpar o arquivo temporário
        os.remove(temp_wav_path)
        
    print("-" * 50)
    print("Experimento de Dependência Monotônica concluído!")
    print(f"Resultados salvos em: '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()