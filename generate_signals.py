# ========================
# generate_signals.py (Ajustado para aceitar variação de fase)
# ========================

import numpy as np
import soundfile as sf

def gerar_sinal_protocolo_A(duracao_s, fs=48000, n_freqs=50, f_min=1000, f_max=15000):
    """
    Gera um sinal com múltiplas frequências e fases aleatórias (baixa compressão).
    """
    freqs = np.linspace(f_min, f_max, n_freqs)
    amplitudes = np.ones_like(freqs) / n_freqs
    fases = np.random.uniform(0, 2 * np.pi, size=n_freqs) # Fases aleatórias
    
    t = np.linspace(0, duracao_s, int(duracao_s * fs), endpoint=False)
    sinal = np.zeros_like(t, dtype=np.float32)
    
    for f, a, p in zip(freqs, amplitudes, fases):
        sinal += a * np.sin(2 * np.pi * f * t + p)
        
    sinal /= np.sqrt(np.mean(sinal**2))
    return sinal

def gerar_sinal_protocolo_B(duracao_s, fs=48000, chirp_rate=500, n_freqs=50, f_min=1000, f_max=15000, vary_phases=False):
    """
    Gera um sinal chirpado.
    - vary_phases: se True, usa fases iniciais aleatórias para as portadoras.
                   se False, usa fases alinhadas (zero).
    """
    freqs_start = np.linspace(f_min, f_max, n_freqs)
    amplitudes = np.ones_like(freqs_start) / n_freqs
    
    if vary_phases:
        fases = np.random.uniform(0, 2 * np.pi, size=n_freqs)
    else:
        fases = np.zeros_like(freqs_start) # Fases alinhadas
    
    t = np.linspace(0, duracao_s, int(duracao_s * fs), endpoint=False)
    sinal = np.zeros_like(t, dtype=np.float32)
    
    for f_start, a, p in zip(freqs_start, amplitudes, fases):
        phase_term = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2) + p
        sinal += a * np.sin(phase_term)
        
    sinal /= np.sqrt(np.mean(sinal**2))
    return sinal