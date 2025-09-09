# ========================
# signal_analyzer.py (VERSÃO FINAL DE DEPURAÇÃO)
# ========================

import numpy as np
import soundfile as sf
from collections import deque
import os
import csv
import time

class SignalAnalyzer:
    def __init__(self, config, output_csv_path):
        self.FS = config.get('FS', 48000)
        self.BLOCK = config.get('BLOCK', 4096)
        self.HOP = self.BLOCK // 2
        self.BAND_MIN = config.get('BAND_MIN', 500)
        self.BAND_MAX = config.get('BAND_MAX', 18000)
        self.PEAK_THRESH = config.get('PEAK_THRESH', 6.0)
        self.MAX_TRACKS = config.get('MAX_TRACKS', 50)
        self.TIMEOUT_BLOCKS = config.get('TIMEOUT_BLOCKS', 20)
        self.SMOOTH = config.get('SMOOTH', 10)
        self.RHO_WINDOW_SIZE = config.get('RHO_WINDOW_SIZE', 25)
        self.TOL_HZ = 3 * (self.FS / self.BLOCK)

        self.tracks = {}
        self.track_id_counter = 0
        self.bins_f = np.fft.rfftfreq(self.BLOCK, d=1/self.FS)
        band_mask = (self.bins_f >= self.BAND_MIN) & (self.bins_f <= self.BAND_MAX)
        self.band_bins = np.where(band_mask)[0]
        self.output_csv_path = output_csv_path
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.output_csv_path):
            with open(self.output_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["trial_id", "condition", "param_compressao", "rho_mean_abs", "timestamp"])

    def _get_track_velocity(self, track):
        hist = track['finst_hist']
        if len(hist) < 5: return 0.0
        y = np.array(hist)[-5:]
        x = np.arange(len(y)) * (self.HOP / self.FS)
        try:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        except np.linalg.LinAlgError:
            return 0.0

    def _process_block(self, block, n0, block_index):
        print(f"\n=============== BLOCO DE ANÁLISE Nº {block_index} ================")
        x = block.astype(np.float64)
        if np.mean(x**2) < 1e-10: 
            print("--> Bloco com energia muito baixa. Ignorando.")
            return []

        win = np.hanning(len(x))
        X = np.fft.rfft(x * win)
        mag = np.abs(X)
        mag_band = mag[self.band_bins]
        noise_floor = np.median(mag_band) + 1e-12
        norm_band = mag_band / noise_floor
        peak_indices_band = np.where(norm_band > self.PEAK_THRESH)[0]
        detected_peaks = set(self.bins_f[self.band_bins[peak_indices_band]])
        print(f"[DEBUG] Picos detectados ({len(detected_peaks)}): {[f'{p:.1f}' for p in sorted(list(detected_peaks))]}")

        if not detected_peaks:
            print("--> Nenhum pico detectado neste bloco.")
            self._handle_timeouts()
            return []

        dt = self.HOP / self.FS
        print(f"[DEBUG] Tracks existentes antes do processamento: {len(self.tracks)}")
        for track_id, track in self.tracks.items():
            velocity = self._get_track_velocity(track)
            track['predicted_f'] = track['f0'] + velocity * dt
            track['seen'] = False
            print(f"  - (Antes) Track ID {track_id}: f0={track['f0']:.1f}, vel={velocity:.1f} Hz/s, previsto={track['predicted_f']:.1f} Hz, hist_len={len(track['finst_hist'])}")

        associations = []
        unmatched_peaks = set(detected_peaks)
        matched_track_ids = set()

        for track_id, track in self.tracks.items():
            for peak_f in unmatched_peaks:
                dist = abs(track['predicted_f'] - peak_f)
                if dist < self.TOL_HZ:
                    associations.append((dist, track_id, peak_f))
        associations.sort()

        for dist, track_id, peak_f in associations:
            if track_id not in matched_track_ids and peak_f in unmatched_peaks:
                print(f"[DEBUG] Associando Track ID {track_id} com Pico {peak_f:.1f} (distância: {dist:.1f} Hz)")
                track = self.tracks[track_id]
                track['f0'] = peak_f
                track['finst_hist'].append(peak_f)
                track['seen'] = True
                track['miss_count'] = 0
                unmatched_peaks.remove(peak_f)
                matched_track_ids.add(track_id)

        unmatched_track_ids = self.tracks.keys() - matched_track_ids
        print(f"[DEBUG] Tracks que não foram associados: {len(unmatched_track_ids)}")
        for track_id in unmatched_track_ids:
            self.tracks[track_id]['miss_count'] += 1

        print(f"[DEBUG] Picos não associados (virarão novos tracks): {len(unmatched_peaks)}")
        for peak_f in unmatched_peaks:
            if len(self.tracks) < self.MAX_TRACKS:
                self.track_id_counter += 1
                new_id = self.track_id_counter
                self.tracks[new_id] = {
                    'f0': peak_f,
                    'finst_hist': deque([peak_f], maxlen=self.SMOOTH * 5), # Aumentado para reter mais histórico
                    'miss_count': 0, 'seen': True, 'predicted_f': peak_f
                }
                print(f"[DEBUG] Criado novo Track ID {new_id} para o pico {peak_f:.1f} Hz")
        
        self._handle_timeouts()
        
        rho_vals_block = []
        print("[DEBUG] Verificando tracks para cálculo de rho:")
        for track_id, track in self.tracks.items():
            if track.get('seen', False):
                hist_len = len(track['finst_hist'])
                print(f"  - Track ID {track_id}: Visto. Comprimento do histórico: {hist_len}/{self.RHO_WINDOW_SIZE}")
                if hist_len >= self.RHO_WINDOW_SIZE:
                    print(f"    --> CALCULANDO RHO PARA TRACK {track_id}")
                    x_win = np.array(track['finst_hist'])[-self.RHO_WINDOW_SIZE:]
                    tau = np.arange(len(x_win))
                    x_ = x_win - x_win.mean()
                    t_ = tau - tau.mean()
                    denom = np.sqrt((x_**2).sum() * (t_**2).sum())
                    if denom > 1e-9:
                        r = (x_ * t_).sum() / denom
                        rho_vals_block.append(r)

        print(f"[DEBUG] Valores de rho calculados neste bloco: {rho_vals_block}")
        return rho_vals_block

    def _handle_timeouts(self):
        remove_keys = []
        for track_id, track in list(self.tracks.items()):
            if track['miss_count'] >= self.TIMEOUT_BLOCKS:
                remove_keys.append(track_id)
        if remove_keys:
            print(f"[DEBUG] Removendo tracks por timeout: {remove_keys}")
        for k in remove_keys:
            self.tracks.pop(k, None)

    def process_file(self, wav_filepath, trial_id, condition, param_compressao):
        print(f"Processando Trial {trial_id} ({condition}, {param_compressao} Hz/s)... ", end="")
        try:
            audio, fs = sf.read(wav_filepath, dtype='float32')
            if fs != self.FS: raise ValueError("Sample rate mismatch")
            if audio.ndim > 1: audio = audio.mean(axis=1)
        except Exception as e:
            print(f"Erro ao ler arquivo: {e}")
            return

        self.tracks = {}
        all_rho_values = []
        n0 = 0
        
        for i in range(0, len(audio) - self.BLOCK, self.HOP):
            block = audio[i : i + self.BLOCK]
            # Chamando a função de processo com o índice do bloco para depuração
            rho_vals_block = self._process_block(block, n0, block_index=i//self.HOP + 1)
            if rho_vals_block:
                all_rho_values.append(np.mean(rho_vals_block))
            n0 += self.HOP
        
        if not all_rho_values:
            rho_mean_abs = 0.0
        else:
            rho_mean_abs = np.mean(np.abs(all_rho_values))
        
        print(f"Concluído. Média |ρ(t)|: {rho_mean_abs:.4f}")

        with open(self.output_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, condition, param_compressao, rho_mean_abs, time.time()])