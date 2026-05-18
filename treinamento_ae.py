import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import segmentar, tamanho_quadro, hop_amostras
from pipeline import espectrograma
from pipeline import aplicar_wiener
from pipeline import extrair_embedding
from pipeline import AutoencoderCondicional

SR, DURACAO_MS, HOP_MS = 16000, 25, 10
DIR_MODELOS = Path(__file__).parent / "models"
NIVEIS_SNR  = [-5, 0, 5, 10]
TIPOS_RUIDO = ["branco", "rosa", "sala"]
FREQS       = [120, 180, 240, 300, 380, 450, 550, 700]

def adicionar_ruido(sinal, snr_db, tipo="branco", seed=0):
    rng   = np.random.default_rng(abs(seed))
    n     = len(sinal)
    pot_s = np.mean(sinal**2) + 1e-12
    if tipo == "rosa":
        b = rng.standard_normal(n)
        f = np.fft.rfftfreq(n); f[0] = 1e-6
        ruido = np.fft.irfft(np.fft.rfft(b) / np.sqrt(f), n=n).astype(np.float32)
    elif tipo == "sala":
        b = rng.standard_normal(n).astype(np.float32)
        ruido = np.convolve(b, np.ones(50)/50, mode="same").astype(np.float32)
    else:
        ruido = rng.standard_normal(n).astype(np.float32)
    pot_r = np.mean(ruido**2) + 1e-12
    ruido *= np.sqrt(pot_s / (10**(snr_db/10)) / pot_r)
    return np.clip(sinal + ruido, -1.0, 1.0).astype(np.float32)

def gerar_sinal_limpo(freq, sr=SR, dur=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, dur, int(sr*dur), endpoint=False)
    s   = sum((1/(k+1)) * np.sin(2*np.pi*freq*(k+1)*t) for k in range(5))
    s  += 0.01 * rng.standard_normal(len(t))
    return (s / (np.abs(s).max() + 1e-12)).astype(np.float32)

def mag(sinal):
    q, _ = espectrograma(segmentar(sinal, SR, DURACAO_MS, HOP_MS))
    return q

def gerar_dataset(n_por_freq=15):
    Xr, Xl, embs = [], [], []
    seed = 0
    for freq in FREQS:
        refs   = [gerar_sinal_limpo(freq, seed=seed+k) for k in range(3)]
        emb_r  = extrair_embedding(np.concatenate([aplicar_wiener(mag(r)) for r in refs]))
        for i in range(n_por_freq):
            sl   = gerar_sinal_limpo(freq, seed=seed+i+100)
            ml   = aplicar_wiener(mag(sl))
            for snr in NIVEIS_SNR:
                for tipo in TIPOS_RUIDO:
                    mr = aplicar_wiener(mag(adicionar_ruido(sl, snr, tipo, seed+i+snr*7)))
                    n  = min(mr.shape[0], ml.shape[0])
                    Xr.append(mr[:n]); Xl.append(ml[:n])
                    embs.append(np.tile(emb_r, (n, 1)))
        seed += 50
    return np.concatenate(Xr), np.concatenate(Xl), np.concatenate(embs)

def treinar(n_por_freq=15, dim_latente=128, epocas=100, lr=3e-4, seed=42):
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    print(f"Gerando dataset: {len(FREQS)} freqs × {n_por_freq} sinais × "
          f"{len(NIVEIS_SNR)} SNRs × {len(TIPOS_RUIDO)} ruidos...")
    Xr, Xl, embs = gerar_dataset(n_por_freq)
    print(f"  {Xr.shape[0]:,} quadros | bins={Xr.shape[1]} | emb={embs.shape[1]}")
    ae = AutoencoderCondicional()
    print(f"Treinando ({epocas} epocas, lr={lr}, latente={dim_latente})...")
    hist = ae.treinar(Xr, Xl, embs, dim_latente=dim_latente,
                      epocas=epocas, lr=lr, seed=seed)
    caminho = DIR_MODELOS / "autoencoder.npz"
    ae.salvar(str(caminho))
    print(f"Salvo em {caminho} | loss: {hist[0]:.6f} -> {hist[-1]:.6f}")
    return {"loss_inicial": hist[0], "loss_final": hist[-1],
            "epocas": len(hist), "n_quadros": Xr.shape[0]}


if __name__ == "__main__":
    treinar()
