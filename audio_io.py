import numpy as np
from scipy.io import wavfile
from pathlib  import Path
import wave, struct, io

SR_ALVO = 16000


def carregar_wav(caminho: str) -> tuple:
    """
    Lê um arquivo WAV e retorna (sinal float32, sr).
    Converte para mono e normaliza para [-1, 1].
    """
    sr, data = wavfile.read(caminho)

    # Converte para float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)

    # Mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != SR_ALVO:
        data = _reamostrar(data, sr, SR_ALVO)

    # Normalizar
    mx = np.abs(data).max()
    if mx > 0:
        data = data / mx

    return data, SR_ALVO

def carregar_bytes(raw_bytes: bytes) -> tuple:
    """
    Carrega áudio a partir de bytes WAV (ex.: upload HTTP).
    Retorna (sinal float32, sr).
    """
    buf = io.BytesIO(raw_bytes)
    return carregar_wav(buf)

def salvar_wav(caminho: str, sinal: np.ndarray, sr: int = SR_ALVO) -> None:
    """Salva sinal float32 como WAV 16-bit."""
    sinal_int = np.clip(sinal * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(caminho, sr, sinal_int)

def _reamostrar(sinal: np.ndarray, sr_orig: int, sr_alvo: int) -> np.ndarray:
    """Reamostragem linear simples (sem librosa)."""
    n_orig  = len(sinal)
    n_alvo  = int(n_orig * sr_alvo / sr_orig)
    indices = np.linspace(0, n_orig - 1, n_alvo)
    return np.interp(indices, np.arange(n_orig), sinal).astype(np.float32)

def bytes_para_array(raw: bytes, sr: int = SR_ALVO,
                     n_canais: int = 1,
                     bits: int = 16) -> np.ndarray:
    """
    Converte bytes PCM bruto para array float32.
    """
    dtype = np.int16 if bits == 16 else np.int32
    data  = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if bits == 16:
        data /= 32768.0
    else:
        data /= 2147483648.0
    if n_canais > 1:
        data = data.reshape(-1, n_canais).mean(axis=1)
    if len(data) == 0:
        return np.zeros(SR_ALVO, dtype=np.float32)
    return data
