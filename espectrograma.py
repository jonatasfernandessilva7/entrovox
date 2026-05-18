import numpy as np

def espectrograma(quadros: np.ndarray, n_fft: int = None):
    """
    Aplica rfft em cada quadro.

    Retorna
    -------
    magnitudes : (n_quadros, n_bins)  — valores ≥ 0
    fases      : (n_quadros, n_bins)  — ângulos em radianos
    """
    if n_fft is None:
        n_fft = quadros.shape[1]
    espectro   = np.fft.rfft(quadros, n=n_fft, axis=1)
    magnitudes = np.abs(espectro)
    fases      = np.angle(espectro)
    return magnitudes, fases


def n_bins(tamanho_quadro: int) -> int:
    return tamanho_quadro // 2 + 1
