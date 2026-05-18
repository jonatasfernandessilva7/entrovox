import numpy as np

def reconstruir(magnitudes: np.ndarray,
                fases:      np.ndarray,
                mascara:    np.ndarray,
                tamanho_quadro: int,
                hop:        int) -> np.ndarray:
    """
    Reconstrói o sinal temporal via IFFT + OLA normalizada.

    Parâmetros
    ----------
    magnitudes     : (n_quadros, n_bins)
    fases          : (n_quadros, n_bins)
    mascara        : (n_quadros,) booleano
    tamanho_quadro : amostras por quadro
    hop            : amostras de avanço entre quadros

    Retorno
    -------
    sinal : array 1-D
    """
    n_quadros  = magnitudes.shape[0]
    comprimento = (n_quadros - 1) * hop + tamanho_quadro
    sinal      = np.zeros(comprimento, dtype=np.float64)
    norm       = np.zeros(comprimento, dtype=np.float64)
    janela     = np.hanning(tamanho_quadro)

    for i in range(n_quadros):
        inicio = i * hop
        fim    = inicio + tamanho_quadro

        if mascara[i]:
            espectro = magnitudes[i] * np.exp(1j * fases[i])
            quadro   = np.real(np.fft.irfft(espectro, n=tamanho_quadro))
            quadro   = quadro[:tamanho_quadro]
        else:
            quadro = np.zeros(tamanho_quadro)

        sinal[inicio:fim] += quadro * janela
        norm [inicio:fim] += janela ** 2

    # Normalização OLA: evita divisão por zero nas bordas
    norm = np.where(norm > 1e-8, norm, 1.0)
    return (sinal / norm).astype(np.float32)
