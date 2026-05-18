import numpy as np

SR_PADRAO    = 16000
DURACAO_MS   = 25.0
HOP_MS       = 10.0

def segmentar(sinal: np.ndarray,
              sr: int = SR_PADRAO,
              duracao_ms: float = DURACAO_MS,
              hop_ms: float = HOP_MS) -> np.ndarray:
    """
    Divide `sinal` em quadros sobrepostos com janela de Hanning.

    Retorna array 2-D (n_quadros, tamanho_quadro).
    """
    tamanho = int(sr * duracao_ms / 1000)
    hop     = int(sr * hop_ms    / 1000)
    janela  = np.hanning(tamanho)

    n_quadros = max(1, 1 + (len(sinal) - tamanho) // hop)
    pad       = max(0, (n_quadros - 1) * hop + tamanho - len(sinal))
    if pad:
        sinal = np.concatenate([sinal, np.zeros(pad)])

    return np.array([
        sinal[i * hop: i * hop + tamanho] * janela
        for i in range(n_quadros)
    ])

def tamanho_quadro(sr: int = SR_PADRAO, duracao_ms: float = DURACAO_MS) -> int:
    return int(sr * duracao_ms / 1000)

def hop_amostras(sr: int = SR_PADRAO, hop_ms: float = HOP_MS) -> int:
    return int(sr * hop_ms / 1000)
