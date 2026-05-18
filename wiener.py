"""
RF03 — Supressão Inicial de Ruído (Filtro de Wiener + Subtração Espectral)

Trata o modelo de convolução do canal acústico (tubo de Eustáquio):

  x(t) = s(t) * h(t) + n(t)

onde:
  s(t) = sinal de fala da fonte
  h(t) = resposta ao impulso do meio (sala, canal)
  n(t) = ruído aditivo estacionário

Esta implementação trata o ruído aditivo n(t) via Wiener espectral.
A deconvolução de h(t) (cepstrum ou WPE) é extensão futura.

Caso especial — frequências próximas (anotações):
  Se dois componentes de frequência distam menos de Δf = SR/N_quadro,
  eles compartilham o mesmo bin FFT e suas energias somam. O filtro
  de Wiener os trata como um único componente, o que pode levar à
  supressão parcial do sinal de interesse junto com o ruído tonal.
  Mitigação: janela de Hanning (já aplicada na segmentação) reduz
  o vazamento espectral, melhorando a separação entre bins adjacentes.

Referências:
  Ephraim & Malah (1984) — MMSE-STSA [IEEE Trans. Acoustics]
  Boll (1979) — Subtração Espectral [IEEE Trans. Acoustics]
  Lugasi et al. (2024) — Wiener Multicanal [IEEE/ACM Trans. ASLP]
  Widrow et al. (1975) — ANC e LMS [Proc. IEEE]
"""
import numpy as np


def estimar_perfil_ruido(magnitudes: np.ndarray,
                          n_quadros: int = 15) -> np.ndarray:
    """
    Estima o perfil de ruído como a média dos primeiros `n_quadros`.
    Pressupõe que o início do áudio contém apenas ruído de fundo —
    hipótese válida para gravações com período de silêncio inicial.

    Para gravações sem silêncio inicial, fornecer `perfil_ruido`
    externamente via `aplicar_wiener(..., perfil_ruido=ref)`.
    """
    n = min(n_quadros, magnitudes.shape[0])
    return np.mean(magnitudes[:n], axis=0)


def aplicar_wiener(magnitudes: np.ndarray,
                   perfil_ruido: np.ndarray = None,
                   n_quadros_ruido: int = 15,
                   beta: float = 0.002,
                   suavizacao: float = 0.98,
                   compensar_ruido_tonal: bool = True) -> np.ndarray:
    """
    Wiener espectral com suavização temporal do ganho.

    O ganho de Wiener por bin é:
      G(f) = sqrt( max(|X(f)|² - |N(f)|², β·|X(f)|²) / |X(f)|² )

    A suavização temporal (α) evita o musical noise — artefato
    característico da subtração espectral pura (Boll, 1979).

    Parâmetros
    ----------
    magnitudes           : (n_quadros, n_bins)
    perfil_ruido         : (n_bins,) — se None, estima dos primeiros quadros
    n_quadros_ruido      : quadros iniciais para estimar o ruído
    beta                 : ganho mínimo — evita supressão total (over-subtraction)
    suavizacao           : coeficiente α de suavização temporal do ganho [0, 1]
    compensar_ruido_tonal: se True, aplica supressão extra em bins
                           com energia de ruído muito concentrada
                           (ruídos tonais: ventilador, AC)

    Retorno
    -------
    magnitudes_filtradas : (n_quadros, n_bins) — valores ≥ 0
    """
    if perfil_ruido is None:
        perfil_ruido = estimar_perfil_ruido(magnitudes, n_quadros_ruido)

    pot_ruido  = perfil_ruido ** 2
    resultado  = np.empty_like(magnitudes)
    ganho_prev = np.ones(magnitudes.shape[1])

    # Detecta bins com ruído tonal (energia muito concentrada no perfil)
    if compensar_ruido_tonal:
        pot_media  = pot_ruido.mean()
        bins_tonais = pot_ruido > 5.0 * pot_media   # bins com ruído 5× acima da média
    else:
        bins_tonais = np.zeros(magnitudes.shape[1], dtype=bool)

    for i, frame in enumerate(magnitudes):
        pot_frame = frame ** 2

        # Ganho de Wiener padrão
        pot_limpa  = np.maximum(pot_frame - pot_ruido, beta * pot_frame)
        ganho_inst = np.sqrt(pot_limpa / (pot_frame + 1e-12))

        # Supressão adicional nos bins com ruído tonal
        # (frequências muito próximas ao ruído — caso especial das anotações)
        if bins_tonais.any():
            ganho_inst[bins_tonais] = np.minimum(
                ganho_inst[bins_tonais], beta * 2
            )

        # Suavização temporal: reduz musical noise
        ganho = suavizacao * ganho_prev + (1.0 - suavizacao) * ganho_inst
        ganho_prev = ganho

        resultado[i] = frame * ganho

    return np.maximum(resultado, 0.0)


def estimar_snr_por_quadro(magnitudes: np.ndarray,
                            perfil_ruido: np.ndarray) -> np.ndarray:
    """
    Estima o SNR (dB) de cada quadro em relação ao perfil de ruído.
    Útil para diagnóstico e para decidir se aplicar realce agressivo ou suave.
    """
    pot_frame = (magnitudes ** 2).mean(axis=1)
    pot_ruido = (perfil_ruido ** 2).mean()
    snr       = 10.0 * np.log10(pot_frame / (pot_ruido + 1e-12))
    return snr
