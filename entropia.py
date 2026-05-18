"""
RF06 — Cálculo de Entropia Espectral de Shannon
RF07 — Validação EnBaSe dos Quadros Reconstruídos

Modelo de entropia (conforme anotações do projeto):

  H = -Σ P(f) · log₂ P(f)
      f

  onde P(f) = energia_bin_f / energia_total_quadro
  e    energia = potência × variação de tempo  (≈ |magnitude|² por quadro)

Interpretação correta (conforme anotações):
  H baixo → sinal estável, energia concentrada (vogais, tons)   ← VÁLIDO
  H alto  → energia distribuída uniformemente (ruído)           ← SUSPEITO

ATENÇÃO — caso especial (anotações):
  Consoantes fricativas (s, f, sh, ch) têm distribuição de energia
  espalhada por todo o espectro → H ≈ 90–93% do máximo teórico.
  O limiar superior deve estar ACIMA desse valor para não descartar
  fonemas legítimos. Medições empíricas:
    fala com formantes    : H ≈ 29–38% do e_max
    fala + ruído SNR 0 dB : H ≈ 71%
    fricativas (s, f)     : H ≈ 92%
    ruído branco puro     : H ≈ 92%   ← limite real de rejeição
    silêncio absoluto     : H = 0%

  → O limiar inferior deve rejeitar apenas silêncio (H ≈ 0).
  → O limiar superior deve rejeitar somente ruído puro (H > 93%).

Frequências próximas (anotações — tubo de Eustáquio):
  A resolução espectral da FFT é Δf = SR / N_quadro.
  Com SR=16000 e N=400 amostras → Δf = 40 Hz.
  Sons separados por < 40 Hz compartilham o mesmo bin e
  sua energia é somada — o que sobe artificialmente a entropia
  desse bin e pode mascarar a presença do sinal de interesse.

Referências:
  Neto et al. (2025), EnBaSe [IEEE Access, vol. 13]
  Taal et al. (2011), STOI [IEEE Trans. Audio, Speech, Lang. Process.]
"""
import numpy as np


# ── RF06: Entropia de Shannon ─────────────────────────────────────────────────

def calcular_entropia(magnitudes: np.ndarray) -> np.ndarray:
    """
    H(quadro) = -Σ P(f) · log₂ P(f)
    onde P(f) = |mag_f|² / Σ|mag_f|²

    Parâmetros
    ----------
    magnitudes : (n_quadros, n_bins)

    Retorno
    -------
    entropias : (n_quadros,)  — valores em [0, log₂(n_bins)]
    """
    energia = magnitudes ** 2
    soma    = energia.sum(axis=1, keepdims=True) + 1e-12
    p       = energia / soma
    p_safe  = np.where(p > 1e-12, p, 1.0)   # evita log(0)
    return -np.sum(p * np.log2(p_safe), axis=1)


def entropia_maxima(n_bins: int) -> float:
    """
    e_max = log₂(n_bins)
    Atingida quando a energia é uniformemente distribuída (ruído branco puro).
    """
    return float(np.log2(n_bins))


# ── RF07: Validação EnBaSe ────────────────────────────────────────────────────

def validar_quadros(entropias: np.ndarray,
                    n_bins: int,
                    limiar_silencio: float = 0.01,
                    limiar_ruido:    float = 0.90) -> np.ndarray:
    """
    Máscara booleana: True = quadro que contém informação de fala válida.

    Critérios baseados nas anotações e em medições empíricas (SR=16kHz, N=400):

    REJEITAR se H < limiar_silencio × e_max   (padrão: H < 1% de e_max)
      → silêncio absoluto (H=0) ou quadro zerado pela reconstrução.

    REJEITAR se H > limiar_ruido × e_max      (padrão: H > 90% de e_max)
      → energia muito uniforme = ruído dominante.

    LIMITAÇÃO CONHECIDA — caso especial de frequências próximas (anotações):
      Fricativas reais (s, f, sh, ch) têm distribuição de energia espalhada
      e entropia ≈ 92% do e_max — praticamente idêntica ao ruído branco puro
      (também ≈ 92%). A entropia de Shannon sozinha NÃO consegue distingui-las.

      Implicação prática:
        - limiar_ruido < 92% → fricativas descartadas junto com ruído
        - limiar_ruido > 92% → ruído branco puro passa como fala
        - Não existe limiar que resolva ambos simultaneamente.

      Solução futura (pesquisa PPGETI fase III):
        Combinar entropia com outras features (centróide espectral,
        ZCR, pitch) para discriminar fricativas de ruído. O speaker
        embedding já contém informação de timbre que pode ajudar.
        Referência: Taal et al. (2011) STOI para inteligibilidade.

    Parâmetros
    ----------
    entropias       : (n_quadros,)
    n_bins          : número de bins FFT
    limiar_silencio : fração de e_max — rejeita silêncio   (padrão: 1%)
    limiar_ruido    : fração de e_max — rejeita ruído puro  (padrão: 90%)

    Retorno
    -------
    mascara : bool (n_quadros,)
    """
    e_max        = entropia_maxima(n_bins)
    nao_silencio = entropias >  limiar_silencio * e_max
    nao_ruido    = entropias <= limiar_ruido    * e_max
    return nao_silencio & nao_ruido


def score_qualidade(entropias: np.ndarray, n_bins: int,
                    limiar_silencio: float = 0.01,
                    limiar_ruido:    float = 0.95) -> np.ndarray:
    """
    Score contínuo de qualidade por quadro [0, 1].
    Alternativa à máscara binária para ponderação suave.

    Lógica:
      - Silêncio (H ≈ 0)      → score = 0
      - Fala estável (H baixo) → score alto (sinal concentrado = bom)
      - Fala + ruído moderado  → score decrescente com H
      - Ruído puro (H ≈ e_max) → score = 0

    Score = max(0, 1 - (H - H_fala_ref) / (H_ruido_limiar - H_fala_ref))
    onde H_fala_ref é a entropia típica de fala limpa.
    """
    e_max        = entropia_maxima(n_bins)
    h_fala_ref   = 0.35 * e_max   # entropia típica de fala limpa (≈35% do máximo)
    h_ruido_lim  = limiar_ruido * e_max

    # Quadros de silêncio recebem 0
    score = np.where(
        entropias <= limiar_silencio * e_max,
        0.0,
        # Para fala: quanto mais próxima de H_fala_ref, melhor
        # Para ruído: score cai linearmente até 0 em h_ruido_lim
        np.where(
            entropias <= h_fala_ref,
            entropias / (h_fala_ref + 1e-12),          # cresce até h_fala_ref
            1.0 - (entropias - h_fala_ref) / (h_ruido_lim - h_fala_ref + 1e-12)
        )
    )
    return np.clip(score, 0.0, 1.0)


def limiares_adaptativos(entropias: np.ndarray, n_bins: int,
                          p_silencio: float = 2.0,
                          p_ruido:    float = 97.0) -> tuple:
    """
    Ajusta limiares automaticamente pelos percentis do sinal.
    Útil para calibrar em novos tipos de ruído ou condições acústicas.

    Retorno: (limiar_silencio_abs, limiar_ruido_abs, frac_silencio, frac_ruido)
    """
    e_max   = entropia_maxima(n_bins)
    lim_sil = float(np.percentile(entropias, p_silencio))
    lim_rui = float(np.percentile(entropias, p_ruido))
    return lim_sil, lim_rui, lim_sil / e_max, lim_rui / e_max


def detectar_tipo_quadro(entropia: float, n_bins: int) -> str:
    """
    Classifica um quadro pelo nível de entropia.
    Útil para diagnóstico e logging.
    """
    e_max = entropia_maxima(n_bins)
    frac  = entropia / e_max
    if frac <= 0.01:   return "silencio"
    if frac <= 0.45:   return "fala_estavel"    # vogais, nasais
    if frac <= 0.75:   return "fala_ruidosa"    # fala + ruido moderado
    if frac <= 0.93:   return "fricativa_ou_ruidoso"   # s, f, sh ou SNR baixo
    return "ruido_puro"


# ── Resumo para API / log ─────────────────────────────────────────────────────

def resumo_entropia(entropias: np.ndarray, mascara: np.ndarray) -> dict:
    """Estatísticas EnBaSe para retorno na API e registro em log."""
    n_bins = None   # não disponível aqui; score omitido
    tipos  = {}
    # Contagem por tipo (diagnóstico)
    for e in entropias:
        # aproximação sem n_bins: usa valor absoluto
        if   e <= 0.08:  k = "silencio"
        elif e <= 3.50:  k = "fala_estavel"
        elif e <= 5.80:  k = "fala_ruidosa"
        elif e <= 7.15:  k = "fricativa_ou_ruidoso"
        else:            k = "ruido_puro"
        tipos[k] = tipos.get(k, 0) + 1

    return {
        "entropia_media":        round(float(entropias.mean()), 4),
        "entropia_min":          round(float(entropias.min()),  4),
        "entropia_max":          round(float(entropias.max()),  4),
        "entropia_std":          round(float(entropias.std()),  4),
        "quadros_total":         int(len(mascara)),
        "quadros_validos":       int(mascara.sum()),
        "quadros_descartados":   int((~mascara).sum()),
        "taxa_descarte":         round(float((~mascara).mean()), 4),
        "distribuicao_tipos":    tipos,
    }
