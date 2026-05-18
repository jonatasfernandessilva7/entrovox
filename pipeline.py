"""
Pipeline de Realce de Fala para Transcrição (PPGETI)
=====================================================
Arquitetura:

  Áudio ruidoso
      │
      ▼ Dupla Decomposição (RF01 + RF02)
      │   Segmentação temporal (quadros 25ms, hop 10ms, Hanning)
      │   FFT por quadro → espectrograma (magnitudes + fases)
      │
      ▼ Supressão Inicial de Ruído — NS (RF03)
      │   Filtro de Wiener com suavização temporal
      │
      ▼ Reconstrução Generativa — VC-Restore (RF04 + RF05)
      │   Speaker embedding: timbre extraído de amostras limpas de referência
      │   Autoencoder condicional: reconstrói espectrograma quadro-a-quadro
      │
      ▼ Validação por Entropia — EnBaSe (RF06 + RF07)
      │   H = -Σ P(f)·log₂P(f),  P(f) = energia_bin / energia_total
      │   H baixo → sinal estável (vogais, tons)        ← válido
      │   H alto  → energia uniforme (ruído puro)       ← rejeitar
      │   Limiar inferior: silêncio (H ≈ 0)
      │   Limiar superior: acima de fricativas (H > 93%) para não
      │                    descartar fonemas legítimos (s, f, sh)
      │
      ▼ Reconstrução Temporal (RF08)
      │   IFFT + Overlap-Add
      │
      ▼ Áudio limpo → entregue ao Whisper (ou qualquer ASR)

Refs: Byun et al. 2024 (VC-Enhance), Neto et al. 2025 (EnBaSe),
      Radford et al. 2022 (Whisper)
"""
import numpy as np
from pathlib import Path

from segmentacao   import segmentar, tamanho_quadro, hop_amostras
from espectrograma import espectrograma
from wiener        import aplicar_wiener
from embedding     import extrair_embedding, carregar_embedding, salvar_embedding
from autoencoder   import AutoencoderCondicional
from entropia      import calcular_entropia, validar_quadros, score_qualidade, resumo_entropia
from reconstrucao  import reconstruir

SR         = 16000
DURACAO_MS = 25
HOP_MS     = 10

DIR_MODELOS = Path(__file__).parent.parent / "models"
PATH_AE     = DIR_MODELOS / "autoencoder.npz"


class PipelineRealce:
    """
    Pipeline de realce de fala para pré-processamento de transcrição.

    Uso básico
    ----------
    pipe = PipelineRealce()
    res  = pipe.processar(sinal_ruidoso)
    audio_limpo = res["sinal_limpo"]   # → Whisper

    Uso com referência de locutor (melhor qualidade)
    ------------------------------------------------
    pipe.registrar_locutor("joao", [ref1, ref2, ref3])
    res = pipe.processar(sinal_ruidoso, id_locutor="joao")
    """

    def __init__(self, caminho_ae: str = None):
        caminho      = caminho_ae or (str(PATH_AE) if PATH_AE.exists() else None)
        self._ae     = AutoencoderCondicional(caminho)
        self._embeds = {}   # id_locutor → embedding
        self._tam    = tamanho_quadro(SR, DURACAO_MS)
        self._hop    = hop_amostras(SR, HOP_MS)

    # ── Gestão de locutores ───────────────────────────────────────────────────

    def registrar_locutor(self, id_locutor: str,
                          amostras_limpas: list,
                          salvar_em: str = None) -> np.ndarray:
        """
        Calcula e armazena o speaker embedding a partir de 2–3 amostras
        limpas de referência (conforme RD03 do projeto PPGETI).

        Parâmetros
        ----------
        id_locutor      : identificador único (ex: "joao")
        amostras_limpas : lista de arrays float32 16kHz
        salvar_em       : caminho .npy para persistência (opcional)
        """
        quadros_concat = []
        for amostra in amostras_limpas:
            q    = segmentar(amostra, SR, DURACAO_MS, HOP_MS)
            mag, _ = espectrograma(q)
            quadros_concat.append(mag)

        emb = extrair_embedding(np.concatenate(quadros_concat, axis=0))
        self._embeds[id_locutor] = emb
        if salvar_em:
            salvar_embedding(emb, salvar_em)
        return emb

    def carregar_locutor(self, id_locutor: str, caminho: str) -> None:
        """Carrega embedding previamente salvo."""
        self._embeds[id_locutor] = carregar_embedding(caminho)

    def locutores(self) -> list:
        return list(self._embeds.keys())

    # ── Pipeline principal ────────────────────────────────────────────────────

    def processar(self, sinal: np.ndarray,
                  id_locutor: str = None,
                  limiar_silencio: float = 0.01,
                  limiar_ruido:    float = 0.90) -> dict:
        """
        Aplica o pipeline completo de realce.

        Parâmetros
        ----------
        sinal       : float32 1-D a 16 kHz
        id_locutor  : usa o embedding registrado para condicionar a reconstrução
        limiar_silencio : rejeita quadros com H < limiar × e_max (silêncio absoluto)
        limiar_ruido    : rejeita quadros com H > limiar × e_max (ruído puro irrecuperável)

        Retorno
        -------
        {
          "sinal_limpo"    : np.ndarray float32 — áudio realçado
          "sinal_original" : np.ndarray float32 — entrada intocada
          "entropia"       : dict  — métricas EnBaSe
          "qualidade"      : float — fracção de quadros válidos (0–1)
          "locutor_usado"  : str | None
          "ae_ativo"       : bool
          "sr"             : int (16000)
        }
        """
        # RF01 — segmentação temporal
        quadros = segmentar(sinal, SR, DURACAO_MS, HOP_MS)

        # RF02 — decomposição espectral
        mag, fases = espectrograma(quadros)

        # RF03 — supressão inicial (NS / 1ª etapa VC-Enhance)
        # Calcula entropia PRÉ-Wiener para detectar silêncio e ruído puro
        # antes que o filtro altere a distribuição espectral
        entropias_pre = calcular_entropia(mag)
        mascara_pre   = validar_quadros(entropias_pre, mag.shape[1],
                                        limiar_silencio, limiar_ruido)
        mag_ns = aplicar_wiener(mag)

        # RF04 — speaker embedding
        # Com referência externa → condicionamento forte (identidade vocal preservada)
        # Sem referência         → usa espectro do próprio sinal pré-limpo (modo degradado)
        if id_locutor and id_locutor in self._embeds:
            emb           = self._embeds[id_locutor]
            locutor_usado = id_locutor
        else:
            emb           = extrair_embedding(mag_ns)
            locutor_usado = None

        # RF05 — reconstrução generativa (VC-Restore / 2ª etapa VC-Enhance)
        mag_rec = self._ae.reconstruir(mag_ns, emb)

        # RF06 — entropia de Shannon por quadro
        entropias = calcular_entropia(mag_rec)

        # RF07 — validação EnBaSe em dois níveis
        # Nível 1 — pré-Wiener (mascara_pre já calculada acima):
        #   rejeita silêncio absoluto (H=0) e ruído puro irrecuperável (H>90% e_max)
        # Nível 2 — pós-reconstrução:
        #   rejeita quadros onde a reconstrução falhou (over-smoothing ou ruído residual)
        # Quadro válido = passa nos dois critérios
        n_bins      = mag_rec.shape[1]
        mascara_pos = validar_quadros(entropias, n_bins, limiar_silencio, limiar_ruido)
        mascara     = mascara_pre & mascara_pos

        # RF08 — reconstrução temporal
        sinal_limpo = reconstruir(mag_rec, fases, mascara, self._tam, self._hop)

        # Alinha comprimento e normaliza
        n           = min(len(sinal), len(sinal_limpo))
        sinal_limpo = sinal_limpo[:n]
        mx          = np.abs(sinal_limpo).max()
        if mx > 1e-6:
            sinal_limpo = sinal_limpo / mx

        # Usa entropias pré-Wiener para o resumo (representa o sinal original)
        meta = resumo_entropia(entropias_pre, mascara)
        meta["entropia_pos_reconstrucao"] = round(float(entropias.mean()), 4)

        return {
            "sinal_limpo":    sinal_limpo.astype(np.float32),
            "sinal_original": sinal[:n].astype(np.float32),
            "entropia":       meta,
            "qualidade":      round(float(mascara.mean()), 4),
            "score_qualidade": round(float(score_qualidade(entropias, n_bins).mean()), 4),
            "locutor_usado":  locutor_usado,
            "ae_ativo":       self._ae.treinado,
            "n_quadros":      int(len(mascara)),
            "sr":             SR,
        }

    def processar_longo(self, sinal: np.ndarray,
                        seg_dur_s: float = 30.0,
                        id_locutor: str = None) -> dict:
        """
        Processa áudios longos em segmentos de `seg_dur_s` segundos.
        Necessário porque o Whisper processa até 30s por vez.
        """
        n_seg  = int(SR * seg_dur_s)
        partes = []
        metas  = []

        for inicio in range(0, len(sinal), n_seg):
            seg = sinal[inicio: inicio + n_seg]
            if len(seg) < self._tam:
                continue
            r = self.processar(seg, id_locutor=id_locutor)
            partes.append(r["sinal_limpo"])
            metas.append(r["entropia"])

        sinal_final = np.concatenate(partes) if partes else sinal.copy()

        meta_total = {
            "entropia_media":      float(np.mean([m["entropia_media"]      for m in metas])),
            "quadros_total":       int  (sum   ([m["quadros_total"]        for m in metas])),
            "quadros_validos":     int  (sum   ([m["quadros_validos"]      for m in metas])),
            "quadros_descartados": int  (sum   ([m["quadros_descartados"]  for m in metas])),
            "taxa_descarte":       float(np.mean([m["taxa_descarte"]       for m in metas])),
            "n_segmentos":         len(metas),
        }

        return {
            "sinal_limpo":   sinal_final.astype(np.float32),
            "entropia":      meta_total,
            "qualidade":     round(1.0 - meta_total["taxa_descarte"], 4),
            "locutor_usado": id_locutor,
            "sr":            SR,
        }
