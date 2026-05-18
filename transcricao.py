"""
Integração com Whisper (OpenAI)
================================
Este módulo é OPCIONAL e requer instalação separada:
  pip install openai-whisper

Funciona como camada entre o pipeline de realce e o modelo ASR.
O pipeline de realce entrega o áudio limpo; este módulo passa para o Whisper
e retorna a transcrição junto com as métricas EnBaSe.

Uso típico
----------
from transcricao import Transcritor

t = Transcritor(modelo="base")                    # tiny/base/small/medium/large
t.pipeline.registrar_locutor("joao", [ref1, ref2])

resultado = t.transcrever(sinal_ruidoso, id_locutor="joao")
print(resultado["texto"])
print(resultado["wer_estimado"])   # se referência fornecida

Uso sem GPU (CPU only, modelo "tiny" ou "base")
-----------------------------------------------
t = Transcritor(modelo="tiny", device="cpu")

Uso via API (endpoint /transcrever)
------------------------------------
Veja api.py — o endpoint aceita WAV e retorna JSON com
texto, confiança, métricas de entropia e metadados.
"""
import numpy as np
import io
import time
from pathlib import Path

# Import condicional — não falha se whisper não estiver instalado
try:
    import whisper as _whisper
    WHISPER_DISPONIVEL = True
except ImportError:
    WHISPER_DISPONIVEL = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline import PipelineRealce, SR
from audio_io import salvar_wav


class Transcritor:
    """
    Transcritor de voz com realce de fala integrado.

    Combina:
      1. PipelineRealce   → limpa o áudio (Wiener + AE + EnBaSe)
      2. Whisper          → transcreve o áudio limpo

    Se o Whisper não estiver instalado, retorna apenas o áudio limpo
    (útil para integração com outros ASRs).
    """

    MODELOS_VALIDOS = ["tiny", "base", "small", "medium", "large"]

    def __init__(self, modelo: str = "base",
                 device: str = "cpu",
                 caminho_ae: str = None,
                 idioma: str = "pt"):
        self.modelo_nome = modelo
        self.device      = device
        self.idioma      = idioma
        self.pipeline    = PipelineRealce(caminho_ae=caminho_ae)
        self._whisper    = None

        if WHISPER_DISPONIVEL:
            print(f"Carregando Whisper '{modelo}' em {device}...")
            self._whisper = _whisper.load_model(modelo, device=device)
            print("Whisper pronto.")
        else:
            print(
                "AVISO: openai-whisper não instalado. "
                "O realce de fala funciona normalmente, mas a transcrição "
                "não estará disponível neste ambiente.\n"
                "Para instalar: pip install openai-whisper"
            )

    @property
    def whisper_disponivel(self) -> bool:
        return self._whisper is not None

    # ── Transcrição principal ─────────────────────────────────────────────────

    def transcrever(self, sinal: np.ndarray,
                    id_locutor: str = None,
                    enhance: bool = True,
                    idioma: str = None,
                    referencia_texto: str = None) -> dict:
        """
        Transcreve um sinal de áudio com realce opcional.

        Parâmetros
        ----------
        sinal            : float32 1-D a 16 kHz
        id_locutor       : id do locutor registrado no pipeline (opcional)
        enhance          : se True, aplica o pipeline de realce antes do ASR
        idioma           : código de idioma (ex: "pt", "en"); None = auto-detect
        referencia_texto : se fornecido, calcula WER aproximado

        Retorno
        -------
        {
          "texto"         : str — transcrição
          "idioma"        : str — idioma detectado
          "confianca"     : float — score médio dos segmentos
          "enhance"       : bool
          "qualidade_audio": float — fracção de quadros válidos (EnBaSe)
          "entropia"      : dict — métricas EnBaSe
          "latencia_ms"   : float
          "wer"           : float | None — Word Error Rate (se referência dada)
          "sinal_limpo"   : np.ndarray — áudio processado (para debug/salvar)
        }
        """
        t0 = time.perf_counter()

        # ── Etapa de realce ───────────────────────────────────────────────────
        if enhance:
            res_realce    = self.pipeline.processar(sinal, id_locutor=id_locutor)
            sinal_asr     = res_realce["sinal_limpo"]
            meta_entropia = res_realce["entropia"]
            qualidade     = res_realce["qualidade"]
        else:
            sinal_asr     = sinal
            meta_entropia = {}
            qualidade     = 1.0

        # ── Transcrição com Whisper ───────────────────────────────────────────
        if self._whisper is not None:
            lang = idioma or self.idioma
            opcoes = _whisper.DecodingOptions(
                language=lang if lang != "auto" else None,
                fp16=False,   # CPU-safe
            )
            # Whisper espera float32 normalizado a 16kHz
            audio_whisper = sinal_asr.astype(np.float32)
            resultado_w   = self._whisper.transcribe(
                audio_whisper,
                language=lang if lang != "auto" else None,
                fp16=False,
                verbose=False,
            )
            texto     = resultado_w["text"].strip()
            lang_det  = resultado_w.get("language", lang)
            # Confiança média dos segmentos (log-prob → prob)
            segmentos = resultado_w.get("segments", [])
            if segmentos:
                log_probs = [s.get("avg_logprob", -1.0) for s in segmentos]
                confianca = float(np.exp(np.mean(log_probs)))
            else:
                confianca = 0.0
        else:
            texto     = "[Whisper não instalado — áudio realçado disponível em 'sinal_limpo']"
            lang_det  = idioma or self.idioma
            confianca = 0.0

        latencia_ms = round((time.perf_counter() - t0) * 1000, 1)

        # ── WER aproximado ────────────────────────────────────────────────────
        wer = None
        if referencia_texto and texto and self._whisper is not None:
            wer = _calcular_wer(referencia_texto, texto)

        return {
            "texto":          texto,
            "idioma":         lang_det,
            "confianca":      round(confianca, 4),
            "enhance":        enhance,
            "qualidade_audio": qualidade,
            "entropia":       meta_entropia,
            "latencia_ms":    latencia_ms,
            "wer":            wer,
            "sinal_limpo":    sinal_asr,
        }

    def transcrever_arquivo(self, caminho_wav: str,
                             id_locutor: str = None,
                             enhance: bool = True) -> dict:
        """Atalho: carrega WAV e transcreve."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from audio_io import carregar_wav
        sinal, _ = carregar_wav(caminho_wav)
        return self.transcrever(sinal, id_locutor=id_locutor, enhance=enhance)

    def comparar_wer(self, sinal: np.ndarray,
                     referencia: str,
                     id_locutor: str = None) -> dict:
        """
        Compara WER com e sem realce — experimento central do PPGETI.
        Equivale ao teste do mês 15 do cronograma.

        Retorno
        -------
        {
          "wer_sem_enhance" : float
          "wer_com_enhance" : float
          "reducao_wer"     : float — redução absoluta em pontos percentuais
          "melhora"         : bool
          "transcricao_base": str
          "transcricao_enhanced": str
        }
        """
        r_base = self.transcrever(sinal, enhance=False,
                                   referencia_texto=referencia)
        r_enh  = self.transcrever(sinal, id_locutor=id_locutor,
                                   enhance=True, referencia_texto=referencia)

        wer_base = r_base["wer"] or 1.0
        wer_enh  = r_enh ["wer"] or 1.0
        reducao  = wer_base - wer_enh

        return {
            "wer_sem_enhance":     round(wer_base, 4),
            "wer_com_enhance":     round(wer_enh,  4),
            "reducao_wer":         round(reducao,  4),
            "melhora":             reducao > 0,
            "qualidade_enhance":   r_enh["qualidade_audio"],
            "entropia":            r_enh["entropia"],
            "transcricao_base":    r_base["texto"],
            "transcricao_enhanced": r_enh["texto"],
        }


# ── Cálculo de WER (sem dependências externas) ────────────────────────────────

def _calcular_wer(referencia: str, hipotese: str) -> float:
    """
    Word Error Rate via distância de edição (Levenshtein) em nível de palavra.
    WER = (S + D + I) / N
    onde S=substituições, D=deleções, I=inserções, N=palavras na referência.
    """
    ref = referencia.lower().split()
    hip = hipotese.lower().split()

    if not ref:
        return 0.0 if not hip else 1.0

    n = len(ref)
    m = len(hip)

    # Matriz de distância de edição
    d = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            custo = 0 if ref[i - 1] == hip[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j]     + 1,     # deleção
                d[i, j - 1]     + 1,     # inserção
                d[i - 1, j - 1] + custo, # substituição
            )

    return round(d[n, m] / n, 4)
