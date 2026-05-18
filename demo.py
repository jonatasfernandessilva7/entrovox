"""
demo.py — Demonstração da arquitetura PPGETI para transcrição.
Testa o pipeline de realce sem depender do Whisper.
"""
import sys, io, time, numpy as np
from pathlib import Path
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).parent))
from api import app

SR = 16000

def tom(freqs, ruido=0.05, snr_db=None, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1.0, SR, endpoint=False)
    s   = sum(np.sin(2*np.pi*f*t) for f in freqs).astype(np.float32)
    s  /= np.abs(s).max() + 1e-12
    if snr_db is not None:
        r   = rng.standard_normal(SR).astype(np.float32)
        pot_s = np.mean(s**2); pot_r = np.mean(r**2)
        r  *= np.sqrt(pot_s / (10**(snr_db/10)) / (pot_r+1e-12))
        s   = np.clip(s + r, -1, 1)
    buf = io.BytesIO()
    wavfile.write(buf, SR, (s*32767).astype(np.int16))
    return buf.getvalue()

def _linha(c="─", n=60): print(c*n)

if __name__ == "__main__":
    client = app.test_client()
    _linha("═")
    print("  PPGETI — Pipeline de Realce de Fala para Transcrição")
    _linha("═")

    # ── DEMO 1: Status ────────────────────────────────────────────────────────
    print("\n── DEMO 1: Status da API")
    r = client.get("/status").get_json()
    print(f"  Whisper:    {r['whisper_disponivel']}")
    print(f"  AE treinado:{r['ae_treinado']}")
    print(f"  Locutores:  {r['locutores']}")

    # ── DEMO 2: Realce básico ─────────────────────────────────────────────────
    print("\n── DEMO 2: Realce de fala (sem locutor registrado)")
    _linha()
    for label, snr in [("limpo", None), ("SNR 10dB", 10), ("SNR 0dB", 0), ("SNR -5dB", -5)]:
        wav = tom([200, 400, 800], snr_db=snr)
        r   = client.post("/realcar",
                          data={"audio": (io.BytesIO(wav), "a.wav")},
                          content_type="multipart/form-data").get_json()
        e   = r["entropia"]
        print(f"  {label:10s} | qualidade={r['qualidade']:.3f} | "
              f"entropia_media={e['entropia_media']:.3f} | "
              f"descartados={e['quadros_descartados']}/{e['quadros_total']}")

    # ── DEMO 3: Registro de locutor ───────────────────────────────────────────
    print("\n── DEMO 3: Registro de locutor (amostras de referência limpas)")
    _linha()
    refs = [tom([200, 600, 1200], seed=i) for i in range(3)]
    r = client.post("/locutor/registrar?id=locutor_a",
                    data={f"audio_{i}": (io.BytesIO(ref), f"ref{i}.wav")
                          for i, ref in enumerate(refs)},
                    content_type="multipart/form-data").get_json()
    print(f"  Locutor registrado: {r}")

    # ── DEMO 4: Realce com locutor ────────────────────────────────────────────
    print("\n── DEMO 4: Realce condicionado ao locutor vs. sem locutor")
    _linha()
    for label, locutor in [("sem locutor", None), ("com locutor", "locutor_a")]:
        wav = tom([200, 600, 1200], snr_db=0, seed=99)
        url = f"/realcar" + (f"?id_locutor={locutor}" if locutor else "")
        r   = client.post(url,
                          data={"audio": (io.BytesIO(wav), "a.wav")},
                          content_type="multipart/form-data").get_json()
        e   = r["entropia"]
        print(f"  {label:15s} | qualidade={r['qualidade']:.3f} | "
              f"locutor_usado={r['locutor_usado']} | "
              f"entropia={e['entropia_media']:.3f}")

    # ── DEMO 5: Endpoint /transcrever (sem Whisper) ───────────────────────────
    print("\n── DEMO 5: Endpoint /transcrever")
    _linha()
    wav = tom([300, 600, 1200], snr_db=5)
    r   = client.post("/transcrever",
                      data={"audio": (io.BytesIO(wav), "a.wav")},
                      content_type="multipart/form-data").get_json()
    print(f"  texto:         '{r['texto']}'")
    print(f"  whisper_ativo: {r['whisper_ativo']}")
    print(f"  qualidade:     {r['qualidade_audio']}")
    print(f"  latencia:      {r['latencia_ms']}ms")

    # ── DEMO 6: WER com referência ────────────────────────────────────────────
    print("\n── DEMO 6: Comparação WER (simulação sem Whisper)")
    _linha()
    from transcricao import _calcular_wer
    ref  = "o gato subiu no telhado"
    hip1 = "o gato subiu no telhado"   # perfeito
    hip2 = "o gato subiu no telhado"   # perfeito com enhance
    hip3 = "gato subiu telhado"         # erros sem enhance
    print(f"  WER perfeito:      {_calcular_wer(ref, hip1):.2f}")
    print(f"  WER com erros:     {_calcular_wer(ref, hip3):.2f}")
    print(f"  WER parcial:       {_calcular_wer(ref, 'o gato subiu no chao'):.2f}")

    # ── DEMO 7: Latência ──────────────────────────────────────────────────────
    print("\n── DEMO 7: Latência do pipeline de realce (10 requisições)")
    _linha()
    lats = []
    for _ in range(10):
        wav = tom([400, 800, 1600], snr_db=5)
        r   = client.post("/realcar",
                          data={"audio": (io.BytesIO(wav), "a.wav")},
                          content_type="multipart/form-data").get_json()
        lats.append(r["latencia_ms"])
    print(f"  media={np.mean(lats):.1f}ms  max={np.max(lats):.1f}ms  "
          f"min={np.min(lats):.1f}ms")
    ok = np.max(lats) < 2000
    print(f"  {'✓' if ok else '✗'} Dentro do limite de 2000ms (RNF02)")

    _linha("═")
    print("  Demonstração concluída.")
    print("  Para transcrição real: pip install openai-whisper")
    _linha("═")
