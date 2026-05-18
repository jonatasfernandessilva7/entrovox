# ENTROVOX
### Filtragem de Ruído com Reconstrução Generativa e Validação Entrópica para Transcrição Robusta de Voz

---

## Visão geral

O ENTROVOX é um pipeline de **pré-processamento de áudio** que limpa o sinal de voz antes de entregá-lo a um modelo de transcrição automática (ASR), com foco no Whisper da OpenAI.

```
Áudio ruidoso
    │
    ▼ RF01/02 — Dupla Decomposição
    │   Segmentação temporal (quadros 25 ms, hop 10 ms, janela Hanning)
    │   FFT por quadro → espectrograma (magnitudes + fases)
    │
    ▼ RF03 — Supressão Inicial de Ruído (NS)
    │   Filtro de Wiener com suavização temporal
    │   Supressão adicional de ruído tonal (ventilador, AC)
    │
    ▼ RF04/05 — Reconstrução Generativa (VC-Restore)
    │   Speaker embedding: [média | variância] espectral por locutor
    │   Autoencoder condicional reconstrói o espectrograma quadro a quadro
    │
    ▼ RF06/07 — Validação EnBaSe
    │   H = −Σ P(f)·log₂P(f),  P(f) = energia_bin / energia_total
    │   H baixo → sinal estável (vogais)          ← válido
    │   H > 90% e_max → ruído puro irrecuperável  ← descartar
    │   H ≈ 0 → silêncio absoluto                 ← descartar
    │
    ▼ RF08 — Reconstrução Temporal
    │   IFFT + Overlap-Add dos quadros válidos
    │
    ▼ Áudio limpo → Whisper (ou qualquer ASR)
```

Referências: Byun et al. 2024 (VC-Enhance), Neto et al. 2025 (EnBaSe), Radford et al. 2022 (Whisper)

---

## Estrutura do projeto

```
entrovox/
├── api.py              # API Flask (endpoints de realce e transcrição)
├── transcricao.py      # Integração com Whisper + cálculo de WER
├── treinamento_ae.py   # Treinamento do Autoencoder Condicional
├── audio_io.py         # Leitura/escrita WAV sem librosa
├── demo.py             # Demonstração e smoke-test
├── requirements.txt    # Dependências
├── models/             # Modelos treinados (gerados automaticamente)
│   └── autoencoder.npz
└── pipeline/           # Módulos do pipeline (RF01–RF08)
    ├── __init__.py     # Reexportações — permite imports flat e modulares
    ├── pipeline.py     # RF01–RF08 orquestrados (PipelineRealce)
    ├── segmentacao.py  # RF01 — janelamento Hanning + zero-padding
    ├── espectrograma.py# RF02 — FFT por quadro
    ├── wiener.py       # RF03 — Wiener espectral com suavização temporal
    ├── embedding.py    # RF04 — speaker embedding por locutor
    ├── autoencoder.py  # RF05 — autoencoder condicional (numpy puro)
    ├── entropia.py     # RF06/07 — entropia de Shannon + validação EnBaSe
    └── reconstrucao.py # RF08 — IFFT + overlap-add
```

---

## Instalação

```bash
pip install -r requirements.txt

# Para transcrição (opcional — o pipeline de realce funciona sem):
pip install openai-whisper
```

---

## Uso rápido

### 1. Treinar o Autoencoder

```bash
python3 treinamento_ae.py
# Salva models/autoencoder.npz
```

### 2. Subir a API

```bash
python3 api.py
# http://localhost:5000
```

Variáveis de ambiente opcionais:

| Variável         | Padrão  | Descrição                        |
|------------------|---------|----------------------------------|
| `PORT`           | `5000`  | Porta da API                     |
| `WHISPER_MODEL`  | `base`  | Modelo Whisper (tiny/base/small) |
| `WHISPER_DEVICE` | `cpu`   | Dispositivo (`cpu` ou `cuda`)    |
| `IDIOMA`         | `pt`    | Idioma padrão da transcrição     |
| `DEBUG`          | `false` | Modo debug do Flask              |

### 3. Endpoints

| Método | Rota                    | Descrição                                  |
|--------|-------------------------|--------------------------------------------|
| `GET`  | `/status`               | Status da API e modelos carregados         |
| `GET`  | `/locutores`            | Lista locutores registrados                |
| `POST` | `/locutor/registrar`    | Registra embedding de um locutor           |
| `POST` | `/realcar`              | Realça o áudio, retorna JSON ou WAV        |
| `POST` | `/transcrever`          | Realce + transcrição Whisper               |
| `POST` | `/transcrever/comparar` | Compara WER com e sem realce (experimento) |

---

## Exemplos de uso

### Realçar áudio (retorna JSON)
```bash
curl -X POST http://localhost:5000/realcar \
     -F "audio=@gravacao.wav"
```
```json
{
  "status": "ok",
  "qualidade": 0.96,
  "entropia": {
    "entropia_media": 3.412,
    "quadros_total": 98,
    "quadros_validos": 94,
    "quadros_descartados": 4,
    "taxa_descarte": 0.0408
  },
  "locutor_usado": null,
  "ae_ativo": true,
  "latencia_ms": 11.2
}
```

### Realçar e baixar WAV limpo
```bash
curl -X POST "http://localhost:5000/realcar?formato=wav" \
     -F "audio=@gravacao.wav" \
     -o audio_limpo.wav
```

### Transcrever com realce
```bash
curl -X POST "http://localhost:5000/transcrever?idioma=pt" \
     -F "audio=@gravacao.wav"
```
```json
{
  "texto": "o sistema de transcrição está funcionando",
  "idioma": "pt",
  "confianca": 0.87,
  "enhance": true,
  "qualidade_audio": 0.96,
  "latencia_ms": 843.1
}
```

### Registrar locutor e transcrever condicionado ao timbre
```bash
# 2–3 amostras limpas de referência por locutor (RD03 do projeto)
curl -X POST "http://localhost:5000/locutor/registrar?id=joao" \
     -F "audio_0=@ref1.wav" \
     -F "audio_1=@ref2.wav" \
     -F "audio_2=@ref3.wav"

curl -X POST "http://localhost:5000/transcrever?id_locutor=joao" \
     -F "audio=@gravacao_ruidosa.wav"
```

### Experimento WER — mês 15 do cronograma
```bash
curl -X POST http://localhost:5000/transcrever/comparar \
     -F "audio=@gravacao.wav" \
     -F "referencia=o texto correto que deveria ser transcrito"
```
```json
{
  "wer_sem_enhance": 0.4286,
  "wer_com_enhance": 0.1429,
  "reducao_wer": 0.2857,
  "melhora": true,
  "transcricao_base": "o texto correto que devia ser transcrito",
  "transcricao_enhanced": "o texto correto que deveria ser transcrito"
}
```

---

## Uso programático

```python
from pipeline import PipelineRealce
from audio_io import carregar_wav

pipe = PipelineRealce()

# Registrar locutor com amostras limpas de referência (opcional)
ref1, _ = carregar_wav("ref1.wav")
ref2, _ = carregar_wav("ref2.wav")
pipe.registrar_locutor("joao", [ref1, ref2])

# Processar áudio ruidoso
sinal, _ = carregar_wav("gravacao_ruidosa.wav")
res = pipe.processar(sinal, id_locutor="joao")

audio_limpo = res["sinal_limpo"]   # float32 a 16 kHz → Whisper
qualidade   = res["qualidade"]     # fração de quadros válidos (0–1)
entropia    = res["entropia"]      # métricas EnBaSe detalhadas
score       = res["score_qualidade"]  # score contínuo (0–1)
```

```python
from transcricao import Transcritor
from audio_io import carregar_wav

t = Transcritor(modelo="base")   # requer: pip install openai-whisper

sinal, _ = carregar_wav("gravacao.wav")
res = t.transcrever(sinal, enhance=True)
print(res["texto"])

# Comparar WER com e sem realce
comp = t.comparar_wer(sinal, referencia="texto de referência")
print(f"Redução de WER: {comp['reducao_wer']:.2%}")
```

---

## Substituição por áudios reais

Para treinar com gravações reais em vez de sinais sintéticos,
substitua `gerar_sinal_limpo()` em `treinamento_ae.py`:

```python
from audio_io import carregar_wav

# Substituir em gerar_dataset():
# sl = gerar_sinal_limpo(freq, seed=seed+i+100)
# por:
sinal, _ = carregar_wav(lista_wavs[i])
```

O dataset deve seguir o formato definido no projeto:

- **RD01:** ≥ 20 locutores, amostras limpas
- **RD02:** Versões ruidosas pareadas (SNR: −5, 0, 5, 10 dB; ruídos: branco, rosa, sala)
- **RD03:** 2–3 amostras limpas por locutor para o speaker embedding

---

## Limitação conhecida — fricativas

Consoantes fricativas (s, f, sh, ch) têm distribuição espectral uniforme
e entropia H ≈ 92% do máximo teórico — estatisticamente idêntica ao ruído
branco puro pela métrica de Shannon. Com o limiar atual (90% do e_max):

- Ruído branco puro → rejeitado ✓
- Fricativas → aceitas (score de qualidade menor, mas não descartadas)
- Fala limpa com formantes → aceita ✓

Solução planejada para a fase III: combinar entropia com taxa de cruzamento
por zero (ZCR) e centróide espectral para discriminar fricativas de ruído.

---

## Dependências

```
numpy
scipy
scikit-learn
joblib
flask
openai-whisper   # opcional
```

Sem pytorch, librosa ou soundfile na inferência.