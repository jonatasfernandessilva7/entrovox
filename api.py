import os, io, time, traceback, sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))
from audio_io import carregar_bytes, bytes_para_array, SR_ALVO
from pipeline import PipelineRealce, SR
from transcricao import Transcritor, WHISPER_DISPONIVEL

app = Flask(__name__)
CORS(app)

MODELO_WHISPER = os.environ.get("WHISPER_MODEL", "base")
DEVICE         = os.environ.get("WHISPER_DEVICE", "cpu")
IDIOMA_PADRAO  = os.environ.get("IDIOMA", "pt")

_pipeline = PipelineRealce()
try:
    _transcritor = Transcritor(modelo=MODELO_WHISPER, device=DEVICE, idioma=IDIOMA_PADRAO)
    _pipeline    = _transcritor.pipeline
except Exception as e:
    print(f"AVISO: Transcritor nao inicializado: {e}")
    _transcritor = None

def _ler_audio(req):
    if "audio" in req.files:
        raw = req.files["audio"].read()
        sinal, _ = carregar_bytes(raw)
    elif req.data:
        try:
            sinal, _ = carregar_bytes(req.data)
        except Exception:
            sinal = bytes_para_array(req.data)
    else:
        raise ValueError("Nenhum audio recebido.")
    return sinal

def _wav_bytes(sinal, sr=SR_ALVO):
    s16 = np.clip(sinal * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, s16)
    buf.seek(0)
    return buf.read()

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status":             "ok",
        "whisper_disponivel": WHISPER_DISPONIVEL,
        "modelo_whisper":     MODELO_WHISPER if WHISPER_DISPONIVEL else None,
        "ae_treinado":        _pipeline._ae.treinado,
        "locutores":          _pipeline.locutores(),
        "sr":                 SR,
        "idioma_padrao":      IDIOMA_PADRAO,
    }), 200

@app.route("/locutores", methods=["GET"])
def listar_locutores():
    return jsonify({"locutores": _pipeline.locutores()}), 200

@app.route("/locutor/registrar", methods=["POST"])
def registrar_locutor():
    id_locutor = request.args.get("id") or request.form.get("id_locutor")
    if not id_locutor:
        return jsonify({"erro": "Parametro 'id' obrigatorio."}), 400
    amostras = []
    for key in request.files:
        if key.startswith("audio"):
            try:
                s, _ = carregar_bytes(request.files[key].read())
                amostras.append(s)
            except Exception as e:
                return jsonify({"erro": f"Falha no arquivo '{key}': {e}"}), 400
    if not amostras:
        return jsonify({"erro": "Envie ao menos 1 arquivo WAV limpo."}), 400
    emb = _pipeline.registrar_locutor(id_locutor, amostras)
    return jsonify({"status": "ok", "id_locutor": id_locutor,
                    "dim_embedding": int(emb.shape[0]), "n_amostras": len(amostras)}), 200

@app.route("/realcar", methods=["POST"])
def realcar():
    id_locutor = request.args.get("id_locutor")
    formato    = request.args.get("formato", "json").lower()
    t0 = time.perf_counter()
    try:
        sinal = _ler_audio(request)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    except Exception as e:
        return jsonify({"erro": f"Falha ao decodificar audio: {e}"}), 400
    try:
        res = _pipeline.processar(sinal, id_locutor=id_locutor)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"erro": f"Falha no pipeline: {e}"}), 500
    lat = round((time.perf_counter() - t0) * 1000, 1)
    if formato == "wav":
        return send_file(io.BytesIO(_wav_bytes(res["sinal_limpo"])),
                         mimetype="audio/wav", as_attachment=True,
                         download_name="audio_realcado.wav")
    return jsonify({"status": "ok", "qualidade": res["qualidade"],
                    "entropia": res["entropia"], "locutor_usado": res["locutor_usado"],
                    "ae_ativo": res["ae_ativo"], "n_quadros": res["n_quadros"],
                    "latencia_ms": lat}), 200

@app.route("/transcrever", methods=["POST"])
def transcrever():
    if _transcritor is None:
        return jsonify({"erro": "Transcritor nao inicializado."}), 503
    id_locutor = request.args.get("id_locutor")
    enhance    = request.args.get("enhance", "true").lower() not in ("false", "0")
    idioma     = request.args.get("idioma", IDIOMA_PADRAO)
    t0 = time.perf_counter()
    try:
        sinal = _ler_audio(request)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    try:
        res = _transcritor.transcrever(sinal, id_locutor=id_locutor,
                                        enhance=enhance, idioma=idioma)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"erro": f"Falha na transcricao: {e}"}), 500
    lat = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify({"texto": res["texto"], "idioma": res["idioma"],
                    "confianca": res["confianca"], "enhance": enhance,
                    "qualidade_audio": res["qualidade_audio"],
                    "entropia": res["entropia"], "latencia_ms": lat,
                    "whisper_ativo": WHISPER_DISPONIVEL}), 200

@app.route("/transcrever/comparar", methods=["POST"])
def comparar_wer():
    if _transcritor is None:
        return jsonify({"erro": "Transcritor nao inicializado."}), 503
    referencia = request.form.get("referencia", "")
    id_locutor = request.form.get("id_locutor") or request.args.get("id_locutor")
    try:
        sinal = _ler_audio(request)
    except ValueError as e:
        return jsonify({"erro": str(e)}), 400
    try:
        res = _transcritor.comparar_wer(sinal, referencia=referencia, id_locutor=id_locutor)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"erro": f"Falha na comparacao: {e}"}), 500
    return jsonify(res), 200

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"API na porta {port} | Whisper: {'sim (' + MODELO_WHISPER + ')' if WHISPER_DISPONIVEL else 'nao instalado'}")
    app.run(host="0.0.0.0", port=port, debug=debug)
