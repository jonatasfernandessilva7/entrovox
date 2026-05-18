"""
RF04 — Speaker Embedding
Vetor de timbre por locutor para condicionar a reconstrução generativa.

O projeto PPGETI exige amostras limpas de referência por locutor (RD03):
  "2–3 amostras limpas de referência por locutor para extrair embedding."

Implementação: [média_por_bin | variância_por_bin] do espectrograma limpo.
Não requer modelo pré-treinado (sem x-vectors, sem d-vectors) — compatível
com o requisito de inferência sem PyTorch (RNF01).
"""
import numpy as np
from pathlib import Path


# ── Extração ──────────────────────────────────────────────────────────────────

def extrair_embedding(magnitudes: np.ndarray) -> np.ndarray:
    """
    Calcula o embedding do locutor como [média | variância] por bin de frequência.

    Captura dois aspectos do timbre:
      - média   : onde o locutor concentra energia (formantes dominantes)
      - variância: consistência dessa concentração ao longo do tempo

    Parâmetros
    ----------
    magnitudes : (n_quadros, n_bins) — espectrograma limpo do locutor

    Retorno
    -------
    embedding : float32 (2 * n_bins,)
    """
    media     = np.mean(magnitudes, axis=0)
    variancia = np.var (magnitudes, axis=0)
    return np.concatenate([media, variancia]).astype(np.float32)


def normalizar(embedding: np.ndarray) -> np.ndarray:
    """Normalização L2 do embedding (distância cosseno)."""
    norma = np.linalg.norm(embedding) + 1e-12
    return embedding / norma


# ── Métricas de similaridade ──────────────────────────────────────────────────

def similaridade_cosseno(e1: np.ndarray, e2: np.ndarray) -> float:
    """
    Similaridade cosseno entre dois embeddings (1 = idênticos, 0 = ortogonais).
    Usada para verificar coerência intra-locutor.
    """
    n1 = np.linalg.norm(e1) + 1e-12
    n2 = np.linalg.norm(e2) + 1e-12
    return float(np.dot(e1 / n1, e2 / n2))


def distancia(e1: np.ndarray, e2: np.ndarray) -> float:
    """Distância cosseno (0 = idênticos)."""
    return 1.0 - similaridade_cosseno(e1, e2)


# ── Persistência ──────────────────────────────────────────────────────────────

def salvar_embedding(embedding: np.ndarray, caminho: str) -> None:
    """Salva embedding em arquivo .npy para reutilização."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho, embedding)


def carregar_embedding(caminho: str) -> np.ndarray:
    """Carrega embedding de arquivo .npy."""
    return np.load(caminho).astype(np.float32)


# ── Banco de embeddings em memória ────────────────────────────────────────────

class BancoEmbeddings:
    """
    Gerenciador de embeddings de múltiplos locutores.
    Permite identificar o locutor mais próximo de um embedding de consulta.
    """

    def __init__(self):
        self._banco: dict = {}   # id → embedding

    def adicionar(self, id_locutor: str, embedding: np.ndarray) -> None:
        self._banco[id_locutor] = normalizar(embedding)

    def identificar(self, embedding: np.ndarray,
                    limiar: float = 0.85) -> tuple:
        """
        Retorna (id_locutor, similaridade) do locutor mais próximo.
        Se a similaridade máxima for menor que `limiar`, retorna (None, sim).
        """
        if not self._banco:
            return None, 0.0

        emb_norm = normalizar(embedding)
        melhor_id  = None
        melhor_sim = -1.0

        for id_loc, emb_ref in self._banco.items():
            sim = float(np.dot(emb_norm, emb_ref))
            if sim > melhor_sim:
                melhor_sim = sim
                melhor_id  = id_loc

        if melhor_sim < limiar:
            return None, melhor_sim
        return melhor_id, melhor_sim

    def listar(self) -> list:
        return list(self._banco.keys())

    def salvar_banco(self, diretorio: str) -> None:
        d = Path(diretorio)
        d.mkdir(parents=True, exist_ok=True)
        for id_loc, emb in self._banco.items():
            np.save(d / f"{id_loc}.npy", emb)

    def carregar_banco(self, diretorio: str) -> None:
        for arq in Path(diretorio).glob("*.npy"):
            id_loc = arq.stem
            self._banco[id_loc] = np.load(arq).astype(np.float32)
