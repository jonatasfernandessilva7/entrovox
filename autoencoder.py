import numpy as np
from pathlib import Path

class AutoencoderCondicional:
    """
    Autoencoder condicional leve para realce espectral.

    Arquitetura:
        Encoder: [n_bins + dim_emb] → ReLU → [dim_latente]
        Decoder: [dim_latente + dim_emb] → ReLU → [n_bins]

    Pesos exportados do treinamento PyTorch como arquivo .npz com chaves:
        enc_W, enc_b, dec_W, dec_b
    """

    def __init__(self, caminho_pesos: str = None):
        self.pesos     = None
        self.treinado  = False
        if caminho_pesos and Path(caminho_pesos).exists():
            self.carregar(caminho_pesos)

    def carregar(self, caminho: str) -> None:
        dados = np.load(caminho)
        self.pesos    = {k: dados[k] for k in dados.files}
        self.treinado = True

    def salvar(self, caminho: str) -> None:
        if self.pesos:
            np.savez(caminho, **self.pesos)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _camada(self, x: np.ndarray, prefixo: str,
                ativacao: str = "relu") -> np.ndarray:
        W = self.pesos[f"{prefixo}_W"]
        b = self.pesos[f"{prefixo}_b"]
        h = x @ W + b
        if ativacao == "relu":
            return self._relu(h)
        if ativacao == "sigmoid":
            return self._sigmoid(h)
        return h   # linear

    def reconstruir(self,
                    magnitudes: np.ndarray,
                    embedding:  np.ndarray) -> np.ndarray:
        """
        Reconstrói o espectrograma limpo quadro a quadro.

        Parâmetros
        ----------
        magnitudes : (n_quadros, n_bins)
        embedding  : (dim_emb,)

        Retorno
        -------
        reconstruido : (n_quadros, n_bins) — valores ≥ 0
        """
        if not self.treinado:
            return magnitudes.copy()

        emb_tile = np.tile(embedding, (magnitudes.shape[0], 1))

        # Encoder
        entrada_enc = np.concatenate([magnitudes, emb_tile], axis=1)
        latente     = self._camada(entrada_enc, "enc", "relu")

        # Decoder (condicionado novamente ao embedding)
        entrada_dec = np.concatenate([latente, emb_tile], axis=1)
        saida       = self._camada(entrada_dec, "dec", "relu")

        return np.maximum(saida, 0.0).astype(np.float32)

    def treinar(self,
                X_ruido:   np.ndarray,
                X_limpo:   np.ndarray,
                embeddings: np.ndarray,
                dim_latente: int = 128,
                epocas:     int  = 50,
                lr:         float = 1e-3,
                batch:      int  = 64,
                seed:       int  = 42) -> list:
        """
        Treina o autoencoder via SGD com backprop manual (numpy puro).

        Parâmetros
        ----------
        X_ruido    : (N, n_bins) — espectrogramas ruidosos (quadros)
        X_limpo    : (N, n_bins) — espectrogramas limpos correspondentes
        embeddings : (N, dim_emb) — embedding de cada quadro
        dim_latente: dimensão do espaço latente
        epocas     : número de épocas
        lr         : taxa de aprendizagem
        batch      : tamanho do mini-batch

        Retorno
        -------
        historico : lista de losses por época
        """
        rng = np.random.default_rng(seed)
        n_bins  = X_ruido.shape[1]
        dim_emb = embeddings.shape[1]
        dim_enc = n_bins + dim_emb
        dim_dec = dim_latente + dim_emb

        # Inicialização He
        escala_enc = np.sqrt(2.0 / dim_enc)
        escala_dec = np.sqrt(2.0 / dim_dec)

        self.pesos = {
            "enc_W": rng.standard_normal((dim_enc, dim_latente)).astype(np.float32) * escala_enc,
            "enc_b": np.zeros(dim_latente, dtype=np.float32),
            "dec_W": rng.standard_normal((dim_dec, n_bins)).astype(np.float32) * escala_dec,
            "dec_b": np.zeros(n_bins, dtype=np.float32),
        }

        N        = X_ruido.shape[0]
        historico = []

        for epoca in range(epocas):
            idx   = rng.permutation(N)
            loss_epoca = 0.0
            n_batches  = 0

            for start in range(0, N, batch):
                b_idx = idx[start: start + batch]
                xr    = X_ruido   [b_idx]
                xl    = X_limpo   [b_idx]
                emb   = embeddings[b_idx]

                # ── Forward ──
                enc_in = np.concatenate([xr, emb], axis=1)
                enc_h  = self._relu(enc_in @ self.pesos["enc_W"] + self.pesos["enc_b"])

                dec_in = np.concatenate([enc_h, emb], axis=1)
                dec_h  = self._relu(dec_in @ self.pesos["dec_W"] + self.pesos["dec_b"])

                erro  = dec_h - xl
                loss  = np.mean(erro ** 2)
                loss_epoca += loss

                # ── Backward ──
                d_dec_h  = 2 * erro / len(b_idx)
                d_dec_h *= (dec_h > 0).astype(np.float32)   # ReLU grad

                d_dec_W  = dec_in.T @ d_dec_h
                d_dec_b  = d_dec_h.sum(axis=0)
                d_dec_in = d_dec_h @ self.pesos["dec_W"].T

                # Gradiente para encoder (parte do dec_in que vem do encoder)
                d_enc_h  = d_dec_in[:, :dim_latente]
                d_enc_h *= (enc_h > 0).astype(np.float32)

                d_enc_W  = enc_in.T @ d_enc_h
                d_enc_b  = d_enc_h.sum(axis=0)

                # ── Atualização ──
                self.pesos["dec_W"] -= lr * d_dec_W
                self.pesos["dec_b"] -= lr * d_dec_b
                self.pesos["enc_W"] -= lr * d_enc_W
                self.pesos["enc_b"] -= lr * d_enc_b

                n_batches += 1

            loss_media = loss_epoca / max(n_batches, 1)
            historico.append(loss_media)

            if (epoca + 1) % 10 == 0:
                print(f"  Época {epoca+1:3d}/{epocas} | loss={loss_media:.6f}")

        self.treinado = True
        return historico
