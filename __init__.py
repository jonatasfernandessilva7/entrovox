"""
pipeline/__init__.py
====================
Reexporta todos os símbolos públicos dos submodulos do pacote,
permitindo imports flat usados em api.py, treinamento_ae.py e demo.py:

    from pipeline import PipelineRealce, SR
    from pipeline import aplicar_wiener, extrair_embedding
"""

# RF01
from .segmentacao  import segmentar, tamanho_quadro, hop_amostras
# RF02
from .espectrograma import espectrograma, n_bins
# RF03
from .wiener       import aplicar_wiener, estimar_perfil_ruido, estimar_snr_por_quadro
# RF04
from .embedding    import (extrair_embedding, normalizar, similaridade_cosseno,
                            distancia, salvar_embedding, carregar_embedding,
                            BancoEmbeddings)
# RF05
from .autoencoder  import AutoencoderCondicional
# RF06/RF07
from .entropia     import (calcular_entropia, entropia_maxima, validar_quadros,
                            score_qualidade, limiares_adaptativos,
                            detectar_tipo_quadro, resumo_entropia)
# RF08
from .reconstrucao import reconstruir
# Orquestrador
from .pipeline     import PipelineRealce, SR, DURACAO_MS, HOP_MS
