"""Detecção de bordas (P5): Sobel."""
import numpy as np
from processamento.filtros import convolucao

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def _luminancia(imagem):
    if imagem.ndim == 3:
        return (0.299 * imagem[..., 0] + 0.587 * imagem[..., 1]
                + 0.114 * imagem[..., 2])
    return imagem.astype(np.float64)


def sobel(imagem, limiar=None):
    """Magnitude do gradiente G = sqrt(Gx² + Gy²), normalizada para 0-255.

    Se limiar for dado (0-255), binariza o resultado.
    """
    lum = _luminancia(imagem)
    gx = convolucao(lum, KX)
    gy = convolucao(lum, KY)
    g = np.hypot(gx, gy)
    maximo = g.max()
    g = g / maximo * 255 if maximo > 0 else g  # imagem uniforme -> tudo 0
    g = np.clip(np.rint(g), 0, 255).astype(np.uint8)
    if limiar is not None:
        g = np.where(g >= limiar, 255, 0).astype(np.uint8)
    return g
