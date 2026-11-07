"""Filtros espaciais (P5): média, mediana, nitidez e ruído sal e pimenta.

Padrão do projeto: funcao(imagem: np.ndarray, **parametros) -> np.ndarray
Imagens uint8, cinza (H, W) ou colorida (H, W, 3).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _por_canal(func):
    """Aplica func em cada canal se a imagem for colorida."""
    def wrapper(imagem, *args, **kw):
        if imagem.ndim == 3:
            canais = [func(imagem[..., c], *args, **kw) for c in range(imagem.shape[2])]
            return np.stack(canais, axis=-1)
        return func(imagem, *args, **kw)
    return wrapper


def _padding(img, r):
    """Replica os pixels da borda (padding 'edge')."""
    return np.pad(img, r, mode="edge")


def convolucao(imagem, kernel):
    """Convolução 2D com padding por replicação. Retorna float64."""
    kernel = np.asarray(kernel, dtype=np.float64)
    r = kernel.shape[0] // 2
    kf = kernel[::-1, ::-1]  # inverte o kernel (convolução, não correlação)

    @_por_canal
    def _conv(img):
        pad = _padding(img.astype(np.float64), r)
        janelas = sliding_window_view(pad, kernel.shape)
        return np.einsum("ijkl,kl->ij", janelas, kf)

    return _conv(imagem)


def _para_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def filtro_media(imagem, tamanho=3):
    """Suaviza: cada pixel vira a média da vizinhança (3, 5 ou 7)."""
    if tamanho not in (3, 5, 7):
        raise ValueError("tamanho deve ser 3, 5 ou 7")
    kernel = np.ones((tamanho, tamanho)) / tamanho**2
    return _para_uint8(convolucao(imagem, kernel))


def filtro_mediana(imagem, tamanho=3):
    """Remove ruído sal e pimenta: pixel vira a mediana da vizinhança (3 ou 5)."""
    if tamanho not in (3, 5):
        raise ValueError("tamanho deve ser 3 ou 5")
    r = tamanho // 2

    @_por_canal
    def _med(img):
        pad = _padding(img, r)
        janelas = sliding_window_view(pad, (tamanho, tamanho))
        return np.median(janelas, axis=(-2, -1)).astype(np.uint8)

    return _med(imagem)


def nitidez(imagem, metodo="unsharp", intensidade=1.0, tamanho=5):
    """Realce de nitidez.

    metodo='laplaciano': g = f - intensidade * laplaciano(f)
    metodo='unsharp'   : g = f + intensidade * (f - borrada(f))
    """
    f = imagem.astype(np.float64)
    if metodo == "laplaciano":
        lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        return _para_uint8(f - intensidade * convolucao(imagem, lap))
    if metodo == "unsharp":
        borrada = convolucao(imagem, np.ones((tamanho, tamanho)) / tamanho**2)
        return _para_uint8(f + intensidade * (f - borrada))
    raise ValueError("metodo deve ser 'laplaciano' ou 'unsharp'")


def ruido_sal_pimenta(imagem, proporcao=0.05, semente=None):
    """Gera ruído sal e pimenta (para o banco de teste). proporcao = fração de pixels."""
    rng = np.random.default_rng(semente)
    saida = imagem.copy()
    sorteio = rng.random(imagem.shape[:2])
    saida[sorteio < proporcao / 2] = 0
    saida[(sorteio >= proporcao / 2) & (sorteio < proporcao)] = 255
    return saida


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255**2 / m)
