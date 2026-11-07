"""
processamento/equalizacao.py
============================
Equalização de histograma — RF-06 do escopo.
Responsável: Pessoa 4 (P4).

Teoria (Gonzalez & Woods, cap. 3)
--------------------------------
Objetivo: redistribuir as intensidades para que o histograma fique o mais
uniforme possível, aumentando o contraste global.

Passos (exatamente os do escopo):
  1. Histograma h(r_k): quantos pixels têm intensidade r_k (k = 0..255).
  2. Probabilidades:   p(r_k) = h(r_k) / N   (N = total de pixels).
  3. CDF:              CDF(r_k) = Σ_{j<=k} h(r_j)  (contagem acumulada).
  4. Tabela de transformação:
        s = round( (L − 1) · (CDF(r) − CDF_min) / (N − CDF_min) )
     CDF_min é o menor valor NÃO NULO da CDF (o primeiro nível presente).
     Subtraí-lo garante que o nível mais escuro presente vá para 0 e que a
     faixa completa 0..255 seja usada.
  5. Nova imagem: g = LUT[f].

Imagens coloridas: equalizar R, G e B separadamente altera as proporções
entre os canais e DISTORCE AS CORES. Por isso convertemos para YCbCr,
equalizamos apenas Y (luminância) e mantemos Cb e Cr (crominância). Também
é oferecida a opção HSV (equaliza o canal V).
"""

from __future__ import annotations

import numpy as np

from .pontuais import L, _validar_imagem, _separar_alfa, _juntar_alfa, eh_colorida


# ---------------------------------------------------------------------------
# Conversões de espaço de cor (implementação própria, ITU-R BT.601 / JPEG)
# ---------------------------------------------------------------------------
def rgb_para_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → YCbCr float64 (faixa completa 0..255, padrão JPEG).

    Y  =  0,299 R + 0,587 G + 0,114 B
    Cb = −0,168736 R − 0,331264 G + 0,5 B + 128
    Cr =  0,5 R − 0,418688 G − 0,081312 B + 128
    """
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]])
    ycc = rgb.astype(np.float64) @ m.T
    ycc[..., 1:] += 128.0
    return ycc


def ycbcr_para_rgb(ycc: np.ndarray) -> np.ndarray:
    """YCbCr float64 → RGB uint8 (inversa de rgb_para_ycbcr, com saturação)."""
    y = ycc[..., 0]
    cb = ycc[..., 1] - 128.0
    cr = ycc[..., 2] - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def rgb_para_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → HSV float64 com H em [0, 1), S em [0, 1] e V em [0, 255]."""
    x = rgb.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    v = x.max(axis=-1)
    c = v - x.min(axis=-1)  # croma
    s = np.where(v > 0, c / np.where(v > 0, v, 1), 0.0)
    h = np.zeros_like(v)
    c_seguro = np.where(c > 0, c, 1)
    mr = (v == r) & (c > 0)
    mg = (v == g) & (c > 0) & ~mr
    mb = (c > 0) & ~mr & ~mg
    h[mr] = ((g - b)[mr] / c_seguro[mr]) % 6
    h[mg] = ((b - r)[mg] / c_seguro[mg]) + 2
    h[mb] = ((r - g)[mb] / c_seguro[mb]) + 4
    h = h / 6.0
    return np.stack([h, s, v], axis=-1)


def hsv_para_rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV (como em rgb_para_hsv) → RGB uint8."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    escolhas_r = [v, q, p, p, t, v]
    escolhas_g = [t, v, v, q, p, p]
    escolhas_b = [p, p, t, v, v, q]
    r = np.choose(i, escolhas_r)
    g = np.choose(i, escolhas_g)
    b = np.choose(i, escolhas_b)
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Núcleo da equalização (canal único)
# ---------------------------------------------------------------------------
def histograma_256(canal: np.ndarray) -> np.ndarray:
    """Histograma de 256 níveis de um canal uint8 (contagens inteiras).

    np.bincount conta quantas vezes cada valor 0..255 aparece — equivale ao
    laço "para cada pixel, h[f] += 1", porém vetorizado.
    """
    return np.bincount(np.asarray(canal, dtype=np.uint8).ravel(), minlength=L).astype(np.int64)


def tabela_equalizacao(canal: np.ndarray) -> np.ndarray:
    """Calcula a LUT de equalização (256 posições, uint8) de um canal.

    s = round((L − 1)·(CDF(r) − CDF_min) / (N − CDF_min))
    """
    hist = histograma_256(canal)
    cdf = np.cumsum(hist)
    n = int(cdf[-1])
    cdf_min = int(cdf[np.flatnonzero(cdf)[0]])
    if n == cdf_min:
        # Imagem com um único nível: não há o que redistribuir (evita divisão por 0)
        return np.arange(L, dtype=np.uint8)
    lut = np.rint((L - 1) * (cdf - cdf_min) / (n - cdf_min))
    return np.clip(lut, 0, L - 1).astype(np.uint8)


def equalizar_canal(canal: np.ndarray) -> np.ndarray:
    """Equaliza um único canal uint8 (H, W)."""
    canal = np.asarray(canal, dtype=np.uint8)
    return tabela_equalizacao(canal)[canal]


def _equalizar_float(canal: np.ndarray) -> np.ndarray:
    """Equaliza um canal float (0..255): quantiza para uint8, aplica a LUT."""
    q = np.clip(np.rint(canal), 0, 255).astype(np.uint8)
    return equalizar_canal(q).astype(np.float64)


# ---------------------------------------------------------------------------
# Função pública (contrato do projeto)
# ---------------------------------------------------------------------------
ESPACOS_VALIDOS = ("ycbcr", "hsv", "rgb")


def equalizar_histograma(imagem: np.ndarray, espaco: str = "ycbcr") -> np.ndarray:
    """Equaliza o histograma da imagem.

    Parâmetros
    ----------
    espaco : para imagens coloridas, em qual representação equalizar:
        'ycbcr' (padrão) → equaliza só a luminância Y; cores preservadas.
        'hsv'            → equaliza só o canal V (valor/brilho).
        'rgb'            → equaliza cada canal separadamente (DIDÁTICO:
                           mostra a distorção de cores que o escopo pede
                           para evitar).
    Imagens em tons de cinza são equalizadas diretamente.

    Retorna: uint8 com o mesmo formato da entrada.
    """
    espaco = espaco.lower()
    if espaco not in ESPACOS_VALIDOS:
        raise ValueError(f"Espaço de cor inválido: {espaco}. Use um de {ESPACOS_VALIDOS}.")
    imagem = _validar_imagem(imagem)
    rgb, alfa = _separar_alfa(imagem)

    if rgb.ndim == 2:
        return equalizar_canal(rgb)

    if not eh_colorida(rgb):
        # Imagem "cinza" armazenada em 3 canais: equaliza um canal e replica
        canal = equalizar_canal(rgb[:, :, 0])
        return _juntar_alfa(np.repeat(canal[:, :, None], 3, axis=2), alfa)

    if espaco == "ycbcr":
        ycc = rgb_para_ycbcr(rgb)
        ycc[..., 0] = _equalizar_float(ycc[..., 0])
        saida = ycbcr_para_rgb(ycc)
    elif espaco == "hsv":
        hsv = rgb_para_hsv(rgb)
        hsv[..., 2] = _equalizar_float(hsv[..., 2])
        saida = hsv_para_rgb(hsv)
    else:  # 'rgb'
        saida = np.stack([equalizar_canal(rgb[:, :, c]) for c in range(3)], axis=-1)

    return _juntar_alfa(saida, alfa)


def detalhes_equalizacao(imagem: np.ndarray) -> dict:
    """Devolve todos os passos intermediários da equalização (para a interface,
    gráficos didáticos e o relatório):

        {
          'canal'            : canal equalizado ('cinza' ou 'Y'),
          'histograma'       : h(r_k) original (256,),
          'probabilidades'   : p(r_k) = h/N (256,),
          'cdf'              : CDF acumulada em contagens (256,),
          'cdf_normalizada'  : CDF / N, em [0, 1] (256,),
          'lut'              : tabela s = T(r) (256,),
          'histograma_eq'    : histograma após equalizar (256,),
          'cdf_normalizada_eq': CDF após equalizar — deve ficar ≈ reta,
          'cdf_min', 'n_pixels'
        }
    """
    imagem = _validar_imagem(imagem)
    rgb, _ = _separar_alfa(imagem)
    if rgb.ndim == 2 or not eh_colorida(rgb):
        canal = rgb if rgb.ndim == 2 else rgb[:, :, 0]
        nome = "cinza"
    else:
        canal = np.clip(np.rint(rgb_para_ycbcr(rgb)[..., 0]), 0, 255).astype(np.uint8)
        nome = "Y"

    hist = histograma_256(canal)
    n = int(hist.sum())
    cdf = np.cumsum(hist)
    lut = tabela_equalizacao(canal)
    hist_eq = histograma_256(lut[canal])
    return {
        "canal": nome,
        "histograma": hist,
        "probabilidades": hist / n,
        "cdf": cdf,
        "cdf_normalizada": cdf / n,
        "lut": lut,
        "histograma_eq": hist_eq,
        "cdf_normalizada_eq": np.cumsum(hist_eq) / n,
        "cdf_min": int(cdf[np.flatnonzero(cdf)[0]]),
        "n_pixels": n,
    }


def linearidade_cdf(cdf_normalizada: np.ndarray) -> float:
    """Erro médio absoluto entre a CDF e a reta ideal (0 → 1), considerando
    apenas os níveis efetivamente ocupados. Quanto menor, mais uniforme o
    histograma. Usado nos testes ("CDF aproximadamente linear")."""
    cdf = np.asarray(cdf_normalizada, dtype=np.float64)
    ideal = (np.arange(L) + 1) / L
    return float(np.mean(np.abs(cdf - ideal)))