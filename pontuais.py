"""
processamento/pontuais.py
=========================
Operações pontuais (transformações de intensidade) — RF-05 do escopo.
Responsável: Pessoa 4 (P4).

Teoria
------
Uma operação pontual calcula cada pixel de saída usando APENAS o valor do
pixel de entrada na mesma posição:  g(x, y) = T[f(x, y)].
Como f assume só 256 valores possíveis (0..255 em uint8), qualquer T pode
ser pré-calculada em uma tabela de consulta (LUT, *look-up table*) com 256
entradas. Aplicar a operação vira uma simples indexação: g = LUT[f].
Isso é O(N), totalmente vetorizado e muito rápido (bem abaixo de 1 s para
Full HD), o que permite pré-visualização em tempo real nos sliders.

Contrato
--------
Todas as funções públicas seguem o padrão do projeto:
    funcao(imagem: np.ndarray, **parametros) -> np.ndarray
- Aceitam imagens em tons de cinza (H, W) ou coloridas (H, W, 3) / (H, W, 4).
- Nunca modificam o array de entrada (funções puras).
- Sempre devolvem uint8 com valores em 0..255 (saturação com np.clip).
"""

from __future__ import annotations

import numpy as np

# Pesos da luminância (ITU-R BT.601), usados no escopo: Y = 0,299R + 0,587G + 0,114B
PESOS_LUMINANCIA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

L = 256  # número de níveis de intensidade em imagens de 8 bits


# ---------------------------------------------------------------------------
# Utilitários internos
# ---------------------------------------------------------------------------
def _validar_imagem(imagem: np.ndarray) -> np.ndarray:
    """Valida o formato da imagem e devolve uma versão uint8 (sem alterar a original).

    Aceita (H, W), (H, W, 1), (H, W, 3) ou (H, W, 4). Imagens float em [0, 1]
    são reescaladas para [0, 255]; demais tipos são saturados e convertidos.
    """
    if imagem is None:
        raise ValueError("Nenhuma imagem foi fornecida.")
    imagem = np.asarray(imagem)
    if imagem.ndim == 3 and imagem.shape[2] == 1:
        imagem = imagem[:, :, 0]
    if imagem.ndim not in (2, 3) or (imagem.ndim == 3 and imagem.shape[2] not in (3, 4)):
        raise ValueError(
            f"Formato de imagem não suportado: {imagem.shape}. "
            "Use (altura, largura) ou (altura, largura, 3/4)."
        )
    if imagem.size == 0:
        raise ValueError("A imagem está vazia.")

    if imagem.dtype == np.uint8:
        return imagem
    if np.issubdtype(imagem.dtype, np.floating) and imagem.max(initial=0) <= 1.0:
        imagem = imagem * 255.0
    return np.clip(np.rint(imagem), 0, 255).astype(np.uint8)


def _separar_alfa(imagem: np.ndarray):
    """Separa o canal alfa (RGBA) para que as operações não o alterem."""
    if imagem.ndim == 3 and imagem.shape[2] == 4:
        return imagem[:, :, :3], imagem[:, :, 3:]
    return imagem, None


def _juntar_alfa(rgb: np.ndarray, alfa):
    """Recoloca o canal alfa, se existir (só faz sentido em saídas coloridas)."""
    if alfa is None or rgb.ndim == 2:
        return rgb
    return np.concatenate([rgb, alfa], axis=2)


def aplicar_lut(imagem: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Aplica uma tabela de transformação (LUT de 256 posições) à imagem.

    g = LUT[f]  — é a forma computacional de g(x, y) = T[f(x, y)].
    O canal alfa (se houver) é preservado.
    """
    imagem = _validar_imagem(imagem)
    lut = np.clip(np.rint(np.asarray(lut, dtype=np.float64)), 0, 255).astype(np.uint8)
    if lut.shape != (L,):
        raise ValueError("A LUT deve ter exatamente 256 posições.")
    rgb, alfa = _separar_alfa(imagem)
    return _juntar_alfa(lut[rgb], alfa)


def luminancia(imagem: np.ndarray) -> np.ndarray:
    """Calcula a luminância Y (float64, 0..255). Em imagens cinza, devolve a própria imagem."""
    imagem = _validar_imagem(imagem)
    rgb, _ = _separar_alfa(imagem)
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    return rgb.astype(np.float64) @ PESOS_LUMINANCIA


def eh_colorida(imagem: np.ndarray) -> bool:
    """True se a imagem tem 3/4 canais E os canais R, G, B não são todos iguais."""
    imagem = np.asarray(imagem)
    if imagem.ndim == 2:
        return False
    r, g, b = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]
    return not (np.array_equal(r, g) and np.array_equal(g, b))


# ---------------------------------------------------------------------------
# 1) Escala de cinza
# ---------------------------------------------------------------------------
def escala_cinza(imagem: np.ndarray) -> np.ndarray:
    """Converte a imagem para tons de cinza (um canal).

    Fórmula (média ponderada pela sensibilidade do olho humano):
        Y = 0,299·R + 0,587·G + 0,114·B
    O verde pesa mais porque o olho é mais sensível a ele. Uma média simples
    (R+G+B)/3 deixaria azuis claros demais e verdes escuros demais.

    Retorna: array uint8 (H, W). Se a imagem já for cinza, devolve uma cópia.
    """
    imagem = _validar_imagem(imagem)
    if imagem.ndim == 2:
        return imagem.copy()
    return np.clip(np.rint(luminancia(imagem)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2) Negativo
# ---------------------------------------------------------------------------
def negativo(imagem: np.ndarray) -> np.ndarray:
    """Inverte as intensidades: g = (L − 1) − f = 255 − f.

    Útil para realçar detalhes claros em regiões escuras (ex.: radiografias).
    Propriedade: negativo(negativo(f)) == f (a operação é sua própria inversa).
    """
    lut = (L - 1) - np.arange(L)
    return aplicar_lut(imagem, lut)


# ---------------------------------------------------------------------------
# 3) Brilho
# ---------------------------------------------------------------------------
BRILHO_MIN, BRILHO_MAX = -100, 100


def ajustar_brilho(imagem: np.ndarray, brilho: float = 0) -> np.ndarray:
    """Clareia (brilho > 0) ou escurece (brilho < 0) a imagem.

    Fórmula: g = f + b, com saturação em [0, 255].
    Sem a saturação, 250 + 10 = 260 "daria a volta" em uint8 e viraria 4
    (overflow) — por isso o cálculo é feito na LUT em float e depois np.clip.

    Parâmetros
    ----------
    brilho : float no intervalo [-100, +100] (slider da interface).
    """
    if not (BRILHO_MIN <= brilho <= BRILHO_MAX):
        raise ValueError(f"O brilho deve estar entre {BRILHO_MIN} e {BRILHO_MAX}.")
    lut = np.arange(L, dtype=np.float64) + float(brilho)
    return aplicar_lut(imagem, lut)


# ---------------------------------------------------------------------------
# 4) Contraste
# ---------------------------------------------------------------------------
CONTRASTE_MIN, CONTRASTE_MAX = 0.0, 2.0


def ajustar_contraste(imagem: np.ndarray, alfa: float = 1.0, centro: float = 128.0) -> np.ndarray:
    """Afasta (alfa > 1) ou aproxima (alfa < 1) as intensidades do centro.

    Fórmula: g = α·(f − 128) + 128, com saturação em [0, 255].
    - α = 1   → imagem inalterada
    - α > 1   → aumenta o contraste (escuros mais escuros, claros mais claros)
    - α < 1   → reduz o contraste; α = 0 → imagem cinza uniforme (128)

    Parâmetros
    ----------
    alfa   : fator de contraste no intervalo [0,0; 2,0] (slider).
    centro : ponto fixo da transformação (padrão 128, meio da escala).
    """
    if not (CONTRASTE_MIN <= alfa <= CONTRASTE_MAX):
        raise ValueError(f"O contraste deve estar entre {CONTRASTE_MIN} e {CONTRASTE_MAX}.")
    f = np.arange(L, dtype=np.float64)
    lut = alfa * (f - centro) + centro
    return aplicar_lut(imagem, lut)


# ---------------------------------------------------------------------------
# 5) Correção gama (usada pela melhoria automática)
# ---------------------------------------------------------------------------
def correcao_gama(imagem: np.ndarray, gama: float = 1.0) -> np.ndarray:
    """Transformação de potência (lei de potência / correção gama).

    Fórmula: g = 255 · (f / 255)^γ
    - γ < 1 → clareia, expandindo os tons escuros (melhor que somar brilho,
      pois não "estoura" os claros nem desperdiça níveis).
    - γ > 1 → escurece, expandindo os tons claros.
    - γ = 1 → identidade.
    """
    if gama <= 0:
        raise ValueError("O valor de gama deve ser positivo.")
    f = np.arange(L, dtype=np.float64) / (L - 1)
    lut = (L - 1) * np.power(f, gama)
    return aplicar_lut(imagem, lut)


def gama_para_alvo(brilho_atual: float, alvo: float = 120.0,
                   gama_min: float = 0.25, gama_max: float = 4.0) -> float:
    """Calcula o γ que leva aproximadamente o brilho médio até o alvo.

    Resolve 255·(m/255)^γ = alvo  →  γ = ln(alvo/255) / ln(m/255).
    (Aproximação: considera só a média; o valor é limitado a [gama_min, gama_max]
    para evitar transformações extremas.)
    """
    m = float(np.clip(brilho_atual, 1.0, 254.0)) / 255.0
    a = float(np.clip(alvo, 1.0, 254.0)) / 255.0
    gama = np.log(a) / np.log(m)
    return float(np.clip(gama, gama_min, gama_max))


# ---------------------------------------------------------------------------
# 6) Alongamento de contraste (contrast stretching) — usado pela melhoria automática
# ---------------------------------------------------------------------------
def alongamento_contraste(imagem: np.ndarray, percentil_inf: float = 1.0,
                          percentil_sup: float = 99.0) -> np.ndarray:
    """Estica linearmente a faixa [r_min, r_max] para ocupar [0, 255].

    Fórmula: g = 255 · (f − r_min) / (r_max − r_min), com saturação.
    r_min e r_max são percentis da luminância (padrão 1% e 99%) e não o
    mínimo/máximo absolutos — assim alguns poucos pixels extremos (ruído)
    não impedem o alongamento.

    Em imagens coloridas a MESMA LUT é aplicada aos três canais, o que
    preserva o matiz aproximadamente (não muda as proporções entre canais de
    forma drástica).
    """
    if not (0 <= percentil_inf < percentil_sup <= 100):
        raise ValueError("Percentis inválidos: use 0 <= inferior < superior <= 100.")
    imagem = _validar_imagem(imagem)
    y = luminancia(imagem)
    r_min, r_max = np.percentile(y, [percentil_inf, percentil_sup])
    if r_max - r_min < 1:
        return imagem.copy()  # imagem uniforme: nada a alongar
    f = np.arange(L, dtype=np.float64)
    lut = (L - 1) * (f - r_min) / (r_max - r_min)
    return aplicar_lut(imagem, lut)


# ---------------------------------------------------------------------------
# 7) Binarização (limiarização) + Otsu
# ---------------------------------------------------------------------------
def limiar_otsu(imagem: np.ndarray) -> int:
    """Calcula o limiar ótimo de Otsu (1979) a partir do histograma.

    Ideia: testar todos os limiares T e escolher o que MAXIMIZA a variância
    entre as classes (fundo e objeto):
        σ²_B(T) = [μ_T·ω(T) − μ(T)]² / [ω(T)·(1 − ω(T))]
    onde
        p_i    = h(i)/N           (probabilidade do nível i)
        ω(T)   = Σ_{i<=T} p_i     (probabilidade da classe 0)
        μ(T)   = Σ_{i<=T} i·p_i   (média acumulada)
        μ_T    = Σ_i i·p_i        (média global)
    Tudo é calculado de forma vetorizada com somas acumuladas (np.cumsum).

    Retorna o limiar T (int). Na binarização, pixels com f > T viram 255
    (convenção do OpenCV). Para usar com a regra "f >= T" do escopo, a função
    binarizar() soma 1 automaticamente.
    """
    y = escala_cinza(imagem)
    hist = np.bincount(y.ravel(), minlength=L).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(L))
    mu_t = mu[-1]
    denominador = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denominador
    sigma_b2[~np.isfinite(sigma_b2)] = 0.0
    if not np.any(sigma_b2 > 0):  # imagem com um único nível
        return int(y.flat[0])
    # Se houver platô de máximos, usa o centro dele (resultado mais estável)
    maximos = np.flatnonzero(sigma_b2 == sigma_b2.max())
    return int(maximos[len(maximos) // 2])


def binarizar(imagem: np.ndarray, limiar: int = 128, automatico: bool = False,
              inverter: bool = False) -> np.ndarray:
    """Separa fundo e objeto em duas classes (0 e 255).

    Fórmula (escopo): g = 255 se f >= T, senão 0.
    Imagens coloridas são convertidas para cinza antes.

    Parâmetros
    ----------
    limiar     : T no intervalo [0, 255] (slider). Ignorado se automatico=True.
    automatico : se True, calcula T pelo método de Otsu.
    inverter   : se True, troca 0 e 255 (objeto escuro em fundo claro).

    Retorna: uint8 (H, W) contendo apenas 0 e 255.
    """
    cinza = escala_cinza(imagem)
    if automatico:
        t = limiar_otsu(cinza) + 1  # Otsu usa f > T; convertemos para f >= T
    else:
        if not (0 <= limiar <= 255):
            raise ValueError("O limiar deve estar entre 0 e 255.")
        t = int(limiar)
    resultado = np.where(cinza >= t, 255, 0).astype(np.uint8)
    return (255 - resultado) if inverter else resultado


# ---------------------------------------------------------------------------
# Catálogo para a interface (P2) e o histórico (P1)
# ---------------------------------------------------------------------------
# Descreve cada operação, seus parâmetros e faixas dos sliders. A interface
# pode montar os controles a partir daqui, e o histórico pode registrar
# "nome + parâmetros" para reaplicar as operações sobre o original.
OPERACOES_PONTUAIS = {
    "escala_cinza": {
        "funcao": escala_cinza,
        "rotulo": "Escala de cinza",
        "parametros": {},
    },
    "negativo": {
        "funcao": negativo,
        "rotulo": "Negativo",
        "parametros": {},
    },
    "ajustar_brilho": {
        "funcao": ajustar_brilho,
        "rotulo": "Brilho",
        "parametros": {"brilho": {"min": BRILHO_MIN, "max": BRILHO_MAX, "passo": 1, "padrao": 0}},
    },
    "ajustar_contraste": {
        "funcao": ajustar_contraste,
        "rotulo": "Contraste",
        "parametros": {"alfa": {"min": CONTRASTE_MIN, "max": CONTRASTE_MAX, "passo": 0.05, "padrao": 1.0}},
    },
    "correcao_gama": {
        "funcao": correcao_gama,
        "rotulo": "Correção gama",
        "parametros": {"gama": {"min": 0.1, "max": 5.0, "passo": 0.05, "padrao": 1.0}},
    },
    "alongamento_contraste": {
        "funcao": alongamento_contraste,
        "rotulo": "Alongamento de contraste",
        "parametros": {
            "percentil_inf": {"min": 0.0, "max": 10.0, "passo": 0.5, "padrao": 1.0},
            "percentil_sup": {"min": 90.0, "max": 100.0, "passo": 0.5, "padrao": 99.0},
        },
    },
    "binarizar": {
        "funcao": binarizar,
        "rotulo": "Binarização",
        "parametros": {
            "limiar": {"min": 0, "max": 255, "passo": 1, "padrao": 128},
            "automatico": {"tipo": "bool", "padrao": False},
            "inverter": {"tipo": "bool", "padrao": False},
        },
    },
}


def descrever_operacao(nome: str, **parametros) -> str:
    """Texto curto para o histórico/relatório. Ex.: 'Contraste (alfa=1.3)'."""
    info = OPERACOES_PONTUAIS.get(nome)
    rotulo = info["rotulo"] if info else nome
    if not parametros:
        return rotulo
    partes = ", ".join(
        f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in parametros.items()
    )
    return f"{rotulo} ({partes})"