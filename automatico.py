"""
processamento/automatico.py
===========================
Melhoria automática ("✨ Melhorar automaticamente") — RF-09 do escopo.
Responsável: Pessoa 4 (P4).

Não usa IA: é um sistema de REGRAS sobre métricas objetivas da própria
imagem (brilho médio, desvio padrão e estimativas de ruído).

Fluxo
-----
  0. Diagnóstico inicial (brilho, contraste, ruído).
  1. Ruído alto   → mediana 3×3.
     (Feito ANTES do brilho/contraste: clarear ou equalizar primeiro
      amplificaria o ruído e o tornaria mais visível.)
  2. Brilho baixo/alto → correção gama com γ calculado para levar o brilho
     médio até o alvo (~120). Gama é melhor que somar uma constante porque
     não satura os tons claros (ou escuros) da imagem.
  3. Contraste baixo → alongamento de contraste (percentis 1–99%);
     se ainda continuar baixo → equalização de histograma (na luminância).
  4. Nada a fazer → "Nenhuma melhoria necessária".

Cada passo é registrado com nome da operação, parâmetros, justificativa
(com os números que motivaram a decisão) e a imagem resultante — assim o
histórico (P1) pode empilhar cada passo e o usuário consegue desfazê-los
um a um, e o relatório (P3) pode explicar o que foi feito e por quê.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .pontuais import (
    _validar_imagem,
    _separar_alfa,
    _juntar_alfa,
    luminancia,
    correcao_gama,
    gama_para_alvo,
    alongamento_contraste,
)
from .equalizacao import equalizar_histograma


# ---------------------------------------------------------------------------
# Mediana 3×3: usa a implementação da P5 (filtros.py) quando disponível.
# ---------------------------------------------------------------------------
def _mediana_propria(imagem: np.ndarray, tamanho: int = 3) -> np.ndarray:
    """Filtro da mediana vetorizado (fallback caso filtros.py ainda não exista).

    Para cada pixel, ordena a vizinhança tamanho×tamanho e pega o valor do
    meio. Bordas tratadas por replicação de pixels (padding 'edge').
    """
    imagem = _validar_imagem(imagem)
    rgb, alfa = _separar_alfa(imagem)
    r = tamanho // 2

    def _canal(c: np.ndarray) -> np.ndarray:
        pad = np.pad(c, r, mode="edge")
        janelas = sliding_window_view(pad, (tamanho, tamanho)).reshape(c.shape[0], c.shape[1], -1)
        k = janelas.shape[-1] // 2
        return np.partition(janelas, k, axis=-1)[..., k].astype(np.uint8)

    if rgb.ndim == 2:
        return _canal(rgb)
    saida = np.stack([_canal(rgb[:, :, i]) for i in range(rgb.shape[2])], axis=-1)
    return _juntar_alfa(saida, alfa)


def _obter_mediana() -> Callable:
    try:  # integração com o módulo da P5
        from .filtros import filtro_mediana  # type: ignore
        return lambda img, tamanho=3: filtro_mediana(img, tamanho=tamanho)
    except Exception:
        return _mediana_propria


def filtro_mediana_3x3(imagem: np.ndarray, tamanho: int = 3) -> np.ndarray:
    """Mediana usada pela melhoria automática (P5 se disponível, senão própria)."""
    return _obter_mediana()(imagem, tamanho=tamanho)


# ---------------------------------------------------------------------------
# Limiares configuráveis (escopo, RF-03)
# ---------------------------------------------------------------------------
@dataclass
class LimiaresDiagnostico:
    """Limiares das regras. Todos podem ser recalibrados com o banco de imagens."""
    brilho_escuro: float = 85.0         # brilho < 85  → muito escura
    brilho_claro: float = 170.0         # brilho > 170 → muito clara
    desvio_baixo: float = 35.0          # desvio < 35  → baixo contraste
    ruido_impulsivo: float = 0.01       # > 1% de pixels impulsivos → ruído sal e pimenta
    sigma_ruido: float = 12.0           # σ estimado > 12 → ruído gaussiano/granulado
    brilho_alvo: float = 120.0          # alvo da correção de brilho
    max_passos: int = 6                 # segurança contra laços


# ---------------------------------------------------------------------------
# Estimativas de ruído
# ---------------------------------------------------------------------------
def _recorte_para_estimativa(y: np.ndarray, lado_max: int = 1024) -> np.ndarray:
    """Usa um recorte central (sem redimensionar, para não 'apagar' o ruído)
    para que a estimativa seja rápida mesmo em imagens de 8 MP."""
    h, w = y.shape
    if h <= lado_max and w <= lado_max:
        return y
    top = max(0, (h - lado_max) // 2)
    left = max(0, (w - lado_max) // 2)
    return y[top:top + lado_max, left:left + lado_max]


def estimar_ruido_impulsivo(imagem: np.ndarray, diferenca: float = 60.0) -> float:
    """Fração de pixels com cara de ruído sal e pimenta.

    Um pixel é considerado impulsivo se for (quase) extremo (≤ 15 ou ≥ 240)
    E diferir da mediana da sua vizinhança 3×3 em mais de `diferenca` níveis.
    Em fotos limpas essa fração fica perto de 0; com sal e pimenta, ela é
    aproximadamente a densidade do ruído.
    """
    y = _recorte_para_estimativa(luminancia(imagem))
    y8 = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    med = _mediana_propria(y8).astype(np.float64)
    extremo = (y <= 15) | (y >= 240)
    impulsivo = extremo & (np.abs(y - med) > diferenca)
    return float(impulsivo.mean())


def estimar_sigma_ruido(imagem: np.ndarray) -> float:
    """Estimativa rápida do desvio padrão do ruído (método de Immerkær, 1996).

    Convolui a imagem com uma máscara que anula regiões suaves e bordas
    lineares (diferença de dois laplacianos):
             [ 1 −2  1]
        M =  [−2  4 −2]
             [ 1 −2  1]
    σ ≈ sqrt(π/2) · Σ|I * M| / (6·(W−2)·(H−2))
    Mede a "variação local elevada" mencionada no escopo.
    """
    y = _recorte_para_estimativa(luminancia(imagem))
    h, w = y.shape
    if h < 3 or w < 3:
        return 0.0
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    janelas = sliding_window_view(y, (3, 3))
    resposta = np.einsum("ijkl,kl->ij", janelas, m)
    return float(np.sqrt(np.pi / 2) * np.abs(resposta).sum() / (6.0 * (w - 2) * (h - 2)))


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------
@dataclass
class Diagnostico:
    brilho: float
    desvio: float
    minimo: int
    maximo: int
    ruido_impulsivo: float
    sigma_ruido: float
    problemas: list = field(default_factory=list)

    @property
    def equilibrada(self) -> bool:
        return not self.problemas

    @property
    def rotulo(self) -> str:
        """Texto curto: 'equilibrada' ou a lista de problemas."""
        return "equilibrada" if self.equilibrada else ", ".join(self.problemas)

    def como_dict(self) -> dict:
        d = asdict(self)
        d["rotulo"] = self.rotulo
        return d

    def texto(self) -> str:
        return (f"Brilho médio {self.brilho:.1f} | desvio {self.desvio:.1f} | "
                f"faixa {self.minimo}–{self.maximo} | ruído impulsivo "
                f"{100 * self.ruido_impulsivo:.2f}% | σ ruído {self.sigma_ruido:.1f} "
                f"→ {self.rotulo}")


def diagnosticar(imagem: np.ndarray, limiares: Optional[LimiaresDiagnostico] = None) -> Diagnostico:
    """Calcula as métricas e classifica a imagem.

    Problemas possíveis: 'muito escura', 'muito clara', 'baixo contraste',
    'ruidosa'. Se nenhum for detectado, a imagem é 'equilibrada'.
    """
    lim = limiares or LimiaresDiagnostico()
    imagem = _validar_imagem(imagem)
    y = luminancia(imagem)
    brilho = float(y.mean())
    desvio = float(y.std())
    imp = estimar_ruido_impulsivo(imagem)
    sigma = estimar_sigma_ruido(imagem)

    problemas = []
    if brilho < lim.brilho_escuro:
        problemas.append("muito escura")
    elif brilho > lim.brilho_claro:
        problemas.append("muito clara")
    if desvio < lim.desvio_baixo:
        problemas.append("baixo contraste")
    if imp > lim.ruido_impulsivo or sigma > lim.sigma_ruido:
        problemas.append("ruidosa")

    return Diagnostico(
        brilho=brilho, desvio=desvio,
        minimo=int(np.floor(y.min())), maximo=int(np.ceil(y.max())),
        ruido_impulsivo=imp, sigma_ruido=sigma, problemas=problemas,
    )


# ---------------------------------------------------------------------------
# Resultado da melhoria
# ---------------------------------------------------------------------------
@dataclass
class PassoMelhoria:
    """Um passo aplicado — pronto para ser empilhado no histórico (P1)."""
    operacao: str            # nome da função (ver REGISTRO_OPERACOES)
    rotulo: str              # texto para a lista de histórico
    parametros: dict         # para reaplicar: REGISTRO_OPERACOES[operacao](img, **parametros)
    justificativa: str       # por que foi aplicado (com números)
    imagem: np.ndarray       # estado da imagem APÓS o passo
    diagnostico_depois: Diagnostico


@dataclass
class ResultadoMelhoria:
    imagem: np.ndarray
    diagnostico_inicial: Diagnostico
    diagnostico_final: Diagnostico
    passos: list = field(default_factory=list)

    @property
    def alterou(self) -> bool:
        return bool(self.passos)

    @property
    def mensagem(self) -> str:
        if not self.passos:
            return "Nenhuma melhoria necessária: a imagem já está equilibrada."
        return f"{len(self.passos)} correção(ões) aplicada(s): " + " → ".join(p.rotulo for p in self.passos)

    def justificativa(self) -> str:
        """Texto completo para exibir na interface ou no relatório."""
        linhas = ["Diagnóstico inicial: " + self.diagnostico_inicial.texto()]
        if not self.passos:
            linhas.append(self.mensagem)
        for i, p in enumerate(self.passos, 1):
            linhas.append(f"{i}. {p.rotulo}: {p.justificativa}")
        linhas.append("Diagnóstico final: " + self.diagnostico_final.texto())
        return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Registro de operações (permite reaplicar a partir do histórico)
# ---------------------------------------------------------------------------
REGISTRO_OPERACOES = {
    "filtro_mediana": filtro_mediana_3x3,
    "correcao_gama": correcao_gama,
    "alongamento_contraste": alongamento_contraste,
    "equalizar_histograma": equalizar_histograma,
}


# ---------------------------------------------------------------------------
# Melhoria automática
# ---------------------------------------------------------------------------
def melhorar_automaticamente(imagem: np.ndarray,
                             limiares: Optional[LimiaresDiagnostico] = None,
                             corrigir_ruido: bool = True,
                             corrigir_brilho: bool = True,
                             corrigir_contraste: bool = True) -> ResultadoMelhoria:
    """Aplica as regras de melhoria e devolve imagem final + passos + justificativas.

    Uso típico na interface:
        res = melhorar_automaticamente(img)
        for passo in res.passos:
            historico.empilhar(passo.rotulo, passo.imagem)   # desfazer passo a passo
        mostrar(res.imagem); mostrar_texto(res.justificativa())
    """
    lim = limiares or LimiaresDiagnostico()
    atual = _validar_imagem(imagem).copy()
    diag_inicial = diagnosticar(atual, lim)
    diag = diag_inicial
    passos: list[PassoMelhoria] = []

    def _registrar(nome: str, rotulo: str, params: dict, justificativa: str):
        nonlocal atual, diag
        atual = REGISTRO_OPERACOES[nome](atual, **params)
        diag = diagnosticar(atual, lim)
        passos.append(PassoMelhoria(nome, rotulo, params, justificativa, atual.copy(), diag))

    # --- Regra 1: ruído ---------------------------------------------------
    if corrigir_ruido and "ruidosa" in diag.problemas:
        motivo = []
        if diag.ruido_impulsivo > lim.ruido_impulsivo:
            motivo.append(f"{100 * diag.ruido_impulsivo:.2f}% dos pixels são impulsos isolados "
                          f"(limite {100 * lim.ruido_impulsivo:.1f}%)")
        if diag.sigma_ruido > lim.sigma_ruido:
            motivo.append(f"variação local estimada σ = {diag.sigma_ruido:.1f} "
                          f"(limite {lim.sigma_ruido:.1f})")
        _registrar("filtro_mediana", "Mediana 3×3", {"tamanho": 3},
                   "Ruído detectado: " + "; ".join(motivo) +
                   ". A mediana remove impulsos preservando bordas; foi aplicada antes "
                   "dos ajustes de brilho/contraste para não amplificar o ruído.")

    # --- Regra 2: brilho (até 2 iterações, pois o γ é uma aproximação) -----
    if corrigir_brilho:
        for _ in range(2):
            if not ({"muito escura", "muito clara"} & set(diag.problemas)):
                break
            if len(passos) >= lim.max_passos:
                break
            gama = round(gama_para_alvo(diag.brilho, lim.brilho_alvo), 3)
            if abs(gama - 1.0) < 0.02:
                break
            estado = "escura" if diag.brilho < lim.brilho_escuro else "clara"
            acao = "clarear (γ < 1)" if gama < 1 else "escurecer (γ > 1)"
            _registrar("correcao_gama", f"Correção gama γ={gama:.2f}", {"gama": gama},
                       f"Imagem {estado}: brilho médio {diag.brilho:.1f} fora da faixa "
                       f"[{lim.brilho_escuro:.0f}, {lim.brilho_claro:.0f}]. "
                       f"γ calculado para {acao} até o alvo ≈ {lim.brilho_alvo:.0f}.")

    # --- Regra 3: contraste ------------------------------------------------
    if corrigir_contraste and "baixo contraste" in diag.problemas and len(passos) < lim.max_passos:
        desvio_antes = diag.desvio
        _registrar("alongamento_contraste", "Alongamento de contraste (1%–99%)",
                   {"percentil_inf": 1.0, "percentil_sup": 99.0},
                   f"Baixo contraste: desvio padrão {desvio_antes:.1f} < {lim.desvio_baixo:.0f} "
                   f"(faixa ocupada {diag.minimo}–{diag.maximo}). "
                   "O alongamento estica a faixa útil para 0–255.")
        if "baixo contraste" in diag.problemas and len(passos) < lim.max_passos:
            _registrar("equalizar_histograma", "Equalização de histograma (luminância)",
                       {"espaco": "ycbcr"},
                       f"Após o alongamento o desvio ainda é {diag.desvio:.1f} "
                       f"(< {lim.desvio_baixo:.0f}); a equalização redistribui os níveis "
                       "para um histograma mais uniforme, sem distorcer as cores.")

    return ResultadoMelhoria(imagem=atual, diagnostico_inicial=diag_inicial,
                             diagnostico_final=diag, passos=passos)


def melhorar(imagem: np.ndarray, **parametros) -> np.ndarray:
    """Versão no padrão do contrato: recebe array e devolve array."""
    return melhorar_automaticamente(imagem, **parametros).imagem