"""P5: gera o banco de imagens de teste, cronometra filtros e escreve o relatório.

Uso (na raiz do projeto):
    python gerar_banco_e_relatorio.py imagens/minha_foto.jpg
Sem argumento, usa uma imagem sintética.
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

from processamento.filtros import (filtro_media, filtro_mediana, nitidez,
                                   ruido_sal_pimenta, mse, psnr)
from processamento.bordas import sobel

IMG, RES = Path("imagens"), Path("resultados")
IMG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)


def carregar_base():
    if len(sys.argv) > 1:
        return np.array(Image.open(sys.argv[1]).convert("RGB"))
    y, x = np.mgrid[0:540, 0:960]
    base = 127 + 80 * np.sin(x / 40) * np.cos(y / 30)
    base[100:300, 200:500] = 230
    base[250:450, 600:900] = 40
    return np.stack([base, base * 0.9, base * 0.8], -1).clip(0, 255).astype(np.uint8)


def cinza(rgb):
    return np.rint(rgb @ [0.299, 0.587, 0.114]).astype(np.uint8)


def tempo(f, *a, **k):
    t = time.perf_counter(); f(*a, **k)
    return time.perf_counter() - t


base = carregar_base()
g = cinza(base)

# 1) Banco de imagens
banco = {
    "colorida": base,
    "cinza": g,
    "escura": (base * 0.3).astype(np.uint8),
    "clara": (255 - (255 - base) * 0.3).astype(np.uint8),
    "baixo_contraste": (110 + (base.astype(float) - 128) * 0.2).astype(np.uint8),
    "ruido_sal_pimenta": ruido_sal_pimenta(base, 0.05, semente=1),
}
for nome, im in banco.items():
    Image.fromarray(im).save(IMG / f"{nome}.png")

L = ["RELATÓRIO DE TESTES - P5 (filtros e bordas)", "=" * 50, ""]

# 2) Qualidade da mediana (cinza, com referência limpa)
L.append("1. Remoção de ruído sal e pimenta (5%) - imagem em cinza")
ruid = ruido_sal_pimenta(g, 0.05, semente=1)
L.append(f"   Sem filtro : MSE={mse(ruid, g):9.2f}  PSNR={psnr(ruid, g):6.2f} dB")
for nome, f, t in [("Média 3x3", filtro_media, 3), ("Mediana 3x3", filtro_mediana, 3),
                   ("Mediana 5x5", filtro_mediana, 5)]:
    r = f(ruid, tamanho=t)
    L.append(f"   {nome:<12}: MSE={mse(r, g):9.2f}  PSNR={psnr(r, g):6.2f} dB")
corrompidos = ruid != g
TOL = 10  # pixel conta como corrigido se ficar a até 10 níveis do original
rest = (np.abs(filtro_mediana(ruid, 3).astype(int) - g.astype(int)) > TOL) & corrompidos
removido = 100 * (1 - rest.sum() / corrompidos.sum())
L.append(f"   Pixels corrompidos corrigidos pela mediana 3x3 (tolerância {TOL}): {removido:.1f}% "
         f"(meta >= 80%: {'ATENDE' if removido >= 80 else 'NAO ATENDE'})")

# 3) Desempenho em Full HD
L += ["", "2. Desempenho em 1920x1080 (cinza) - meta < 5 s"]
hd = np.array(Image.fromarray(g).resize((1920, 1080)))
for nome, f, kw in [("Média 3x3", filtro_media, {"tamanho": 3}),
                    ("Média 7x7", filtro_media, {"tamanho": 7}),
                    ("Mediana 3x3", filtro_mediana, {"tamanho": 3}),
                    ("Mediana 5x5", filtro_mediana, {"tamanho": 5}),
                    ("Nitidez unsharp", nitidez, {}),
                    ("Sobel", sobel, {})]:
    s = tempo(f, hd, **kw)
    L.append(f"   {nome:<16}: {s:6.2f} s  [{'OK' if s < 5 else 'LENTO'}]")

# 4) Todas as operações em cor e cinza
L += ["", "3. Todas as operações em imagem colorida e em cinza (sem erro)"]
for nome, im in [("colorida", base), ("cinza", g)]:
    for fn, f in [("média", filtro_media), ("mediana", filtro_mediana),
                  ("nitidez", nitidez), ("sobel", sobel)]:
        out = f(im)
        L.append(f"   {nome:<9}{fn:<8}: OK shape={out.shape} dtype={out.dtype}")

# imagens de exemplo
Image.fromarray(ruid).save(RES / "ruidosa.png")
Image.fromarray(filtro_mediana(ruid, 3)).save(RES / "mediana.png")
Image.fromarray(sobel(g)).save(RES / "sobel.png")
Image.fromarray(nitidez(g)).save(RES / "nitidez.png")

(RES / "relatorio_testes.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
