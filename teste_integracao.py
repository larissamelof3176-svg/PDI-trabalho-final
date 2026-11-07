import numpy as np
from processamento.automatico import *   # só testa se os imports funcionam
from processamento.filtros import filtro_mediana
from processamento.bordas import sobel
from processamento.equalizacao import equalizar_histograma

img = np.random.default_rng(0).integers(0, 256, (100, 100), dtype=np.uint8)
print(equalizar_histograma(img).shape, sobel(
    img).shape, filtro_mediana(img).shape)
