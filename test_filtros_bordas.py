import numpy as np
import pytest
from processamento.filtros import (filtro_media, filtro_mediana, nitidez,
                                   ruido_sal_pimenta, mse, psnr)
from processamento.bordas import sobel


@pytest.fixture
def limpa():
    x = np.tile(np.linspace(50, 200, 64), (64, 1))
    return x.astype(np.uint8)


def test_media_imagem_constante():
    img = np.full((20, 20), 100, np.uint8)
    for t in (3, 5, 7):
        assert np.array_equal(filtro_media(img, t), img)


def test_mediana_reduz_ruido(limpa):
    ruidosa = ruido_sal_pimenta(limpa, 0.05, semente=1)
    filtrada = filtro_mediana(ruidosa, 3)
    assert mse(filtrada, limpa) < mse(ruidosa, limpa)
    assert psnr(filtrada, limpa) > psnr(ruidosa, limpa)


def test_sobel_uniforme_zero():
    img = np.full((30, 30), 80, np.uint8)
    assert sobel(img).max() == 0


def test_sobel_detecta_borda():
    img = np.zeros((30, 30), np.uint8); img[:, 15:] = 255
    s = sobel(img)
    assert s[:, 14:16].max() == 255 and s[:, :10].max() == 0


def test_sobel_binario():
    img = np.zeros((30, 30), np.uint8); img[:, 15:] = 255
    assert set(np.unique(sobel(img, limiar=100))) <= {0, 255}


def test_colorida_e_dtype():
    img = np.random.default_rng(0).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    for out in (filtro_media(img), filtro_mediana(img), nitidez(img)):
        assert out.shape == img.shape and out.dtype == np.uint8


def test_opencv_se_disponivel(limpa):
    cv2 = pytest.importorskip("cv2")
    ruidosa = ruido_sal_pimenta(limpa, 0.05, semente=2)
    ref = cv2.medianBlur(ruidosa, 3)
    assert np.abs(filtro_mediana(ruidosa, 3).astype(int) - ref).max() <= 1
