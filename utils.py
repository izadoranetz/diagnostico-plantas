"""
Funções utilitárias para processamento de imagens e cálculo de métricas
baseadas no notebook Diagnostico_Katafuchi_Tokunaga.ipynb
"""
import numpy as np
from PIL import Image
from pathlib import Path
from skimage.color import deltaE_ciede2000, rgb2lab
import cv2


IMG_SIZE = (256, 256)


def cria_par_ab(color_img: Image.Image) -> Image.Image:
    """
    Cria um par AB a partir de uma imagem colorida.
    Canal A: escala de cinza
    Canal B: imagem colorida original
    """
    B = color_img.convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
    A = B.convert("L").convert("RGB")
    AB = Image.new("RGB", (IMG_SIZE[0] * 2, IMG_SIZE[1]))
    AB.paste(A, (0, 0))
    AB.paste(B, (IMG_SIZE[0], 0))
    return AB


def prepare_image_for_inference(image: Image.Image) -> tuple:
    """
    Prepara uma imagem para inferência.
    Retorna: (imagem_grayscale, imagem_colorida_redimensionada)
    """
    img_rgb = image.convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
    img_gray = img_rgb.convert("L").convert("RGB")
    return img_gray, img_rgb


def load_rgb(image: Image.Image) -> np.ndarray:
    """
    Carrega imagem RGB como numpy array uint8 (H, W, 3).
    """
    img_array = np.asarray(image.convert("RGB"))
    return img_array


def resize_to(img: np.ndarray, target_hw: tuple) -> np.ndarray:
    """
    Redimensiona img para (H, W) do target.
    """
    H, W = target_hw
    img_pil = Image.fromarray(img)
    resized = img_pil.resize((W, H), Image.BICUBIC)
    return np.asarray(resized)


def leaf_mask_from_rgb(rgb: np.ndarray, white_thr: int = 240) -> np.ndarray:
    """
    Cria máscara para remover fundo branco/claro.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r < white_thr) | (g < white_thr) | (b < white_thr)
    return mask.astype(np.uint8)


def de2000_map(real_rgb_u8: np.ndarray, fake_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Calcula o mapa de diferença de cor CIEDE2000 entre duas imagens RGB.
    """
    real = real_rgb_u8.astype(np.float32) / 255.0
    fake = fake_rgb_u8.astype(np.float32) / 255.0
    return deltaE_ciede2000(rgb2lab(real), rgb2lab(fake))


def metric_top_p_mean(de_map: np.ndarray, leaf_mask: np.ndarray, top_p: float = 0.02) -> float:
    """
    Calcula a média dos top P% maiores valores de DeltaE2000 na máscara.
    Útil para detecção de anomalias.
    """
    vals = de_map[leaf_mask > 0]
    if vals.size == 0:
        return 0.0
    k = max(1, int(np.ceil(top_p * vals.size)))
    topk = np.partition(vals, -k)[-k:]
    return float(topk.mean())


def metric_concentration_top_q_energy(de_map: np.ndarray, leaf_mask: np.ndarray, top_q: float = 0.01) -> float:
    """
    Calcula a fração de energia concentrada nos top Q% maiores valores.
    Proxy para localização da anomalia.
    """
    vals = de_map[leaf_mask > 0]
    if vals.size == 0:
        return 0.0
    k = max(1, int(np.ceil(top_q * vals.size)))
    topk = np.partition(vals, -k)[-k:]
    denom = float(vals.sum()) + 1e-12
    return float(topk.sum() / denom)


def calculate_hsl_error_pixelwise(img_real: np.ndarray, img_fake: np.ndarray, mask=None):
    """
    Calcula erro de cor no espaço HSV (melhor que HLS para plantas).

    Foco:
    - H (Hue): Mudança de cor (verde -> amarelo/marrom = doença)
    - S (Saturation): Perda de saturação (planta murcha)
    - V (Value): Escurecimento (necrose)

    Returns:
        float: Score de erro normalizado [0, 1]
               Valores altos = maior anomalia
    """
    # Converter RGB para HSV
    real_hsv = cv2.cvtColor(img_real, cv2.COLOR_RGB2HSV).astype(np.float32)
    fake_hsv = cv2.cvtColor(img_fake, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Separar canais
    h_real, s_real, v_real = cv2.split(real_hsv)
    h_fake, s_fake, v_fake = cv2.split(fake_hsv)

    # Tratamento especial do Hue (circular)
    diff_h = np.abs(h_real - h_fake)
    diff_h = np.minimum(diff_h, 180 - diff_h)  # Distância circular

    # Diferenças lineares
    diff_s = np.abs(s_real - s_fake)
    diff_v = np.abs(v_real - v_fake)

    # Aplicar máscara
    if mask is not None:
        mask_bin = (mask > 0).astype(np.float32)
        diff_h *= mask_bin
        diff_s *= mask_bin
        diff_v *= mask_bin
        num_pixels = np.sum(mask_bin)
    else:
        num_pixels = img_real.shape[0] * img_real.shape[1]

    if num_pixels == 0:
        return 0.0

    # Pesos baseados em importância para detecção de doenças
    weight_h = 0.5
    weight_s = 0.35
    weight_v = 0.15

    # Normalizar cada componente para [0, 1]
    score_h = np.sum(diff_h) / (num_pixels * 90.0)   # Hue máx = 90 (metade de 180)
    score_s = np.sum(diff_s) / (num_pixels * 255.0)  # Saturation máx = 255
    score_v = np.sum(diff_v) / (num_pixels * 255.0)  # Value máx = 255

    # Score final ponderado
    total_score = (weight_h * score_h +
                   weight_s * score_s +
                   weight_v * score_v)

    return total_score

