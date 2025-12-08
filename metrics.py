"""
Módulo para cálculo de métricas de análise de imagens
Baseado no notebook Diagnostico_Katafuchi_Tokunaga
"""
import numpy as np
import cv2
from skimage.color import rgb2lab, deltaE_ciede2000
from typing import Tuple


def leaf_mask_from_rgb(rgb: np.ndarray, white_thr: int = 240) -> np.ndarray:
    """
    Gera uma máscara binária para identificar a região da folha (não-branco)
    
    Args:
        rgb: imagem RGB em formato uint8 (H, W, 3)
        white_thr: limiar para considerar pixel como branco/fundo
        
    Returns:
        máscara binária onde True = folha, False = fundo
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (r < white_thr) | (g < white_thr) | (b < white_thr)
    return mask


def de2000_map(real_rgb_u8: np.ndarray, fake_rgb_u8: np.ndarray) -> np.ndarray:
    """
    Calcula o mapa de diferença de cor CIEDE2000 entre duas imagens
    
    Args:
        real_rgb_u8: imagem original em RGB uint8
        fake_rgb_u8: imagem reconstruída em RGB uint8
        
    Returns:
        mapa de diferenças CIEDE2000
    """
    real = real_rgb_u8.astype(np.float32) / 255.0
    fake = fake_rgb_u8.astype(np.float32) / 255.0
    return deltaE_ciede2000(rgb2lab(real), rgb2lab(fake))


def metric_top_p_mean(de_map: np.ndarray, leaf_mask: np.ndarray, top_p: float = 0.02) -> float:
    """
    Calcula a média dos top p% maiores valores de erro
    
    Args:
        de_map: mapa de diferenças CIEDE2000
        leaf_mask: máscara da folha
        top_p: percentual dos maiores valores (default: 2%)
        
    Returns:
        média dos top p% valores
    """
    vals = de_map[leaf_mask]
    if vals.size == 0:
        return 0.0
    k = max(1, int(np.ceil(top_p * vals.size)))
    topk = np.partition(vals, -k)[-k:]
    return float(topk.mean())


def metric_concentration_top_q_energy(de_map: np.ndarray, leaf_mask: np.ndarray, 
                                      top_q: float = 0.01) -> float:
    """
    Calcula a fração de energia concentrada nos top q% maiores valores
    
    Args:
        de_map: mapa de diferenças CIEDE2000
        leaf_mask: máscara da folha
        top_q: percentual dos maiores valores (default: 1%)
        
    Returns:
        fração de energia (0 a 1)
    """
    vals = de_map[leaf_mask]
    if vals.size == 0:
        return 0.0
    k = max(1, int(np.ceil(top_q * vals.size)))
    topk = np.partition(vals, -k)[-k:]
    denom = float(vals.sum()) + 1e-12
    return float(topk.sum() / denom)


def calculate_hsl_error_pixelwise(img_real: np.ndarray, img_fake: np.ndarray, 
                                   mask: np.ndarray = None) -> float:
    """
    Calcula erro de cor no espaço HSV (Hue, Saturation, Value)
    
    Foco:
    - H (Hue): Mudança de cor (verde -> amarelo/marrom = doença)
    - S (Saturation): Perda de saturação (planta murcha)
    - V (Value): Escurecimento (necrose)
    
    Args:
        img_real: imagem original RGB uint8
        img_fake: imagem reconstruída RGB uint8
        mask: máscara opcional da folha
        
    Returns:
        score de erro normalizado [0, 1], valores altos = maior anomalia
    """
    # Converter RGB para HSV
    real_hsv = cv2.cvtColor(img_real, cv2.COLOR_RGB2HSV).astype(np.float32)
    fake_hsv = cv2.cvtColor(img_fake, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Separar canais
    h_real, s_real, v_real = cv2.split(real_hsv)
    h_fake, s_fake, v_fake = cv2.split(fake_hsv)
    
    # Tratamento especial do Hue (circular, varia de 0-180 no OpenCV)
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
    weight_h = 0.5    # Hue: mais importante
    weight_s = 0.35   # Saturation: média importância
    weight_v = 0.15   # Value: menor importância
    
    # Normalizar cada componente para [0, 1]
    score_h = np.sum(diff_h) / (num_pixels * 90.0)   # Hue máx = 90
    score_s = np.sum(diff_s) / (num_pixels * 255.0)  # Saturation máx = 255
    score_v = np.sum(diff_v) / (num_pixels * 255.0)  # Value máx = 255
    
    # Score final ponderado
    total_score = (weight_h * score_h + 
                   weight_s * score_s + 
                   weight_v * score_v)
    
    return total_score


def calculate_all_metrics(real_rgb: np.ndarray, fake_rgb: np.ndarray, 
                         mask: np.ndarray = None) -> dict:
    """
    Calcula todas as métricas de uma vez
    
    Args:
        real_rgb: imagem original RGB uint8
        fake_rgb: imagem reconstruída RGB uint8
        mask: máscara opcional da folha (se None, será gerada automaticamente)
        
    Returns:
        dicionário com todas as métricas
    """
    # Gerar máscara se não fornecida
    if mask is None:
        mask = leaf_mask_from_rgb(real_rgb, white_thr=240)
    
    # Calcular mapa de diferenças CIEDE2000
    de_map = de2000_map(real_rgb, fake_rgb)
    
    # Calcular métricas
    ciede_sum = float(np.sum(de_map[mask]))
    top2_mean = metric_top_p_mean(de_map, mask, top_p=0.02)
    top1_energy = metric_concentration_top_q_energy(de_map, mask, top_q=0.01)
    hsl_error = calculate_hsl_error_pixelwise(real_rgb, fake_rgb, mask.astype(np.uint8))
    
    return {
        "ciede2000_sum": round(ciede_sum, 2),
        "top2pct_mean_deltaE": round(top2_mean, 2),
        "top1pct_energy_fraction": round(top1_energy, 4),
        "hsl_error": round(hsl_error, 4)
    }
