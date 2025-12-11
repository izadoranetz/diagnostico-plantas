"""
Módulo para realizar inferência com o modelo pix2pix treinado
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple

from model_loader import load_pix2pix_generator


class Pix2PixInference:
    """Classe para realizar inferência com o modelo pix2pix"""
    
    def __init__(self, weights_path: str, device='cpu', image_size: int = 256):
        """
        Inicializa o sistema de inferência
        
        Args:
            weights_path: caminho para os pesos do gerador
            device: dispositivo ('cpu' ou 'cuda')
            image_size: tamanho da imagem (default: 256)
        """
        self.device = device
        self.image_size = image_size
        # Modelo aceita 1 canal (grayscale) e produz 3 canais (RGB)
        self.model = load_pix2pix_generator(
            weights_path, 
            device, 
            in_channels=1, 
            out_channels=3
        )
        
    def preprocess_image(self, pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Preprocessa a imagem para inferência (matching notebook preprocessing)
        
        Args:
            pil_img: imagem PIL em RGB
            
        Returns:
            tensor para o modelo (grayscale 1-channel), imagem original RGB, imagem em escala de cinza
        """
        # Redimensionar e converter para RGB
        img_rgb = pil_img.convert("RGB").resize(
            (self.image_size, self.image_size), 
            Image.BICUBIC
        )
        
        # Converter para escala de cinza (1 canal para o modelo)
        img_gray = img_rgb.convert("L")
        
        # Converter para numpy
        img_rgb_np = np.asarray(img_rgb).astype(np.uint8)
        img_gray_np = np.asarray(img_gray).astype(np.uint8)
        
        # Preparar tensor para o modelo (1 canal)
        # Normalizar para [-1, 1] como no treinamento
        tensor = np.asarray(img_gray).astype(np.float32) / 255.0
        tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        tensor = (tensor * 2.0) - 1.0
        tensor = tensor.to(self.device)
        
        return tensor, img_rgb_np, img_gray_np
    
    @torch.no_grad()
    def reconstruct(self, pil_img: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
        """
        Reconstrói a imagem colorida a partir da entrada em escala de cinza
        
        Args:
            pil_img: imagem PIL em RGB
            
        Returns:
            tupla com (imagem original, imagem cinza, imagem reconstruída, tensor de entrada) 
            primeiros 3 em RGB uint8, último é tensor para Grad-CAM
        """
        # Preprocessar
        tensor, img_rgb_np, img_gray_np = self.preprocess_image(pil_img)
        
        # Inferência
        fake_tensor = self.model(tensor)
        
        # Desnormalizar de [-1, 1] para [0, 1]
        fake_tensor = (fake_tensor + 1.0) / 2.0
        fake_tensor = torch.clamp(fake_tensor, 0.0, 1.0)
        
        # Converter para numpy uint8
        fake_np = fake_tensor[0].cpu().permute(1, 2, 0).numpy()
        fake_np = (fake_np * 255).astype(np.uint8)
        
        return img_rgb_np, img_gray_np, fake_np, tensor


def create_inference_engine(weights_path: str = None, device: str = 'cpu') -> Pix2PixInference:
    """
    Cria uma instância do motor de inferência
    
    Args:
        weights_path: caminho para os pesos (default: weights/latest_net_G.pth)
        device: dispositivo ('cpu' ou 'cuda')
        
    Returns:
        instância configurada de Pix2PixInference
    """
    if weights_path is None:
        weights_path = Path(__file__).parent / "weights" / "latest_net_G.pth"
    
    return Pix2PixInference(str(weights_path), device=device)
