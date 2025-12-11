"""
Módulo para gerar visualizações Grad-CAM
Baseado no notebook diagnostico_plantas.ipynb
"""
import torch
import torch.nn as nn
import numpy as np
import cv2


class GradCAM:
    """
    Classe para gerar mapas de calor Grad-CAM
    Captura ativações e gradientes de uma camada específica
    """
    
    def __init__(self, model, target_layer=None):
        """
        Inicializa o Grad-CAM
        
        Args:
            model: modelo PyTorch
            target_layer: camada alvo (se None, usa última convolucional)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Se não especificado, encontrar última camada convolucional
        if target_layer is None:
            target_layer = self._find_last_conv_layer()
        
        self.target_layer = target_layer
    
    def _find_last_conv_layer(self):
        """Encontra a última camada convolucional do modelo"""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                last_conv = module
        return last_conv
    
    def _register_hooks(self):
        """Registra hooks para capturar ativações e gradientes"""
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach().clone()
        
        forward_h = self.target_layer.register_forward_hook(forward_hook)
        backward_h = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [forward_h, backward_h]
    
    def _remove_hooks(self):
        """Remove hooks registrados"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_heatmap(self, input_tensor, target_output=None):
        """
        Gera mapa de calor Grad-CAM
        
        Args:
            input_tensor: tensor de entrada (grayscale)
            target_output: tensor de saída alvo (imagem colorida original)
                          se None, usa o canal com maior ativação
        
        Returns:
            mapa de calor normalizado entre [0, 1]
        """
        # Limpar estado anterior
        self._remove_hooks()
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self._register_hooks()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Determinar alvo para backward
        if target_output is not None:
            # Usar diferença entre saída e alvo
            target = torch.mean((output - target_output) ** 2)
        else:
            # Usar canal com maior ativação média
            channel_means = output.mean(dim=[2, 3])
            target_channel = torch.argmax(channel_means[0]).item()
            target = output[:, target_channel].mean()
        
        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # Verificar se capturou gradientes e ativações
        if self.gradients is None or self.activations is None:
            print("⚠️ Grad-CAM: hooks não capturaram dados, usando fallback")
            return self._generate_fallback_heatmap()
        
        # Calcular pesos dos gradientes (Global Average Pooling)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Combinar pesos com ativações
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # Aplicar ReLU (manter apenas valores positivos)
        heatmap = torch.relu(heatmap)
        
        # Normalizar para [0, 1]
        min_val = heatmap.min()
        max_val = heatmap.max()
        
        if (max_val - min_val) > 1e-8:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            # Mapa plano, usar fallback
            heatmap = self._generate_fallback_heatmap()
        
        # Converter para numpy
        heatmap_np = heatmap.squeeze().cpu().detach().numpy()
        
        # Garantir que é 2D
        if heatmap_np.ndim == 0:
            heatmap_np = np.ones((32, 32)) * heatmap_np.item()
        elif heatmap_np.ndim == 1:
            size = int(np.sqrt(len(heatmap_np)))
            if size * size == len(heatmap_np):
                heatmap_np = heatmap_np.reshape(size, size)
            else:
                heatmap_np = np.ones((32, 32))
        elif heatmap_np.ndim > 2:
            heatmap_np = heatmap_np[0] if heatmap_np.shape[0] > 0 else heatmap_np[0, 0]
        
        # Aplicar contraste (opcional)
        if heatmap_np.max() - heatmap_np.min() < 0.3:
            heatmap_np = np.clip(heatmap_np * 1.5, 0, 1)
        
        # Remover hooks
        self._remove_hooks()
        
        return heatmap_np
    
    def _generate_fallback_heatmap(self):
        """Gera mapa de calor fallback (gradiente radial)"""
        h, w = 32, 32
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        heatmap = np.exp(-dist / (max(h, w) / 4))
        return heatmap
    
    def overlay_heatmap_on_image(self, heatmap, image_rgb, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Sobrepõe mapa de calor na imagem original
        
        Args:
            heatmap: mapa de calor 2D normalizado [0, 1]
            image_rgb: imagem RGB uint8
            alpha: transparência do overlay (0-1)
            colormap: colormap OpenCV
        
        Returns:
            imagem com overlay RGB uint8
        """
        h, w = image_rgb.shape[:2]
        
        # Redimensionar heatmap para o tamanho da imagem
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Aplicar colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Converter imagem para uint8 se necessário
        if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
            image_uint8 = (image_rgb * 255).astype(np.uint8)
        else:
            image_uint8 = image_rgb
        
        # Misturar imagens
        overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


def generate_simple_heatmap_with_contrast(model, image_tensor):
    """
    Gera mapa de calor simples baseado em ativações (sem gradientes)
    Função auxiliar do notebook: gerar_mapa_simples_com_contraste
    
    Args:
        model: modelo PyTorch
        image_tensor: tensor de entrada
    
    Returns:
        mapa de calor 2D normalizado
    """
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Encontrar última camada convolucional
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        # Fallback: gradiente radial
        h, w = 32, 32
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return np.exp(-dist / (max(h, w) / 3))
    
    # Registrar hook
    handle = last_conv.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)
    
    handle.remove()
    
    if not activations:
        # Fallback
        h, w = 32, 32
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return np.exp(-dist / (max(h, w) / 3))
    
    # Usar média das ativações como mapa de importância
    heatmap = activations[0].mean(dim=1, keepdim=True)
    heatmap_np = heatmap.squeeze().cpu().numpy()
    
    # Garantir 2D
    if heatmap_np.ndim == 0:
        heatmap_np = np.full((32, 32), heatmap_np.item())
    elif heatmap_np.ndim == 1:
        size = int(np.sqrt(len(heatmap_np)))
        if size * size == len(heatmap_np):
            heatmap_np = heatmap_np.reshape(size, size)
        else:
            heatmap_np = np.full((32, 32), np.mean(heatmap_np))
    elif heatmap_np.ndim > 2:
        heatmap_np = heatmap_np.squeeze()
        if heatmap_np.ndim != 2:
            heatmap_np = np.full((32, 32), np.mean(heatmap_np))
    
    # Normalizar
    min_val = heatmap_np.min()
    max_val = heatmap_np.max()
    if (max_val - min_val) < 1e-8:
        heatmap_np = np.zeros_like(heatmap_np)
    else:
        heatmap_np = (heatmap_np - min_val) / (max_val - min_val)
    
    # Aplicar contraste
    heatmap_np = np.clip((heatmap_np - 0.5) * 2.0 + 0.5, 0, 1)
    
    return heatmap_np
