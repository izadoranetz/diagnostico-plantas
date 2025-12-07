"""
Funções para carregar e usar o modelo pix2pix para reconstrução de cor.
Baseado no framework pytorch-CycleGAN-and-pix2pix.
"""
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys
import os

# Importar classes necessárias do pytorch-CycleGAN-and-pix2pix
networks = None
create_model = None

# Tentar importar do repositório pytorch-CycleGAN-and-pix2pix
try:
    # Primeiro, tentar adicionar o caminho do repositório ao sys.path
    repo_path = Path(__file__).parent / "pytorch-CycleGAN-and-pix2pix"
    if repo_path.exists():
        sys.path.insert(0, str(repo_path))
    
    from models import networks
    from models import create_model
except ImportError:
    # Tentar importar se estiver no PYTHONPATH
    try:
        from models import networks
        from models import create_model
    except ImportError:
        # Se não conseguir importar, continuar sem as classes
        # A função load_pix2pix_model terá um fallback
        pass


def load_pix2pix_model(checkpoint_dir: str, model_name: str = "ramularia_colorrec_pix2pix", 
                       epoch: str = "latest", device: str = None):
    """
    Carrega o modelo pix2pix treinado.
    
    Args:
        checkpoint_dir: Diretório onde estão os checkpoints
        model_name: Nome do modelo
        epoch: Época a carregar ('latest' ou número)
        device: Device para carregar o modelo ('cuda' ou 'cpu')
    
    Returns:
        Modelo carregado e opções do modelo
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_path = Path(checkpoint_dir) / model_name
    
    # Tentar carregar usando a função create_model do framework
    if create_model is not None:
        try:
            opt = {
                'model': 'pix2pix',
                'netG': 'unet_256',
                'input_nc': 3,
                'output_nc': 3,
                'ngf': 64,
                'norm': 'batch',
                'no_dropout': False,
                'checkpoints_dir': checkpoint_dir,
                'name': model_name,
                'gpu_ids': [0] if device == 'cuda' else [],
                'isTrain': False
            }
            
            model = create_model(opt)
            model.setup(opt)
            if epoch == 'latest':
                model.load_networks(epoch)
            else:
                model.load_networks(int(epoch))
            
            return model, opt
        except Exception as e:
            print(f"Erro ao carregar modelo usando create_model: {e}")
    
    # Fallback: carregar diretamente o gerador usando networks
    if networks is None:
        raise ImportError(
            "Não foi possível importar o módulo networks. "
            "Certifique-se de que o pytorch-CycleGAN-and-pix2pix está disponível.\n"
            "Você pode clonar o repositório em:\n"
            "git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git\n"
            "E garantir que está no mesmo diretório ou no PYTHONPATH."
        )
    
    # Criar gerador
    generator = networks.define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG='unet_256',
        norm='batch',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=[0] if device == 'cuda' else []
    )
    
    # Carregar pesos
    checkpoint_path_obj = Path(checkpoint_dir) / model_name
    
    if epoch == 'latest':
        # Procurar o último checkpoint
        latest_file = checkpoint_path_obj / "latest_net_G.pth"
        if latest_file.exists():
            checkpoint_file = latest_file
        else:
            # Procurar todos os checkpoints e pegar o mais recente
            checkpoints = list(checkpoint_path_obj.glob("*_net_G.pth"))
            if checkpoints:
                checkpoint_file = max(checkpoints, key=lambda x: x.stat().st_mtime)
            else:
                raise FileNotFoundError(
                    f"Nenhum checkpoint encontrado em {checkpoint_path_obj}\n"
                    f"Certifique-se de que o modelo foi treinado e os checkpoints foram salvos."
                )
    else:
        checkpoint_file = checkpoint_path_obj / f"{epoch}_net_G.pth"
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_file}\n"
            f"Verifique o caminho do checkpoint."
        )
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    generator.to(device)
    
    return generator, {'device': device, 'checkpoint_path': str(checkpoint_file)}


def preprocess_image_for_model(image: Image.Image, device: str = None):
    """
    Prepara uma imagem para inferência no modelo pix2pix.
    
    Args:
        image: PIL Image em escala de cinza ou RGB
        device: Device para o tensor
    
    Returns:
        Tensor pronto para o modelo
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Garantir que é RGB (3 canais)
    if image.mode != 'RGB':
        img_rgb = image.convert('RGB')
    else:
        img_rgb = image
    
    # Redimensionar para 256x256
    img_rgb = img_rgb.resize((256, 256), Image.BICUBIC)
    
    # Converter para tensor e normalizar de [0, 255] para [-1, 1]
    img_array = np.array(img_rgb).astype(np.float32)
    img_array = img_array / 127.5 - 1.0  # Normalizar para [-1, 1]
    
    # Converter de (H, W, C) para (C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    return img_tensor


def postprocess_model_output(tensor: torch.Tensor):
    """
    Converte a saída do modelo de volta para imagem PIL.
    
    Args:
        tensor: Tensor de saída do modelo (1, 3, H, W) em [-1, 1]
    
    Returns:
        PIL Image RGB
    """
    # Converter de (1, C, H, W) para (H, W, C)
    img_array = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Desnormalizar de [-1, 1] para [0, 255]
    img_array = (img_array + 1.0) * 127.5
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img_array, mode='RGB')


def inference_colorization(model, image: Image.Image, device: str = None):
    """
    Executa inferência para colorizar uma imagem em escala de cinza.
    
    Args:
        model: Modelo pix2pix (pode ser objeto model ou generator direto)
        image: PIL Image em escala de cinza
        device: Device para processamento
    
    Returns:
        PIL Image RGB reconstruída
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocessar
    input_tensor = preprocess_image_for_model(image, device)
    
    # Executar inferência
    with torch.no_grad():
        # Se for um objeto model do framework
        if hasattr(model, 'netG'):
            fake = model.netG(input_tensor)
        # Se for o generator diretamente
        elif isinstance(model, nn.Module):
            fake = model(input_tensor)
        else:
            raise ValueError("Modelo não reconhecido")
    
    # Postprocessar
    output_image = postprocess_model_output(fake)
    
    return output_image

