"""
Módulo para carregar o modelo pix2pix treinado
Baseado no notebook diagnostico_plantas.ipynb
"""
import torch
import torch.nn as nn
from pathlib import Path


class UNetDown(nn.Module):
    """Bloco de downsampling da U-Net"""
    
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Bloco de upsampling da U-Net com skip connection"""
    
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], dim=1)
        return x


class GeneratorUNet(nn.Module):
    """
    Generator U-Net para Pix2Pix
    Arquitetura do notebook: 8 camadas down, 7 camadas up + final
    """
    
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorUNet, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = UNetDown(in_channels, 64, normalize=False)    # 256x256 -> 128x128
        self.down2 = UNetDown(64, 128)                              # 128x128 -> 64x64
        self.down3 = UNetDown(128, 256)                             # 64x64 -> 32x32
        self.down4 = UNetDown(256, 512)                             # 32x32 -> 16x16
        self.down5 = UNetDown(512, 512)                             # 16x16 -> 8x8
        self.down6 = UNetDown(512, 512)                             # 8x8 -> 4x4
        self.down7 = UNetDown(512, 512)                             # 4x4 -> 2x2
        self.down8 = UNetDown(512, 512, normalize=False)            # 2x2 -> 1x1
        
        # Decoder (upsampling)
        self.up1 = UNetUp(512, 512, dropout=0.5)                    # 1x1 -> 2x2
        self.up2 = UNetUp(1024, 512, dropout=0.5)                   # 2x2 -> 4x4
        self.up3 = UNetUp(1024, 512, dropout=0.5)                   # 4x4 -> 8x8
        self.up4 = UNetUp(1024, 512)                                # 8x8 -> 16x16
        self.up5 = UNetUp(1024, 256)                                # 16x16 -> 32x32
        self.up6 = UNetUp(512, 128)                                 # 32x32 -> 64x64
        self.up7 = UNetUp(256, 64)                                  # 64x64 -> 128x128
        
        # Camada final
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),         # 128x128 -> 256x256
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder com skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder com skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


def load_pix2pix_generator(weights_path: str, device='cpu', in_channels=1, out_channels=3):
    """
    Carrega o generator do pix2pix a partir do arquivo de pesos
    
    Args:
        weights_path: caminho para o arquivo .pth com os pesos
        device: dispositivo ('cpu' ou 'cuda')
        in_channels: canais de entrada (1 para grayscale)
        out_channels: canais de saída (3 para RGB)
        
    Returns:
        modelo carregado em modo eval
    """
    # Criar modelo
    model = GeneratorUNet(in_channels=in_channels, out_channels=out_channels)
    
    # Carregar pesos
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")
    
    # Carregar state_dict diretamente (formato do modelo_final.pth)
    state_dict = torch.load(weights_path, map_location=device)
    
    # Se for um dict com chaves, tentar encontrar o state_dict
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Carregar pesos no modelo
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

