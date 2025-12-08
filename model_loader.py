"""
Módulo para carregar o modelo pix2pix treinado
Baseado no repositório pytorch-CycleGAN-and-pix2pix
"""
import torch
import torch.nn as nn
from pathlib import Path


class UnetGenerator(nn.Module):
    """Generator com arquitetura U-Net para pix2pix"""
    
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        Args:
            input_nc: número de canais de entrada
            output_nc: número de canais de saída
            num_downs: número de downsampling na U-Net
            ngf: número de filtros no primeiro layer
            norm_layer: tipo de normalização
            use_dropout: se usa dropout
        """
        super(UnetGenerator, self).__init__()
        
        # Construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8, ngf * 8, input_nc=None, submodule=None, 
            norm_layer=norm_layer, innermost=True
        )
        
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                norm_layer=norm_layer, use_dropout=use_dropout
            )
        
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
            norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block,
            norm_layer=norm_layer
        )
        
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer
        )

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Define um bloco da U-Net com skip connection"""
    
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4,
            stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc,
                kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc,
                kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc,
                kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def load_pix2pix_generator(weights_path: str, device='cpu'):
    """
    Carrega o gerador pix2pix treinado
    
    Args:
        weights_path: caminho para o arquivo de pesos (latest_net_G.pth)
        device: dispositivo para carregar o modelo ('cpu' ou 'cuda')
    
    Returns:
        modelo carregado em modo de avaliação
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")
    
    # Criar o modelo com a mesma arquitetura do treinamento
    model = UnetGenerator(
        input_nc=3,
        output_nc=3,
        num_downs=8,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False
    )
    
    # Carregar os pesos
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Colocar em modo de avaliação
    model.eval()
    model.to(device)
    
    return model
