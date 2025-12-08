"""
Aplicação Streamlit para diagnóstico de doenças em plantas
Baseado no artigo de Katafuchi e Tokunaga (2020)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Importar módulos customizados
from inference import create_inference_engine
from metrics import calculate_all_metrics, leaf_mask_from_rgb, de2000_map

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico de Plantas",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache do modelo
@st.cache_resource
def load_model():
    """Carrega o modelo uma única vez"""
    weights_path = Path(__file__).parent / "weights" / "latest_net_G.pth"
    return create_inference_engine(str(weights_path), device='cpu')


def create_visualization(original, gray, reconstructed, de_map, mask):
    """
    Cria visualização com 5 imagens lado a lado
    
    Args:
        original: imagem original RGB
        gray: imagem em escala de cinza
        reconstructed: imagem reconstruída
        de_map: mapa de diferenças CIEDE2000
        mask: máscara da folha
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Entrada (escala de cinza)
    axes[0].imshow(gray)
    axes[0].set_title("Entrada (Escala de Cinza)", fontsize=12)
    axes[0].axis('off')
    
    # Original
    axes[1].imshow(original)
    axes[1].set_title("Original", fontsize=12)
    axes[1].axis('off')
    
    # Reconstruída
    axes[2].imshow(reconstructed)
    axes[2].set_title("Reconstruída", fontsize=12)
    axes[2].axis('off')
    
    # Mapa de calor ΔE2000
    # Normalizar para visualização
    de_normalized = de_map.copy()
    de_normalized[~mask] = 0  # Zerar fundo
    
    im = axes[3].imshow(de_normalized, cmap='hot')
    axes[3].set_title("Mapa de Erro (ΔE2000)", fontsize=12)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    # Grad-CAM style overlay (mapa de erro sobre imagem original)
    axes[4].imshow(original)
    axes[4].imshow(de_normalized, cmap='jet', alpha=0.5)
    axes[4].set_title("Grad-CAM Style\n(Erro Sobreposto)", fontsize=12)
    axes[4].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    # Cabeçalho
    st.markdown('<h1 class="main-header">🌿 Diagnóstico de Doenças em Plantas</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Análise baseada em reconstrução de cores com Pix2Pix</p>', 
                unsafe_allow_html=True)
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Sobre")
        st.markdown("""
        Este sistema utiliza um modelo **Pix2Pix** treinado para analisar 
        folhas de plantas através da reconstrução de cores.
        
        **Como funciona:**
        1. A imagem colorida é convertida para escala de cinza
        2. O modelo reconstrói a versão colorida
        3. Diferenças entre original e reconstrução são calculadas
        4. Métricas CIEDE2000 e HSL quantificam as diferenças
        
        **Baseado em:**
        - Katafuchi & Tokunaga (2020)
        - Arquitetura Pix2Pix (Isola et al., 2017)
        """)
        
        st.header("📊 Métricas")
        st.markdown("""
        **CIEDE2000 Sum**: Diferença total de cor
        
        **HSL Error**: Erro em tonalidade e saturação
        
        **Top 2% Mean**: Média dos maiores erros
        
        **Top 1% Energy**: Concentração de erro
        """)
    
    # Área principal
    st.header("📤 Upload da Imagem")
    uploaded_file = st.file_uploader(
        "Selecione uma imagem de folha",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceitos: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Mostrar nome do arquivo
        st.info(f"📁 Arquivo carregado: **{uploaded_file.name}**")
        input_image = Image.open(uploaded_file)
        
        # Botão de análise
        if st.button("🔬 Analisar Imagem", type="primary", use_container_width=True):
            with st.spinner("Processando análise..."):
                try:
                    # Carregar modelo
                    model = load_model()
                    
                    # Realizar inferência
                    original, gray, reconstructed = model.reconstruct(input_image)
                    
                    # Gerar máscara
                    mask = leaf_mask_from_rgb(original, white_thr=240)
                    
                    # Calcular mapa de diferenças
                    de_map = de2000_map(original, reconstructed)
                    
                    # Calcular métricas
                    metrics = calculate_all_metrics(original, reconstructed, mask)
                    
                    # Exibir resultados
                    st.success("✅ Análise concluída!")
                    
                    # Visualizações
                    st.header("📊 Visualizações")
                    fig = create_visualization(original, gray, reconstructed, de_map, mask)
                    st.pyplot(fig)
                    
                    # Métricas detalhadas
                    st.header("📈 Métricas Calculadas")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "CIEDE2000 Sum",
                            f"{metrics['ciede2000_sum']:.2f}",
                            help="Soma total das diferenças de cor na região da folha"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "Top 2% Mean ΔE",
                            f"{metrics['top2pct_mean_deltaE']:.2f}",
                            help="Média dos 2% maiores valores de erro"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "HSL Error",
                            f"{metrics['hsl_error']:.4f}",
                            help="Erro combinado de Hue, Saturation e Lightness"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric(
                            "Top 1% Energy Fraction",
                            f"{metrics['top1pct_energy_fraction']:.4f}",
                            help="Fração de energia concentrada nos 1% maiores erros"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    

                    
                except Exception as e:
                    st.error(f"❌ Erro durante a análise: {str(e)}")
                    st.exception(e)
    else:
        # Instruções quando não há imagem
        st.info("👆 Faça upload de uma imagem de folha para começar a análise")
       


if __name__ == "__main__":
    main()
