"""
Aplicação Streamlit para diagnóstico de doenças em plantas
Baseado no artigo de Katafuchi e Tokunaga (2020)
"""
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import io
import sys

# Importar módulos customizados
from utils import (
    prepare_image_for_inference,
    load_rgb,
    resize_to,
    leaf_mask_from_rgb,
    de2000_map,
    metric_top_p_mean,
    metric_concentration_top_q_energy,
    calculate_hsl_error_pixelwise,
    IMG_SIZE
)

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
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .healthy {
        background-color: #C8E6C9;
        color: #2E7D32;
    }
    .diseased {
        background-color: #FFCDD2;
        color: #C62828;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(checkpoint_path: str = None):
    """
    Carrega o modelo pix2pix com cache.
    """
    try:
        from model_utils import load_pix2pix_model, inference_colorization
        
        if checkpoint_path and Path(checkpoint_path).exists():
            model, opt = load_pix2pix_model(checkpoint_path)
            return model, opt, inference_colorization
        else:
            return None, None, None
    except Exception as e:
        st.warning(f"Erro ao carregar modelo: {e}")
        return None, None, None


def perform_analysis(original_image: Image.Image, reconstructed_image: Image.Image):
    """
    Realiza todas as análises na imagem comparando original com reconstruída.
    """
    # Preparar imagens
    orig_array = load_rgb(original_image)
    fake_array = load_rgb(reconstructed_image)
    
    # Redimensionar para mesmo tamanho se necessário
    H, W = fake_array.shape[:2]
    orig_resized = resize_to(orig_array, (H, W))
    
    # Criar máscara da folha
    leaf_mask = leaf_mask_from_rgb(orig_resized, white_thr=240)
    
    # Calcular mapa CIEDE2000
    de_map = de2000_map(orig_resized, fake_array)
    
    # Calcular métricas
    ciede_sum = float(np.sum(de_map[leaf_mask > 0]))
    score_detect = metric_top_p_mean(de_map, leaf_mask, top_p=0.02)
    score_loc = metric_concentration_top_q_energy(de_map, leaf_mask, top_q=0.01)
    score_hsl = calculate_hsl_error_pixelwise(orig_resized, fake_array, leaf_mask)
    
    # Calcular média DeltaE2000
    de_mean = float(np.mean(de_map))
    
    return {
        'ciede_sum': ciede_sum,
        'ciede_mean': de_mean,
        'top2pct_mean': score_detect,
        'top1pct_energy': score_loc,
        'hsl_error': score_hsl,
        'de_map': de_map,
        'leaf_mask': leaf_mask,
        'original_resized': orig_resized,
        'reconstructed': fake_array
    }


def classify_disease(metrics: dict, threshold_ciede: float = 350000, threshold_hsl: float = 0.15):
    """
    Classifica se a planta está doente baseado nas métricas.
    """
    diagnosis_a = "DOENTE" if metrics['ciede_sum'] > threshold_ciede else "SAUDÁVEL"
    diagnosis_b = "DOENTE" if metrics['hsl_error'] > threshold_hsl else "SAUDÁVEL"
    
    # Diagnóstico combinado (maioria dos votos)
    votes_doente = sum([
        metrics['ciede_sum'] > threshold_ciede,
        metrics['hsl_error'] > threshold_hsl,
        metrics['top2pct_mean'] > 20.0  # Limiar adicional baseado na experiência
    ])
    
    diagnosis_final = "DOENTE" if votes_doente >= 2 else "SAUDÁVEL"
    
    return {
        'A': diagnosis_a,
        'B': diagnosis_b,
        'FINAL': diagnosis_final,
        'confidence': abs(metrics['ciede_sum'] - threshold_ciede) / threshold_ciede if threshold_ciede > 0 else 0
    }


def create_visualizations(metrics: dict, gray_image: Image.Image):
    """
    Cria visualizações das análises.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 1. Imagem em escala de cinza (entrada)
    axes[0].imshow(np.asarray(gray_image.convert("L")), cmap="gray")
    axes[0].set_title("Entrada (Escala de Cinza)", fontsize=10)
    axes[0].axis("off")
    
    # 2. Imagem original
    axes[1].imshow(metrics['original_resized'])
    axes[1].set_title("Original (Real)", fontsize=10)
    axes[1].axis("off")
    
    # 3. Imagem reconstruída
    axes[2].imshow(metrics['reconstructed'])
    axes[2].set_title("Reconstruída (Fake)", fontsize=10)
    axes[2].axis("off")
    
    # 4. Mapa de erro CIEDE2000
    de_map_normalized = metrics['de_map'] / (np.percentile(metrics['de_map'], 99) + 1e-6)
    de_map_normalized = np.clip(de_map_normalized, 0, 1)
    im = axes[3].imshow(metrics['de_map'], cmap='hot')
    axes[3].set_title(f"ΔE2000 (Score: {metrics['ciede_sum']:.0f})", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    # 5. Sobreposição do mapa de erro na imagem original
    axes[4].imshow(metrics['original_resized'])
    heatmap_overlay = axes[4].imshow(metrics['de_map'], cmap='jet', alpha=0.5)
    axes[4].set_title("Mapa de Anomalia", fontsize=10)
    axes[4].axis("off")
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Diagnóstico de Doenças em Plantas</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detecção de anomalias usando pix2pix e análise CIEDE2000</p>', 
                unsafe_allow_html=True)
    
    # Sidebar para configurações
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Opção de checkpoint do modelo
        checkpoint_path = st.text_input(
            "Caminho do Checkpoint do Modelo",
            value="checkpoints/ramularia_colorrec_pix2pix",
            help="Caminho para o diretório do checkpoint do modelo pix2pix treinado"
        )
        
        # Limiares ajustáveis
        st.subheader("📊 Limiares de Classificação")
        threshold_ciede = st.slider(
            "Limiar CIEDE2000",
            min_value=0,
            max_value=1000000,
            value=350000,
            step=10000,
            help="Score acima deste valor indica doença"
        )
        
        threshold_hsl = st.slider(
            "Limiar HSL Error",
            min_value=0.0,
            max_value=1.0,
            value=0.15,
            step=0.01,
            help="Erro HSL acima deste valor indica doença"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Sobre")
        st.markdown("""
        Esta aplicação usa o método proposto por **Katafuchi e Tokunaga (2020)** 
        para detecção de doenças em plantas baseado em:
        
        - Reconstrução de cor usando pix2pix
        - Análise de diferença de cor (CIEDE2000)
        - Análise de erro HSL
        - Métricas de localização de anomalias
        """)
    
    # Área principal
    st.header("📤 Upload de Imagem")
    
    uploaded_file = st.file_uploader(
        "Faça upload de uma imagem de planta para análise",
        type=['jpg', 'jpeg', 'png'],
        help="Imagem deve ser colorida (RGB)"
    )
    
    if uploaded_file is not None:
        try:
            # Carregar imagem
            original_image = Image.open(uploaded_file).convert("RGB")
            
            # Mostrar imagem original
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagem Original")
                st.image(original_image, use_container_width=True)
            
            # Carregar modelo
            with st.spinner("Carregando modelo..."):
                model, opt, inference_func = load_model(checkpoint_path)
            
            if model is None:
                st.error("⚠️ Modelo não encontrado ou não foi possível carregar.")
                
                # Verificar o que está faltando
                checkpoint_dir = Path(checkpoint_path)
                if not checkpoint_dir.exists():
                    st.warning(f"❌ Diretório não existe: `{checkpoint_path}`")
                else:
                    checkpoints = list(checkpoint_dir.glob("*_net_G.pth"))
                    if not checkpoints:
                        st.warning(f"❌ Nenhum checkpoint encontrado em: `{checkpoint_path}`")
                
                st.info("""
                **Para usar a aplicação, você precisa:**
                
                1. **Treinar o modelo pix2pix** seguindo o notebook:
                   - `notebook/Diagnostico_Katafuchi_Tokunaga.ipynb`
                
                2. **Ter o checkpoint salvo** no diretório:
                   - `checkpoints/ramularia_colorrec_pix2pix/latest_net_G.pth`
                   - Ou atualizar o caminho do checkpoint na barra lateral
                
                3. **Ter o repositório pytorch-CycleGAN-and-pix2pix disponível**:
                   ```bash
                   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
                   ```
                   E garantir que está no mesmo diretório ou no PYTHONPATH.
                
                **Nota:** Sem o modelo, apenas a pré-visualização da imagem será exibida.
                """)
                
                # Mostrar pré-processamento mesmo sem modelo
                gray_img, resized_img = prepare_image_for_inference(original_image)
                with col2:
                    st.subheader("Pré-processamento (Escala de Cinza)")
                    st.image(gray_img, use_container_width=True)
                    
                st.warning("Análise completa requer o modelo treinado.")
            else:
                # Preparar imagem para inferência
                with st.spinner("Processando imagem..."):
                    gray_img, resized_img = prepare_image_for_inference(original_image)
                    
                    # Executar inferência
                    with st.spinner("Executando reconstrução de cor..."):
                        reconstructed_img = inference_func(model, gray_img)
                
                with col2:
                    st.subheader("Imagem Reconstruída")
                    st.image(reconstructed_img, use_container_width=True)
                
                # Realizar análise
                with st.spinner("Calculando métricas..."):
                    metrics = perform_analysis(resized_img, reconstructed_img)
                    diagnosis = classify_disease(metrics, threshold_ciede, threshold_hsl)
                
                # Exibir resultados
                st.header("📊 Resultados da Análise")
                
                # Diagnóstico
                diagnosis_class = "diseased" if diagnosis['FINAL'] == "DOENTE" else "healthy"
                diagnosis_html = f"""
                <div class="diagnosis-box {diagnosis_class}">
                    Diagnóstico Final: {diagnosis['FINAL']}
                </div>
                """
                st.markdown(diagnosis_html, unsafe_allow_html=True)
                
                # Métricas detalhadas
                st.subheader("📈 Métricas Calculadas")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "CIEDE2000 Sum",
                        f"{metrics['ciede_sum']:,.0f}",
                        delta=f"{metrics['ciede_sum'] - threshold_ciede:,.0f}" if metrics['ciede_sum'] != 0 else None,
                        help="Soma total da diferença de cor CIEDE2000 na máscara da folha"
                    )
                    st.metric(
                        "CIEDE2000 Média",
                        f"{metrics['ciede_mean']:.2f}",
                        help="Média do erro de cor CIEDE2000"
                    )
                
                with col2:
                    st.metric(
                        "Top 2% Mean ΔE2000",
                        f"{metrics['top2pct_mean']:.2f}",
                        help="Média dos top 2% maiores erros (métrica de detecção)"
                    )
                    st.metric(
                        "Top 1% Energy Fraction",
                        f"{metrics['top1pct_energy']:.4f}",
                        help="Fração de energia concentrada nos top 1% erros (localização)"
                    )
                
                with col3:
                    st.metric(
                        "HSL Error",
                        f"{metrics['hsl_error']:.4f}",
                        delta=f"{metrics['hsl_error'] - threshold_hsl:.4f}" if metrics['hsl_error'] != 0 else None,
                        help="Erro ponderado no espaço de cor HSV"
                    )
                    confidence_pct = min(100, max(0, diagnosis['confidence'] * 100))
                    st.metric(
                        "Confiança",
                        f"{confidence_pct:.1f}%",
                        help="Confiança no diagnóstico baseada na distância ao limiar"
                    )
                
                # Diagnósticos individuais
                st.subheader("🔍 Diagnósticos por Métrica")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info(f"**Método A (CIEDE2000):** {diagnosis['A']}")
                with col2:
                    st.info(f"**Método B (HSL Error):** {diagnosis['B']}")
                with col3:
                    st.success(f"**Resultado Final:** {diagnosis['FINAL']}")
                
                # Visualizações
                st.subheader("🎨 Visualizações")
                fig = create_visualizations(metrics, gray_img)
                st.pyplot(fig)
                
                # Download dos resultados
                st.subheader("💾 Exportar Resultados")
                
                # Salvar figura
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Baixar Visualizações",
                        data=buf,
                        file_name="analise_planta.png",
                        mime="image/png"
                    )
                
                with col2:
                    # Resumo em texto
                    summary = f"""
DIAGNÓSTICO DE PLANTA
=====================

Resultado: {diagnosis['FINAL']}
Confiança: {confidence_pct:.1f}%

MÉTRICAS:
---------
CIEDE2000 Sum: {metrics['ciede_sum']:,.0f}
CIEDE2000 Média: {metrics['ciede_mean']:.2f}
Top 2% Mean ΔE2000: {metrics['top2pct_mean']:.2f}
Top 1% Energy Fraction: {metrics['top1pct_energy']:.4f}
HSL Error: {metrics['hsl_error']:.4f}

DIAGNÓSTICOS:
-------------
Método A (CIEDE2000): {diagnosis['A']}
Método B (HSL Error): {diagnosis['B']}
Diagnóstico Final: {diagnosis['FINAL']}

LIMIARES UTILIZADOS:
--------------------
CIEDE2000 Threshold: {threshold_ciede:,}
HSL Error Threshold: {threshold_hsl:.2f}
"""
                    st.download_button(
                        label="📄 Baixar Relatório (TXT)",
                        data=summary,
                        file_name="relatorio_diagnostico.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"❌ Erro ao processar imagem: {str(e)}")
            st.exception(e)
    else:
        # Instruções iniciais
        st.info("""
        👆 **Faça upload de uma imagem de planta para começar a análise.**
        
        A aplicação irá:
        1. Converter a imagem para escala de cinza
        2. Reconstruir as cores usando o modelo pix2pix treinado
        3. Comparar a imagem original com a reconstruída
        4. Calcular métricas de anomalia (CIEDE2000, HSL Error)
        5. Classificar a planta como SAUDÁVEL ou DOENTE
        """)


if __name__ == "__main__":
    main()
