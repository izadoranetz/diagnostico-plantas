import streamlit as st

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
    </style>
""", unsafe_allow_html=True)

# Função principal da aplicação
def main():
    # Header
    st.markdown('<h1 class="main-header">🌿 Diagnóstico de Doenças em Plantas</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detecção de anomalias usando pix2pix e CIEDE2000</p>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()