# Diagnóstico de Doenças em Plantas

Aplicação Streamlit para detecção de anomalias em plantas usando aprendizado profundo, baseada no método proposto por **Katafuchi e Tokunaga (2020)** no artigo "Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection".

**🆕 Sistema refatorado para seguir exatamente a implementação do notebook `diagnostico_plantas.ipynb`**

> 📓 **Notebook de Referência**: O notebook original com o treinamento e implementação completa está disponível no Google Colab:
> [https://colab.research.google.com/drive/1jvj0GIocm_QFgZN2_-LFkDvV9XJLQuNX](https://colab.research.google.com/drive/1jvj0GIocm_QFgZN2_-LFkDvV9XJLQuNX)

## 📋 Sobre

Esta aplicação utiliza um modelo **Pix2Pix U-Net** para reconstrução de cor de imagens de plantas em escala de cinza. Ao comparar a imagem original com a reconstruída, o sistema detecta anomalias (possíveis doenças) através de análises de diferença de cor usando:

- **CIEDE2000**: Métrica de diferença de cor perceptual no espaço LAB
- **HSL Error**: Análise de erro no espaço de cor HSV (Hue, Saturation, Value)
- **Grad-CAM**: Visualização das regiões de atenção do modelo durante a reconstrução
- **Métricas de localização**: Top 2% Mean ΔE e Top 1% Energy para quantificar anomalias

## 🏗️ Arquitetura

O sistema utiliza uma **U-Net Generator** com:
- **Entrada**: Imagem em escala de cinza (1 canal, 256x256)
- **Saída**: Imagem RGB reconstruída (3 canais, 256x256)
- **Encoder**: 8 camadas de downsampling (64→128→256→512×5)
- **Decoder**: 7 camadas de upsampling com skip connections + camada final
- **Dropout**: 0.5 nas primeiras 3 camadas do decoder
- **Ativação**: Tanh na saída ([-1, 1])

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd diagnostico-plantas
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure o modelo

A aplicação procura automaticamente por modelos treinados em:

1. **Prioritário**: `weights/modelo_final.pth` (modelo do notebook)

Certifique-se de ter pelo menos um desses arquivos no diretório `weights/`.

### 4. Estrutura do projeto

```
diagnostico-plantas/
├── app.py                    # Aplicação Streamlit principal
├── model_loader.py           # Carregamento do modelo U-Net
├── inference.py              # Pipeline de inferência
├── gradcam.py                # Grad-CAM para explicabilidade
├── metrics.py                # Cálculo de métricas CIEDE2000 e HSL
├── diagnosis.py              # Lógica de diagnóstico (threshold)
├── requirements.txt          # Dependências Python
├── weights/                  # Modelos treinados
│   ├── modelo_final.pth      # Modelo do notebook (prioritário)
│   └── latest_net_G.pth      # Modelo legacy (fallback)
├── notebook/                 # Notebook de referência
│   └── diagnostico_plantas.ipynb
├── data/                     # Dados de teste
├── REFACTORING_SUMMARY.md    # Documentação da refatoração
├── COMPARISON.md             # Antes vs Depois
└── README.md
```

## 💻 Uso

### Executar a aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`.

### Funcionalidades

1. **Upload de Imagem**: Faça upload de uma imagem de folha (JPG, PNG)
2. **Análise Automática**: A aplicação irá:
   - Converter a imagem RGB para escala de cinza (1 canal)
   - Reconstruir as cores usando o modelo U-Net
   - Calcular métricas de anomalia (CIEDE2000, HSL Error)
   - Gerar visualização Grad-CAM da atenção do modelo
3. **Visualizações**: Veja 5 painéis:
   - **Painel 1**: Entrada em escala de cinza
   - **Painel 2**: Imagem original RGB
   - **Painel 3**: Imagem reconstruída pelo modelo
   - **Painel 4**: Mapa de erro CIEDE2000 (hot colormap)
   - **Painel 5**: Grad-CAM - atenção do modelo durante reconstrução

### Pipeline Completo

```python
from inference import create_inference_engine
from gradcam import GradCAM
from metrics import calculate_all_metrics

# 1. Carregar modelo
model = create_inference_engine('weights/modelo_final.pth')

# 2. Inferir
original, gray, reconstructed, input_tensor = model.reconstruct(image)

# 3. Calcular métricas
mask = leaf_mask_from_rgb(original)
de_map = de2000_map(original, reconstructed)
metrics = calculate_all_metrics(original, reconstructed, mask)

# 4. Gerar Grad-CAM
gradcam = GradCAM(model.model)
heatmap = gradcam.generate_heatmap(input_tensor)
```

## 📊 Métricas Utilizadas

### 1. CIEDE2000 Sum
Soma total da diferença de cor CIEDE2000 na máscara da folha. Métrica perceptual que considera diferenças de cor como humanos as percebem. Valores altos indicam maior diferença entre original e reconstruída.

**Limiar do notebook**: 136759 (score acima indica folha doente)

### 2. Top 2% Mean ΔE2000
Média dos top 2% maiores erros de cor. Útil para detecção de anomalias concentradas em regiões específicas da folha.

### 3. Top 1% Energy Fraction
Fração de energia concentrada nos top 1% erros. Proxy para localização da anomalia - valores altos indicam anomalias bem localizadas.

### 4. HSL Error
Erro ponderado no espaço de cor HSV:
- **50% Hue (H)**: Mudança de cor (verde → amarelo/marrom = doença)
- **35% Saturation (S)**: Perda de saturação (planta murcha)
- **15% Value (V)**: Escurecimento (necrose)

### 5. Grad-CAM
Visualização das regiões onde o modelo concentrou sua atenção durante a reconstrução. Usa mapa de calor (heatmap), as áreas vermelhas indicam alta ativação do modelo.

## 🧪 Como Funciona

O algoritmo funciona em quatro etapas principais:

1. **Conversão para Grayscale**: 
   - A imagem RGB é convertida para escala de cinza (1 canal)
   - Normalizada para [-1, 1] como no treinamento

2. **Reconstrução de Cor**: 
   - O modelo U-Net (treinado em plantas saudáveis) reconstrói as cores RGB
   - Plantas saudáveis terão reconstrução similar à original
   - Plantas doentes terão diferenças significativas devido às cores anômalas

3. **Análise de Diferença**:
   - Comparação pixel a pixel entre original e reconstruída no espaço LAB
   - Cálculo de CIEDE2000 (métrica perceptual de diferença de cor)
   - Geração de mapas de erro e métricas quantitativas

4. **Explicabilidade com Grad-CAM**:
   - Captura das ativações e gradientes da última camada convolucional
   - Geração de mapa de calor mostrando regiões de atenção do modelo
   - Overlay na imagem original para interpretação visual

## 🔧 Detalhes Técnicos

### Modelo U-Net
- **Entrada**: [1, 1, 256, 256] - Grayscale
- **Saída**: [1, 3, 256, 256] - RGB
- **Arquitetura**: 8 layers down + 7 layers up + final
- **Normalização**: BatchNorm em todas camadas exceto down1 e down8
- **Dropout**: 0.5 nas primeiras 3 camadas up (up1, up2, up3)
- **Ativação final**: Tanh (saída em [-1, 1])

### Preprocessing
```python
# Conversão para grayscale
img_gray = img_rgb.convert("L")  # PIL Image (H, W)

# Normalização
tensor = (img_gray / 255.0) * 2.0 - 1.0  # [-1, 1]
tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
```

### Grad-CAM Implementation
```python
class GradCAM:
    def generate_heatmap(self, input_tensor):
        # Forward pass + captura de ativações
        output = model(input_tensor)
        
        # Backward pass + captura de gradientes
        target.backward()
        
        # Combinar: weights = GAP(gradients)
        weights = torch.mean(gradients, dim=[2, 3])
        heatmap = sum(weights * activations)
        
        return relu(normalize(heatmap))
```

## 📚 Documentação Adicional

- **REFACTORING_SUMMARY.md**: Documentação completa das mudanças realizadas
- **COMPARISON.md**: Comparação visual Antes vs Depois da refatoração
- **notebook/diagnostico_plantas.ipynb**: Implementação de referência original

## 📝 Referências

- Katafuchi, R., & Tokunaga, T. (2020). Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection. arXiv preprint arXiv:2011.14306.
- Isola, P., et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.

