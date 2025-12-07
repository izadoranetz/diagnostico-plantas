# Diagnóstico de Doenças em Plantas

Aplicação Streamlit para detecção de anomalias em plantas usando aprendizado profundo, baseada no método proposto por **Katafuchi e Tokunaga (2020)** no artigo "Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection".

## 📋 Sobre

Esta aplicação utiliza um modelo pix2pix para reconstrução de cor de imagens de plantas. Ao comparar a imagem original com a reconstruída, o sistema detecta anomalias (possíveis doenças) através de análises de diferença de cor usando:

- **CIEDE2000**: Métrica de diferença de cor perceptual
- **HSL Error**: Análise de erro no espaço de cor HSV
- **Métricas de localização**: Identificação de regiões específicas com anomalias

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

### 3. Configure o modelo pix2pix

Para usar a aplicação, você precisa ter o modelo pix2pix treinado. Siga estes passos:

#### Opção A: Usando modelo já treinado

1. Certifique-se de ter os checkpoints do modelo treinado em:
   ```
   checkpoints/ramularia_colorrec_pix2pix/
   ```

2. O checkpoint deve conter arquivos como:
   - `latest_net_G.pth` (ou `{epoch}_net_G.pth`)
   - `train_opt.txt` (ou `opt.txt`)

#### Opção B: Treinar o modelo

1. Siga o notebook `notebook/Diagnostico_Katafuchi_Tokunaga.ipynb` para treinar o modelo

2. Ou clone o repositório pytorch-CycleGAN-and-pix2pix:
   ```bash
   git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
   ```

3. Treine o modelo seguindo as instruções do notebook

### 4. Estrutura do projeto

```
diagnostico-plantas/
├── app.py                 # Aplicação Streamlit principal
├── utils.py               # Funções utilitárias de processamento
├── model_utils.py         # Funções para carregar e usar o modelo
├── requirements.txt       # Dependências Python
├── notebook/              # Notebook com treinamento e análise
│   └── Diagnostico_Katafuchi_Tokunaga.ipynb
├── checkpoints/           # Diretório para checkpoints do modelo
│   └── ramularia_colorrec_pix2pix/
└── README.md
```

## 💻 Uso

### Executar a aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`.

### Funcionalidades

1. **Upload de Imagem**: Faça upload de uma imagem de planta (JPG, PNG)
2. **Análise Automática**: A aplicação irá:
   - Converter a imagem para escala de cinza
   - Reconstruir as cores usando o modelo pix2pix
   - Calcular métricas de anomalia
   - Classificar a planta como SAUDÁVEL ou DOENTE
3. **Visualizações**: Veja:
   - Imagem original vs reconstruída
   - Mapa de diferença de cor (CIEDE2000)
   - Mapa de anomalia sobreposto na imagem
4. **Exportar Resultados**: Baixe as visualizações e relatório em texto

### Configurações Ajustáveis

Na barra lateral, você pode ajustar:
- **Caminho do Checkpoint**: Localização do modelo treinado
- **Limiar CIEDE2000**: Threshold para classificação (padrão: 350000)
- **Limiar HSL Error**: Threshold para análise HSL (padrão: 0.15)

## 📊 Métricas Utilizadas

### 1. CIEDE2000 Sum
Soma total da diferença de cor CIEDE2000 na máscara da folha. Valores altos indicam maior diferença entre original e reconstruída.

### 2. Top 2% Mean ΔE2000
Média dos top 2% maiores erros de cor. Útil para detecção de anomalias concentradas.

### 3. Top 1% Energy Fraction
Fração de energia concentrada nos top 1% erros. Proxy para localização da anomalia.

### 4. HSL Error
Erro ponderado no espaço de cor HSV, considerando:
- **Hue (H)**: Mudança de cor (verde → amarelo/marrom = doença)
- **Saturation (S)**: Perda de saturação (planta murcha)
- **Value (V)**: Escurecimento (necrose)

## 🧪 Como Funciona

O algoritmo funciona em três etapas principais:

1. **Reconstrução de Cor**: 
   - A imagem colorida é convertida para escala de cinza
   - O modelo pix2pix (treinado em plantas saudáveis) reconstrói as cores
   - Plantas saudáveis terão reconstrução similar à original
   - Plantas doentes terão diferenças significativas

2. **Análise de Diferença**:
   - Comparação pixel a pixel entre original e reconstruída
   - Cálculo de métricas de diferença de cor
   - Geração de mapas de anomalia

3. **Classificação**:
   - Comparação das métricas com limiares configuráveis
   - Diagnóstico combinado usando múltiplas métricas
   - Geração de confiança no resultado

## 📝 Referências

- Katafuchi, R., & Tokunaga, T. (2020). Image-based Plant Disease Diagnosis with Unsupervised Anomaly Detection. arXiv preprint arXiv:2011.14306.
- [Repositório pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## ⚠️ Notas Importantes

- O modelo precisa ser treinado em imagens de plantas saudáveis para funcionar corretamente
- A qualidade do diagnóstico depende da qualidade do modelo treinado
- Os limiares padrão podem precisar ser ajustados conforme seu dataset específico
- Para melhor precisão, treine o modelo com imagens do mesmo tipo de planta que deseja diagnosticar

## 🐛 Troubleshooting

### Erro: "Modelo não encontrado"
- Verifique se o caminho do checkpoint está correto na barra lateral
- Certifique-se de que os arquivos do checkpoint existem
- Verifique se o repositório pytorch-CycleGAN-and-pix2pix está disponível

### Erro ao importar módulos
- Certifique-se de que todas as dependências foram instaladas: `pip install -r requirements.txt`
- Verifique se o PyTorch está instalado corretamente

### Imagens não processando
- Verifique se a imagem está em formato RGB
- Certifique-se de que o tamanho da imagem é razoável (não muito grande)

## 📄 Licença

Este projeto é baseado em trabalhos acadêmicos e código de código aberto. Consulte as licenças dos projetos originais.
