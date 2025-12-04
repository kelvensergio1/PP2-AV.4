# PP2-AV.4

---
# Classificador de Tumores Cerebrais

Este projeto utiliza um modelo de Deep Learning para classificar imagens de ressonância magnética (MRI) em quatro categorias:

- Glioma
- Meningioma
- Sem Tumor
- Pituitário

O modelo foi treinado no Google Colab e a aplicação final foi construída usando Streamlit, permitindo ao usuário enviar uma imagem e receber a previsão do tipo de tumor.

##  Funcionalidades

- Upload de imagens (JPG/PNG)
- Pré-processamento automático
- Classificação em tempo real
- Interface simples e responsiva
- Registro opcional de interações em banco de dados

##  Estrutura do Projeto
/
├── app.py
├── model_brain_tumor.h5
├── requirements.txt
└── README.md


##  Como executar o projeto

### 1. Instalar dependências

pip install -r requirements.txt

### 2. Executar o aplicativo

streamlit run app.py


O Streamlit abrirá a interface no navegador.

##  Dataset utilizado

O conjunto de dados escolhido contém imagens de tumores cerebrais divididas em quatro classes. Foi usado no treinamento do modelo CNN implementado no Colab.

##  Modelo treinado

- Rede Neural Convolucional (CNN)
- Imagens redimensionadas para 150×150
- Normalização entre 0 e 1
- Acurácia final obtida: ≈ 91%

## Tecnologias utilizadas

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Streamlit

## 👥 Equipe

- Integrantes do grupo: Kelven Sérgio ; Davi Pedro
- Disciplina: PP2 – Projeto Prático


