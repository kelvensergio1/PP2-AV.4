import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from database import criar_tabela, registrar_interacao
from datetime import datetime

# ------------------------------------------------
# 1. Carregar modelo treinado
# ------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = CNN()
model.load_state_dict(torch.load("modelo_treinado.pth", map_location="cpu"))
model.eval()

# ------------------------------------------------
# 2. Transforms
# ------------------------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# ------------------------------------------------
# 3. Criar tabela no banco ao iniciar o app
# ------------------------------------------------
criar_tabela()

# ------------------------------------------------
# 4. Interface Streamlit
# ------------------------------------------------
st.title("Classificador de Imagens – Projeto PP2")

uploaded = st.file_uploader("Envie uma imagem JPG ou PNG", type=["jpg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagem enviada", width=200)

    # Pré-processamento
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)
        conf, predicted = torch.max(prob, 1)

    classes = ["Classe 0", "Classe 1"]
    predicao = classes[predicted.item()]
    confianca = conf.item()

    st.subheader("Resultado:")
    st.write(f"**Predição:** {predicao}")
    st.write(f"**Confiança:** {confianca:.4f}")

    # ------------------------------------------------
    # 5. Registrar interação no banco
    # ------------------------------------------------
    registrar_interacao(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        uploaded.name,
        predicao,
        float(confianca)
    )

    st.success("Interação registrada no banco de dados!")
