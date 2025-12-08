# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime

# ------------------------------------------------
# 1. Carregar modelo treinado (cache para otimizar)
# ------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_brain_tumor.h5")

model = load_model()

# ------------------------------------------------
# 2. Banco de dados SQLite
# ------------------------------------------------
DB_NAME = "interacoes.db"

def criar_tabela():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_hora TEXT,
            nome_arquivo TEXT,
            predicao TEXT,
            probabilidade REAL
        )
    """)
    conn.commit()
    conn.close()

def registrar_interacao(data_hora, nome_arquivo, predicao, prob):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO interacoes (data_hora, nome_arquivo, predicao, probabilidade)
        VALUES (?, ?, ?, ?)
    """, (data_hora, nome_arquivo, predicao, prob))
    conn.commit()
    conn.close()

criar_tabela()

# ------------------------------------------------
# 3. Interface Streamlit
# ------------------------------------------------
st.title("Classificação de Tumores Cerebrais – Projeto PP2")
st.write("Envie uma imagem de ressonância para identificar o tipo de tumor.")

uploaded_file = st.file_uploader("Envie a imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

classes = ["glioma", "meningioma", "notumor", "pituitary"]

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem carregada", width=300)

    # Pré-processamento igual ao treino
    img_resized = img.resize((150, 150))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predição
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    predicao = classes[class_idx]
    confianca = float(predictions[0][class_idx])

    st.subheader("Resultado:")
    st.write(f"**Predição:** {predicao.upper()}")
    st.write(f"**Confiança:** {confianca:.4f}")

    # Registrar interação no banco
    registrar_interacao(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        uploaded_file.name,
        predicao,
        confianca
    )

    st.success("Interação registrada no banco de dados!")

# ------------------------------------------------
# 4. Histórico das interações (opcional)
# ------------------------------------------------
if st.checkbox("Mostrar histórico de interações"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM interacoes ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    st.write(rows)
