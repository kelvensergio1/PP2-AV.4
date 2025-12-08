# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import os
import io

# ---------- CONFIGURAÇÕES ----------
st.set_page_config(page_title="Classificador Tumores Cerebrais", layout="centered")

DB_PATH = "predictions.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Labels na ordem correta do modelo
labels = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------- CARREGAR MODELO ----------
@st.cache_resource
def load_model_safe():
    try:
        model = tf.keras.models.load_model("model_brain_tumor.h5")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

model = load_model_safe()

# ---------- BANCO DE DADOS ----------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            input_shape TEXT
        )
    """)

    conn.commit()
    conn.close()

def log_prediction(filename, predicted_class, confidence, input_shape, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute(
        "INSERT INTO predictions (timestamp, filename, predicted_class, confidence, input_shape) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(), filename, predicted_class, float(confidence), input_shape)
    )

    conn.commit()
    conn.close()

# Criar tabela se não existir
init_db()

# ---------- INTERFACE ----------
st.title(" Classificador de Tumores Cerebrais (MRI)")
st.markdown("Envie uma imagem de ressonância magnética para classificação. Todas as predições serão salvas em `predictions.db`.")

uploaded_file = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Pré-processamento
def preprocess_image(img_pil):
    img = img_pil.convert("RGB").resize((150, 150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Quando o usuário envia uma imagem
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    st.image(img, caption="Imagem enviada", width=400)

    # Salvar imagem localmente
    fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    save_path = os.path.join(UPLOAD_DIR, fname)
    img.save(save_path)

    if model is None:
        st.error(" Modelo não encontrado. Verifique se model_brain_tumor.h5 está na mesma pasta do app.py.")
    else:
        if st.button("Classificar"):
            x = preprocess_image(img)
            preds = model.predict(x)

            idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            predicted_class = labels[idx]

            st.success(f"### Resultado: **{predicted_class.upper()}**")
            st.write(f"Confiança: **{confidence*100:.2f}%**")

            # Log no banco
            log_prediction(fname, predicted_class, confidence, str(x.shape))

            st.info(" Predição salva no banco de dados.")

# ---------- VISUALIZAR LOGS ----------
st.markdown("---")

if st.button("Mostrar últimas 10 predições"):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, filename, predicted_class, confidence FROM predictions ORDER BY id DESC LIMIT 10"
    ).fetchall()
    conn.close()

    if rows:
        for r in rows:
            st.write(f" {r[0]} —  {r[1]} —  {r[2]} —  {r[3]*100:.2f}%")
    else:
        st.info("Nenhuma predição registrada ainda.")
