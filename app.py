# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import os
import io

# ---------- config ----------
st.set_page_config(page_title="Classificador Tumores Cerebrais", layout="centered")
DB_PATH = "predictions.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- carregar modelo ----------
@st.cache_resource
def load_model_safe():
    try:
        m = tf.keras.models.load_model("model_brain_tumor.h5")
        return m
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None

model = load_model_safe()
labels = ["glioma", "meningioma", "notumor", "pituitary"]

# ---------- banco sqlite (criação) ----------
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            input_shape TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(filename, predicted_class, confidence, input_shape, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('INSERT INTO predictions (timestamp, filename, predicted_class, confidence, input_shape) VALUES (?,?,?,?,?)',
              (datetime.now().isoformat(), filename, predicted_class, float(confidence), input_shape))
    conn.commit()
    conn.close()

init_db()

# ---------- UI ----------
st.title(" Classificador de Tumores Cerebrais (MRI)")
st.markdown("Envie uma imagem de ressonância magnética para classificação. Cada predição será registrada no banco `predictions.db`.")

uploaded_file = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg","jpeg","png"])

def preprocess_pil(img_pil):
    img = img_pil.convert("RGB").resize((150,150))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

if uploaded_file is not None:
    # show
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(img, caption="Imagem enviada", width=400)

    # save locally
    fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    local_path = os.path.join(UPLOAD_DIR, fname)
    img.save(local_path)

    if model is None:
        st.error("Modelo não carregado. Verifique model_brain_tumor.h5 na mesma pasta do app.py.")
    else:
        if st.button("Classificar"):
            x = preprocess_pil(img)
            preds = model.predict(x)
            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = labels[idx]
            st.success(f"Classe predita: **{label}** — Confiança: **{conf*100:.2f}%**")
            # salva log no DB
            input_shape = str(x.shape)
            log_prediction(fname, label, conf, input_shape)
            st.write("✅ Predição registrada no banco de dados (predictions.db).")

# botão para mostrar logs
if st.button("Mostrar últimas predições (10)"):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT timestamp, filename, predicted_class, confidence FROM predictions ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    if rows:
        for r in rows:
            st.write(f"{r[0]} — {r[1]} — {r[2]} — {r[3]*100:.2f}%")
    else:
        st.info("Ainda não há predições registradas.")
