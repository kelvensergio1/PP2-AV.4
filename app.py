import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Carregar modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_brain_tumor.h5")
    return model

model = load_model()

st.title("Classificação de Tumores Cerebrais")
st.write("Upload de uma imagem de ressonância para identificar o tipo de tumor.")

uploaded_file = st.file_uploader("Envie a imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Imagem carregada", use_column_width=True)

    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)

    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    result = classes[class_idx]

    st.write("### **Resultado da Classificação:**")
    st.write(f"**{result.upper()}**")

