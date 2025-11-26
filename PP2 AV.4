import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("model_brain_tumor.h5")

st.title("Classificação de Tumores Cerebrais")

uploaded_file = st.file_uploader("Envie uma imagem de MRI", type=["jpg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    st.write("Predição:", labels[class_index])
