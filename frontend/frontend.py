import streamlit as st
import requests
from PIL import Image  # Importa o PIL para redimensionar

# URL da sua API
API_URL = "http://127.0.0.1:8000/predict"

st.title("Classificador de imagens animais")

# Upload da imagem
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abre e redimensiona a imagem para exibição menor
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((250, 250))  # Reduz a imagem proporcionalmente

    st.image(image, caption="Imagem enviada", use_column_width=False)

    if st.button("Classificar"):
        # Envia o arquivo original para a API (não a imagem redimensionada!)
        uploaded_file.seek(0)  # Garante que o arquivo está no início antes de enviar
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        try:
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Classe prevista: **{result['class']}**")
                st.info(f"Confiança: {result['confidence']}%")
            else:
                st.error("Erro na predição. Verifique a API.")
        except Exception as e:
            st.error(f"Erro ao conectar com a API: {e}")
