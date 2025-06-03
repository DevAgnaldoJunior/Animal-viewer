from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse #Permite retornar um JSON personalizado como resposta da API.
import tensorflow as tf
from tensorflow.keras.models import load_model #Carrega um modelo já treinado
from PIL import Image #Permite abrir, redimensionar, e converter imagens.
import numpy as np #manipulação de vetores, matrizes e tensores.
import os #criar e salvar arquivos
import uuid #Gera IDs únicos para as imagens

app = FastAPI()
model = load_model("model/cnn_animals10.h5")


class_names = [
    "cachorro", "cavalo", "elefante", "borboleta", "galinha",
    "gato", "vaca", "ovelha", "aranha", "esquilo"
]

# Caminho para salvar imagens
upload_folder = "uploads"
os.makedirs(upload_folder, exist_ok=True)

img_size = (128,128)

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Adiciona o "batch"
    return image_array


@app.post("/predict")
async def predict(file:UploadFile = File(...)):
    
    # Salvar a imagem (para reuso futuro) ideia de Continuous Learning
    image_id = str(uuid.uuid4()) #cria um nome único para a imagem salva.
    file_path = os.path.join(upload_folder, f"{image_id}_#{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Reabrir a imagem salva para processar
    image_array = preprocess_image(open(file_path, "rb"))

    #predição
    predictions = model.predict(image_array)
    predict_class = class_names[np.argmax(predictions)] #pega a posição da maior probabilidade (classe prevista).
    confidence = float(np.max(predictions))

    return JSONResponse({
        "class": predict_class,
        "confidence": round(confidence*100,2),
        "saved_as": file_path
    })

#http://127.0.0.1:8000/docs
