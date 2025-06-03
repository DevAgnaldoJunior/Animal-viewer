from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os

# Caminho para o diretório de validação ou teste
val_dir = 'data/split/val'  # ou 'data/test'

# Parâmetros
img_size = (224, 224)
batch_size = 32

# Gerador de dados
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Carrega o modelo
model = load_model("model/cnn_animals10.h5")

# Avalia o modelo
loss, accuracy = model.evaluate(val_generator)
print(f"Acurácia: {accuracy * 100:.2f}%")
