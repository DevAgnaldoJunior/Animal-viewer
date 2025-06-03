import tensorflow as tf
from tensorflow.keras import layers, models

train_ds = tf.data.Dataset.load("data/preprocessed/train_ds")
val_ds = tf.data.Dataset.load("data/preprocessed/val_ds")
test_ds = tf.data.Dataset.load("data/preprocessed/test_ds")


model = tf.keras.Sequential([ #pilha linear de camadas — onde a saída de uma é a entrada da próxima. Sequential quando o fluxo da rede é direto, sem bifurcações ou múltiplas entradas/saídas.
    layers.Input(shape =(128,128,3)), #formato entrada da imagem e 3 canais de cor RGB

    layers.Conv2D(32, (3,3), activation ='relu', padding = "same"), #Cria 32 filtros (ou kernels) que vão varrer a imagem em janelas de 3x3 pixels.

    layers.MaxPooling2D(pool_size=(2,2)), #reduz a dimensão pela metade e Para cada região da imagem, pega o maior valor (daí o “max”).

    layers.Conv2D(64, (3,3), activation = 'relu', padding="same"),
    layers.MaxPooling2D(pool_size = (2,2)),

    layers.Conv2D(128, (3,3), activation = 'relu', padding="same"),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Dropout(0.3), # Regularização com Dropout (ajuda a evitar overfitting). "desliga" aleatoriamente algumas conexões durante o treino, forçando a rede a não depender demais de alguns neurônios.

    layers.Conv2D(256,(3,3), activation = "relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2,2)),


    layers.Flatten(), #Transforma 3D → vetor 1D

    layers.Dense(256, activation = 'relu'), #	Aprende relações finais

    layers.Dropout(0.5), 

    layers.Dense(10,activation = 'softmax') #Gera saída probabilística
])


model.compile(
    optimizer = "adam", #Qual otimizador usar (como ele ajusta os pesos)
    loss = "categorical_crossentropy" ,#Qual função de perda (mede o erro)
    metrics = ["accuracy"] #Quais métricas acompanhar

)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # para parar quando não estiver mais melhorando:


history = model.fit(
    train_ds,
    validation_data =  val_ds,
    epochs = 30, # passada completa por todos os dados de treino
    callbacks = [callback]
)


test_loss, test_accuracy = model.evaluate(test_ds) # Roda o modelo nos dados de teste

print(f"Acurácia do teste: {test_accuracy * 100:.2f}%")


model.save("model/cnn_animals10.h5")

