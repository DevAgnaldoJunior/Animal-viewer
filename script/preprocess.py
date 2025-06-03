import tensorflow as tf
from tensorflow.keras import layers

train_dir = "data/split/train"
val_dir = "data/split/val"
test_dir = "data/split/test"

tamanho_imagem = (128,128)
batch_size = 32 # número de imagens que serão processadas juntas por vez (um "lote").

train_ds = tf.keras.utils.image_dataset_from_directory( #carrega as imagens de cada pasta de classe.
    train_dir,
    image_size = tamanho_imagem,
    batch_size = batch_size,
    label_mode = "categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size = tamanho_imagem,
    batch_size = batch_size,
    label_mode = "categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size = tamanho_imagem,
    batch_size = batch_size,
    label_mode = "categorical"
)

normalization_layer = layers.Rescaling(1./255) #Imagens vêm com pixels entre 0 e 255. Aqui você normaliza para valores entre 0 e 1


data_augmentation = tf.keras.Sequential([ #gera novas variações da imagem (usado apenas na base de treinos)
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y:(data_augmentation(x,training =True), y)) #x=imagem, y= rotulo - map() função que aplica uma transformação a cada item do dataset.
train_ds = train_ds.map(lambda x,y :(normalization_layer(x),y))

val_ds = val_ds.map(lambda x,y:(normalization_layer(x),y))

test_ds = test_ds.map(lambda x,y:(normalization_layer(x), y))

#salvando os dados processados
tf.data.experimental.save(train_ds, "data/preprocessed/train_ds")
tf.data.experimental.save(val_ds, "data/preprocessed/val_ds")
tf.data.experimental.save(test_ds, "data/preprocessed/test_ds")

