import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

VERI_KLASORU = "organize_veri_Seti"
BATCH_SIZE = 48
IMAGE_SIZE = (200, 200)

train_ds = tf.keras.utils.image_dataset_from_directory(
    VERI_KLASORU,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VERI_KLASORU,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb"
)
SINIF_ISIMLERI = train_ds.class_names

train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.prefetch(buffer_size= AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size= AUTOTUNE)
