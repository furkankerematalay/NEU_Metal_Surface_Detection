from data_loader import train_ds, val_ds
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, RandomFlip, RandomRotation, Dropout, BatchNormalization, RandomContrast
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 1. MODEL MİMARİSİNİN İNŞASI
model = models.Sequential()

# Girdi Kapısı ve Dinamik Ön İşleme (RGB -> Grayscale Dönüşümü)
model.add(Input(shape=(200, 200, 3)))
model.add(RandomFlip("horizontal_and_vertical"))
model.add(RandomRotation(0.2))
model.add(RandomContrast(0.1))
model.add(layers.Rescaling(1./255))
model.add(layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x)))

# 1. Evrişim Bloğu (32 Filtre)
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

# 2. Evrişim Bloğu (64 Filtre)
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# 3. Evrişim Bloğu (128 Filtre)
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# Karar Mekanziası (Global Pooling ve L2 Regülarizasyonlu Dense)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(6, activation='softmax', kernel_regularizer=l2(0.001)))


# 2. DERLEME (COMPİLE)
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 3. AKILLI DONANIM AJANLARI (CALLBACKS)
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

lr_schedular = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# 4. EĞİTİM MARATONU (FİT)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stop, lr_schedular]
)

# 5. ÖĞRENİLEN BEYNİ FİZİKSEL DİSKE KAYDET (MÜHÜRLE)
KAYIT_YOLU = "model_v1.h5"
model.save(KAYIT_YOLU)
print(f"\n[DONANIM] Eğitim tamamlandı. Yapay Zeka beyni kalıcı olarak '{KAYIT_YOLU}' dosyasına mühürlendi.")