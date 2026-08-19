from data_loader import train_ds, val_ds
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, RandomFlip, RandomRotation, Dropout, BatchNormalization, RandomContrast, Lambda
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


base_model = ResNet50(weights ='imagenet', include_top =False, input_shape =(200,200,3))
base_model.trainable= False
model = Sequential()
model.add(Input(shape=(200,200,3)))
model.add(Lambda(lambda x: preprocess_input(x)))


model.add(RandomFlip("horizontal_and_vertical"))
model.add(RandomRotation(0.2))
model.add(RandomContrast(0.1))

model.add(base_model)

model.add(GlobalAveragePooling2D())      # Matrisleri düzleştir
model.add(Dropout(0.3))                  # Karar verirken acele etmesin, ezberlemesin
model.add(Dense(6, activation='softmax'))

model.compile(
    optimizer =Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 15,
    restore_best_weights = True,
    verbose = 1
)

lr_schedular = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor=0.2,
    patience=3,
    min_lr= 0.00001,
    verbose = 1
)

history= model.fit(
    train_ds,
    validation_data= val_ds,
    epochs= 15,
    callbacks =[early_stop,lr_schedular]
)

KAYIT_YOLU = "model_resnet50_v1.keras"
model.save(KAYIT_YOLU)
print(f"\n[DONANIM] ResNet50 entegrasyonu tamamlandı. Mühürlendi: '{KAYIT_YOLU}'")

base_model.trainable = True
fine_tune_at = len(base_model.layers)-50

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Toplam ResNet50 Katman Sayısı: {len(base_model.layers)}")
print(f"Eğitime Açılan Son Katman Sayısı: {len(base_model.layers) - fine_tune_at}")


model.compile(
    optimizer=Adam(learning_rate = 0.00001),
    loss ='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop_fine = EarlyStopping(
    monitor='val_accuracy',
    patience=10,                  # Erken durdurma sabrı
    restore_best_weights=True,
    verbose=1
)

lr_schedular_fine = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor =0.2,
    patience = 2,
    min_lr= 1e-7,
    verbose=1
)

history_fine = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs =20,
    callbacks=[early_stop, lr_schedular]
)

NİHAİ_KAYIT_YOLU = "model_resnet50_fine_tuned.keras"
model.save(NİHAİ_KAYIT_YOLU)
print(f"\n[DONANIM] Fine-Tuning başarıyla tamamlandı. Yeni mühürlü model: '{NİHAİ_KAYIT_YOLU}'")